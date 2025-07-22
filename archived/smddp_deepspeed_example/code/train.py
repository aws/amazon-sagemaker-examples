import argparse
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, get_scheduler, SchedulerType
from datasets import load_from_disk
import torch
import torch.distributed as dist

from utils import create_dataloaders, StubDataset
import functools
import deepspeed
try:
    backend = "smddp"
    import smdistributed.dataparallel.torch.torch_smddp
except ModuleNotFoundError:
    backend = "nccl"
    print("Warning: SMDDP not found on this image, falling back to NCCL!")

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--model_id",
    type=str,
    default="meta-llama/Llama-2-7b-chat-hf",
    help="Model id to use for training.",
  )
  parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train for.")
  parser.add_argument("--max_steps", type=int, default=None, help="Number of epochs to train for.")
  parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="Batch size to use for training.",
  )
  parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate to use for training.")
  parser.add_argument("--optimizer", type=str, default="adamw_hf", help="Learning rate to use for training.")
  parser.add_argument("--seed", type=int, default=42, help="Seed to use for training.")
  parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")

  parser.add_argument(
    "--gradient_checkpointing",
    type=bool,
    default=True,
    help="Whether to use gradient checkpointing to save memory.",
  )
  parser.add_argument(
    "--bf16",
    type=bool,
    default=True if torch.cuda.get_device_capability()[0] == 8 else False,
    help="Whether to use bf16.",
  )
  parser.add_argument(
    "--max_train_steps",
    type=int,
    default=None,
    help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
  )
  parser.add_argument(
    "--learning_rate",
    type=float,
    default=5e-5,
    help="Initial learning rate (after the potential warmup period) to use.",
  )
  parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
  )
  parser.add_argument(
    "--lr_scheduler_type",
    type=SchedulerType,
    default="linear",
    help="The scheduler type to use.",
    choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
  )
  parser.add_argument(
    "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
  )
  parser.add_argument(
    "--deepspeed_config", type=str, help="Path to deepspeed config json"
  )

  parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
  parser.add_argument("--model_dir",type=str,default="/opt/ml/model")
  parser.add_argument("--cache_dir",type=str,default=None)
  args = parser.parse_known_args()
  return args

def training_function(args):
  # smddp example specifically tailored for p4d(e) instance types.
  local_rank = dist.get_rank() % 8
  seed = args.seed
  set_seed(seed)
  torch.cuda.set_device(local_rank)

  dataset = {
    'train': StubDataset(),
    'validation': StubDataset()
  }
    
  dtype = torch.bfloat16

  from transformers import LlamaConfig
  configuration = LlamaConfig(use_cache=False)
  from transformers.models.llama import LlamaForCausalLM
  with deepspeed.zero.Init(dtype=dtype, enabled=True):
    model = AutoModelForCausalLM.from_config(configuration)
  model.gradient_checkpointing_enable()

  train_dataset = dataset["train"]
  eval_dataset = dataset["validation"]
  train_dataloader, eval_dataloader = create_dataloaders(
    train_dataset, eval_dataset, dist.get_rank(), dist.get_world_size(), 
    seed, args.batch_size, args.batch_size)
 
  no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
  optimizer_grouped_parameters = [{
      "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
      "weight_decay": args.weight_decay,
    },{
      "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
      "weight_decay": 0.0,
    }] 

  optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

   # Scheduler and math around the number of training steps.
  overrode_max_train_steps = False
  num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
  if dist.get_rank()==0:
    print(f"Number of update steps per epoch {num_update_steps_per_epoch}")
  if args.max_train_steps is None:
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True

  lr_scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
    num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
  )

  model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    model_parameters=model.parameters(),
    config=args.deepspeed_config
  )
  device = torch.device(f"cuda:{local_rank}")
  for epoch in range(args.num_train_epochs):
    model.train()
    total_steps=0
    ds_loss = torch.zeros(2).to(local_rank)

    for batch_idx, batch in enumerate(train_dataloader):
      batch = {k: v.to(device) for k, v in batch.items()}  
      output = model(**batch)
      if dist.get_rank() == 0: print(f"Processing training batch {batch_idx}")
      loss = output["loss"]
      loss.backward()
      ds_loss[0] += loss.item()
      ds_loss[1] += len(batch["input_ids"])
      optimizer.zero_grad()
      lr_scheduler.step()
      total_steps += 1
      if args.max_steps is not None and total_steps > args.max_steps:
        break
    
    torch.distributed.all_reduce(ds_loss, op=torch.distributed.ReduceOp.SUM)
    train_loss = ds_loss[0] / ds_loss[1]
    train_ppl = torch.exp(train_loss)

    if dist.get_rank()==0:
      print(f"******{epoch=}: {train_ppl=} {train_loss=}******")
    
    model.eval()
    eval_loss = 0
    ds_eval_loss = torch.zeros(2).to(local_rank)
    for steps, batch in enumerate(eval_dataloader):
      batch = {k: v.to(device) for k, v in batch.items()}

      if dist.get_rank() == 0: print(f"Performing validation on training batch {batch_idx}")
      with torch.no_grad():
        outputs = model(**batch)
      loss = outputs["loss"]
      ds_eval_loss[0] += loss.item()
      ds_eval_loss[1] += len(batch["input_ids"])
      if args.max_steps is not None and steps > args.max_steps:
        break

    torch.distributed.all_reduce(ds_eval_loss, op=torch.distributed.ReduceOp.SUM)
    eval_loss = ds_eval_loss[0] / ds_eval_loss[1]
    eval_ppl = torch.exp(eval_loss)

    if dist.get_rank()==0:
      print(f"*******{epoch=}: {eval_ppl=} {eval_loss=}*******")
    
    if args.max_steps is not None and total_steps > args.max_steps:
        break

  if dist.get_rank() == 0:
    print("Training done!")
  dist.barrier()

def main():
  deepspeed.init_distributed(dist_backend=backend)  

  args, _ = parse_args()
  training_function(args)

if __name__ == "__main__":
  main()
