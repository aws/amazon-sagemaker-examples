
import torch
from torch.utils.data import DataLoader
from transformers import (
    default_data_collator
)
import os

def create_dataloaders(train_dataset,eval_dataset,rank,world_size,seed,train_batch_size,eval_batch_size):
    
    train_sampler = torch.utils.data.DistributedSampler(
                train_dataset,
                shuffle=True,
                seed=seed,
                rank=rank,
                num_replicas=world_size,
                drop_last=True,
            )
    
    eval_sampler = torch.utils.data.DistributedSampler(
                eval_dataset,
                shuffle=True,
                seed=seed,
                rank=rank,
                num_replicas=world_size,
                drop_last=True,
            )

    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, collate_fn=default_data_collator, batch_size=train_batch_size, pin_memory=True,drop_last=True
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,sampler=eval_sampler, collate_fn=default_data_collator, batch_size=eval_batch_size, pin_memory=True,drop_last=True

    )

    return train_dataloader,eval_dataloader

def save_model(model, tokenizer, output_dir,rank):
    """Helper method to save model when using FSDP."""

    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        FullStateDictConfig,
        StateDictType,
    )
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state_dict = model.state_dict()
        """ 
            To save optimizer states as well in case you want to resume training
            on the same dataset, see: https://pytorch.org/docs/stable/fsdp.html
        """
        if rank==0:
            torch.save(cpu_state_dict,os.path.join(output_dir,"model_weights.pt"))
            tokenizer.save_pretrained(output_dir)