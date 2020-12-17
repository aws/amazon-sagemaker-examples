# coding=utf-8
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Modifications Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# ==================
import csv
import os
import sys
import time
import argparse
import random
import h5py
from tqdm import tqdm, trange
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
import math
from apex import amp
import multiprocessing
import pprint
import json
import subprocess

import modeling
from schedulers import PolyWarmUpScheduler

from utils import format_step, get_world_size, get_rank
from utils import is_main_process
from apex.parallel import DistributedDataParallel as DDP
from smdistributed.modelparallel.torch.optimizers import FusedLAMB
from schedulers import LinearWarmUpScheduler
from apex.parallel.distributed import flat_dist_call
import amp_C
import apex_C
from apex.amp import _amp_state
# SMP: Import smp library
import smdistributed.modelparallel.torch as smp
import configparser

from concurrent.futures import ProcessPoolExecutor

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

skipped_steps = 0

# Track whether a SIGTERM (cluster time up) has been handled
timeout_sent = False

import signal
# handle SIGTERM sent from the scheduler and mark so we
# can gracefully save & exit
def signal_handler(sig, frame):
    global timeout_sent
    timeout_sent = True

signal.signal(signal.SIGTERM, signal_handler)


#Workaround because python functions are not picklable
class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed
    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)

def create_pretraining_dataset(input_file, max_pred_length, shared_list, args, worker_init):
    train_data = pretraining_dataset(input_file=input_file, max_pred_length=max_pred_length)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=args.train_batch_size * args.n_gpu,
                                  num_workers=4, worker_init_fn=worker_init,
                                  pin_memory=True)
    return train_dataloader, input_file


class pretraining_dataset(Dataset):

    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids',
                'next_sentence_labels']
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs[0])

    def __getitem__(self, index):
        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [
            torch.from_numpy(input[index].astype(np.int64)) if indice < 5 else torch.from_numpy(
                np.asarray(input[index].astype(np.int64))) for indice, input in enumerate(self.inputs)]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero(as_tuple=False)
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]

        return [input_ids, segment_ids, input_mask,
                masked_lm_labels, next_sentence_labels]

class BertPretrainingCriterion(torch.nn.Module):
    def __init__(self, vocab_size):
        super(BertPretrainingCriterion, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size
    def forward(self, prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels):
        masked_lm_loss = self.loss_fn(prediction_scores.view(-1, self.vocab_size), masked_lm_labels.view(-1))
        next_sentence_loss = self.loss_fn(seq_relationship_score.view(-1, 2), next_sentence_labels.view(-1))
        total_loss = masked_lm_loss + next_sentence_loss
        return total_loss


def aws_s3_sync(source, destination):
    """aws s3 sync in quiet mode and time profile"""
    import time, subprocess
    cmd = ["aws", "s3", "sync", "--quiet", source, destination]
    print(f"Syncing files from {source} to {destination}")
    start_time = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()
    end_time = time.time()
    print("Time Taken to Sync: ", (end_time-start_time))
    return

def sync_local_checkpoints_to_s3(local_path, s3_path):
    """ sample function to sync checkpoints from local path to s3 """

    import boto3, botocore
    #check if local path exists
    if not os.path.exists(local_path):
        raise RuntimeError("Provided local path {local_path} does not exist. Please check")

    #check if s3 bucket exists
    s3 = boto3.resource('s3')
    if 's3://' not in s3_path:
        raise ValueError("Provided s3 path {s3_path} is not valid. Please check")

    s3_bucket = s3_path.replace('s3://','').split('/')[0]
    print(f"Destination S3 Bucket which will store model artifacts: {s3_bucket}")
    try:
        s3.meta.client.head_bucket(Bucket=s3_bucket)
    except botocore.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            raise RuntimeError('S3 bucket does not exist. Please check')
    aws_s3_sync(local_path, s3_path)
    return

def sync_s3_checkpoints_to_local(local_path, s3_uri):
    """ sample function to sync checkpoints from s3 to local path """

    import boto3
    #try to create path if it does not exist
    if not os.path.exists(local_path):
        print(f"Provided local path {local_path} does not exist. Creating...")
        try:
            os.makedirs(local_path)
        except Exception as e:
            raise RuntimeError(f"failed to create {local_path}")

    #check if s3 bucket exists
    s3 = boto3.resource('s3')
    if not s3_uri.startswith("s3://"):
        raise ValueError("Provided s3 uri {s3_uri} is not valid. Please check")

    s3_bucket = s3_uri.replace('s3://','').split('/')[0]
    print(f"S3 Bucket: {s3_bucket}")
    try:
        s3.meta.client.head_bucket(Bucket=s3_bucket)
    except Exception as e:
        raise e
    aws_s3_sync(s3_uri, local_path)
    return

def parse_arguments():

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain .hdf5 files  for the task.")

    parser.add_argument("--config_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The BERT model config")

    parser.add_argument("--bert_model", default="bert-large-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="The initial checkpoint to start training from.")

    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_predictions_per_seq",
                        default=80,
                        type=int,
                        help="The maximum total of masked tokens in input sequence")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps",
                        default=1000,
                        type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.01,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=os.getenv('LOCAL_RANK', -1),
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=4211,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=0,
                        type=int,
                        help="Mixed precision training")
    parser.add_argument('--amp',
                        default=0,
                        type=int,
                        help="Mixed precision training")
    parser.add_argument('--loss_scale',
                        type=float, default=0.0,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--log_freq',
                        type=float, default=1.0,
                        help='frequency of logging loss.')
    parser.add_argument('--checkpoint_activations',
                        default=0,
                        type=int,
                        help="Whether to use gradient checkpointing")
    parser.add_argument("--resume_from_checkpoint",
                        default=0,
                        type=int,
                        help="Whether to resume training from checkpoint.")
    parser.add_argument("--s3_checkpoint_uri",
                        default=None,
                        type=str,
                        help="S3 Checkpoint URI")
    parser.add_argument('--resume_step',
                        type=int,
                        default=-1,
                        help="Step to resume training from.")
    parser.add_argument('--num_steps_per_checkpoint',
                        type=int,
                        default=100,
                        help="Number of update steps until a model checkpoint is saved to disk.")
    parser.add_argument('--skip_checkpoint',
                        default=0,
                        type=int,
                        help="Whether to save checkpoints")
    parser.add_argument('--phase2',
                        default=0,
                        type=int,
                        help="Whether to train with seq len 512")
    parser.add_argument('--allreduce_post_accumulation',
                        default=0,
                        type=int,
                        help="Whether to do allreduces during gradient accumulation steps.")
    parser.add_argument('--allreduce_post_accumulation_fp16',
                        default=0,
                        type=int,
                        help="Whether to do fp16 allreduce post accumulation.")
    parser.add_argument('--phase1_end_step',
                        type=int,
                        default=7038,
                        help="Number of training steps in Phase1 - seq len 128")
    parser.add_argument('--init_loss_scale',
                        type=int,
                        default=2**20,
                        help="Initial loss scaler value")
    parser.add_argument("--do_train",
                        default=0,
                        type=int,
                        help="Whether to run training.")
    parser.add_argument("--use_env",
                        default=0,
                        type=int,
                        help="Whether to read local rank from ENVVAR")
    parser.add_argument('--disable_progress_bar',
                        default=0,
                        type=int,
                        help='Disable tqdm progress bar')
    parser.add_argument('--steps_this_run', type=int, default=1000,
                        help='If provided, only run this many steps before exiting')
    parser.add_argument('--ddp', type=int, default=0)
    parser.add_argument('--smp', type=int, default=0)
    parser.add_argument('--num_microbatches', type=int, default=1)
    parser.add_argument('--pipeline', type=str, default='interleaved')
    parser.add_argument('--apply_optimizer', type=int, default=1)
    parser.add_argument('--use_sequential', type=int, default=0)
    parser.add_argument('--param_weight', type=float, default=0.3)
    parser.add_argument('--overlapping_allreduce', type=int, default=1)
    parser.add_argument('--save-full',
                        default=0,
                        type=int,
                        help='Save the full model')
    parser.add_argument('--partial-checkpoint', type=bool, default=True)
    parser.add_argument('--mp_parameters', type=str, default='')
    
    args = parser.parse_args()

    args.fp16 = args.fp16 or args.amp

    if args.resume_from_checkpoint:
        args.steps_this_run += 100

    if args.steps_this_run < 0:
        args.steps_this_run = args.max_steps
        
    return args

def setup_training(args):

    assert (torch.cuda.is_available())

    if args.smp > 0:
        # Initialize SMP. The configuration is obtained from the parameters passed to
        # the Sagemaker PyTorch estimator.
        smp.init()

    # SMP: Set the device to the GPU ID used by the current process.
    # Input tensors should be transferred to this device.
    torch.cuda.set_device(smp.local_rank())
    device = torch.device("cuda", smp.local_rank())
    args.n_gpu = 1

    #if args.local_rank == -1:
    #    device = torch.device("cuda")
    #    args.n_gpu = torch.cuda.device_count()
    #    args.allreduce_post_accumulation = False
    #    args.allreduce_post_accumulation_fp16 = False
    #else:
    #    torch.cuda.set_device(args.local_rank)
    #    device = torch.device("cuda", args.local_rank)
    #    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    #    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    #    args.n_gpu = 1

    if args.gradient_accumulation_steps == 1:
        args.allreduce_post_accumulation = False
        args.allreduce_post_accumulation_fp16 = False

    print("device: {} n_gpu: {}, mp_rank: {}, rank: {}, distributed training: {}, 16-bits training: {}".format(
        device, args.n_gpu, smp.mp_rank(), smp.rank(), bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, batch size {} should be divisible".format(
            args.gradient_accumulation_steps, args.train_batch_size))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    if not args.resume_from_checkpoint and os.path.exists(args.output_dir) and (
            os.listdir(args.output_dir) and any([i.startswith('ckpt') for i in os.listdir(args.output_dir)])):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    if (not args.resume_from_checkpoint or not os.path.exists(args.output_dir)) and is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)

    return device, args

def prepare_model_and_optimizer(args, device):

    # Prepare model
    config = modeling.BertConfig.from_json_file(args.config_file)

    # Padding for divisibility by 8
    if config.vocab_size % 8 != 0:
        config.vocab_size += 8 - (config.vocab_size % 8)

    if args.use_sequential > 0:
        config.use_sequential = True
    else:
        config.use_sequential = False

    modeling.ACT2FN["bias_gelu"] = modeling.bias_gelu_training
    model = modeling.BertForPreTraining(config)
    model.checkpoint_activations(args.checkpoint_activations)
    if args.smp > 0:
        # SMP: Use the DistributedModel container to provide the model
        # to be partitioned across different ranks. For the rest of the script,
        # the returned DistributedModel object should be used in place of
        # the model provided for DistributedModel class instantiation.
        model = smp.DistributedModel(model)

    checkpoint = None
    if not args.resume_from_checkpoint:
        global_step = 0
    else:
        if not args.init_checkpoint:
            if not args.s3_checkpoint_uri:
                raise ValueError("Need to set s3_checkpoint_uri, if init_checkpoint not set")
            if smp.local_rank() == 0:
                sync_s3_checkpoints_to_local(args.output_dir, args.s3_checkpoint_uri)
            smp.barrier()
        if args.resume_step == -1 and not args.init_checkpoint:
            model_names = [f for f in os.listdir(args.output_dir) if ".pt" in f]
            args.resume_step = max([int(x.split('.pt')[0].split('_')[1].strip()) for x in model_names])

        global_step = args.resume_step if not args.init_checkpoint else 0

        # SMP: Load a model that was saved with smp.save
        if not args.init_checkpoint:
            checkpoint = smp.load(os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step)), partial=args.partial_checkpoint)
        else:
            checkpoint = smp.load(args.init_checkpoint)

        model.load_state_dict(checkpoint['model'], strict=False)

        if args.phase2 and not args.init_checkpoint:
            global_step -= args.phase1_end_step
        if is_main_process():
            print("resume step from ", args.resume_step)

    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = FusedLAMB(optimizer_grouped_parameters,
                          lr=args.learning_rate)
    if args.smp > 0:
        # SMP: Use Distributed Optimizer which allows the loading of optimizer state for a distributed model
        # Also provides APIs to obtain local optimizer state for the current mp_rank.
        optimizer = smp.DistributedOptimizer(optimizer)
    lr_scheduler = PolyWarmUpScheduler(optimizer,
                                       warmup=args.warmup_proportion,
                                       total_steps=args.max_steps)

    if args.fp16:
        if args.loss_scale == 0:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale="dynamic", cast_model_outputs=torch.float16)
        else:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=args.loss_scale, cast_model_outputs=torch.float16)
        amp._amp_state.loss_scalers[0]._loss_scale = args.init_loss_scale


    if args.resume_from_checkpoint:
        if args.phase2 or args.init_checkpoint:
            keys = list(checkpoint['optimizer']['state'].keys())
            #Override hyperparameters from previous checkpoint
            for key in keys:
                checkpoint['optimizer']['state'][key]['step'] = global_step
            for iter, item in enumerate(checkpoint['optimizer']['param_groups']):
                checkpoint['optimizer']['param_groups'][iter]['step'] = global_step
                checkpoint['optimizer']['param_groups'][iter]['t_total'] = args.max_steps
                checkpoint['optimizer']['param_groups'][iter]['warmup'] = args.warmup_proportion
                checkpoint['optimizer']['param_groups'][iter]['lr'] = args.learning_rate
        optimizer.load_state_dict(checkpoint['optimizer'])  # , strict=False)
        # Restore AMP master parameters
        if args.fp16:
            optimizer._lazy_init_maybe_master_weights()
            optimizer._amp_stash.lazy_init_called = True
            optimizer.load_state_dict(checkpoint['optimizer'])
            for param, saved_param in zip(amp.master_params(optimizer), checkpoint['master params']):
                param.data.copy_(saved_param.data)

    #if args.local_rank != -1:
    #    if not args.allreduce_post_accumulation:
    #        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())
    #    else:
    #        flat_dist_call([param.data for param in model.parameters()], torch.distributed.broadcast, (0,) )
    #elif args.n_gpu > 1:
    #    model = torch.nn.DataParallel(model)

    criterion = BertPretrainingCriterion(config.vocab_size)

    return model, optimizer, lr_scheduler, checkpoint, global_step, criterion

def take_optimizer_step(args, optimizer, model, overflow_buf, global_step):

    global skipped_steps
    if args.allreduce_post_accumulation:
        # manually allreduce gradients after all accumulation steps
        # check for Inf/NaN
        # 1. allocate an uninitialized buffer for flattened gradient
        loss_scale = _amp_state.loss_scalers[0].loss_scale() if args.fp16 else 1
        master_grads = [p.grad for p in amp.master_params(optimizer) if p.grad is not None]
        flat_grad_size = sum(p.numel() for p in master_grads)
        allreduce_dtype = torch.float16 if args.allreduce_post_accumulation_fp16 else torch.float32
        flat_raw = torch.empty(flat_grad_size, device='cuda', dtype=allreduce_dtype)
        # 2. combine unflattening and predivision of unscaled 'raw' gradient
        allreduced_views = apex_C.unflatten(flat_raw, master_grads)
        overflow_buf.zero_()
        amp_C.multi_tensor_scale(65536,
            overflow_buf,
            [master_grads, allreduced_views],
            loss_scale / (get_world_size() * args.gradient_accumulation_steps))
        # 3. sum gradient across ranks. Because of the predivision, this averages the gradient
        torch.distributed.all_reduce(flat_raw)
        # 4. combine unscaling and unflattening of allreduced gradient
        overflow_buf.zero_()
        amp_C.multi_tensor_scale(65536,
            overflow_buf,
            [allreduced_views, master_grads],
            1./loss_scale)
        # 5. update loss scale
        if args.fp16:
            scaler = _amp_state.loss_scalers[0]
            old_overflow_buf = scaler._overflow_buf
            scaler._overflow_buf = overflow_buf
            had_overflow = scaler.update_scale()
            scaler._overfloat_buf = old_overflow_buf
        else:
            had_overflow = 0
        # 6. call optimizer step function
        if had_overflow == 0:
            optimizer.step()
            global_step += 1
        else:
            # Overflow detected, print message and clear gradients
            skipped_steps += 1
            if is_main_process():
                scaler = _amp_state.loss_scalers[0]
            if _amp_state.opt_properties.master_weights:
                for param in optimizer._amp_stash.all_fp32_from_fp16_params:
                    param.grad = None
        for param in model.parameters():
            param.grad = None
    else:
        if args.apply_optimizer > 0:
            optimizer.step()
        # optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None
        global_step += 1

    return global_step

# SMP: Define smp step. Pass the necessary arguments for the train_step call
# and return any tensors needed outside
@smp.step
def smp_step(args, device, input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels, model, optimizer, criterion, step):
    rval = train_step(args, device, input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels, model, optimizer, criterion, step)
    return rval

def train_step(args, device, input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels, model, optimizer, criterion, step):
    prediction_scores, seq_relationship_score = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

    loss = criterion(prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels)
    if args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu.

    # divisor = args.gradient_accumulation_steps
    # if args.gradient_accumulation_steps > 1:
    #    if not args.allreduce_post_accumulation:
    #         this division was merged into predivision
    #        loss = loss / args.gradient_accumulation_steps
    #        divisor = 1.0
    if args.fp16:
        with amp.scale_loss(loss, optimizer, delay_overflow_check=args.allreduce_post_accumulation) as scaled_loss:
            model.backward(scaled_loss)
    else:
        if args.smp > 0:
            model.backward(loss)
        else:
            loss.backward()
    return loss

def test_step(args, device, input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels, model, criterion, step):
    prediction_scores, seq_relationship_score = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
    loss = criterion(prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels)
    if args.n_gpu > 1:
        loss = loss.mean()
    return loss.item()

def main():
    global timeout_sent

    args = parse_arguments()

    random.seed(args.seed + args.local_rank)
    np.random.seed(args.seed + args.local_rank)
    torch.manual_seed(args.seed + args.local_rank)
    torch.cuda.manual_seed(args.seed + args.local_rank)
    worker_init = WorkerInitObj(args.seed + args.local_rank)

    device, args = setup_training(args)

    # Prepare optimizer
    model, optimizer, lr_scheduler, checkpoint, global_step, criterion = prepare_model_and_optimizer(args, device)

    raw_train_start = None
    most_recent_ckpts_paths = []
    average_loss = 0.0  # averaged loss every args.log_freq steps
    epoch = 0
    training_steps = 0
    test_losses = []

    pool = ProcessPoolExecutor(1)

    # Note: We loop infinitely over epochs, termination is handled via iteration count
    while True:
        thread = None
        restored_data_loader = None
        if not args.resume_from_checkpoint or epoch > 0 or (args.phase2 and global_step < 1) or args.init_checkpoint:
            files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if
                    os.path.isfile(os.path.join(args.input_dir, f)) and 'training' in f]
            files.sort()
            num_files = len(files)
            random.Random(args.seed + epoch).shuffle(files)
            f_start_id = 0
        else:
            f_start_id = checkpoint['files'][0]
            files = checkpoint['files'][1:]
            args.resume_from_checkpoint = False
            num_files = len(files)
            # may not exist in all checkpoints
            epoch = checkpoint.get('epoch', 0)
            restored_dataloader = checkpoint.get('data_loader', None)

        shared_file_list = {}

        if torch.distributed.is_initialized() and get_world_size() > num_files:
            remainder = get_world_size() % num_files
            data_file = files[(f_start_id*get_world_size()+get_rank() + remainder*f_start_id)%num_files]
        else:
            data_file = files[(f_start_id*get_world_size()+get_rank())%num_files]

        previous_file = data_file

        if restored_data_loader is None:
            train_data = pretraining_dataset(data_file, args.max_predictions_per_seq)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                        batch_size=args.train_batch_size * args.n_gpu,
                                        num_workers=4, worker_init_fn=worker_init,
                                        pin_memory=True)
            # shared_file_list["0"] = (train_dataloader, data_file)
        else:
            train_dataloader = restored_data_loader
            restored_data_loader = None

        overflow_buf = None
        if args.allreduce_post_accumulation:
            overflow_buf = torch.cuda.IntTensor([0])

        for f_id in range(f_start_id + 1 , len(files)):
            if get_world_size() > num_files:
                data_file = files[(f_id*get_world_size()+get_rank() + remainder*f_id)%num_files]
            else:
                data_file = files[(f_id*get_world_size()+get_rank())%num_files]

            previous_file = data_file

            dataset_future = pool.submit(create_pretraining_dataset, data_file, args.max_predictions_per_seq, shared_file_list, args, worker_init)

            train_iter = tqdm(train_dataloader, desc="Iteration", disable=args.disable_progress_bar) if is_main_process() else train_dataloader

            if raw_train_start is None:
                raw_train_start = time.time()

            for step, batch in enumerate(train_iter):
                training_steps += 1
                batch = [t.to(device) for t in batch]
                input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels = batch
                if args.do_train:
                    from smdistributed.modelparallel.test.torch.utils import verify, dump_model
                    model.train()
                    if args.smp > 0:
                        loss_mbs = smp_step(args, device, input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels, model, optimizer, criterion, step)
                        loss = loss_mbs.reduce_mean()
                        if smp.rank() == 0:
                            print("Loss:", loss.item())
                    else:
                        loss = train_step(args, device, input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels, model, optimizer, criterion, step)
                    divisor=1
                    average_loss += loss.item()

                    if training_steps % args.gradient_accumulation_steps == 0:
                        lr_scheduler.step()  # learning rate warmup
                        global_step = take_optimizer_step(args, optimizer, model, overflow_buf, global_step)

                    if global_step >= args.steps_this_run or timeout_sent:
                        train_time_raw = time.time() - raw_train_start
                        last_num_steps = int(training_steps / args.gradient_accumulation_steps) % args.log_freq
                        last_num_steps = args.log_freq if last_num_steps == 0 else last_num_steps
                        average_loss = torch.tensor(average_loss, dtype=torch.float32).cuda()
                        average_loss = average_loss / (last_num_steps * divisor)
                        if (torch.distributed.is_initialized()):
                            average_loss /= get_world_size()
                            torch.distributed.all_reduce(average_loss)
                        final_loss = loss.item()
                    elif training_steps % (args.log_freq * args.gradient_accumulation_steps) == 0:
                        average_loss = 0


                    if global_step >= args.steps_this_run or training_steps % (
                            args.num_steps_per_checkpoint * args.gradient_accumulation_steps) == 0 or timeout_sent:
                        if smp.dp_rank() == 0 and not args.skip_checkpoint:
                            if args.resume_step < 0 or not args.phase2:
                                output_save_file = os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step))
                            else:
                                output_save_file = os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step + args.phase1_end_step))
                            if args.do_train:
                                save_dict = {
                                            'model': model.local_state_dict(),
                                            'optimizer': optimizer.local_state_dict(),
                                            'files': [f_id] + files,
                                            'epoch': epoch,
                                            'data_loader': None if global_step >= args.steps_this_run else train_dataloader}
                                if args.fp16:
                                    save_dict['master params'] = list(amp.master_params(optimizer))
                                # SMP: Checkpoint mp_rank specific state
                                smp.save(save_dict, output_save_file, partial=True)

                                most_recent_ckpts_paths.append(output_save_file)
                                if len(most_recent_ckpts_paths) > 3 and (args.smp == 0 or smp.dp_rank() == 0):
                                    ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
                                    os.remove(ckpt_to_be_removed+f"_{smp.mp_rank()}")

                        # Exiting the training due to hitting max steps, or being sent a
                        # timeout from the cluster scheduler
                        if global_step >= args.steps_this_run or timeout_sent:
                            del train_dataloader
                            # thread.join()
                            if smp.dp_rank() == 0 and args.save_full:
                                output_save_file = os.path.join(args.output_dir, "ckpt_{}.pt".format(global_step))
                                save_dict = {
                                            'model': model.local_state_dict(),
                                            'optimizer': optimizer.local_state_dict(),
                                            'files': [f_id] + files,
                                            'epoch': epoch,
                                            'data_loader': None if global_step >= args.steps_this_run else train_dataloader}
                                if args.fp16:
                                    save_dict['master params'] = list(amp.master_params(optimizer))
                                # SMP: Save a single checkpoint containing entire model parameters
                                smp.save(save_dict, output_save_file, partial=False)
                            smp.barrier()
                            if smp.local_rank() == 0:
                                print(f"Start syncing model checkpoints to s3")
                                base_s3_path = os.path.dirname(os.path.dirname(os.getenv('SM_MODULE_DIR', '')))
                                curr_host = os.getenv('SM_CURRENT_HOST')
                                full_s3_path = f'{base_s3_path}/checkpoints/{curr_host}/'
                                sync_local_checkpoints_to_s3(local_path=args.output_dir, s3_path=full_s3_path)
                                print(f"Finished syncing model checkpoints to s3")
                            return args, final_loss, train_time_raw, global_step
                else:
                    model.eval()
                    with torch.no_grad():
                        loss = test_step(args, device, input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels, model, criterion, step)
                        print(f"global_step {global_step} Test Loss:", loss)
                        test_losses.append(loss)
                    global_step += 1
                    if global_step >= args.steps_this_run:
                        return sum(test_losses) / len(test_losses)

            del train_dataloader
            # thread.join()
            # Make sure pool has finished and switch train_dataloader
            # NOTE: Will block until complete
            train_dataloader, data_file = dataset_future.result(timeout=None)
        epoch += 1


if __name__ == "__main__":
    pprint.pprint(dict(os.environ), width = 1)
    now = time.time()
    args = parse_arguments()
    if args.do_train:
        args, final_loss, train_time_raw, global_step = main()
        gpu_count = args.n_gpu
        global_step += args.phase1_end_step if (args.phase2 and args.resume_step > 0) else 0
        if args.resume_step == -1:
            args.resume_step = 0
        if torch.distributed.is_initialized():
            gpu_count = get_world_size()
        if is_main_process():
            e2e_time = time.time() - now
            training_perf = args.train_batch_size * args.gradient_accumulation_steps * gpu_count\
                            * (global_step - args.resume_step + skipped_steps) / train_time_raw
            print(f"[SMP_METRIC]__Bert__Loss__{final_loss:.3f}")
            print(f"[SMP_METRIC]__Bert__Time_to_train__{train_time_raw:.4f}")
            print(f"[SMP_METRIC]__Bert__Sequences/second__{training_perf:.2f}")
    else:
        avg_loss = main()
    sys.exit(0)
