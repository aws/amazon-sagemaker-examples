
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import default_data_collator

from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

class StubDataset(Dataset):
  def __len__(self): return dist.get_world_size()*2
  def __getitem__(self, index):
    block_size = 4096
    return {
      'input_ids': torch.randint(1, 31580, (block_size,)),
      'attention_mask': torch.randint(0, 2, (block_size,)),
      'labels': torch.randint(1, 31579, (block_size,))
    }

def create_dataloaders(train_dataset, eval_dataset, rank, world_size, seed, 
                       train_batch_size, eval_batch_size):
  train_sampler = torch.utils.data.DistributedSampler(
    train_dataset, shuffle=True, seed=seed, rank=rank, num_replicas=world_size, 
    drop_last=True,)
  eval_sampler = torch.utils.data.DistributedSampler(
    eval_dataset, shuffle=True, seed=seed, rank=rank, num_replicas=world_size, 
    drop_last=True,)
  train_dataloader = DataLoader(
    train_dataset, sampler=train_sampler, collate_fn=default_data_collator, 
    batch_size=train_batch_size, pin_memory=True,drop_last=True)
  eval_dataloader = DataLoader(
    eval_dataset,sampler=eval_sampler, collate_fn=default_data_collator, 
    batch_size=eval_batch_size, pin_memory=True,drop_last=True)
  return train_dataloader,eval_dataloader

def get_module_class_from_name(module, name):
  modules_children = list(module.children())
  if module.__class__.__name__ == name:
    return module.__class__
  elif len(modules_children) == 0:
    return
  else:
    for child_module in modules_children:
      module_class = get_module_class_from_name(child_module, name)
      if module_class is not None:
        return module_class