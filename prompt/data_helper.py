import os
import json
import zipfile
import random
import zipfile
import torch

import numpy as np
from io import BytesIO
from functools import partial
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from time import time
import datasets

def create_dataloaders(args):
    train_prefix = np.load(args.train_prefix_path)
    train_suffix = np.load(args.train_suffix_path)
    val_prefix = np.load(args.val_prefix_path)
    val_suffix = np.load(args.val_suffix_path)

    if args.train_chunk:
        train_prefix = train_prefix[:args.train_chunk]
        train_suffix = train_suffix[:args.train_chunk]

    train_negsuffix = None
    val_negsuffix = None
    if args.useneg:
        train_negsuffix = np.load(args.train_negsuffix_path)
        val_negsuffix = np.load(args.val_negsuffix_path)

    train_dataset = GenDataset(args, train_prefix, train_suffix, train_negsuffix)
    val_dataset = GenDataset(args, val_prefix, val_suffix, val_negsuffix)
    
    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
    else:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=args.batch_size,
                                        sampler=train_sampler,
                                        drop_last=True)
    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args.val_batch_size,
                                      sampler=val_sampler,
                                      drop_last=False)
    return train_dataloader, val_dataloader


def create_predict_dataloaders(args):
    if args.test_path.endswith('.json'):
        with open(args.test_path, 'r', encoding='utf8') as f:
            test_data = json.load(f)
    else:
        test_data = datasets.load_from_disk(args.test_path)
    
    test_dataset = ClassifyDataset(args, test_data)
    
    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
    else:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    test_sampler = SequentialSampler(test_dataset)
    
    test_dataloader = dataloader_class(test_dataset,
                                      batch_size=args.test_batch_size,
                                      sampler=test_sampler,
                                      drop_last=False)
    return test_dataloader

class GenDataset(Dataset):

    def __init__(self,
                 args,
                 prefixes,
                 suffixes,
                 neg_suffixes=None
                 ):
       
        self.max_input_length = args.max_input_length

        self.args = args
        
        self.prefixes = prefixes.astype(np.int64)[:, -args.prefix_len:]
        self.suffixes = suffixes.astype(np.int64)[:, :args.suffix_len]
        if args.useneg:
            self.neg_suffixes = neg_suffixes

    def __len__(self) -> int:
        return len(self.prefixes)


    def __getitem__(self, idx: int) -> dict:
        prefix_ids = self.prefixes[idx]
        suffix_ids = self.suffixes[idx]
        input_ids = np.concatenate([prefix_ids, suffix_ids], axis=0)
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        if not self.args.useneg:
            data = dict(
                input_ids=input_ids,
            )
            return data

        else:
            neg_suffix_ids = self.neg_suffixes[idx]
            neg_input_ids = np.concatenate([prefix_ids, neg_suffix_ids], axis=0)
            neg_input_ids = torch.tensor(neg_input_ids, dtype=torch.long)
            if np.all(neg_suffix_ids == suffix_ids):
                mask = torch.tensor(False, dtype=torch.bool)
            else:
                mask = torch.tensor(True, dtype=torch.bool)
            # mask为1的要计算Loss，为0的不需要
            data = dict(
                input_ids=input_ids,
                neg_input_ids=neg_input_ids,
                mask=mask
            )
            return data