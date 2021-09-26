from re import T
import torch
from torch.utils.data import DataLoader
from torch.utils.data import SequentialSampler
from util import collate_fn
from load_data.dataset import BaseDataSet
from load_data.dataset import build_transforms
import os


def build_data(args, is_train=True):
    transforms = build_transforms(args, is_train=is_train)
    if is_train:
        dataset = BaseDataSet(
            os.path.join(args.data_root, args.dataset, "train.txt"),
            transforms=transforms,
            mode="RGB",
        )
        data_loader = DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
        )
    else:
        dataset = BaseDataSet(
            os.path.join(args.data_root, args.dataset, "test.txt"),
            transforms=transforms,
            mode="RGB",
        )
        data_loader = DataLoader(
            dataset,
            collate_fn=collate_fn,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
        )
    return data_loader
