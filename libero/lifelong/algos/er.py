import collections

import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
from torch.utils.data import ConcatDataset, RandomSampler
import multiprocessing

from libero.lifelong.algos.base import Sequential
from libero.lifelong.datasets import TruncatedSequenceDataset
from libero.lifelong.utils import *


def cycle(dl):
    while True:
        for data in dl:
            yield data


def merge_datas(x, y):
    if isinstance(x, (dict, collections.OrderedDict)):
        if isinstance(x, dict):
            new_x = dict()
        else:
            new_x = collections.OrderedDict()

        for k in x.keys():
            new_x[k] = merge_datas(x[k], y[k])
        return new_x
    elif isinstance(x, torch.FloatTensor) or isinstance(x, torch.LongTensor):
        return torch.cat([x, y], 0)


class ER(Sequential):
    """
    The experience replay policy.
    """

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        # we truncate each sequence dataset to a buffer, when replay is used,
        # concate all buffers to form a single replay buffer for replay.
        self.datasets = []
        self.descriptions = []
        self.buffer = None

    def start_task(self, task):
        super().start_task(task)
        if self.current_task > 0:
            multiprocessing.set_start_method("fork", force=True)
            # WARNING: currently we have a fixed size memory for each task.
            # We use a fixed proportion of the dataset for replay. The default is 20%.
            buffers = [
                TruncatedSequenceDataset(dataset, int(len(dataset) * self.cfg.lifelong.replay_ratio)) # self.cfg.lifelong.n_memories)
                for dataset in self.datasets
            ]
            buf = ConcatDataset(buffers)
            self.buffer = cycle(
                DataLoader(
                    buf,
                    batch_size=self.cfg.train.batch_size,
                    num_workers=self.cfg.train.num_workers,
                    sampler=RandomSampler(buf),
                    persistent_workers=True,
                )
            )

    def end_task(self, dataset, task_id, benchmark):
        self.datasets.append(dataset)

    def observe(self, data):
        multiprocessing.set_start_method("fork", force=True)
        if self.buffer is not None:
            buf_data = next(self.buffer)
            data = merge_datas(data, buf_data)

        data = self.map_tensor_to_device(data)

        self.optimizer.zero_grad()
        with amp.autocast('cuda', dtype=torch.float16):
            loss = self.policy.compute_loss(data)
        self.scaler.scale(self.loss_scale * loss).backward()
        if self.cfg.train.grad_clip is not None:
            self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.train.grad_clip
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()
