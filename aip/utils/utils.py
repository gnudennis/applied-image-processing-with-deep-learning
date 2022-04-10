import os

import torch

__all__ = ['get_dataloader_workers', 'try_all_gpus']


def get_dataloader_workers(batch_size):
    return min([os.cpu_count(), batch_size if batch_size > 0 else 0, 8])


def try_all_gpus():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]
