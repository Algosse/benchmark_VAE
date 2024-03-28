import random

import numpy as np
import torch
import torch.distributed as dist


def set_seed(seed: int):
    """
    Functions setting the seed for reproducibility on ``random``, ``numpy``,
    and ``torch``

    Args:

        seed (int): The seed to be applied
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_cpu_stats_over_ranks(stat_dict, world_size):
    keys = sorted(stat_dict.keys())
    allreduced = allreduce(torch.stack([torch.as_tensor(stat_dict[k]).detach().cuda().float() for k in keys]), average=True, world_size=world_size).cpu()
    return {k: allreduced[i].item() for (i, k) in enumerate(keys)}

def allreduce(x, average, world_size):
    if world_size > 1:
        dist.all_reduce(x, dist.ReduceOp.SUM)
    return x / world_size if average else x