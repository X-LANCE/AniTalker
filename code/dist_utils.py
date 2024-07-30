from typing import List
from torch import distributed


def barrier():
    if distributed.is_initialized():
        distributed.barrier()
    else:
        pass


def broadcast(data, src):
    if distributed.is_initialized():
        distributed.broadcast(data, src)
    else:
        pass


def all_gather(data: List, src):
    if distributed.is_initialized():
        distributed.all_gather(data, src)
    else:
        data[0] = src


def get_rank():
    if distributed.is_initialized():
        return distributed.get_rank()
    else:
        return 0


def get_world_size():
    if distributed.is_initialized():
        return distributed.get_world_size()
    else:
        return 1


def chunk_size(size, rank, world_size):
    extra = rank < size % world_size
    return size // world_size + extra