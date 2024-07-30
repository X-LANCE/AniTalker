import shutil
import os
from dist_utils import *


def use_cached_dataset_path(source_path, cache_path):
    if get_rank() == 0:
        if not os.path.exists(cache_path):
            # shutil.rmtree(cache_path)
            print(f'copying the data: {source_path} to {cache_path}')
            shutil.copytree(source_path, cache_path)
    barrier()
    return cache_path