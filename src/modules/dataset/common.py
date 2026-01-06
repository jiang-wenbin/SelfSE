import numpy as np
import pandas as pd
import random
import torch
from typing import Optional
from multiprocessing import Manager
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
import time
import math
import os
import re
import getpass
import shutil
import torch.nn.functional as F

from modules.utils.common import cost_time
from modules.utils.logging import logger


@dataclass
class VarData:
    data: torch.Tensor
    lengths: Optional[torch.Tensor] = None # a list of torch.Tensor


@cost_time
def create_mem_cache(index_map, wav_store):
    clean_data, noisy_data = {}, {}
    for idx in tqdm(index_map, desc='loading'):
        wavid, _, _ = index_map[idx]
        if wavid not in clean_data:
            clean_data[wavid] = wav_store['clean'][wavid][:]
            noisy_data[wavid] = wav_store['noisy'][wavid][:]
    return {'clean': clean_data, 'noisy': noisy_data}


@cost_time
def create_shared_dict(index_map, wav_store):
    manager = Manager()
    shared_dict = manager.dict()
    for idx in tqdm(index_map, desc='loading'):
        wavid, _, _ = index_map[idx]
        if wavid not in shared_dict:
            shared_dict[wavid] = (wav_store['clean'][wavid]
                                  [:], wav_store['noisy'][wavid][:])
    return shared_dict


def create_cache(h5_file, dest='/dev/shm', name=None):
    dest_path = Path(dest).joinpath(getpass.getuser())
    if name:
        dest_path = dest_path.joinpath(name)
    cache_h5_file = dest_path.joinpath(h5_file)
    cache_h5_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_h5_file.exists() is False:
        logger.info(f'{cache_h5_file} not exists, copying...')
        t = time.perf_counter()
        shutil.copy2(h5_file, cache_h5_file)
        logger.info('cost time: {:.3f} s'.format(time.perf_counter() - t))
    else:
        size_ok = math.isclose(os.path.getsize(cache_h5_file), 
                               os.path.getsize(h5_file))
        time_ok = math.isclose(os.path.getmtime(cache_h5_file),
                               os.path.getmtime(h5_file))  # mtime !
        if size_ok is False or time_ok is False:
            logger.info(f'{cache_h5_file} not match, copying...')
            t = time.perf_counter()
            shutil.copy2(h5_file, cache_h5_file)
            logger.info('cost time: {:.3f} s'.format(time.perf_counter() - t))
    return cache_h5_file


def clear_cache(dest='/dev/shm'):
    user_cache_path = Path(dest).joinpath(getpass.getuser())
    safe_remove(user_cache_path)


def safe_remove(target, safe_path='/dev/shm'):
    if not Path(target).relative_to(safe_path):
        logger.info(f'{target} is not relative to {safe_path}')
        return
    logger.info(f'remove {target}')
    if os.path.isfile(target) or os.path.islink(target):
        os.remove(target)  # remove the file
    elif os.path.isdir(target):
        shutil.rmtree(target)  # remove dir and all contains
    else:
        logger.info(f'{target} is not a file or dir.')
            

def read_csv(csv_file, deny_list=None):
    dataframe = pd.read_csv(csv_file, index_col=0)
    deny_list = []
    if deny_list:
        with open(deny_list) as f:
            deny_list = [item.strip() for item in f.readlines()]
    dataframe = dataframe[~dataframe.index.isin(deny_list)]    
    return dataframe


def get_chunk_info(data_info, chunk_samples=-1, min_chunk=0.5):
    '''get chunk info
    data_info: dataframe or dict
    return: [(wav_id, start, end), ...]
    '''
    if data_info is None:
        return None
    if isinstance(data_info, pd.DataFrame):
        data_info = data_info.to_dict(orient='index')
    chunk_info = []
    for wav_id in data_info:
        if chunk_samples == -1:  # read all samples
            chunk_info.append((wav_id, 0, None))
            continue
        duration = data_info[wav_id]['duration']
        if duration//chunk_samples == 0:
            chunk_info.append((wav_id, 0, None))
            continue
        for j in range(duration//chunk_samples):
            chunk_info.append((wav_id, j*chunk_samples, (j+1)*chunk_samples))
        if duration % chunk_samples > min_chunk * chunk_samples:  # the remainder
            chunk_info.append((wav_id, (j+1)*chunk_samples, None))    
    return chunk_info


def random_choice(id_list, deny_list=None, n=1):
    chosen_list = []
    for _ in range(max(len(id_list), 200)):
        chosen_id = random.choice(id_list)
        if deny_list and chosen_id in deny_list:
            continue
        chosen_list.append(chosen_id)
        if len(chosen_list) == n:
            break
    assert len(chosen_list) == n
    return chosen_list[0] if n == 1 else chosen_list


def check_wav_value(z, eps=1e-8):
    if np.sum(z ** 2) < eps or np.isnan(z).any():
        return False
    return True