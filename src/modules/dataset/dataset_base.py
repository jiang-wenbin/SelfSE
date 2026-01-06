from scipy import signal
import random
import numpy as np
import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
from collections import OrderedDict
from modules.utils.logging import logger
from modules.utils.common import wave_pad_sample, get_memory_info, get_rank
from .common import read_csv, get_chunk_info
from .data_store import WavStore


class DatasetBase:
    def __init__(self, conf, subset):
        '''
        subset: ['train', 'valid', 'test']
        '''
        super().__init__()
        self.conf = conf
        self.subset = subset
        self.pairs = conf['data'].get('pairs', 'noisy2clean')
        self.unpaired = conf['data'][subset].get('unpaired', False)
        self.mix_param = conf['data'].get('mix_param')
        self.add_rir = self.conf['data'].get('add_rir', 0)
        self.split_wav = conf['data'][subset]['split_wav']
        self.sample_rate = conf['data']['sample_rate']
        self.cache = conf['data'].get('cache', None)
        self.check_value = conf['data'].get('check_value', False)
        noisy_csv = self.resolve_path('noisy_csv', subset)
        noisy_store = self.resolve_path('noisy_store', subset)
        clean_csv = self.resolve_path('clean_csv', subset)        
        clean_store = self.resolve_path('clean_store', subset)
        noise_csv = self.resolve_path('noise_csv')
        noise_store = self.resolve_path('noise_store')
        self.data_info = {
            'noisy': self.read_csv(noisy_csv),
            'clean': self.read_csv(clean_csv),
            'noise': self.read_csv(noise_csv)
        }
        base_dir = conf['data'][subset].get('base_dir')
        self.data_store = {
            'noisy': WavStore(noisy_store, self.data_info['noisy'], name=self.conf['name'],
                              base_dir=base_dir, cache=self.cache) if noisy_store or noisy_csv else None,
            'clean': WavStore(clean_store, self.data_info['clean'], name=self.conf['name'],
                              base_dir=base_dir, cache=self.cache) if clean_store or clean_csv else None,
            'noise': WavStore(noise_store, self.data_info['noise'], name=self.conf['name'],
                              base_dir=base_dir, cache=self.cache) if noise_store or noise_csv else None,
        }
        if self.split_wav:
            self.chunk_samples = (conf['data']['chunk_size'] - 1) * conf['stft']['hop_length'] + \
                conf['stft']['win_length']
        else:
            self.chunk_samples = -1
        self.chunk_info = {
            'noisy': get_chunk_info(self.data_info['noisy'], self.chunk_samples),
            'clean': get_chunk_info(self.data_info['clean'], self.chunk_samples),
            'noise': get_chunk_info(self.data_info['noise'], self.chunk_samples)
        }
        self.chunk_info_grouped = self.group_chunk_info()
        if self.add_rir > 0:
            rir_csv = self.resolve_path('rir_csv')
            rir_store = self.resolve_path('rir_store')
            self.data_info['rir'] = read_csv(rir_csv) if rir_csv else None
            self.data_store['rir'] = WavStore(rir_store, self.data_info['rir'], 
                cache=self.cache, name=self.conf['name']) if rir_store or rir_csv else None
            self.update_rir_info(rir_csv.with_name('RIR_table_simple.csv'))
        logger.info(self.get_msg())
        logger.info(get_memory_info())
        
    def get_msg(self):
        msg = f'[{self.subset}] chunk_samples: {self.chunk_samples}; '
        len_info = [len(self.chunk_info[key]) if self.chunk_info[key] else 0 for key in self.chunk_info]
        msg = msg + '|n_chunk|' + f' noisy: {len_info[0]}, clean: {len_info[1]}, noise: {len_info[2]}'
        dur_msg = ''
        for key in self.data_info:
            if key not in ['noisy', 'clean', 'noise', 'rir']:
                continue
            if self.data_info[key] is not None:
                dur_msg = dur_msg + ' {{{}}} max: {:.2f} s, min: {:.2f} s, mean: {:.2f} s'.format(
                    key, 
                    self.data_info[key]['duration'].max()/self.sample_rate,
                    self.data_info[key]['duration'].min()/self.sample_rate,
                    self.data_info[key]['duration'].mean()/self.sample_rate)
        if len(dur_msg) > 0:
            msg = msg + '; |duration|' + dur_msg
        if self.unpaired:
            msg = msg + f'; unpaired: {self.unpaired}'
        return msg
        
    def resolve_path(self, key, subset=None):
        """
        Resolve the path based on the key and subset.

        Args:
            key (str): The key to lookup in the configuration.
            subset (str, optional): The subset to lookup in the configuration.

        Returns:
            str or list of str: The resolved path(s).
        """
        sub_conf = self.conf['data'][subset] if subset else self.conf['data']
        file_path = sub_conf.get(key)
        if file_path is None:
            logger.debug(f'return None: {key}, {subset}')
            return None
        data_dir = Path(self.conf['data']['data_dir'])
        if isinstance(file_path, list) or OmegaConf.is_list(file_path):
            return [data_dir.joinpath(item) for item in file_path]
        else:
            n_split = sub_conf.get('split_dataset', 1)
            if n_split > 1:
                file_path = file_path.format(n_split, get_rank())
            return data_dir.joinpath(file_path)
    
    def read_csv(self, csv_file):
        """
        Read one or multiple CSV files into a pandas DataFrame.

        Args:
            csv_file (Path or list of Path): The path(s) to the CSV file(s).

        Returns:
            pd.DataFrame: The data from the CSV file(s).
        """
        if not csv_file:
            return None
        if isinstance(csv_file, list):
            df_list = [read_csv(item).assign(store_idx=i) for i, item in enumerate(csv_file)]            
            return pd.concat(df_list)
        else:
            return read_csv(csv_file)
    
    @property
    def snr(self):
        return random.randint(self.mix_param['snr_lower'], self.mix_param['snr_upper'])
    
    @property
    def snr_n1(self):
        if 'n1_lower' in self.mix_param: # only for noiser2noisy
            return random.randint(self.mix_param['n1_lower'], self.mix_param['n1_upper'])
        else:
            return self.snr

    def group_chunk_info(self):
        def _group_list(chunk_list):
            if chunk_list is None:
                return None
            grouped_data = OrderedDict()
            for wav_id, start, end in chunk_list:
                if wav_id not in grouped_data:
                    grouped_data[wav_id] = []
                grouped_data[wav_id].append((start, end))
            return grouped_data

        return {
            key: _group_list(self.chunk_info[key])
            for key in self.chunk_info.keys()
        }
        
    def get_chunk(self, index, key='noisy', min_chunk=0.5):
        index = index % len(self.chunk_info[key]) # remainder
        wav_id, begin, end = self.chunk_info[key][index]
        if end is None:
            return wav_id, begin, end
        max_begin = begin + min_chunk * self.chunk_samples
        begin_random = random.randint(begin, max_begin)  # random begin
        return wav_id, begin_random, begin_random + self.chunk_samples
    
    def get_wav_data(self, wav_id, key):
        return self.data_store[key][wav_id]

    def check_wav_data(self, wav_data):
        if self.chunk_samples == -1:
            return wav_data
        data_len = wav_data.shape[-1]
        if data_len > self.chunk_samples:
            start_idx = np.random.randint(0, data_len - self.chunk_samples + 1)
            wav_data = wav_data[..., start_idx:start_idx + self.chunk_samples]
        else:
            n_pad = self.chunk_samples - data_len
            if n_pad > 0 and 'test' not in self.subset:
                wav_data = np.pad(wav_data, (0, n_pad))
        return wav_data
    
    @property
    def rir(self):
        return self.get_rir_data()

    def update_rir_info(self, rir_csv_ext):
        def wavfile2id(wavfile):
            wav_id = '_'.join(wavfile.split('/')[2:])
            return wav_id.replace('.wav', '')
        ext_info = pd.read_csv(rir_csv_ext)
        msg = 'rir.csv: {}, {}: {}'.format(
            len(self.data_store['rir'].id_list), rir_csv_ext.name, len(ext_info))
        ext_info = ext_info[ext_info['T60_WB'] >= self.mix_param['lower_t60']]
        ext_info = ext_info[ext_info['T60_WB'] <= self.mix_param['upper_t60']]
        ext_info['wav_id'] = ext_info['file'].apply(wavfile2id)
        rir_info = self.data_info['rir']
        rir_info_new = rir_info[rir_info.index.isin(ext_info['wav_id'])]
        logger.info(f'{msg}, filtered: {len(rir_info_new)}')
        self.data_info['rir'] = rir_info_new
        self.data_store['rir'].update_info(rir_info_new)

    def get_rir_data(self, chunk_samples=None):
        rir_data = self.data_store['rir'].sample_data()
        if rir_data.ndim > 1:
            rir_data = rir_data[:, 0] # only use the first channel
        if chunk_samples is not None:
            rir_data = wave_pad_sample(rir_data, chunk_samples)
        return rir_data

    def add_pyreverb(self, clean_speech):
        reverb_speech = signal.fftconvolve(clean_speech, self.rir, mode='full')
        reverb_speech = reverb_speech[0: clean_speech.shape[0]]
        return reverb_speech
    