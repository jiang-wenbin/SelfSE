# pytorch map-style dataset
from torch.utils.data import Dataset
import torch
import random
import numpy as np
from omegaconf import OmegaConf
from modules.utils.logging import logger
from modules.utils import audio
from modules.utils.common import get_rank
from .common import read_csv, get_chunk_info, check_wav_value
from .dataset_base import DatasetBase


class DatasetMap(Dataset, DatasetBase):
    def __init__(self, conf, subset):
        Dataset.__init__(self)
        DatasetBase.__init__(self, conf, subset)
        
    def __getitem__(self, index):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError


class WavDataset(DatasetMap):
    '''map-style dataset'''
    def __init__(self, conf, subset='train'):
        super().__init__(conf, subset)
        assert self.data_store['noisy']
        self.normalize = conf.get('normalize') # for the diffusion based model
        
    def __getitem__(self, index):
        wav_id, begin, end = self.get_chunk(index)
        noisy_wav = self.get_wav_data(wav_id, 'noisy')
        noisy_wav = self.check_wav_data(noisy_wav[begin:end])
        if self.normalize:
            normfac = noisy_wav.abs().max()
            noisy_wav = noisy_wav / normfac
        data = {'noisy_wav': noisy_wav, 'wav_id': wav_id}
        if self.unpaired is True:
            wav_id = self.data_store['clean'].sample_id(deny_list=[wav_id])
        if self.chunk_info['clean'] is not None:
            clean_wav = self.get_wav_data(wav_id, 'clean')
            clean_wav = self.check_wav_data(clean_wav[begin:end])
            if self.normalize:
                clean_wav = clean_wav / normfac
            data['clean_wav'] = clean_wav
        if self.label_dict is not None and len(self.label_dict) > 0:
            data['label'] = self.label_dict[wav_id]
        if self.check_value:
            if check_wav_value(clean_wav) is False or check_wav_value(noisy_wav) is False:
                new_index = np.random.randint(0, len(self)-1) # sampling new index
                logger.info(f'check value failed, {index} -> {new_index}')
                data = self[new_index]
        return data

    def __len__(self):
        return len(self.chunk_info['clean'] if self.chunk_info['clean'] else self.chunk_info['noisy'])


class NoiseDataset(DatasetMap):
    def __init__(self, conf, subset='train'):
        super().__init__(conf, subset)
        assert self.data_store['noise']

    def get_noise(self, index=None):
        if not index:
            index = random.randint(0, len(self))
        wav_id, begin, end = self.get_chunk(index, 'noise')
        noise = self.get_wav_data(wav_id, 'noise')
        return self.check_wav_data(noise[begin:end])

    def __getitem__(self, index):
        noise = self.get_noise()
        return noise
    
    def __len__(self):
        # should larger than batch size
        return len(self.chunk_info['noise'])


class OnlineNoisyDataset(DatasetMap):
    '''
    noisy2clean: clean + noise -> clean
    noisy2noisy: clean + noise1 -> clean + noise2
    noisier2noisy: clean + noise1 + noise2 -> clean + noise1
    noisier2noisy_v2: noisy + noise -> noisy
    '''
    def __init__(self, conf, subset='train'):
        super().__init__(conf, subset)
        if self.pairs == 'noisier2noisy_v2':
            assert (self.data_store['noisy'] and self.data_store['noise'])
        else:
            assert (self.data_store['clean'] and self.data_store['noise'])
        if self.add_rir > 0:
            assert self.data_store['rir']
        logger.info(f"pairs: {self.pairs}, [{self.mix_param.snr_lower}, {self.mix_param.snr_upper}] dB")
        self.suppression_ratio = self.mix_param.get('suppression_ratio')
        if self.suppression_ratio:
            logger.info(f'suppression_ratio: {self.suppression_ratio} dB')
        
    def get_clean(self, index):
        key = 'noisy' if self.pairs == 'noisier2noisy_v2' else 'clean'
        wav_id, begin, end = self.get_chunk(index, key)
        clean = self.get_wav_data(wav_id, key)
        clean = clean[begin:end]
        clean = self.check_wav_data(clean)
        if random.random() < self.add_rir:
            clean = self.add_pyreverb(clean)
        if clean.sum() == 0:
            new_index = np.random.randint(0, index)
            logger.error(f'clean_wav is 0, {index} -> {new_index}')
            return self.get_clean(new_index)
        return wav_id, clean

    def get_noise(self, index=None):
        if not index:
            index = random.randint(0, len(self))
        wav_id, begin, end = self.get_chunk(index, 'noise')
        noise = self.get_wav_data(wav_id, 'noise')
        return self.check_wav_data(noise[begin:end])

    def __getitem__(self, index):
        wav_id, clean = self.get_clean(index)
        noise = self.get_noise()
        snr = self.snr_n1
        clean, _, noisy, _ = audio.snr_mixer(self.mix_param, clean, noise, snr)
        if self.pairs in ['noisy2clean', 'noisier2noisy_v2']:
            if self.suppression_ratio:
                target_snr = snr + self.suppression_ratio
                _, _, target, target_rms_level = audio.snr_mixer(self.mix_param, clean, noise, target_snr)
                noisy = audio.normalize(noisy, target_rms_level)    
            else:
                target = clean
        elif self.pairs == 'noisy2noisy':
            noise = self.get_noise()
            _, _, target, target_rms_level = audio.snr_mixer(self.mix_param, clean, noise, self.snr)
            noisy = audio.normalize(noisy, target_rms_level)
        elif self.pairs == 'noisier2noisy':
            noise = self.get_noise()
            _, _, noisier, noisier_rms_level = audio.snr_mixer(self.mix_param, noisy, noise, self.snr)
            target = noisy
            noisy = noisier
            target = audio.normalize(target, noisier_rms_level)
        else:
            raise ValueError(f'pairs: {self.pairs}')
        data = {'clean_wav': target.astype(np.float32), 
                'noisy_wav': noisy.astype(np.float32), 
                'wav_id': wav_id}
        return data
        # return target.astype(np.float32), noisy.astype(np.float32), wav_id

    def __len__(self):
        if self.pairs == 'noisier2noisy_v2':
            length = len(self.chunk_info['noisy'])
        else:
            length = len(self.chunk_info['clean'])
        return length

