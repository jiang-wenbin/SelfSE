import torch
import functools
from torch.nn.utils.rnn import pad_sequence
from modules.utils.common import torch_float32, torch_int32, EPS
from .common import VarData

class AudioCollateFn():
    def __init__(self, conf):
        self.conf = conf
        if 'feats' in conf['data']:
            self.feat_padding_value = conf['data']['feats'].get('padding_value', EPS)
        self.padding = functools.partial(pad_sequence, batch_first=True)

    def get_padded(self, data_list, key, order, padding_value=0):
        padded = self.padding(
            [torch_float32(data_list[i][key]) for i in order], 
            padding_value=padding_value)
        return padded
    
    def __call__(self, data_list):
        '''
        data_list: a list of data obtain by dataset
        return: dict with keys of ['wav_id', 'noisy_wav', 
        'clean_wav', 'wav_lengths', 'label', 'label_lengths']
        '''
        batch_data = {}
        order = None
        data_item = data_list[0]
        assert 'noisy_wav' in data_item or 'feats' in data_item
        
        if 'noisy_wav' in data_item:
            wav_length = torch_int32([x['noisy_wav'].shape[0] for x in data_list])
            order = torch.argsort(wav_length, descending=True)
            wav_lengths = torch_int32([data_list[i]['noisy_wav'].shape[0] for i in order])
            batch_data['noisy_wav'] = self.get_padded(data_list, 'noisy_wav', order)
            batch_data['wav_lengths'] = wav_lengths
        
        if 'feats' in data_item:
            if order is None:
                feats_length = torch_int32([x['feats'].shape[0] for x in data_list])
                order = torch.argsort(feats_length, descending=True)
            feats_lengths = torch_int32([data_list[i]['feats'].shape[0] for i in order])
            batch_data['feats'] = VarData(
                self.get_padded(data_list, 'feats', order, self.feat_padding_value),
                feats_lengths)
        
        if 'wav_id' in data_item:
            batch_data['wav_id'] = [data_list[i]['wav_id'] for i in order]

        if 'clean_wav' in data_item:
            batch_data['clean_wav'] = self.get_padded(data_list, 'clean_wav', order)
        
        if 'teacher_wav' in data_item:
            batch_data['teacher_wav'] = self.get_padded(data_list, 'teacher_wav', order)

        if 'label' in data_item:
            if isinstance(data_item['label'], int): # int
                padded_labels = torch_int32([data_list[i]['label'] for i in order])
                label_lengths = torch_int32([1 for i in order])
            if isinstance(data_item['label'], float): # float
                padded_labels = torch_float32([data_list[i]['label'] for i in order])
                label_lengths = torch_int32([1 for i in order])
            elif isinstance(data_item['label'], list): # a list of label
                sorted_labels = [torch_int32(data_list[i]['label']) for i in order]
                label_lengths = torch_int32([len(data_list[i]['label']) for i in order])
                padded_labels = self.padding(sorted_labels, padding_value=-1)
            else:
                raise TypeError(type(data_item['label']).__name__)
            batch_data['label'] = VarData(padded_labels, label_lengths)

        return batch_data
