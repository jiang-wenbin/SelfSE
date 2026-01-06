from torch import nn
from modules.blocks.blocks import *
from modules.blocks.DPRNN import DPRNN
from modules.blocks.cbam import CBAM
from modules.utils.logging import logger
from modules.utils.common import merge_dicts


class UNet(ModelBase):
    def __init__(self, model_conf):
        super().__init__(model_conf)
        # the default configuration
        default_conf = {
            'debug': False, 'mask': True, 'mask_bound': 'tanh', 'mag_decoder': False,
            'spec_decoder': False, 'norm': 'batch', 'block_att': True, 'in_channels': 3,
            'rnn': {'name': 'DPRNN', 'num_layers': 2, 'bidirectional': False, 'hidden_size': 128}
        }
        self.model_conf = merge_dicts(default_conf, model_conf) # merge
        
        assert (self.model_conf['mag_decoder'] ^ self.model_conf['spec_decoder'])
        param = self.model_conf['param']
        norm = self.model_conf['norm']
        self.mask_bound = self.model_conf['mask_bound']

        param['encoder'][0][0] = self.model_conf['in_channels']
        param['decoder'][-1][1] = self.model_conf['in_channels']
        self.encoder = nn.ModuleList(
            [ConvBlock(*item, norm=norm) for item in param['encoder']])
        self.decoder = nn.ModuleList(
            [TransConvBlock(*item, norm=norm) for item in param['decoder']])
        self.block_att = nn.ModuleList([
            CBAM(item[1]) if self.model_conf['block_att']  is True else nn.Identity() for item in param['encoder']])

        rnn_name = self.model_conf['rnn']['name']
        rnn_kwargs = {
            'encoder_dim': param['encoder'][-1][1],
            'num_layers': self.model_conf['rnn']['num_layers'],
            'bidirectional': self.model_conf['rnn']['bidirectional']
        }
        if rnn_name == 'DPRNN':
            self.rnn_block = DPRNN(**rnn_kwargs, hidden_size=self.model_conf['rnn']['hidden_size'])
        elif rnn_name == 'DPTransformer':
            self.rnn_block = DPTransformer(**rnn_kwargs)
        else:
            raise NotImplementedError

    def forward(self, x):
        # x: shape of [batch size, in_channels, n_fft//2+1, T]
        e = x
        e_list = []
        for i, layer in enumerate(self.encoder):
            e = layer(e)
            e = self.block_att[i](e)
            e_list.append(e)
            if self.model_conf['debug']:
                logger.debug(f'encoder_{i}: {e.shape}')
        rnn_out = self.rnn_block(e)
        if self.model_conf['debug']:
            logger.debug(f'rnn_out: {rnn_out.shape}')
        idx = len(e_list)
        d = rnn_out
        for i, layer in enumerate(self.decoder):
            idx = idx - 1
            # torch.cat((d, e_list[idx]), 1))
            d = layer(padded_cat(d, e_list[idx]))
            if self.model_conf['debug']:
                logger.debug(f'decoder_{i}: {d.shape}')
        if self.model_conf['mask']:
            d = apply_mask_bound(d, self.mask_bound)
        if self.model_conf['spec_decoder']:
            return d, None
        else:
            return None, d

