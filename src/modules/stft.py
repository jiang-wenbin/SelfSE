import torch
from asteroid_filterbanks import make_enc_dec
from asteroid_filterbanks.transforms import from_torch_complex, to_torch_complex
from copy import deepcopy
from .utils.logging import logger

class STFT(torch.nn.Module):
    '''Speech Enhancement Base Class'''

    def __init__(self, stft_conf, fft_m=1):
        # fft_m: fft multipler
        super().__init__()
        self.stft_conf = deepcopy(stft_conf)
        self.stft_conf.update({'n_fft': self.stft_conf['n_fft']*fft_m})
        self.stft_name = self.stft_conf.pop('name', 'torch')
        self.mag_compress = self.stft_conf.pop('mag_compress', None)
        self.spec_factor = self.stft_conf.pop('spec_factor', None)
        if self.stft_name == 'torch':
            window_func = getattr(
                torch, '{}_window'.format(self.stft_conf.pop('window', 'hann')))
            window = window_func(self.stft_conf['win_length'])
            self.register_buffer('window', window)
        elif self.stft_name == 'filterbank':  # filterbank
            self.stft_conf.update({'n_filters': self.stft_conf['n_filters']*fft_m})
            self.stft_encoder, self.stft_decoder = make_enc_dec(
                'stft', **self.stft_conf)
        else:
            raise NotImplementedError

    def apply_stft(self, x):
        '''
        x: shape of [n_batch, n_samples]
        return: shape of [n_batch, 1, n_fft/2+1, n_frames]
        '''
        if self.stft_name == 'torch':
            X_stft = torch.stft(x, **self.stft_conf,
                                window=self.window, return_complex=True)
        elif self.stft_name == 'filterbank':  # filterbank
            tf_rep = self.stft_encoder(torch.unsqueeze(x, 1))
            X_stft = to_torch_complex(tf_rep)
            if self.stft_conf.get('onesided', None) is False:
                raise NotImplementedError
        else:
            raise NotImplementedError
        X_stft = self.appy_mag_compress(X_stft)
        if self.spec_factor:
            X_stft = self.spec_factor * X_stft
        return torch.unsqueeze(X_stft, 1)

    def apply_istft(self, X_stft, **kwargs):
        '''
        X_stft: shape of [n_batch, 1, n_fft/2+1, n_frames]
        return: shape of [n_batch, n_samples]
        '''
        if self.spec_factor:
            X_stft = X_stft / self.spec_factor
        X_stft = self.appy_mag_compress(X_stft, inv=True)
        if self.stft_name == 'torch':
            X_stft = X_stft.squeeze(1)
            return torch.istft(X_stft, **self.stft_conf, window=self.window)
        elif self.stft_name == 'filterbank':  # filterbank
            x = self.stft_decoder(from_torch_complex(X_stft))
            if self.stft_conf.get('onesided', None) is False:
                raise NotImplementedError
            return torch.squeeze(x, 1)
        else:
            raise NotImplementedError

    def appy_mag_compress(self, X_stft, inv=False):
        if self.mag_compress is not None:
            mag = X_stft.abs() #.clip(EPS)
            phase = X_stft.angle()
            if type(self.mag_compress) is float:
                factor = self.mag_compress
                if inv is True:
                    factor = 1.0/factor
                compressed_mag = torch.pow(mag, factor)
            elif self.mag_compress == 'log1p':
                compressed_mag = torch.expm1(mag) if inv is True else torch.log1p(mag)
            else:
                logger.error(f'{self.mag_compress} error')
                compressed_mag = mag
            X_stft = compressed_mag * torch.exp(1j * phase)
        return X_stft
