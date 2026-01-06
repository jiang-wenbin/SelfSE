from omegaconf import OmegaConf
from torch.linalg import vector_norm
import torch
import torch.nn.functional as F
import numpy as np
from modules.utils.common import EPS, power_of_2
from modules.loss.base import BaseLoss

# STFT-based Loss -------------------------------------
# based on https://github.com/facebookresearch/denoiser/blob/main/denoiser/stft_loss.py


def stft(x, fft_size, hop_size, win_length, window):
    '''Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    '''
    x_stft = torch.stft(x, fft_size, hop_size, win_length,
                        window, return_complex=True)
    real = x_stft.real
    imag = x_stft.imag

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    mag = torch.sqrt(torch.clamp(real ** 2 + imag **
                     2, min=1e-7)).transpose(2, 1)
    return x_stft, mag


class SpectralConvergengeLoss(torch.nn.Module):
    '''Spectral convergence loss module.'''

    def __init__(self):
        '''Initilize spectral convergence loss module.'''
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        '''Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        '''
        return torch.norm(y_mag - x_mag, p='fro') / torch.norm(y_mag, p='fro')


class LogSTFTMagnitudeLoss(torch.nn.Module):
    '''Log STFT magnitude loss module.'''

    def __init__(self):
        '''Initilize los STFT magnitude loss module.'''
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        '''Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        '''
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class ComplexSpectrumLoss(torch.nn.Module):
    def __init__(self):
        super(ComplexSpectrumLoss, self).__init__()

    def forward(self, x_stft, y_stft):
        '''Calculate forward propagation.
        Args:
            x_stft (Tensor): Spectrogram of predicted signal (B, #frames, #freq_bins).
            y_stft (Tensor): Spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        '''
        return F.l1_loss(x_stft, y_stft)


class STFTLoss(torch.nn.Module):
    '''STFT loss module.'''

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window='hann_window'):
        '''Initialize STFT loss module.'''
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer('window', getattr(torch, window)(win_length))
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()
        self.complex_spctrum_loss = ComplexSpectrumLoss()

    def forward(self, x, y):
        '''Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        '''
        x_stft, x_mag = stft(x, self.fft_size, self.shift_size,
                             self.win_length, self.window)
        y_stft, y_mag = stft(y, self.fft_size, self.shift_size,
                             self.win_length, self.window)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)
        complex_loss = self.complex_spctrum_loss(x_stft, y_stft)

        return sc_loss, mag_loss, complex_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    '''Multi resolution STFT loss module.'''

    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window='hann_window'):
        '''Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
            factor (float): a balancing factor across different losses.
        '''
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        '''Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        '''
        sc_loss = 0.0
        mag_loss = 0.0
        complex_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l, complex_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
            complex_loss += complex_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)
        complex_loss /= len(self.stft_losses)

        return sc_loss, mag_loss, complex_loss


# SNR, SI_SRN, wSDR Loss --------------------------------------------

def _power_sum(z, keepdim=True):
    return torch.sum(z ** 2, dim=-1, keepdim=keepdim)


def _inner(a, b, keepdim=True):
    return torch.sum(a * b, dim=-1, keepdim=keepdim)


def _remove_mean(x, dim=-1):
    return x - x.mean(dim=dim, keepdim=True)


def snr(s, x):
    '''Compute SNR
    Args:
        s: Reference signal (ground truth) of shape [n_batch, n_sample]
        x: Enhanced/separated signal of shape [n_batch, n_sample]
    Return: SNR of shape [n_batch, 1]
    '''
    assert s.shape == x.shape
    n = s - x
    _snr = 10 * torch.log10(_power_sum(s).clip(EPS) / _power_sum(n).clip(EPS))
    return _snr


def si_snr(s, x, zero_mean=True):
    '''Compute scale invariant SNR, SI-SNR
    Args:
        s: Reference signal (ground truth) of shape [n_batch, n_sample]
        x: Enhanced/separated signal of shape [n_batch, n_sample]
    Return: SI-SNR of shape [n_batch, 1]
    '''
    assert s.shape == x.shape
    if zero_mean:
        x = _remove_mean(x)
        s = _remove_mean(s)
    proj_x = (_inner(x, s)/_power_sum(s).clip(EPS)) * s  # projection of x on s
    n = x - proj_x
    _si_snr = 10 * \
        torch.log10(_power_sum(proj_x).clip(EPS) / _power_sum(n).clip(EPS))
    return _si_snr


def wSDR(noisy, clean, est, eps=1e-8):
    # shape [n_batch, n_samples]
    def _SDR(x, x_est):
        return _inner(x, x_est) / (
            vector_norm(x, dim=1, keepdim=True) * vector_norm(x_est, dim=1, keepdim=True) + eps)
    noise = noisy - clean
    noise_est = noisy - est
    a = _power_sum(clean) / (_power_sum(clean) + _power_sum(noise) + eps)
    _wSDR = a * _SDR(clean, est) + (1 - a) * _SDR(noise, noise_est)
    return _wSDR


# Loss module -------------------------------------------------

class SI_SNRLoss(BaseLoss):
    def __init__(self, loss_conf):
        super().__init__(loss_conf)

    def forward(self, clean, est, *args):
        return -torch.mean(si_snr(clean, est))


class SNRLoss(BaseLoss):
    def __init__(self, loss_conf):
        super().__init__(loss_conf)

    def forward(self, clean, est, *args):
        return -torch.mean(snr(clean, est))


class wSDRLoss(BaseLoss):
    def __init__(self, loss_conf):
        super().__init__(loss_conf)

    def forward(self, clean, est, noisy):
        return -torch.mean(wSDR(noisy, clean, est))


class CMSELoss(BaseLoss):
    '''Complex MSE loss'''

    def __init__(self, loss_conf):
        super().__init__(loss_conf)

    def forward(self, clean, est):
        l = (F.mse_loss(clean.real, est.real) +
             F.mse_loss(clean.imag, est.imag)) / 2
        return l


class LogMSELoss(BaseLoss):
    def __init__(self, loss_conf):
        super().__init__(loss_conf)        

    def forward(self, clean, est, *args):        
        mse = F.mse_loss(clean, est)
        return torch.log(mse + EPS)


class MR_STFT_SNRLoss(BaseLoss):
    '''Multi-resolution STFT and SNR'''

    def __init__(self, loss_conf):
        super().__init__(loss_conf)
        win_lengths = loss_conf.get('win_lengths', None)
        self.stft_weight = loss_conf.get('stft_weight', 0.1)
        self.snr_weight = loss_conf.get('snr_weight', 0.9)
        if win_lengths is None:
            fft_sizes = [1024, 2048, 512]
            hop_sizes = [120, 240, 50]
            win_lengths = [600, 1200, 240]
        else:
            if type(win_lengths) is str:
                win_lengths = [int(item) for item in win_lengths.split(',')]
            fft_sizes = [power_of_2(item) for item in win_lengths]
            hop_sizes = [item//4 for item in win_lengths]
        self.stft_losses = MultiResolutionSTFTLoss(fft_sizes, hop_sizes, win_lengths)

    def forward(self, clean, est, *args):
        sc_loss, mag_loss, _ = self.stft_losses(est, clean)
        stft_mag_loss = sc_loss + mag_loss
        si_snr_loss = -torch.mean(si_snr(clean, est))
        l = self.stft_weight * stft_mag_loss + self.snr_weight * si_snr_loss
        return l


class SI_STFTLoss(torch.nn.Module):
    '''Scale invariant STFT loss module.'''

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window='hann_window'):
        '''Initialize STFT loss module.'''
        super().__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer('window', getattr(torch, window)(win_length))

    def forward(self, x, y):
        '''Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        '''
        _, x_mag = stft(x, self.fft_size, self.shift_size,
                        self.win_length, self.window)
        _, y_mag = stft(y, self.fft_size, self.shift_size,
                        self.win_length, self.window)
        return -torch.mean(si_snr(y_mag, x_mag))


class SI_STFT_SNRLoss(BaseLoss):
    '''Scale Invariant STFT and SNR'''

    def __init__(self, loss_conf):
        super().__init__(loss_conf)
        win_lengths = loss_conf.get('win_lengths', None)
        self.stft_weight = loss_conf.get('stft_weight', 0.1)
        self.snr_weight = loss_conf.get('snr_weight', 0.9)
        if win_lengths is None:
            fft_sizes = [1024, 2048, 512]
            hop_sizes = [120, 240, 50]
            win_lengths = [600, 1200, 240]
        else:
            if type(win_lengths) is str:
                win_lengths = [int(item) for item in win_lengths.split(',')]
            fft_sizes = [power_of_2(item) for item in win_lengths]
            hop_sizes = [item//4 for item in win_lengths]
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [SI_STFTLoss(fs, ss, wl)]

    def forward(self, clean, est, *args):
        stft_mag_loss = 0.0
        for f in self.stft_losses:
            stft_mag_loss += f(est, clean)
        stft_mag_loss /= len(self.stft_losses)
        si_snr_loss = -torch.mean(si_snr(clean, est))
        l = self.stft_weight * stft_mag_loss + self.snr_weight * si_snr_loss
        return l


def test(conf):
    noisy = torch.randn((8, 2*16000))
    clean = torch.randn((8, 2*16000))
    est = torch.randn((8, 2*16000))
    # est = -clean
    print('SI_SNR:', -torch.mean(si_snr(clean, est)))
    print('SNR:', snr(clean, est))
    print('wSDR:', wSDR(noisy, clean, est))
    print('SI_STFT_SNRLoss:')
    for stft_weight, snr_weight in ([0.5, 0.5], [1.0, 0.0], [0.0, 1.0]):
        loss_fun = SI_STFT_SNRLoss(stft_weight, snr_weight)
        print(stft_weight, snr_weight, loss_fun(clean, est))


if __name__ == '__main__':
    conf = OmegaConf.create({
        'yaml': 'conf/UnSE_mag.yaml',
        'cmd': 'test'})
    conf.merge_with_cli()
    conf = OmegaConf.merge(OmegaConf.load(conf.yaml), conf)
    eval(conf.cmd)(conf)