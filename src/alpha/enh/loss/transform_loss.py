# loss in the transform domain
import torch
import torch.nn.functional as F
import functools
from omegaconf import OmegaConf
from modules.loss.base import BaseLoss
from torchaudio.compliance.kaldi import fbank
from torchaudio.transforms import MelSpectrogram
from modules.loss.torch_pesq import PesqLoss
from modules.utils.common import power_of_2


class _Mel_loss(torch.nn.Module):
    def __init__(self, sample_rate=16000, n_fft=400, n_mels=128,  **kwargs):
        super().__init__()
        self.mel_transform = MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft,
                                            n_mels=n_mels, **kwargs)

    def forward(self, x, y):
        '''Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: MSE distance of the melspectrogram            
        '''
        mel_x = self.mel_transform(x)
        mel_y = self.mel_transform(y)
        return F.mse_loss(mel_x, mel_y)


class MelSpecLoss(torch.nn.Module):
    '''Mel-scale spectrogram loss'''

    def __init__(self, loss_conf):
        super().__init__(loss_conf)
        win_lengths = loss_conf.get('win_lengths', None)
        self.mel_weight = loss_conf.get('mel_weight', 1e-5)
        self.L1_weight = loss_conf.get('L1_weight', 1.0)
        if win_lengths is None:
            fft_list = [256, 512, 768, 1024]  # , 1536, 2048
            mel_list = [32, 64, 96, 128]  # , 192, 256
        else:
            if type(win_lengths) is str:
                win_lengths = [int(item) for item in win_lengths.split(',')]
            fft_list = [power_of_2(item) for item in win_lengths]
            n_mels = [item//8 for item in win_lengths]
        self.melSpec_losses = torch.nn.ModuleList()
        mel_kwargs = {'sample_rate': conf.get('sample_rate', 16000)}
        for n_fft, n_mels in zip(fft_list, mel_list):
            mel_kwargs.update({'n_fft': n_fft, 'n_mels': n_mels})
            _loss = _Mel_loss(**mel_kwargs)
            self.melSpec_losses += [_loss]

    def forward(self, clean, est, *args):
        mel_loss = 0.0
        for f in self.melSpec_losses:
            mel_loss += f(est, clean)
        mel_loss /= len(self.melSpec_losses)
        L1_loss = F.l1_loss(clean, est)
        l = self.mel_weight * mel_loss + self.L1_weight * L1_loss
        return l


class FBankLoss(torch.nn.Module):
    '''FBankLoss with kaldi'''

    def __init__(self, loss_conf):
        super().__init__(loss_conf)
        fbank_param = OmegaConf.load(loss_conf.get('fbank_conf', 'conf/fbank.yaml'))
        self.fbank_transform = functools.partial(fbank, **fbank_param)
        self.FBank_weight = loss_conf.get('FBank_weight', None)
        self.L1_weight = loss_conf.get('L1_weight', None)

    def forward(self, clean, est, *args):
        fbank_clean = self.fbank_transform(clean)
        fbank_est = self.fbank_transform(est)
        l = F.mse_loss(fbank_clean, fbank_est)
        if self.FBank_weight:
            l = self.FBank_weight * l
        if self.L1_weight:
            l = l + self.L1_weight * F.l1_loss(clean, est)
        return l


class PESQLoss(BaseLoss):
    '''PESQ Loss'''
    def __init__(self, loss_conf):
        super().__init__(loss_conf)
        sample_rate = loss_conf.get('sample_rate', 16000)
        self.pesq_loss = PesqLoss(1.0, sample_rate=sample_rate).eval()
        for param in self.pesq_loss.parameters():
            param.requires_grad = False

    def forward(self, clean, est, *args):
        _pesq_loss = self.pesq_loss(clean, est)
        return torch.mean(_pesq_loss).float()


def test(conf):
    noisy = torch.randn((8, 2*16000))
    clean = torch.randn((8, 2*16000))
    est = torch.randn((8, 2*16000))
    print('MelSpecLoss:')
    for L1_weight, mel_weight in ([0.5, 0.5], [1.0, 0.0], [0.0, 1.0], [1.0, 1e-5]):
        loss_fun = MelSpecLoss(L1_weight, mel_weight)
        print(L1_weight, mel_weight, loss_fun(clean, est))
    

if __name__ == '__main__':
    conf = OmegaConf.create({
        'yaml': 'conf/UnSE_mag.yaml',
        'cmd': 'test'})
    conf.merge_with_cli()
    conf = OmegaConf.merge(OmegaConf.load(conf.yaml), conf)
    eval(conf.cmd)(conf)
