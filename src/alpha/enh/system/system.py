import torch
import sys
import random
import numpy as np
import torch.optim.lr_scheduler as LR_scheduler
import modules.model.arch as model_arch
from modules.utils.common import EPS, torch_float32, refine_state_dict, \
    check_model_isnan, merge_dicts, resolve_path
from modules.utils.logging import logger
from modules.utils import torch_audio
from modules.system import BaseModule
from modules.stft import STFT
from omegaconf import OmegaConf
from modules.utils import metrics
from .. import loss as loss_module
from .. import model as models


class BaseSE(BaseModule):
    def __init__(self, conf):
        super().__init__(conf)
        self.stft = STFT(self.stft_conf)
        self.model_class = getattr(models, conf['model']['name'], None)
        self.resolve_model_param()
        self.mix_param = conf['data'].get('mix_param', None)
        self.loss_name = conf['loss'].get('name')
        self.loss = self.get_loss(conf['loss'])
        self.valid_loss = self.init_valid_loss(conf.get('valid_loss', None))
        self.feature = None
        self.deepfilter = self.init_deepfilter(conf)
        self.model_conf = OmegaConf.to_container(self.conf['model'], resolve=True)
        self.check_nan = conf['system'].get('check_nan', False)
        self.rms_norm = conf['system'].get('rms_norm', False) # apply RMS normalization

    def resolve_model_param(self):
        model_param = self.conf['model'].get('param', None)
        if type(model_param) is str:
            self.conf['model']['param'] = getattr(model_arch, model_param, None)
            
    def get_loss(self, loss_conf):
        name = loss_conf.get('name')
        if not name:
            return None
        if hasattr(loss_module, name):
            loss = getattr(loss_module, name)(loss_conf)
        elif hasattr(torch.nn, name):
            loss = getattr(torch.nn, name)()
        else:
            raise ValueError(f'Loss name error: {name}')
        return loss

    def init_valid_loss(self, loss_conf):        
        return self.get_loss(loss_conf) if loss_conf else None
    
    def init_model(self, model_conf):
        model_path = model_conf.get('ckpt')
        if model_path:
            model_path = resolve_path(model_path)
            logger.info(f'Model initialize with: {model_path}')
            model_dict = torch.load(model_path)
            if 'state_dict' in model_dict:  # pytorch lightning ckpt
                model_dict = refine_state_dict(model_dict['state_dict'])
            self.model.load_state_dict(model_dict)
            
    def init_deepfilter(self, conf):
        deepfilter = conf.get('deepfilter', None)
        if deepfilter:
            assert self.mask is True
            deepfilter.update({
                'n_freq': conf['feature']['n_feature'] if self.feature else
                conf['stft']['n_fft']//2 + 1})
        return deepfilter

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def configure_optimizers(self):
        if self.conf.get('xformer_opt', False):
            self.conf['warmup'] = True  # set the warmup flag
            optimizer = torch.optim.Adam(self.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
            def lr_lambda(epoch): return 4e-4 * (0.98 ** (max((epoch-1)//2, 0)))
            scheduler = LR_scheduler.LambdaLR(optimizer, lr_lambda)
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler}}
        else:
            return super().configure_optimizers()

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        # update params
        optimizer.step(closure=optimizer_closure)
        # manually warm up lr without a scheduler
        if self.conf.get('warmup', False):
            k, d_model, warmup_steps = 0.2, 64, 4000
            n_step = self.trainer.global_step + 1  # start from 1
            if n_step <= warmup_steps:
                lr = k * d_model**(-0.5) * min(n_step ** (-0.5), n_step * (warmup_steps ** (-1.5)))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                # print('Adjusting learning rate to {:.4e}.'.format(lr))

    @property
    def snr(self):
        return random.randint(self.mix_param.snr_lower, self.mix_param.snr_upper)
    
    def _calc_stft(self, x):
        x_stft = self.stft.apply_stft(x)
        x_mag = torch.abs(x_stft)
        x_phase = torch.angle(x_stft)
        return x_mag, x_phase
    
    def _calc_istft(self, x_mag, x_phase, **kwargs):
        x_stft = x_mag * torch.exp(1.0j*x_phase)
        x = self.stft.apply_istft(x_stft, **kwargs)
        return x
    
    def test_stft(self, x):
        # x: wave
        x_mag, x_phase = self._calc_stft(x)
        x_r = self._calc_istft(x_mag, x_phase)
        score = metrics.eval(x, x_r, self.sample_rate)
        print(score)


class UniSE(BaseSE):
    # Single head speech enhancement, using magnitude and/or complex stft as input
    def __init__(self, conf):
        super().__init__(conf)
        in_channels = 0
        self.spec_data = conf['model'].get('spec_data', True)
        self.mag_data = conf['model'].get('mag_data', False)
        if self.spec_data:
            in_channels = in_channels + 2
        if self.mag_data:
            in_channels = in_channels + 1
        
        # the default model configuration
        model_conf = {
            "in_channels": in_channels,
            "mag_decoder": False,
            "spec_decoder": True,
            'debug': self.debug,
            'rnn': {'bidirectional': True}
        }
        self.model_conf = merge_dicts(model_conf, self.conf['model']) # merge
        self.model = self.model_class(self.model_conf)
        
        # extra loss
        self.ext_loss = {}
        if conf.get('mag_loss', None):            
            self.ext_loss['mag_loss'] = (getattr(torch.nn, conf.get('mag_loss_fun', 'MSELoss'))(), 
                                         conf['mag_loss'])
        if conf.get('spec_loss', None):
            self.ext_loss['spec_loss'] = (loss_module.CMSELoss(), conf['spec_loss'])
        if self.ext_loss:
            logger.info(f"ext_loss: {self.ext_loss}")
        self.extra_loss_log2 = conf.get('extra_loss_log2', False)
        
    def apply_loss_fun(self, clean_wav, est_wav, clean_spec, est_spec,
                       clean_mag=None, est_mag=None):
        loss = 0.0
        loss_fun_w = self.conf.get('loss_fun_w', 1.0)
        if loss_fun_w > 0: # time-domain loss
            _loss = loss_fun_w * self.loss(clean_wav, est_wav) 
            self.log("train/time_loss", _loss, sync_dist=True, batch_size=self.batch_size)
            loss = loss + _loss
        if 'spec_loss' in self.ext_loss: # complex spectrum loss
            _loss_fun, loss_w = self.ext_loss['spec_loss']
            _spec_loss = _loss_fun(clean_spec, est_spec)
            if self.extra_loss_log2 is True:
                _spec_loss = torch.log2(_spec_loss.clip(EPS))
            _spec_loss = loss_w * _spec_loss
            self.log("train/spec_loss", _spec_loss, sync_dist=True, batch_size=self.batch_size)
            loss = loss + _spec_loss
        if 'mag_loss' in self.ext_loss: # magnitude spectrum loss
            _loss_fun, loss_w = self.ext_loss['mag_loss']
            if est_mag is None:
                est_mag = torch.abs(est_spec)
            _mag_loss = _loss_fun(clean_mag, est_mag)
            if self.extra_loss_log2 is True:
                _mag_loss = torch.log2(_mag_loss.clip(EPS))
            _mag_loss = loss_w * _mag_loss
            self.log("train/mag_loss", _mag_loss, sync_dist=True, batch_size=self.batch_size)
            loss = loss + _mag_loss
        return loss
    
    def forward(self, noisy_wav, clean_wav=None, train=True, ret_mask=False):
        noisy_spec = self.stft.apply_stft(noisy_wav)
        noisy_mag = torch.abs(noisy_spec)
        data_list = []
        if self.spec_data:
            data_list.extend([noisy_spec.real, noisy_spec.imag])
        if self.mag_data:
            data_list.append(noisy_mag)
        noisy_input = torch.cat(data_list, dim=1)
        est_spec, est_mag = self.model(noisy_input)
        if est_spec is not None: # output of spec_decoder
            est_spec = torch.complex(est_spec[:,0,...], est_spec[:,1,...]).unsqueeze(1)
            est_spec = est_spec * noisy_spec if self.mask else est_spec
            if est_mag is not None: # output of mag_decoder
                est_mag = est_mag * noisy_mag if self.mask else est_mag
                if self.conf.get('noisy_phase', True):
                    noisy_phase = torch.angle(noisy_spec)
                    final_spec = (est_spec + est_mag * torch.exp(1.0j*noisy_phase))/2
                else: # don't use it !
                    final_mag = (torch.abs(est_spec) + est_mag) / 2
                    est_phase = torch.angle(torch.complex(est_spec.real.clip(EPS), 
                                                          est_spec.imag.clip(EPS))) # fix nan
                    final_spec = final_mag * torch.exp(1.0j*est_phase)
            else:
                est_mag = None
                final_spec = est_spec
        else:
            est_spec = None # don't set spec_loss !!!
            if est_mag is not None: # output of mag_decoder
                est_mag = est_mag * noisy_mag if self.mask else est_mag
                noisy_phase = torch.angle(noisy_spec) # use noisy phase
                final_spec =  est_mag * torch.exp(1.0j*noisy_phase)
            else:
                raise ValueError('both est_spec and est_mag are None')
        
        est_wav = self.stft.apply_istft(final_spec)
        if self.rms_norm is True:
            est_wav = torch_audio.normalize_with_ref(est_wav, noisy_wav)
        if not train or clean_wav is None:
            return None, est_wav
        
        if (not self.model.training) and self.valid_loss is not None:
            _loss = self.valid_loss(clean_wav, est_wav)
        else:
            if len(self.ext_loss) > 0:
                clean_spec = self.stft.apply_stft(clean_wav)
                clean_mag = torch.abs(clean_spec)
                _loss = self.apply_loss_fun(clean_wav, est_wav, clean_spec, est_spec,
                                            clean_mag, est_mag)
            else:
                if self.loss_name == 'wSDRLoss': # require noisy wav
                    _loss = self.loss(clean_wav, est_wav, noisy_wav)
                elif self.loss_name in ['CMSELoss']: # frequency domain loss
                    _loss = self.loss(self.stft.apply_stft(clean_wav), est_spec)
                else:
                    _loss = self.loss(clean_wav, est_wav)
        if self.check_nan:
            data_file = f"exp/debug/{self.global_step}.pt"
            if check_model_isnan(self.model.state_dict()) or torch.isnan(est_wav).any() or _loss.isnan():
                logger.info("check_nan, model: {}, est_wav: {}, loss: {}".format(
                    check_model_isnan(self.model.state_dict()), torch.isnan(est_wav).any(), _loss.isnan()))
                torch.save([_loss, noisy_wav, clean_wav, est_wav, self.model.state_dict()], data_file)
                sys.exit(0)
        if ret_mask is True:
            return _loss, est_wav, est_spec, est_mag
        return _loss, est_wav

