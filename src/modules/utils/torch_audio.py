# -*- coding: utf-8 -*-
"""
pytorch version of https://github.com/microsoft/DNS-Challenge/blob/master/audiolib.py
"""
import torch

EPS = torch.finfo(float).eps

def is_clipped(audio, clipping_threshold=0.99):
    return torch.any(torch.abs(audio) > clipping_threshold)

def normalize_with_ref(audio, ref_audio):
    '''Normalize the signal with refrence audio'''
    # rms = (audio ** 2).mean() ** 0.5
    # ref_rms = (ref_audio ** 2).mean() ** 0.5
    rms = torch.mean(audio ** 2, dim=1, keepdim=True) ** 0.5
    ref_rms = torch.mean(ref_audio ** 2, dim=1, keepdim=True) ** 0.5
    scalar = ref_rms / (rms+EPS)
    audio = audio * scalar
    return audio

def normalize(audio, target_level=-25):
    '''Normalize the signal to the target level'''
    rms = torch.mean(audio ** 2, dim=1, keepdim=True) ** 0.5
    scalar = 10 ** (target_level / 20) / (rms+EPS)
    audio = audio * scalar
    return audio

def calc_rms(audio):
    return torch.mean(audio ** 2, dim=1, keepdim=True) ** 0.5

def calc_max(audio):
    audio_max, _ = torch.max(torch.abs(audio), dim=1, keepdim=True)
    return audio_max

def snr_mixer(params, clean, noise, snr=None, target_level=-25, clipping_threshold=0.99):
    '''Function to mix clean speech and noise at various SNR levels
    clean, noise: shape of [n_batch, n_samples]
    snr: shape of [n_batch, 1]
    '''
    if clean.shape != noise.shape:
        raise Exception("clean.shape: {}, noise.shape: {}".format(clean.shape, noise.shape))
    if snr is None:
        snr = torch.randint(params["snr_lower"], params["snr_upper"], (clean.shape[0],1)).type_as(clean)
    # Normalizing to -25 dB FS
    clean_max = calc_max(clean)
    clean = clean/(clean_max+EPS)
    clean = normalize(clean, target_level)
    rmsclean = calc_rms(clean)
    
    noise_max = calc_max(noise)
    noise = noise/(noise_max+EPS)
    noise = normalize(noise, target_level)
    rmsnoise = calc_rms(noise)

    # Set the noise level for a given SNR
    noisescalar = rmsclean / (10**(snr/20)) / (rmsnoise+EPS)
    noisenewlevel = noise * noisescalar

    # Mix noise and clean speech
    noisyspeech = clean + noisenewlevel
    
    # Randomly select RMS value between -15 dBFS and -35 dBFS and normalize noisyspeech with that value
    # There is a chance of clipping that might happen with very less probability, which is not a major issue. 
    noisy_rms_level = torch.randint_like(snr, params['target_level_lower'], params['target_level_upper'])
    rmsnoisy = calc_rms(noisyspeech)
    scalarnoisy = 10 ** (noisy_rms_level / 20) / (rmsnoisy+EPS)
    noisyspeech = noisyspeech * scalarnoisy
    clean = clean * scalarnoisy
    noisenewlevel = noisenewlevel * scalarnoisy
    
    # Final check to see if there are any amplitudes exceeding +/- 1. If so, normalize all the signals accordingly
    if is_clipped(noisyspeech) == True:
        noisyspeech_maxamplevel = calc_max(noisyspeech)
        noisyspeech_maxamplevel = noisyspeech_maxamplevel/(clipping_threshold-EPS)
        noisyspeech = noisyspeech/noisyspeech_maxamplevel
        clean = clean/noisyspeech_maxamplevel
        noisenewlevel = noisenewlevel/noisyspeech_maxamplevel
        noisy_rms_level = (20*torch.log10(scalarnoisy/noisyspeech_maxamplevel*(rmsnoisy+EPS))).int()

    return clean, noisenewlevel, noisyspeech, noisy_rms_level


def test_snr_mixer(conf):
    n_batch = 8
    params = {"snr_lower": 0, "snr_upper": 10,
              "target_level_lower": -35, "target_level_upper": -15}
    snr = torch.randint(params["snr_lower"], params["snr_upper"], (n_batch,1))
    clean = torch.rand(n_batch, 16000*4) # n_batch, n_samples
    noise = torch.rand(n_batch, 16000*4)
    assert is_clipped(clean) == True
    assert is_clipped(0.9*clean) == False
    clean_norm = normalize(clean)        
    _, _, noisy, _ = snr_mixer(params, clean, noise, snr)
    
from omegaconf import OmegaConf    
if __name__ == '__main__':
    conf = OmegaConf.create({"cmd": "test_snr_mixer"})
    conf.merge_with_cli()
    eval(conf.cmd)(conf)