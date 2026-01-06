import numpy as np
from pystoi import stoi
from pesq import pesq
from tqdm import tqdm
import torch
from .common import EPS, compact_dict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


def vec_l2norm(x):
    return np.linalg.norm(x, 2)


def _power_sum(x):
    return np.sum(x ** 2, axis=-1)


def _inner(a, b):
    return np.sum(a * b, axis=-1)


def _remove_mean(x, axis=-1):
    return x - x.mean(axis=axis)


def snr(x, s):
    """Compute SNR
    Arguments:
        x: vector, enhanced/separated signal
        s: vector, reference signal (ground truth)    
    """
    assert s.shape == x.shape
    n = s - x
    _snr = 10 * np.log10(_power_sum(s).clip(EPS) / _power_sum(n).clip(EPS))
    # 由于截断误差，当噪声能量非常小（信噪比高于80dB），两者计算结果误差较大
    # _snr = 20 * np.log10(vec_l2norm(s).clip(EPS) / vec_l2norm(n).clip(EPS))
    return _snr


def si_snr(x, s, zero_mean=True):
    """
    Compute Si-SNR
    Arguments:
        x: vector, enhanced/separated signal
        s: vector, reference signal (ground truth)
    """
    if zero_mean is True:
        x = _remove_mean(x)
        s = _remove_mean(s)
    proj_x = (_inner(x, s)/_power_sum(s).clip(EPS)) * s  # projection of x on s
    n = x - proj_x
    _si_snr = 10 * np.log10(_power_sum(proj_x).clip(EPS) /
                            _power_sum(n).clip(EPS))
    return _si_snr


# for MetricGAN
def pesq_eval(pred_wav, target_wav):
    """Normalized PESQ (to 0-1)"""
    try:
        pesq_s = pesq(fs=16000, ref=target_wav.numpy(), deg=pred_wav.numpy(), mode="wb")
        pesq_s = (0.5 + pesq_s)/5
    except:
        pesq_s = 0.0
    return pesq_s


def eval(ref, degraded, fs=16000, mode='wb', extended=True):
    """evaluate speech quality
    """
    if torch.is_tensor(ref):
        ref = ref.squeeze().detach().cpu().numpy()
    if torch.is_tensor(degraded):
        degraded = degraded.squeeze().detach().cpu().numpy()

    stoi_s = stoi(ref, degraded, fs, extended)
    if extended is False:
        stoi_s = 100 * stoi_s
    try:
        pesq_s = pesq(fs, ref, degraded, mode)  # wb = wideband
    except Exception as e:
        # print(e)
        pesq_s = 0.0
    si_snr_s = si_snr(degraded, ref)
    return compact_dict({'PESQ': pesq_s, 'eSTOI': stoi_s, 'SI_SNR': si_snr_s})


def eval_batch(clean_wav, est_wav, fs=16000):
    # evaluate batch of wav
    n_batch, _ = est_wav.shape
    clean_wav_list, est_wav_list = [], []
    for i in range(n_batch):
        clean_wav_list.append(clean_wav[i, :])
        est_wav_list.append(est_wav[i, :])
    return eval_list(clean_wav_list, est_wav_list, fs)


def eval_list(clean_wav_list, est_wav_list, fs=16000, num_workers=8, disable_tqdm=True):
    # evaluate list of wav
    assert len(clean_wav_list) == len(est_wav_list)
    future_tasks = []
    score_list = []
    if num_workers > 1:
        with ProcessPoolExecutor(num_workers) as executor:
            for clean_wav, est_wav in zip(clean_wav_list, est_wav_list):
                future_tasks.append(executor.submit(eval, clean_wav, est_wav, fs))
            for f in tqdm(future_tasks, desc="Metrics", disable=disable_tqdm):
                score = f.result()
                score_list.append([score['PESQ'], score['eSTOI'], score['SI_SNR']])
    else:
        for clean_wav, est_wav in tqdm(zip(clean_wav_list, est_wav_list),
                                       total=len(clean_wav_list), desc="Metrics",
                                       disable=disable_tqdm):
            score = eval(clean_wav, est_wav, fs)
            score_list.append([score['PESQ'], score['eSTOI'], score['SI_SNR']])
    return zip(*score_list)
