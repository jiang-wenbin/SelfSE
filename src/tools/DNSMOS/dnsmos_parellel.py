from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
import numpy as np
import math
import soundfile as sf
import librosa
import onnxruntime as ort
import numpy.polynomial.polynomial as poly
from tqdm import tqdm
from tools import compact_dict

# Coefficients for polynomial fitting
COEFS_SIG = np.array([9.651228012789436761e-01, 6.592637550310214145e-01, 
                    7.572372955623894730e-02])
COEFS_BAK = np.array([-3.733460011101781717e+00,2.700114234092929166e+00,
                    -1.721332907340922813e-01])
COEFS_OVR = np.array([8.924546794696789354e-01, 6.609981731940616223e-01,
                    7.600269530243179694e-02])

def audio_logpowspec(audio, nfft=320, hop_length=160, sr=16000):
    powspec = (np.abs(librosa.core.stft(audio, n_fft=nfft, hop_length=hop_length)))**2
    logpowspec = np.log10(np.maximum(powspec, 10**(-12)))
    return logpowspec.T

def worker(audio_clips_list, input_length=9, sig_model_path='DNSMOS/model_bak/sig.onnx', bak_ovr_model_path='DNSMOS/model_bak/bak_ovr.onnx'):
    # DNSMOS worker
    id_list = []
    predicted_mos_sig = []
    predicted_mos_bak = []
    predicted_mos_ovr = []
    session_sig = ort.InferenceSession(sig_model_path)
    session_bak_ovr = ort.InferenceSession(bak_ovr_model_path)
    if type(audio_clips_list) is not list:
        audio_clips_list = [audio_clips_list]
    for fpath in audio_clips_list:
        audio, fs = sf.read(fpath)
        if len(audio)<2*fs:
            print('Audio clip is too short. Skipped processing ', fpath.name)
            continue
        len_samples = int(input_length*fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)
        
        num_hops = int(np.floor(len(audio)/fs) - input_length)+1
        hop_len_samples = fs        
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx*hop_len_samples) : int((idx+input_length)*hop_len_samples)]
            input_features = np.array(audio_logpowspec(audio=audio_seg, sr=fs)).astype('float32')[np.newaxis,:,:]

            onnx_inputs_sig = {inp.name: input_features for inp in session_sig.get_inputs()}
            mos_sig = poly.polyval(session_sig.run(None, onnx_inputs_sig), COEFS_SIG)
                
            onnx_inputs_bak_ovr = {inp.name: input_features for inp in session_bak_ovr.get_inputs()}
            mos_bak_ovr = session_bak_ovr.run(None, onnx_inputs_bak_ovr)

            mos_bak = poly.polyval(mos_bak_ovr[0][0][1], COEFS_BAK)
            mos_ovr = poly.polyval(mos_bak_ovr[0][0][2], COEFS_OVR)
            
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
        id_list.append(fpath.stem)
        predicted_mos_sig.append(np.mean(predicted_mos_sig_seg))
        predicted_mos_bak.append(np.mean(predicted_mos_bak_seg))
        predicted_mos_ovr.append(np.mean(predicted_mos_ovr_seg))
    return id_list, predicted_mos_sig, predicted_mos_bak, predicted_mos_ovr

def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    return [arr[i:i + n] for i in range(0, len(arr), n)]

def run_eval_wavlist(conf):    
    test_path = Path(conf.test_dir)
    audio_clips_list = list(test_path.expanduser().glob("*.wav"))
    if len(audio_clips_list) == 0:
        print("There is no wav in: ", conf.test_dir)
        return
    future_tasks, data_list = [], []
    pool = ProcessPoolExecutor(conf.n_workers)
    audio_clips_list_chunk = chunks(audio_clips_list, conf.n_workers)
    for audio_clips in audio_clips_list_chunk:
        future_tasks.append(pool.submit(worker, audio_clips))        
    for f in tqdm(future_tasks):
        data_list.append(f.result())    
    ID_list, SIG_list, BAK_list, OVR_list = [], [], [], []
    for id, sig, bak, ovr in data_list:         
        ID_list.extend(id)
        SIG_list.extend(sig)
        BAK_list.extend(bak)
        OVR_list.extend(ovr)
    data = [ID_list, SIG_list, BAK_list, OVR_list]
    df = pd.DataFrame(data=zip(*data), columns=["id", 'SIG', 'BAK', 'OVR']).sort_values(by="id")
    out_csv = conf.out_csv
    if out_csv is None:
        name = "{}_{}".format(test_path.parent.parent.name, test_path.parent.name)
        Path(conf.out_dir).mkdir(exist_ok=True)
        out_csv = Path(conf.out_dir).joinpath(f"{name}.csv")
    print("to csv:", out_csv)
    df.to_csv(out_csv, index=None)
    df_mean = df.mean()
    result = compact_dict({"SIG": df_mean["SIG"], "BAK": df_mean["BAK"], "OVR": df_mean["OVR"]})
    print(result)

from omegaconf import OmegaConf
if __name__=="__main__":
    conf = OmegaConf.create({
         "cmd": "run_eval_wavlist",
         "n_workers": 10,
         "test_dir": "exp/logs/MagHomoSE_CRN_Fusion/1a/clean",
         "out_dir": "exp/DNSMOS",
         "out_csv": None
         })
    conf.merge_with_cli()
    eval(conf.cmd)(conf)