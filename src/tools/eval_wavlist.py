import pandas as pd
from pathlib import Path
from omegaconf import OmegaConf
import soundfile as sf
from modules.utils import metrics
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from modules.utils.common import compact_dict
from copy import deepcopy

def eval_wavfile(ref_file, est_file, fs=16000, truncate=True, mode='wb', extended=True):
    ref, fs1 = sf.read(ref_file)
    est, fs2 = sf.read(est_file)
    assert fs1 == fs
    assert fs2 == fs
    if truncate is True:
        est = est[:ref.shape[0]]
        ref = ref[:est.shape[0]]
    score = metrics.eval(ref, est, fs, mode, extended)
    return (ref_file.stem, score['PESQ'], score['eSTOI'], score['SI_SNR'])

    
def eval_wavlist(pairlist, n_workers=16, mode='wb', extended=True):
    """evalutate wav list
    Args:
        pairlist (list): pairlist of wav, [(ref, estimated), ...]
    Return:
        [(name, score1, score2, ...), ...]
    """
    future_tasks = []
    data_list = []
    pool = ProcessPoolExecutor(n_workers)
    for (ref_file, est_file) in pairlist:
        future_tasks.append(pool.submit(eval_wavfile, ref_file, est_file, 
                                        mode=mode, extended=extended))
    for f in tqdm(future_tasks):
        data_list.append(f.result())
    return data_list


def run_eval_wavlist(conf):
    pairlist = []    
    est_dir = Path(conf.est_dir).expanduser()
    ref_dir = Path(conf.ref_dir).expanduser()
    subset = conf.get('subset', None)
    if subset:
        df = pd.read_csv(subset, index_col=0, sep=" ", names=['id', 'wav_path', 'dur'])
        id_list = [str(item) for item in df.index.to_list()]
        for wav_id in id_list:
            pairlist.append([ref_dir.joinpath(f"{wav_id}.wav"), est_dir.joinpath(f"{wav_id}.wav")])
    else:
        for est_wav in est_dir.glob("*.wav"):
            pairlist.append([ref_dir.joinpath(est_wav.name), est_wav])
    data_list = eval_wavlist(pairlist, conf.n_workers, 
                             conf.get('mode', 'wb'), conf.get('extended', True))
    
    metric_list = conf.metric_list.split(",")
    columns = deepcopy(metric_list)
    columns.insert(0, "id")
    df = pd.DataFrame(data=data_list, columns=columns).sort_values(by="id")
    out_csv = conf.out_csv
    if out_csv is not None:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        print("to csv:", out_csv)
        df.to_csv(out_csv, index=None)
    
    if conf.key_list is not None:
        for key in conf.key_list.split(","):
            print(key, compact_dict(df[df.id.str.contains(key)][metric_list].mean()))
    print("Average", compact_dict(df[metric_list].mean()))


def run_eval_wavfile(conf):
    print(eval_wavfile(Path(conf.ref_file), Path(conf.est_file)))


if __name__ == '__main__':
    conf = OmegaConf.create({
         "cmd": "run_eval_wavlist",
         "n_workers": 16,
         "ref_dir": "exp/logs/SelfSE/3i9d1/clean",
         "est_dir": "exp/logs/SelfSE/3i9d1/denoise",
         "out_dir": "exp/results",
         "out_csv": None,
         "metric_list": "PESQ,eSTOI,SI_SNR",
         "key_list": None
         })
    conf.merge_with_cli()
    eval(conf.cmd)(conf)
    