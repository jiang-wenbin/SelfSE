import torch
from tqdm import tqdm
import librosa
from pathlib import Path
from .utils import audio, torch_signal as signal
from .utils.logging import logger
from .train import get_model


def denoise(conf):
    def _pad_wav(wav, conf):
        return signal.padding(wav, conf['stft']['win_length'], conf['stft']['hop_length'])
    device = conf.get('device', 'cuda')
    assert 'denoise' in conf
    out_dir = Path(conf.denoise.out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    if conf.denoise.get('wavlist', None):
        with open(conf.denoise.wavlist) as f:
            content = f.readlines()
            wav_list = [Path(wav.strip()) for wav in content]
    else:
        wav_list = list(Path(conf.denoise.wav_dir).expanduser().rglob('*.wav'))
    logger.info(f'wav_list: {wav_list[:3]} ...')
    model = get_model(conf).to(device)
    model.eval()  # evaluation mode
    for _wav in tqdm(wav_list):
        y, sr = librosa.load(_wav, sr=conf.get('sr', None))
        y_tensor = torch.as_tensor(y, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            est_wav = model(_pad_wav(y_tensor, conf))[1].squeeze()
            est_wav = est_wav[0:y_tensor.shape[1]] # remove the padded
        est_wav = est_wav.cpu().numpy()
        audio.audiowrite(out_dir.joinpath(_wav.name), est_wav, sample_rate=sr)
