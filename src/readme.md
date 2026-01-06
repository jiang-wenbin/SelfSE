## 1. Runtime Environment

### Create a Conda Environment

```bash
conda create -n speech python=3.10
conda activate speech
```

### Install Dependencies

Required packages:
- numpy==1.26.4
- torch==2.1.0
- lightning==2.2.5
- librosa==0.10.2
- onnxruntime-gpu==1.15.0
- h5py==3.11

Install with pip:

```bash
pip install numpy==1.26.4
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install lightning==2.2.5 librosa==0.10.2 onnxruntime-gpu==1.15.0 h5py==3.11
pip install ipython pandas matplotlib omegaconf rich pytest seaborn asteroid_filterbanks
pip install pystoi tensorboard einops loguru editdistance joblib pesq
```

### Test PyTorch

Run the PyTorch test:

```bash
python test/test_pytorch.py
```

## 2. Evaluation

### Navigate to the Example Directory
```bash
cd examples/voicebank
. path.sh
```

### Evaluate Results Using Evaluation Tools
Run the following command to calculate PESQ, eSTOI, and SI-SNR metrics:
```bash
python -m tools.eval_wavlist \
  ref_dir=exp/logs/SelfSE/3i9d1/clean \
  est_dir=exp/logs/SelfSE/3i9d1/denoise
```
{'PESQ': 2.98, 'eSTOI': 0.855, 'SI_SNR': 18.933}

To calculate DNSMOS scores, run:
```bash
python -m tools.DNSMOS.dnsmos_local -t exp/logs/SelfSE/3i9d1/denoise
```
{'SIG': 3.447, 'BAK': 4.037, 'OVRL': 3.175, 'P808_MOS': 3.497}

### Run Speech Enhancement (Inference)
Alternatively, you can perform speech enhancement directly using the pre-trained model:

```bash
python -m modules.launch cmd=denoise \
  root_dir=exp/logs/SelfSE/3i9d1 \
  model.ckpt=exp/logs/SelfSE/3i9d1/model.ckpt \
  denoise.wav_dir=exp/logs/SelfSE/3i9d1/noisy \
  denoise.out_dir=exp/logs/SelfSE/3i9d1/denoise
```