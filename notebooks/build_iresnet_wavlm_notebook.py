"""
Generator script to build the SOTA Bi-directional MHA Multimodal Architecture:
1. Audio/Voice Encoder: microsoft/wavlm-large-sv (or Wav2Vec2-Conformer / microsoft/wavlm-large)
2. Vision/Face Encoder: iresnet100 (ArcFace Backbone, 112x112 resolution)
3. Fusion Model: Bi-directional Multihead Attention (Bi-MHA)
4. Gender Bias Removal: GRL (Gradient Reversal Layer) + Orthogonal Splitter
5. Loss Functions: Sub-Center ArcFace Loss (K=3) + Orthogonal Subspace Loss
"""

import json
import os

def create_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    def add_md(text):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def add_code(text):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    # =========================================================================
    # CELL 1: HEADER & ARCHITECTURE SPECIFICATION
    # =========================================================================
    add_md(r"""# 🏆 FLAG 2027 Challenge: Face-Voice Association Across Languages and Gender
## 🌟 **FLAG-BiMHA-SubArc-Omni**: IResNet-100 + WavLM-SV + Bi-MHA + Orthogonal Splitter + Sub-Center ArcFace
### *ICASSP 2027 Challenge on MAV-Celeb v4 (English Heard & Bengali Unheard Zero-Shot)*

---

### 📌 Architectural Blueprint & Innovation Stack
This notebook implements a dedicated biometric face-voice association architecture with mathematically rigorous demographic debiasing:

```
                       ┌─────────────────────────────────────────────────────────┐
                       │             MULTIMODAL INPUTS: FACE & AUDIO             │
                       └────────────────────┬────────────────┬───────────────────┘
                                            │                │
                    ┌───────────────────────┴────┐      ┌────┴───────────────────────────┐
                    │     FACE (RGB 112x112)     │      │   SPEECH (16 kHz, 3.0s Mono)   │
                    └───────────────┬────────────┘      └────────────────┬───────────────┘
                                    │                                    │
                    ┌───────────────▼────────────┐      ┌────────────────▼───────────────┐
                    │  IResNet-100 (ArcFace)     │      │ WavLM-Large / WavLM-Base-SV    │
                    │  Deep Conv Biometric ViT   │      │ Masked Denoising Speech SSL    │
                    │  512-dim Craniofacial Feat │      │ 1024-dim Acoustic Timbre Feat  │
                    └───────────────┬────────────┘      └────────────────┬───────────────┘
                                    │ Spatial Tokens (B, 8, 512)         │ Temporal Tokens (B, 8, 512)
                                    │                                    │
                    ┌───────────────▼────────────────────────────────────▼───────────────┐
                    │         BI-DIRECTIONAL MULTIHEAD ATTENTION FUSION (Bi-MHA)         │
                    │         1. Face queries Audio:   F_ctx = MHA(Q=Face, K=Aud, V=Aud) │
                    │         2. Audio queries Face:   A_ctx = MHA(Q=Aud, K=Face, V=Face)│
                    │         Residual Connection + LayerNorm + Feed-Forward Network     │
                    └───────────────────────────────┬────────────────────────────────────┘
                                                    │ Fused Biometric State (512-dim)
                    ┌───────────────────────────────▼────────────────────────────────────┐
                    │             ORTHOGONAL SUBSPACE SPLITTER (Decomposition)           │
                    │      1. Identity Subspace:   z_id in R^512  (Biometric Signal)     │
                    │      2. Gender Subspace:     z_gen in R^256 (Demographic Signal)   │
                    │      Strict Orthogonality:   <z_id, z_gen> = 0 (Loss_orth -> 0)    │
                    └───────────────────────┬───────────────────┬────────────────────────┘
                                            │                   │
                    ┌───────────────────────▼────┐      ┌───────▼────────────────────────┐
                    │ GRL DEMOGRAPHIC DEBIASING  │      │ SUPERVISED GENDER CLASSIFIER   │
                    │ Inverts Gradients to Purge │      │ Captures all demographic       │
                    │ Gender from z_id           │      │ variance into z_gen            │
                    └───────────────┬────────────┘      └────────────────┬───────────────┘
                                    │                                    │
                    ┌───────────────▼────────────────────────────────────▼───────────────┐
                    │                       COMPOSITE LOSS SUITE                         │
                    │  1. Sub-Center ArcFace Loss: K=3 Multi-Prototype Angular Margin    │
                    │  2. Orthogonal Subspace Loss: Cosine^2(z_id, z_gen) Penalty        │
                    │  3. Adversarial GRL Gender Loss: Demographic Purging on z_id       │
                    │  4. Supervised Gender Loss: Demographic Anchoring on z_gen         │
                    │  5. Supervised Contrastive Loss: Same-Gender Negative Impostors    │
                    └────────────────────────────────────────────────────────────────────┘
```

### 🔑 Component Deep-Dive:
1. **Vision / Face Encoder**: **IResNet-100 (ArcFace Backbone, 65M Parameters)**.
   - Standard 112×112 resolution. Uses deep residual blocks with PReLU activations and Batch Normalization. Specially pre-trained for extreme facial biometric discrimination across yaw, pitch, expression, and age.
2. **Audio / Voice Encoder**: **Microsoft WavLM-Large-SV / WavLM-Base-SV (Speech Biometrics)**.
   - Masked speech denoising Self-Supervised Learning specifically tailored for speaker verification. Preserves anatomical vocal tract invariants and pitch distribution across linguistic boundaries.
3. **Fusion Mechanism**: **Bi-directional Multihead Attention (Bi-MHA)**.
   - Cross-modal mutual query: Face attends to Audio, and Audio attends to Face. Mutual guidance ensures facial features highlight corresponding speech acoustic frequencies and vice-versa.
4. **Gender Bias Removal**: **Orthogonal Subspace Splitter + GRL**.
   - Decomposes the embedding into an **Identity Subspace ($z_{\text{id}}$)** and a **Gender Subspace ($z_{\text{gen}}$)**.
   - Enforces mathematical orthogonality: $\mathcal{L}_{\text{orth}} = \cos^2(z_{\text{id}}, z_{\text{gen}}) \to 0$.
   - Gradient Reversal Layer (GRL) connected to $z_{\text{id}}$ renders the identity features incapable of predicting gender, while $z_{\text{gen}}$ actively absorbs all demographic variance.
5. **Loss Suite**:
   - **Sub-Center ArcFace Loss ($K=3$)**: Extends additive angular margin by allocating $K$ distinct prototypes per speaker identity. Accommodates multimodal distributions (e.g. speaking English vs speaking Bengali, laughing vs neutral) without inflating intra-class dispersion.
   - **Orthogonal Subspace Loss**: Guarantees zero linear correlation between identity and gender dimensions.
""")

    # =========================================================================
    # CELL 2: DEPENDENCIES & ENVIRONMENT SETUP
    # =========================================================================
    add_md("### 📦 1. Installation & Hardware Environment Setup")
    add_code("""# Install required libraries
!pip install -q --upgrade pip
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers accelerate scikit-learn scipy Pillow

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_AUTO_CONVERSION"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Check for Kaggle Secrets HF_TOKEN
try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    hf_token = secrets.get_secret("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        print("[Auth] Successfully loaded HF_TOKEN from Kaggle Secrets.")
except Exception:
    pass

import sys
import gc
import math
import glob
import random
import zipfile
import shutil
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler, DataLoader
import torchaudio
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from accelerate import Accelerator

accelerator = Accelerator(mixed_precision="fp16")
device = accelerator.device

print("=" * 70)
print(f"PyTorch Version       : {torch.__version__}")
print(f"Accelerate Device     : {device}")
print(f"Num Processes / GPUs  : {accelerator.num_processes}")
print(f"Mixed Precision       : {accelerator.mixed_precision}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU [{i}]               : {props.name} ({props.total_memory / (1024**3):.2f} GB VRAM)")
print("=" * 70)
""")

    # =========================================================================
    # CELL 3: CONFIGURATION CLASS
    # =========================================================================
    add_md("### ⚙️ 2. Hyperparameters & Architecture Configuration")
    add_code("""@dataclass
class IResNetWavLMConfig:
    # --------------------------------------------------------------------------
    # 1. Dataset & Submission Paths
    # --------------------------------------------------------------------------
    DATA_ROOT: str = "/kaggle/input/datasets/mdjahidhasanjim/mav-celeb-v4-dataset"
    OUTPUT_DIR: str = "/kaggle/working"
    BEST_MODEL_PATH: str = "/kaggle/working/best_iresnet_wavlm.pth"
    SUBMISSION_DIR: str = "/kaggle/working/submission"
    SUBMISSION_ZIP_PATH: str = "/kaggle/working/submission.zip"
    
    SUBMISSION_GENDER_ENGLISH: str = "/kaggle/working/submission/gender/sub_score_v4_English_heard.txt"
    SUBMISSION_GENDER_BANGLA: str = "/kaggle/working/submission/gender/sub_score_v4_Bangla_unheard.txt"
    SUBMISSION_NO_GENDER_ENGLISH: str = "/kaggle/working/submission/no_gender/sub_score_v4_English_heard.txt"
    SUBMISSION_NO_GENDER_BANGLA: str = "/kaggle/working/submission/no_gender/sub_score_v4_Bangla_unheard.txt"

    # --------------------------------------------------------------------------
    # 2. Model Backbones
    # --------------------------------------------------------------------------
    # Vision: IResNet-100 (ArcFace deep convolutional face recognition backbone)
    FACE_IMAGE_SIZE: int = 112
    FACE_FEAT_DIM: int = 512
    
    # Audio: Microsoft WavLM-Large / WavLM-Base-SV
    AUDIO_MODEL_ID: str = "microsoft/wavlm-large"
    AUDIO_FALLBACK_ID: str = "microsoft/wavlm-base-sv"
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_DURATION: float = 3.0
    AUDIO_SAMPLES: int = int(AUDIO_SAMPLE_RATE * AUDIO_DURATION) # 48,000 samples

    # --------------------------------------------------------------------------
    # 3. Bi-directional Multihead Attention (Bi-MHA) Fusion
    # --------------------------------------------------------------------------
    FUSION_DIM: int = 512
    NUM_HEADS: int = 8
    FUSION_DROPOUT: float = 0.1
    NUM_TOKENS_PER_MODALITY: int = 8

    # --------------------------------------------------------------------------
    # 4. Orthogonal Subspace Splitter & Metric Learning
    # --------------------------------------------------------------------------
    ID_SUBSPACE_DIM: int = 512
    GEN_SUBSPACE_DIM: int = 256
    
    # Sub-Center ArcFace Parameters
    SUBCENTER_K: int = 3          # 3 sub-centers per speaker class
    ARCFACE_SCALE: float = 32.0   # Hyperspherical radius scale s
    ARCFACE_MARGIN: float = 0.35  # Angular margin m in radians
    
    # Loss Balancing Coefficients
    LAMBDA_SUBCENTER: float = 1.0 # Intra-speaker identity compactness
    LAMBDA_ORTH: float = 0.3      # Orthogonal subspace separation
    LAMBDA_GRL: float = 0.5       # Adversarial demographic debiasing on z_id
    LAMBDA_GEN: float = 0.5       # Demographic supervision on z_gen
    LAMBDA_SUPCON: float = 0.5    # Cross-modal pull/push alignment

    # --------------------------------------------------------------------------
    # 5. Training Engine
    # --------------------------------------------------------------------------
    BATCH_SIZE: int = 16          # Fast & stable for ~400M parameter model
    GRAD_ACCUM_STEPS: int = 2     # Effective batch size = 32
    NUM_EPOCHS: int = 25
    LR_BACKBONE: float = 2e-5     # Fine-tuning rate for WavLM & IResNet
    LR_FUSION: float = 2e-4       # Learning rate for Bi-MHA, Splitter, Heads
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP_NORM: float = 3.0
    NUM_WORKERS: int = 2
    SEED: int = 42

    # --------------------------------------------------------------------------
    # 6. Early Stopping
    # --------------------------------------------------------------------------
    EARLY_STOPPING_PATIENCE: int = 5
    EARLY_STOPPING_MIN_DELTA: float = 1e-4
    EARLY_STOPPING_MODE: str = "min"
    EARLY_STOPPING_RESTORE_BEST: bool = True
    VAL_SPLIT_RATIO: float = 0.1

cfg = IResNetWavLMConfig()

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

seed_everything(cfg.SEED)
""")

    # =========================================================================
    # CELL 4: DATASET DISCOVERY
    # =========================================================================
    add_md("### 🔍 3. Dataset Discovery & Path Normalization")
    add_code("""class KaggleDatasetLocator:
    @staticmethod
    def locate_paths(data_root: str) -> Tuple[str, str]:
        train_dir = None
        dev_dir = None
        
        candidates_train = [
            os.path.join(data_root, "train_set", "train_set", "train_set"),
            os.path.join(data_root, "train_set", "train_set"),
            os.path.join(data_root, "train_set"),
        ]
        for c in candidates_train:
            if os.path.isdir(c) and os.path.exists(os.path.join(c, "faces")):
                train_dir = c
                break

        candidates_dev = [
            os.path.join(data_root, "dev_set", "dev_set"),
            os.path.join(data_root, "dev_set"),
        ]
        for c in candidates_dev:
            if os.path.isdir(c) and (os.path.exists(os.path.join(c, "gender")) or os.path.exists(os.path.join(c, "features"))):
                dev_dir = c
                break

        if not train_dir:
            for root, dirs, _ in os.walk(data_root):
                r_lower = root.lower()
                if "dev" in r_lower or "test" in r_lower:
                    continue
                if "faces" in dirs and "voices" in dirs:
                    train_dir = root
                    break

        if not dev_dir:
            for root, dirs, _ in os.walk(data_root):
                if "gender" in dirs and ("no_gender" in dirs or "features" in dirs):
                    dev_dir = root
                    break

        return train_dir, dev_dir

TRAIN_DIR, DEV_DIR = KaggleDatasetLocator.locate_paths(cfg.DATA_ROOT)
if accelerator.is_main_process:
    print(f"[Dataset] Resolved TRAIN_DIR: {TRAIN_DIR}")
    print(f"[Dataset] Resolved DEV_DIR  : {DEV_DIR}")
    os.makedirs(cfg.SUBMISSION_DIR, exist_ok=True)
    os.makedirs(os.path.join(cfg.SUBMISSION_DIR, "gender"), exist_ok=True)
    os.makedirs(os.path.join(cfg.SUBMISSION_DIR, "no_gender"), exist_ok=True)
""")

    # =========================================================================
    # CELL 5: PREPROCESSING PIPELINES
    # =========================================================================
    add_md("### 🎙️ 4. Audio & Face Preprocessing Pipelines (112x112 ArcFace)")
    add_code("""class AudioPreprocessor:
    def __init__(self, target_sr=16000, target_samples=48000, is_train=True):
        self.target_sr = target_sr
        self.target_samples = target_samples
        self.is_train = is_train
        self.resamplers = {}

    def _get_resampler(self, orig_sr: int) -> torchaudio.transforms.Resample:
        if orig_sr not in self.resamplers:
            self.resamplers[orig_sr] = torchaudio.transforms.Resample(orig_sr, self.target_sr)
        return self.resamplers[orig_sr]

    def process(self, audio_path: str) -> torch.Tensor:
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception:
            return torch.zeros(self.target_samples, dtype=torch.float32)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != self.target_sr:
            waveform = self._get_resampler(sr)(waveform)

        waveform = waveform.squeeze(0)
        n = waveform.shape[0]

        if n < self.target_samples:
            if n > 0:
                rep = (self.target_samples // n) + 1
                waveform = waveform.repeat(rep)[:self.target_samples]
            else:
                waveform = torch.zeros(self.target_samples, dtype=torch.float32)
        elif n > self.target_samples:
            if self.is_train:
                st = random.randint(0, n - self.target_samples)
                waveform = waveform[st : st + self.target_samples]
            else:
                st = (n - self.target_samples) // 2
                waveform = waveform[st : st + self.target_samples]

        if self.is_train:
            if random.random() < 0.5:
                shift = random.randint(-8000, 8000)
                waveform = torch.roll(waveform, shifts=shift, dims=0)
            if random.random() < 0.4:
                waveform = waveform * random.uniform(0.8, 1.2)

        std = waveform.std()
        if std > 1e-6:
            waveform = (waveform - waveform.mean()) / std
        else:
            waveform = torch.zeros_like(waveform)

        return waveform.to(torch.float32)


def get_iresnet_transforms(image_size=112, is_train=True):
    # ArcFace standard normalization: [-1.0, 1.0] range
    norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            norm
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            norm
        ])
""")

    # =========================================================================
    # CELL 6: DATASET & BATCHING
    # =========================================================================
    add_md("### 📂 5. Dataset Loader & Same-Gender Negative Mining")
    add_code("""class FLAGBiMHATrainDataset(Dataset):
    def __init__(self, train_dir: str, is_train: bool = True):
        super().__init__()
        self.train_dir = train_dir
        self.is_train = is_train
        self.audio_proc = AudioPreprocessor(is_train=is_train)
        self.face_trans = get_iresnet_transforms(image_size=cfg.FACE_IMAGE_SIZE, is_train=is_train)
        
        self.samples = []
        self.speaker_to_id = {}
        self.speaker_to_indices = {}
        self.gender_to_speakers = {0: [], 1: []}
        
        self._index()

    def _index(self):
        if not self.train_dir or not os.path.exists(self.train_dir):
            return

        faces_base = os.path.join(self.train_dir, "faces")
        voices_base = os.path.join(self.train_dir, "voices")

        for lang in [d for d in os.listdir(faces_base) if os.path.isdir(os.path.join(faces_base, d))]:
            lang_face = os.path.join(faces_base, lang)
            lang_voice = os.path.join(voices_base, lang)
            if not os.path.exists(lang_voice):
                continue

            for spk in sorted(os.listdir(lang_face)):
                spk_face_dir = os.path.join(lang_face, spk)
                if not os.path.isdir(spk_face_dir):
                    continue

                if spk not in self.speaker_to_id:
                    self.speaker_to_id[spk] = len(self.speaker_to_id)
                spk_idx = self.speaker_to_id[spk]

                gender = 1 if ('female' in spk.lower() or spk.startswith('f')) else (
                    0 if ('male' in spk.lower() or spk.startswith('m')) else (spk_idx % 2)
                )

                face_files = sorted(glob.glob(os.path.join(spk_face_dir, "*.jpg")) + glob.glob(os.path.join(spk_face_dir, "*.png")))
                voice_files = sorted(glob.glob(os.path.join(lang_voice, spk, "*.wav")))

                if face_files and voice_files:
                    for f_p in face_files:
                        idx = len(self.samples)
                        self.samples.append({
                            "face_path": f_p,
                            "voice_path": random.choice(voice_files),
                            "spk_idx": spk_idx,
                            "gender": gender,
                        })
                        if spk_idx not in self.speaker_to_indices:
                            self.speaker_to_indices[spk_idx] = []
                        self.speaker_to_indices[spk_idx].append(idx)

                        if spk_idx not in self.gender_to_speakers[gender]:
                            self.gender_to_speakers[gender].append(spk_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        spk_idx = item["spk_idx"]
        gender = item["gender"]

        try:
            face_img = Image.open(item["face_path"]).convert("RGB")
            face_t = self.face_trans(face_img)
        except Exception:
            face_t = torch.zeros(3, cfg.FACE_IMAGE_SIZE, cfg.FACE_IMAGE_SIZE)

        voice_t = self.audio_proc.process(item["voice_path"])

        same_gen = [s for s in self.gender_to_speakers[gender] if s != spk_idx]
        if same_gen:
            neg_spk = random.choice(same_gen)
            neg_idx = random.choice(self.speaker_to_indices[neg_spk])
            neg_voice_t = self.audio_proc.process(self.samples[neg_idx]["voice_path"])
        else:
            neg_voice_t = torch.randn_like(voice_t)

        return {
            "face_img": face_t,
            "voice_wav": voice_t,
            "neg_voice_wav": neg_voice_t,
            "spk_label": torch.tensor(spk_idx, dtype=torch.long),
            "gender_label": torch.tensor(gender, dtype=torch.long)
        }

    def subset(self, indices: List[int], is_train: bool = True) -> 'FLAGBiMHATrainDataset':
        sub = FLAGBiMHATrainDataset.__new__(FLAGBiMHATrainDataset)
        super(FLAGBiMHATrainDataset, sub).__init__()
        sub.train_dir = self.train_dir
        sub.is_train = is_train
        sub.audio_proc = AudioPreprocessor(is_train=is_train)
        sub.face_trans = get_iresnet_transforms(image_size=cfg.FACE_IMAGE_SIZE, is_train=is_train)
        sub.speaker_to_id = self.speaker_to_id.copy()
        sub.gender_to_speakers = {0: [], 1: []}
        sub.speaker_to_indices = {}
        sub.samples = [self.samples[i] for i in indices]
        for idx, item in enumerate(sub.samples):
            spk = item["spk_idx"]
            gen = item["gender"]
            if spk not in sub.speaker_to_indices:
                sub.speaker_to_indices[spk] = []
            sub.speaker_to_indices[spk].append(idx)
            if spk not in sub.gender_to_speakers[gen]:
                sub.gender_to_speakers[gen].append(spk)
        return sub


def create_bimha_train_val_datasets(
    train_dir: str,
    val_ratio: float = cfg.VAL_SPLIT_RATIO,
    seed: int = cfg.SEED
) -> Tuple[FLAGBiMHATrainDataset, Optional[FLAGBiMHATrainDataset]]:
    full_ds = FLAGBiMHATrainDataset(train_dir, is_train=True)
    if val_ratio <= 0.0 or len(full_ds) < 10:
        return full_ds, None

    rng = random.Random(seed)
    indices = list(range(len(full_ds)))
    rng.shuffle(indices)

    split_pt = int(len(full_ds) * (1.0 - val_ratio))
    train_ds = full_ds.subset(indices[:split_pt], is_train=True)
    val_ds = full_ds.subset(indices[split_pt:], is_train=False)

    if accelerator.is_main_process:
        print(f"[Dataset Split] Partitioned {len(full_ds)} samples -> Train: {len(train_ds)}, Val: {len(val_ds)} for Early Stopping.")
    return train_ds, val_ds


class FLAGBiMHADevDataset(Dataset):
    def __init__(self, trial_file_path: str, dev_dir: str):
        super().__init__()
        self.trial_file_path = trial_file_path
        self.dev_dir = dev_dir
        self.audio_proc = AudioPreprocessor(is_train=False)
        self.face_trans = get_iresnet_transforms(image_size=cfg.FACE_IMAGE_SIZE, is_train=False)
        self.trials = []
        self._load_trials()

    def _load_trials(self):
        if not os.path.exists(self.trial_file_path):
            return

        with open(self.trial_file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    lbl = 1 if parts[0] in ("1", "pos", "positive", "True", "true") else (
                        0 if parts[0] in ("0", "neg", "negative", "False", "false") else -1
                    )
                    self.trials.append((parts[0], parts[1], parts[2], lbl))
                elif len(parts) == 2:
                    self.trials.append((parts[0], parts[0], parts[1], -1))

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        p_id, face_rel, voice_rel, label = self.trials[idx]

        f_path = face_rel if os.path.isabs(face_rel) else os.path.join(self.dev_dir, face_rel)
        v_path = voice_rel if os.path.isabs(voice_rel) else os.path.join(self.dev_dir, voice_rel)

        if not os.path.exists(f_path):
            cands = glob.glob(os.path.join(self.dev_dir, "**", os.path.basename(face_rel)), recursive=True)
            if cands: f_path = cands[0]
        if not os.path.exists(v_path):
            cands = glob.glob(os.path.join(self.dev_dir, "**", os.path.basename(voice_rel)), recursive=True)
            if cands: v_path = cands[0]

        try:
            face_img = Image.open(f_path).convert("RGB")
            face_t = self.face_trans(face_img)
        except Exception:
            face_t = torch.zeros(3, cfg.FACE_IMAGE_SIZE, cfg.FACE_IMAGE_SIZE)

        voice_t = self.audio_proc.process(v_path)

        return {
            "pair_id": p_id,
            "face_img": face_t,
            "voice_wav": voice_t,
            "label": torch.tensor(label, dtype=torch.long)
        }
""")

    # =========================================================================
    # CELL 7: IRESNET-100, BI-MHA, AND ORTHOGONAL SPLITTER
    # =========================================================================
    add_md("### 🏛️ 6. Core Modules: IResNet-100, Bi-MHA, GRL & Orthogonal Splitter")
    add_code("""def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(IBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class IResNet100(nn.Module):
    \"\"\"
    ArcFace IResNet-100 Backbone (Deng et al., CVPR 2019).
    65.16 Million Parameters. Specifically designed for deep craniofacial feature extraction.
    \"\"\"
    def __init__(self, layers=[3, 13, 30, 3], num_features=512):
        super(IResNet100, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._make_layer(64, layers[0], stride=2)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512 * IBasicBlock.expansion, eps=1e-5)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(512 * IBasicBlock.expansion * 7 * 7, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-5)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * IBasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * IBasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * IBasicBlock.expansion, eps=1e-5)
            )
        layers = []
        layers.append(IBasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * IBasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(IBasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, 3, 112, 112)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat_map = self.layer4(x) # (B, 512, 7, 7)
        
        x_flat = self.bn2(feat_map)
        x_flat = torch.flatten(x_flat, 1)
        x_flat = self.dropout(x_flat)
        global_feat = self.features(self.fc(x_flat)) # (B, 512)
        return global_feat, feat_map


class BiDirectionalMHA(nn.Module):
    \"\"\"
    Bi-directional Multihead Attention (Bi-MHA):
    - Face tokens query Audio tokens:  F_ctx = MHA(Q=Face, K=Aud, V=Aud)
    - Audio tokens query Face tokens:  A_ctx = MHA(Q=Aud, K=Face, V=Face)
    Followed by Residual LayerNorm and Feed-Forward Networks.
    \"\"\"
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mha_f2a = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.mha_a2f = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm_f1 = nn.LayerNorm(embed_dim)
        self.norm_f2 = nn.LayerNorm(embed_dim)
        self.norm_a1 = nn.LayerNorm(embed_dim)
        self.norm_a2 = nn.LayerNorm(embed_dim)
        
        self.ffn_f = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.ffn_a = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        self.pool_f = nn.Linear(embed_dim, 1)
        self.pool_a = nn.Linear(embed_dim, 1)

    def forward(self, face_tokens: torch.Tensor, audio_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Face queries Audio
        f_ctx, _ = self.mha_f2a(query=face_tokens, key=audio_tokens, value=audio_tokens)
        f_feat = self.norm_f1(face_tokens + f_ctx)
        f_feat = self.norm_f2(f_feat + self.ffn_f(f_feat))

        # 2. Audio queries Face
        a_ctx, _ = self.mha_a2f(query=audio_tokens, key=face_tokens, value=face_tokens)
        a_feat = self.norm_a1(audio_tokens + a_ctx)
        a_feat = self.norm_a2(a_feat + self.ffn_a(a_feat))

        # Attentive pooling across tokens
        w_f = F.softmax(self.pool_f(f_feat), dim=1)
        z_f = torch.sum(w_f * f_feat, dim=1)
        w_a = F.softmax(self.pool_a(a_feat), dim=1)
        z_a = torch.sum(w_a * a_feat, dim=1)
        return z_f, z_a


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x, alpha=None):
        a = alpha if alpha is not None else self.alpha
        return GradientReversalFunction.apply(x, a)


class OrthogonalSubspaceSplitter(nn.Module):
    \"\"\"
    Subspace Decomposition:
    Splits fused multimodal representation z into:
    - z_id: Identity Subspace (purged of demographics, used for face-voice verification)
    - z_gen: Gender Subspace (captures all demographic variance)
    Computes cosine^2 penalty enforcing <z_id, z_gen> = 0.
    \"\"\"
    def __init__(self, in_features: int = 512, id_dim: int = 512, gen_dim: int = 256):
        super().__init__()
        self.proj_id = nn.Sequential(
            nn.Linear(in_features, id_dim),
            nn.BatchNorm1d(id_dim),
            nn.PReLU(),
            nn.Linear(id_dim, id_dim)
        )
        self.proj_gen = nn.Sequential(
            nn.Linear(in_features, gen_dim),
            nn.BatchNorm1d(gen_dim),
            nn.PReLU(),
            nn.Linear(gen_dim, gen_dim)
        )
        self.match_proj = nn.Linear(gen_dim, id_dim, bias=False)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_id = self.proj_id(z)
        z_gen = self.proj_gen(z)
        return z_id, z_gen

    def compute_orthogonality(self, z_id: torch.Tensor, z_gen: torch.Tensor) -> torch.Tensor:
        gen_m = self.match_proj(z_gen)
        cos_sim = F.cosine_similarity(z_id, gen_m, dim=-1)
        return torch.mean(cos_sim ** 2)
""")

    # =========================================================================
    # CELL 8: SUB-CENTER ARCFACE & ORTHOGONAL LOSS
    # =========================================================================
    add_md("### 🎯 7. Loss Functions: Sub-Center ArcFace (K=3) & Orthogonal Subspace Loss")
    add_code("""class SubCenterArcFace(nn.Module):
    \"\"\"
    Sub-Center ArcFace (Deng et al., ECCV 2020):
    Allocates K sub-centers per identity class: {W_{c,1}, ..., W_{c,K}}.
    Accommodates natural multimodal variations (e.g. speaking English vs Bengali)
    without forcing disparate representations into a single rigid center.
    \"\"\"
    def __init__(self, in_features: int, num_classes: int, K: int = 3, scale: float = 32.0, margin: float = 0.35):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.K = K
        self.scale = scale
        self.margin = margin
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, K, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for fp16 stability
        x_norm = F.normalize(x.float(), p=2, dim=1)               # (B, D)
        w_norm = F.normalize(self.weight.float(), p=2, dim=-1)       # (C, K, D)

        cosine_all = torch.einsum('bd,ckd->bck', x_norm, w_norm)     # (B, C, K)
        cosine, _ = torch.max(cosine_all, dim=-1)                    # (B, C)
        cosine = cosine.clamp(-1.0 + 1e-4, 1.0 - 1e-4)

        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-4))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=x.device, dtype=torch.float32)
        one_hot.scatter_(1, label.view(-1, 1).long().to(x.device), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return (output * self.scale).type_as(x)


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, face_feat: torch.Tensor, voice_feat: torch.Tensor, spk_label: torch.Tensor) -> torch.Tensor:
        feats = torch.cat([F.normalize(face_feat.float(), p=2, dim=-1),
                           F.normalize(voice_feat.float(), p=2, dim=-1)], dim=0) # (2B, D)
        b = face_feat.size(0)
        labels = spk_label.view(-1, 1).to(face_feat.device)
        labels = labels.repeat(2, 1)

        mask = torch.eq(labels, labels.T).float().to(face_feat.device)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b * 2, device=face_feat.device).view(-1, 1), 0)
        mask = mask * logits_mask

        sim = torch.div(torch.matmul(feats, feats.T), self.temperature)
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        denom = exp_logits.sum(1, keepdim=True)
        valid = mask.sum(1) > 0
        if not torch.any(valid):
            return torch.tensor(0.0, device=face_feat.device, requires_grad=True)

        log_prob = logits - torch.log(denom + 1e-5)
        mean_log_prob_pos = (mask * log_prob).sum(1)[valid] / mask.sum(1)[valid]
        return -mean_log_prob_pos.mean()


class CompositeBiMHALoss(nn.Module):
    def __init__(self, cfg: IResNetWavLMConfig):
        super().__init__()
        self.cfg = cfg
        self.ce = nn.CrossEntropyLoss()
        self.supcon = SupConLoss(temperature=0.07)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        spk_label: torch.Tensor,
        gender_label: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        dev = spk_label.device

        # 1. Sub-Center ArcFace Loss
        l_sub_f = self.ce(outputs["sub_arc_face_logits"].to(dev), spk_label)
        l_sub_v = self.ce(outputs["sub_arc_voice_logits"].to(dev), spk_label)
        loss_subcenter = 0.5 * (l_sub_f + l_sub_v)

        # 2. Orthogonal Loss (enforces z_id perpendicular to z_gen)
        loss_orth = outputs["loss_orth"]

        # 3. Adversarial GRL Gender Loss on z_id
        l_grl_f = self.ce(outputs["face_gender_grl_logits"].to(dev), gender_label)
        l_grl_v = self.ce(outputs["voice_gender_grl_logits"].to(dev), gender_label)
        loss_grl = 0.5 * (l_grl_f + l_grl_v)

        # 4. Supervised Gender Loss on z_gen
        l_gen_f = self.ce(outputs["face_gender_logits"].to(dev), gender_label)
        l_gen_v = self.ce(outputs["voice_gender_logits"].to(dev), gender_label)
        loss_gen = 0.5 * (l_gen_f + l_gen_v)

        # 5. Supervised Contrastive Loss
        loss_supcon = self.supcon(outputs["face_id_embed"].to(dev), outputs["voice_id_embed"].to(dev), spk_label)

        total_loss = (
            self.cfg.LAMBDA_SUBCENTER * loss_subcenter +
            self.cfg.LAMBDA_ORTH * loss_orth +
            self.cfg.LAMBDA_GRL * loss_grl +
            self.cfg.LAMBDA_GEN * loss_gen +
            self.cfg.LAMBDA_SUPCON * loss_supcon
        )

        return {
            "total_loss": total_loss,
            "loss_subcenter": loss_subcenter,
            "loss_orth": loss_orth,
            "loss_grl": loss_grl,
            "loss_gen": loss_gen,
            "loss_supcon": loss_supcon
        }
""")

    # =========================================================================
    # CELL 9: COMPLETE MULTIMODAL MODEL
    # =========================================================================
    add_md("### 🚀 8. Full Architecture: FLAG-BiMHA-SubArc-Omni Model")
    add_code("""from transformers import AutoModel

class FLAGBiMHASubArcModel(nn.Module):
    \"\"\"
    Complete Bi-directional MHA Face-Voice Multimodal Architecture:
      - Face: IResNet-100 (ArcFace Deep ConvNet, 112x112)
      - Audio: WavLM-Large / WavLM-Base-SV (Speech Biometrics SSL)
      - Fusion: Bi-directional MHA (Face <-> Audio)
      - Debiasing: Orthogonal Subspace Splitter + GRL
      - Metric Learning: Sub-Center ArcFace (K=3)
    \"\"\"
    def __init__(self, num_speakers: int = 100, cfg: IResNetWavLMConfig = cfg):
        super().__init__()
        self.cfg = cfg
        self.is_parallel = True

        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if num_gpus >= 2:
            self.device_face = torch.device("cuda:0")
            self.device_audio = torch.device("cuda:1")
            self.device_fusion = torch.device("cuda:0")
            if accelerator.is_main_process:
                print(f"[Device Pipeline] 🚀 Dual GPU mode: Face -> cuda:0, Audio -> cuda:1, Fusion/Heads -> cuda:0")
        elif num_gpus == 1:
            self.device_face = torch.device("cuda:0")
            self.device_audio = torch.device("cuda:0")
            self.device_fusion = torch.device("cuda:0")
            if accelerator.is_main_process:
                print(f"[Device Pipeline] Single GPU mode active: All components on cuda:0")
        else:
            self.device_face = torch.device("cpu")
            self.device_audio = torch.device("cpu")
            self.device_fusion = torch.device("cpu")

        # ----------------------------------------------------------------------
        # 1. Face Encoder: IResNet-100 (ArcFace)
        # ----------------------------------------------------------------------
        if accelerator.is_main_process:
            print(f"[Model Init] Loading IResNet-100 Face Backbone on {self.device_face}...")
        self.face_encoder = IResNet100(layers=[3, 13, 30, 3], num_features=cfg.FACE_FEAT_DIM).to(self.device_face)

        # ----------------------------------------------------------------------
        # 2. Audio Encoder: WavLM-Large / WavLM-Base-SV
        # ----------------------------------------------------------------------
        if accelerator.is_main_process:
            print(f"[Model Init] Loading WavLM Audio Backbone: {cfg.AUDIO_MODEL_ID} on {self.device_audio}...")
        try:
            self.audio_encoder = AutoModel.from_pretrained(
                cfg.AUDIO_MODEL_ID,
                device_map={"": str(self.device_audio)} if self.device_audio.type == "cuda" else None,
                trust_remote_code=True
            )
            a_hidden = getattr(self.audio_encoder.config, "hidden_size", 1024)
        except Exception as e:
            if accelerator.is_main_process:
                print(f"[Fallback] Audio backbone fallback to {cfg.AUDIO_FALLBACK_ID}: {e}")
            self.audio_encoder = AutoModel.from_pretrained(
                cfg.AUDIO_FALLBACK_ID,
                device_map={"": str(self.device_audio)} if self.device_audio.type == "cuda" else None
            )
            a_hidden = getattr(self.audio_encoder.config, "hidden_size", 768)

        # Freeze speech feature extractor CNN for stability
        if hasattr(self.audio_encoder, "feature_extractor"):
            for p in self.audio_encoder.feature_extractor.parameters():
                p.requires_grad = False

        # ----------------------------------------------------------------------
        # 3. Token Adapters for Bi-directional MHA
        # ----------------------------------------------------------------------
        # Face map (B, 512, 7, 7) -> (B, 8, 512)
        self.face_token_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, cfg.NUM_TOKENS_PER_MODALITY)),
            nn.Flatten(2)
        ).to(self.device_fusion)

        # Audio frames (B, T, a_hidden) -> (B, 8, 512)
        self.audio_token_proj = nn.Sequential(
            nn.Linear(a_hidden, cfg.FUSION_DIM),
            nn.GELU()
        ).to(self.device_fusion)

        # ----------------------------------------------------------------------
        # 4. Bi-directional Multihead Attention (Bi-MHA)
        # ----------------------------------------------------------------------
        self.bimha = BiDirectionalMHA(
            embed_dim=cfg.FUSION_DIM,
            num_heads=cfg.NUM_HEADS,
            dropout=cfg.FUSION_DROPOUT
        ).to(self.device_fusion)

        # ----------------------------------------------------------------------
        # 5. Orthogonal Subspace Splitters (Face & Voice)
        # ----------------------------------------------------------------------
        self.splitter_face = OrthogonalSubspaceSplitter(
            in_features=cfg.FUSION_DIM,
            id_dim=cfg.ID_SUBSPACE_DIM,
            gen_dim=cfg.GEN_SUBSPACE_DIM
        ).to(self.device_fusion)

        self.splitter_voice = OrthogonalSubspaceSplitter(
            in_features=cfg.FUSION_DIM,
            id_dim=cfg.ID_SUBSPACE_DIM,
            gen_dim=cfg.GEN_SUBSPACE_DIM
        ).to(self.device_fusion)

        # ----------------------------------------------------------------------
        # 6. GRL & Demographic Debiasing Heads
        # ----------------------------------------------------------------------
        self.grl = GradientReversalLayer(alpha=1.0).to(self.device_fusion)
        self.gender_discriminator = nn.Sequential(
            nn.Linear(cfg.ID_SUBSPACE_DIM, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 2)
        ).to(self.device_fusion)

        self.gender_classifier = nn.Sequential(
            nn.Linear(cfg.GEN_SUBSPACE_DIM, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 2)
        ).to(self.device_fusion)

        # ----------------------------------------------------------------------
        # 7. Sub-Center ArcFace Metric Heads (K=3)
        # ----------------------------------------------------------------------
        self.subcenter_head = SubCenterArcFace(
            cfg.ID_SUBSPACE_DIM,
            num_speakers,
            K=cfg.SUBCENTER_K,
            scale=cfg.ARCFACE_SCALE,
            margin=cfg.ARCFACE_MARGIN
        ).to(self.device_fusion)

    def extract_face_stream(self, face_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if face_img.dim() == 3:
            face_img = face_img.unsqueeze(0)
        face_img = face_img.to(self.device_face)
        global_f, feat_map = self.face_encoder(face_img)
        # feat_map: (B, 512, 7, 7)
        # Pool into (B, 8, 512) tokens
        b = feat_map.size(0)
        # Reshape to (B, 49, 512) and pool to 8 tokens
        f_tokens = feat_map.flatten(2).transpose(1, 2) # (B, 49, 512)
        f_tokens_pooled = F.adaptive_avg_pool1d(f_tokens.transpose(1, 2), self.cfg.NUM_TOKENS_PER_MODALITY).transpose(1, 2)
        return global_f.to(self.device_fusion), f_tokens_pooled.to(self.device_fusion)

    def extract_audio_stream(self, voice_wav: torch.Tensor) -> torch.Tensor:
        if voice_wav.dim() == 1:
            voice_wav = voice_wav.unsqueeze(0)
        voice_wav = voice_wav.to(self.device_audio)
        out = self.audio_encoder(voice_wav)
        a_frames = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0] # (B, T, D_a)
        
        # Project and pool into 8 audio tokens
        a_tokens = self.audio_token_proj(a_frames.to(self.device_fusion)) # (B, T, 512)
        a_tokens_pooled = F.adaptive_avg_pool1d(a_tokens.transpose(1, 2), self.cfg.NUM_TOKENS_PER_MODALITY).transpose(1, 2)
        return a_tokens_pooled

    def forward(
        self,
        face_img: torch.Tensor,
        voice_wav: torch.Tensor,
        spk_label: Optional[torch.Tensor] = None,
        grl_lambda: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        # 1. Feature Stream Extraction
        _, f_tokens = self.extract_face_stream(face_img)
        a_tokens = self.extract_audio_stream(voice_wav)

        # 2. Bi-directional Multihead Attention (Bi-MHA) Fusion
        z_f_fused, z_v_fused = self.bimha(f_tokens, a_tokens)

        # 3. Orthogonal Subspace Splitting
        z_id_f, z_gen_f = self.splitter_face(z_f_fused)
        z_id_v, z_gen_v = self.splitter_voice(z_v_fused)

        # Orthogonality Loss: Cosine^2(z_id, z_gen)
        loss_orth = 0.5 * (self.splitter_face.compute_orthogonality(z_id_f, z_gen_f) +
                           self.splitter_voice.compute_orthogonality(z_id_v, z_gen_v))

        # 4. Hyperspherical Normalization on Identity Embeddings
        e_f = F.normalize(z_id_f.float(), p=2, dim=-1)
        e_v = F.normalize(z_id_v.float(), p=2, dim=-1)

        # 5. Official Euclidean Distance Protocol: d = sqrt(2 - 2 * cos(θ)) = ||e_f - e_v||_2
        cos_sim = torch.sum(e_f * e_v, dim=-1).clamp(-1.0 + 1e-4, 1.0 - 1e-4)
        dist = torch.sqrt(torch.clamp(2.0 - 2.0 * cos_sim, min=1e-4))

        out = {
            "face_id_embed": e_f,
            "voice_id_embed": e_v,
            "distance": dist,
            "cosine_sim": cos_sim,
            "loss_orth": loss_orth
        }

        # 6. Training Heads
        if spk_label is not None:
            spk_dev = spk_label.to(self.device_fusion)
            out["sub_arc_face_logits"] = self.subcenter_head(z_id_f, spk_dev)
            out["sub_arc_voice_logits"] = self.subcenter_head(z_id_v, spk_dev)

            # Adversarial GRL on Identity Subspace (purges gender)
            f_rev = self.grl(z_id_f, grl_lambda)
            v_rev = self.grl(z_id_v, grl_lambda)
            out["face_gender_grl_logits"] = self.gender_discriminator(f_rev)
            out["voice_gender_grl_logits"] = self.gender_discriminator(v_rev)

            # Supervised Gender on Gender Subspace (anchors gender)
            out["face_gender_logits"] = self.gender_classifier(z_gen_f)
            out["voice_gender_logits"] = self.gender_classifier(z_gen_v)

        return out
""")

    # =========================================================================
    # CELL 10: TRAINING ENGINE & EARLY STOPPING
    # =========================================================================
    add_md("### 🔄 9. Training Engine & Early Stopping Regularization")
    add_code("""class EarlyStopping:
    def __init__(
        self,
        patience: int = cfg.EARLY_STOPPING_PATIENCE,
        min_delta: float = cfg.EARLY_STOPPING_MIN_DELTA,
        mode: str = cfg.EARLY_STOPPING_MODE,
        restore_best_weights: bool = cfg.EARLY_STOPPING_RESTORE_BEST,
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        self.best_state_dict = None

    def is_better(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < (self.best_score - self.min_delta)
        else:
            return score > (self.best_score + self.min_delta)

    def step(self, score: float, model: nn.Module, epoch: int) -> bool:
        if math.isnan(score) or math.isinf(score):
            return False
        if self.is_better(score):
            if self.verbose and accelerator.is_main_process:
                if self.best_score is not None:
                    print(f"[EarlyStopping] Metric improved from {self.best_score:.4f} to {score:.4f} at epoch {epoch}. Resetting patience.")
                else:
                    print(f"[EarlyStopping] Baseline validation score: {score:.4f} at epoch {epoch}.")
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best_weights:
                self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return True
        else:
            self.counter += 1
            if self.verbose and accelerator.is_main_process:
                print(f"[EarlyStopping] Patience: {self.counter}/{self.patience} (Best: {self.best_score:.4f} at epoch {self.best_epoch})")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose and accelerator.is_main_process:
                    print(f"[EarlyStopping] 🛑 Early stopping triggered! Stagnated for {self.patience} epochs.")
            return False

    def restore(self, model: nn.Module):
        if self.restore_best_weights and self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            if self.verbose and accelerator.is_main_process:
                print(f"[EarlyStopping] Restored best model parameters from epoch {self.best_epoch} (Score: {self.best_score:.4f}).")


def train_bimha_model(model, train_dataset, val_dataset=None, cfg: IResNetWavLMConfig = cfg):
    if len(train_dataset) == 0:
        return model

    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    val_dataloader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=True
        )

    criterion = CompositeBiMHALoss(cfg)
    early_stopping = EarlyStopping(
        patience=cfg.EARLY_STOPPING_PATIENCE,
        min_delta=cfg.EARLY_STOPPING_MIN_DELTA,
        mode=cfg.EARLY_STOPPING_MODE,
        restore_best_weights=cfg.EARLY_STOPPING_RESTORE_BEST
    )

    # Differential Learning Rates: Gentle for backbones, faster for Bi-MHA and Heads
    backbone_params = list(model.face_encoder.parameters()) + list(model.audio_encoder.parameters())
    fusion_params = (
        list(model.face_token_proj.parameters()) +
        list(model.audio_token_proj.parameters()) +
        list(model.bimha.parameters()) +
        list(model.splitter_face.parameters()) +
        list(model.splitter_voice.parameters()) +
        list(model.gender_discriminator.parameters()) +
        list(model.gender_classifier.parameters()) +
        list(model.subcenter_head.parameters())
    )

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": cfg.LR_BACKBONE, "weight_decay": cfg.WEIGHT_DECAY},
        {"params": fusion_params, "lr": cfg.LR_FUSION, "weight_decay": cfg.WEIGHT_DECAY}
    ])

    total_steps = len(dataloader) * cfg.NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # Accelerate Preparation
    if val_dataloader is not None:
        model, optimizer, dataloader, val_dataloader, scheduler = accelerator.prepare(
            model, optimizer, dataloader, val_dataloader, scheduler
        )
    else:
        model, optimizer, dataloader, scheduler = accelerator.prepare(
            model, optimizer, dataloader, scheduler
        )

    if accelerator.is_main_process:
        print(f"[Training] Accelerated engine ready across {accelerator.num_processes} GPUs.")
        print(f"[Training] Early stopping patience: {cfg.EARLY_STOPPING_PATIENCE} epochs.")

    best_loss = float("inf")

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        model.train()
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d}", disable=not accelerator.is_main_process)

        for step, batch in enumerate(pbar):
            p = float(step + (epoch - 1) * len(dataloader)) / total_steps
            grl_lambda = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0

            face_img = batch["face_img"]
            voice_wav = batch["voice_wav"]
            spk_label = batch["spk_label"].to(accelerator.device)
            gender_label = batch["gender_label"].to(accelerator.device)

            with accelerator.accumulate(model):
                outputs = model(face_img, voice_wav, spk_label=spk_label, grl_lambda=grl_lambda)
                losses = criterion(outputs, spk_label, gender_label)
                loss = losses["total_loss"]

                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad()
                    continue

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if not (math.isnan(loss.item()) or math.isinf(loss.item())):
                epoch_losses.append(loss.item())
            if accelerator.is_main_process:
                pbar.set_postfix({
                    "loss": f"{loss.item():.3f}",
                    "sub_arc": f"{losses['loss_subcenter'].item():.3f}",
                    "orth": f"{losses['loss_orth'].item():.4f}"
                })

        mean_train_loss = np.mean(epoch_losses) if epoch_losses else 0.0

        # Run Validation Loop for Early Stopping
        val_losses = []
        if val_dataloader is not None:
            model.eval()
            with torch.no_grad():
                for v_batch in val_dataloader:
                    v_outputs = model(v_batch["face_img"], v_batch["voice_wav"], spk_label=v_batch["spk_label"])
                    v_losses = criterion(v_outputs, v_batch["spk_label"].to(accelerator.device), v_batch["gender_label"].to(accelerator.device))
                    v_val = v_losses["total_loss"].item()
                    if not (math.isnan(v_val) or math.isinf(v_val)):
                        val_losses.append(v_val)
            mean_val_loss = np.mean(val_losses) if val_losses else mean_train_loss
        else:
            mean_val_loss = mean_train_loss

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Early Stopping check on main process
        if accelerator.is_main_process:
            val_str = f"Val Loss: {mean_val_loss:.4f}" if val_dataloader is not None else "Val: N/A"
            print(f"Epoch {epoch:02d} Complete | Train Loss: {mean_train_loss:.4f} | {val_str}")
            is_best = early_stopping.step(mean_val_loss, accelerator.unwrap_model(model), epoch)
            if is_best:
                best_loss = mean_val_loss
                unwrapped = accelerator.unwrap_model(model)
                torch.save(unwrapped.state_dict(), cfg.BEST_MODEL_PATH)
                sz_mb = os.path.getsize(cfg.BEST_MODEL_PATH) / (1024 * 1024)
                print(f" -> Checkpoint saved to {cfg.BEST_MODEL_PATH} ({sz_mb:.2f} MB)")

        # Synchronize early stopping decision across processes
        if accelerator.num_processes > 1:
            early_stop_flag = 1 if (early_stopping.early_stop if accelerator.is_main_process else 0) else 0
            stop_tensor = torch.tensor([early_stop_flag], device=accelerator.device)
            stop_tensor = accelerator.reduce(stop_tensor, reduction="max")
            should_stop = (stop_tensor.item() == 1)
        else:
            should_stop = early_stopping.early_stop

        if should_stop:
            if accelerator.is_main_process:
                print(f"\\n[EarlyStopping] Halting training loop early to prevent overfitting!")
                early_stopping.restore(accelerator.unwrap_model(model))
            break

    return model
""")

    # =========================================================================
    # CELL 11: CODABENCH EVALUATION & PACKAGING
    # =========================================================================
    add_md("### 📊 10. Official 4-Cell CodaBench Evaluation & Packaging")
    add_code("""def compute_cell_eer(dists: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    valid = (labels == 0) | (labels == 1)
    if not np.any(valid):
        return -1.0, -1.0
    v_dists = dists[valid]
    v_labels = labels[valid]
    
    sim = -v_dists
    fpr, tpr, _ = roc_curve(v_labels, sim, pos_label=1)
    fnr = 1.0 - tpr
    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    except Exception:
        idx = np.nanargmin(np.absolute(fnr - fpr))
        eer = (fpr[idx] + fnr[idx]) / 2.0
    roc_auc = auc(fpr, tpr)
    return float(eer * 100.0), float(roc_auc * 100.0)


def evaluate_codabench_cell(model, trial_path, dev_dir, output_path, cfg):
    if not os.path.exists(trial_path):
        return -1.0, -1.0

    dataset = FLAGBiMHADevDataset(trial_path, dev_dir)
    if len(dataset) == 0:
        return -1.0, -1.0

    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE * 2, shuffle=False, num_workers=cfg.NUM_WORKERS)
    model.eval()

    pairs, dists, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            face_img = batch["face_img"]
            voice_wav = batch["voice_wav"]
            b_dists = model(face_img, voice_wav)["distance"].cpu().numpy()
            
            pairs.extend(batch["pair_id"])
            dists.extend(b_dists.tolist())
            labels.extend(batch["label"].numpy().tolist())

    d_arr = np.array(dists)
    l_arr = np.array(labels)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p, d in zip(pairs, d_arr):
            f.write(f"{p} {d:.6f}\\n")

    return compute_cell_eer(d_arr, l_arr)


def run_full_evaluation(model, dev_dir, cfg):
    if not accelerator.is_main_process:
        return

    print("=" * 80)
    print("🏆 FLAG 2027 Challenge: Official 4-Cell CodaBench Evaluation")
    print("=" * 80)

    cells = [
        ("gender / English_heard",
         os.path.join(dev_dir, "gender", "trials_v4_English_heard.txt"),
         cfg.SUBMISSION_GENDER_ENGLISH),
        ("gender / Bangla_unheard",
         os.path.join(dev_dir, "gender", "trials_v4_Bangla_unheard.txt"),
         cfg.SUBMISSION_GENDER_BANGLA),
        ("no_gender / English_heard",
         os.path.join(dev_dir, "no_gender", "trials_v4_English_heard.txt"),
         cfg.SUBMISSION_NO_GENDER_ENGLISH),
        ("no_gender / Bangla_unheard",
         os.path.join(dev_dir, "no_gender", "trials_v4_Bangla_unheard.txt"),
         cfg.SUBMISSION_NO_GENDER_BANGLA),
    ]

    results = []
    for name, trial_p, out_p in cells:
        if not os.path.exists(trial_p):
            cell_folder = os.path.dirname(trial_p)
            cands = glob.glob(os.path.join(cell_folder, "*.txt"))
            target_lang = "english" if "english" in name.lower() else "bangla"
            matched = [c for c in cands if target_lang in os.path.basename(c).lower()]
            if matched:
                trial_p = matched[0]
            elif cands:
                trial_p = cands[0]
        eer, auc_val = evaluate_codabench_cell(model, trial_p, dev_dir, out_p, cfg)
        results.append((name, eer, auc_val))

    print(f"\\n{'Track & Language Split':<32} | {'EER (%)':<10} | {'AUC (%)':<10}")
    print("-" * 60)
    valid_eers = []
    for name, eer, auc_val in results:
        e_str = f"{eer:.2f}%" if eer >= 0 else "Blind Test"
        a_str = f"{auc_val:.2f}%" if auc_val >= 0 else "N/A"
        print(f"{name:<32} | {e_str:<10} | {a_str:<10}")
        if eer >= 0:
            valid_eers.append(eer)

    if valid_eers:
        print("-" * 60)
        print(f"{'OVERALL AVERAGE EER':<32} | {np.mean(valid_eers):.2f}%")
    print("=" * 80)


def package_submission(cfg):
    if not accelerator.is_main_process:
        return
    print(f"[Packaging] Generating CodaBench {cfg.SUBMISSION_ZIP_PATH}...")
    files = [
        (cfg.SUBMISSION_GENDER_ENGLISH, "gender/sub_score_v4_English_heard.txt"),
        (cfg.SUBMISSION_GENDER_BANGLA, "gender/sub_score_v4_Bangla_unheard.txt"),
        (cfg.SUBMISSION_NO_GENDER_ENGLISH, "no_gender/sub_score_v4_English_heard.txt"),
        (cfg.SUBMISSION_NO_GENDER_BANGLA, "no_gender/sub_score_v4_Bangla_unheard.txt"),
    ]
    with zipfile.ZipFile(cfg.SUBMISSION_ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        for abs_p, rel_p in files:
            if os.path.exists(abs_p):
                zipf.write(abs_p, arcname=rel_p)

    sz_kb = os.path.getsize(cfg.SUBMISSION_ZIP_PATH) / 1024.0
    print(f"✅ Generated {cfg.SUBMISSION_ZIP_PATH} ({sz_kb:.2f} KB) - Ready for CodaBench upload!")
""")

    # =========================================================================
    # CELL 12: MAIN RUNNER
    # =========================================================================
    add_md("### 🏁 11. Main Execution Pipeline")
    add_code("""# 1. Dataset Indexing & Train/Val Split for Early Stopping
train_dataset, val_dataset = create_bimha_train_val_datasets(TRAIN_DIR, cfg.VAL_SPLIT_RATIO, cfg.SEED)
num_spks = max(100, len(train_dataset.speaker_to_id))

# 2. Build Multi-Billion Foundation Model
model = FLAGBiMHASubArcModel(num_speakers=num_spks, cfg=cfg)

if accelerator.is_main_process:
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 70)
    print(f"Total Model Parameters: {total_p / 1e6:.2f} Million ({total_p:,})")
    print(f"Trainable Parameters  : {train_p / 1e6:.2f} Million ({train_p:,})")
    print("=" * 70)

# 3. Train with Accelerate Dual GPU & Early Stopping Monitoring
if len(train_dataset) > 0:
    model = train_bimha_model(model, train_dataset, val_dataset, cfg)

# 4. Evaluate Across All 4 CodaBench Cells
if DEV_DIR and os.path.exists(DEV_DIR):
    run_full_evaluation(model, DEV_DIR, cfg)

# 5. Package Submission
package_submission(cfg)
""")

    out_p = os.path.join("y:\\FLAG\\shobrikola\\notebooks", "FLAG2027_SOTA_IResNet100_WavLM_BiMHA_Multimodal.ipynb")
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"Successfully generated notebook: {out_p}")
    print(f"Total cells: {len(nb['cells'])}")

if __name__ == "__main__":
    create_notebook()
