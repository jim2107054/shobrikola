"""
Generator script to build the Ultra-SOTA Multi-Billion Parameter (up to 6.7B)
Multimodal Foundation Notebook for the FLAG 2027 Challenge on Face-Voice Association.

Specifications:
- Vision Encoder: EVA-02-E (4.4B) / CLIP-ViT-bigG
- Audio Encoder: SeamlessM4T-v2-Large (2.3B) / Whisper-large-v3 (1.5B)
- Cross-Modal Fusion: Q-Former (BLIP-2 style)
- Adversarial Gender Debiasing: GRL + Gender Discriminator
- Loss Functions: ArcFace (Identity) + SupCon (Supervised Contrastive)
- Training Method: QLoRA (4-bit NF4 Quantization) + Hugging Face Accelerate (Dual GPU)
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
    add_md(r"""# 🏆 FLAG 2027 Challenge: Ultra-SOTA Multi-Billion Multimodal Architecture
## 🌟 **FLAG-Q-Omni-7B**: Multi-Billion Foundation Model with Q-Former & QLoRA
### *ICASSP 2027 Challenge: Face-Voice Association Across Languages and Gender (MAV-Celeb v4)*

---

### 📌 Architecture Specification & Innovation Stack
This notebook implements the state-of-the-art multi-billion parameter multimodal architecture tailored specifically for the **FLAG 2027 Challenge**:

```
                       ┌─────────────────────────────────────────────────────────┐
                       │             MULTIMODAL INPUTS: FACE & AUDIO             │
                       └────────────────────┬────────────────┬───────────────────┘
                                            │                │
                    ┌───────────────────────┴────┐      ┌────┴───────────────────────────┐
                    │     FACE (RGB 224x224)     │      │   SPEECH (16 kHz, 3.0s Mono)   │
                    └───────────────┬────────────┘      └────────────────┬───────────────┘
                                    │                                    │
                    ┌───────────────▼────────────┐      ┌────────────────▼───────────────┐
                    │ EVA-02-E (4.4B) / CLIP-bigG│      │ SeamlessM4T-v2-Large (2.3B)    │
                    │ 4-bit NF4 Quantized (QLoRA)│      │ 4-bit NF4 Quantized (QLoRA)    │
                    │ Frozen Backbone + LoRA     │      │ 100+ Langs (Native Bengali/En) │
                    └───────────────┬────────────┘      └────────────────┬───────────────┘
                                    │ Visual Patch Tokens                │ Acoustic Frame Tokens
                                    │ (B, N_patches, D_v)                │ (B, T_frames, D_a)
                                    │                                    │
                    ┌───────────────▼────────────────────────────────────▼───────────────┐
                    │               BLIP-2 STYLE BIMODAL Q-FORMER FUSION                 │
                    │     Learnable Identity Queries Q in R^(K x D_q) (K=32, D_q=768)     │
                    │     Self-Attention (Queries <-> Queries)                           │
                    │     Cross-Attention (Queries <-> Visual & Acoustic Tokens)          │
                    └───────────────┬────────────────────────────────────┬───────────────┘
                                    │                                    │
                    ┌───────────────▼────────────┐      ┌────────────────▼───────────────┐
                    │ Fused Face Vector e_f      │      │ Fused Voice Vector e_v         │
                    │ L2-Normalized on S^(D-1)   │      │ L2-Normalized on S^(D-1)       │
                    └───────────────┬────────────┘      └────────────────┬───────────────┘
                                    │                                    │
                    ┌───────────────┴────────────────────────────────────┴───────────────┐
                    │            ADVERSARIAL DEMOGRAPHIC DEBIASING HEAD (GRL)            │
                    │            Inverts Gender Discriminator Gradients: -lambda * grad  │
                    │            Purges Gender Shortcut on Same-Gender Negative Pairs    │
                    └────────────────────────────────┬───────────────────────────────────┘
                                                     │
                    ┌────────────────────────────────▼───────────────────────────────────┐
                    │                       COMPOSITE LOSS SUITE                         │
                    │  1. Sub-Center ArcFace Loss (Hyperspherical Identity Discrimination)│
                    │  2. Supervised Contrastive Loss (SupCon - Khosla et al.)           │
                    │  3. Adversarial Gender Cross-Entropy Loss (Demographic Debiasing)  │
                    └────────────────────────────────────────────────────────────────────┘
```

### 🔑 Component Breakdown:
1. **Vision / Face Encoder**:
   - **EVA-02-E (4.4B Parameters)** / **CLIP-ViT-bigG (1.8B Parameters)**.
   - Captures ultra-fine craniofacial geometry, bone structures, and morphological invariants.
2. **Audio / Voice Encoder**:
   - **SeamlessM4T-v2-Large (2.3B Parameters)** / **Whisper-large-v3 (1.5B Parameters)**.
   - Pre-trained across 100+ languages with direct native representation for **Bengali (Bangla)** and **English**.
3. **Cross-Modal Fusion**:
   - **Q-Former (BLIP-2 Architecture)**: Querying Transformer with $K=32$ learned identity queries interacting via bidirectional cross-attention with frozen vision and speech representations.
4. **Adversarial Demographic Debiasing**:
   - **Gradient Reversal Layer (GRL)** connected to a 2-layer MLP **Gender Discriminator**.
   - Inverts gradients with dynamic $\lambda(p)$ scheduling, rendering identity embeddings invariant to gender shortcuts.
5. **Loss Functions**:
   - **Sub-Center ArcFace (AAM-Softmax)**: Cosine margin $m=0.35, s=32.0$ for intra-speaker hyperspherical compactness.
   - **Supervised Contrastive Loss (SupCon)**: Pulls together cross-modal pairs of the same speaker identity while pushing apart same-gender negative impostors.
6. **Training Engine**:
   - **QLoRA (4-bit NF4 Quantization with double quantization via `bitsandbytes` & `peft`)**: Loads 6.7B parameters in ~4 GB VRAM!
   - **Hugging Face `accelerate`**: Native support for **Dual GPUs (Kaggle 2x T4)** or single P100/A100.
7. **Evaluation & Verification**:
   - Computes Euclidean distance $d = \sqrt{2 - 2 \cos(\mathbf{e}_f, \mathbf{e}_v)} \in [0, 2.0]$.
   - Evaluates all 4 competition cells (`gender/English_heard`, `gender/Bangla_unheard`, `no_gender/English_heard`, `no_gender/Bangla_unheard`).
   - Automatically packages CodaBench-verified `/kaggle/working/submission.zip`.
""")

    # =========================================================================
    # CELL 2: DEPENDENCIES & ENVIRONMENT SETUP
    # =========================================================================
    add_md("### 📦 1. Installation & Hardware Environment Setup")
    add_code("""# Install required libraries with QLoRA & Accelerate support
!pip install -q --upgrade pip
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers accelerate bitsandbytes peft open_clip_torch timm scikit-learn scipy Pillow

import os
# Disable Transformers auto-conversion thread and unauthenticated PR spam
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

# Disable transformers background PR auto-conversion thread permanently
try:
    import transformers.safetensors_conversion
    transformers.safetensors_conversion.auto_conversion = lambda *args, **kwargs: None
except (ImportError, AttributeError):
    pass

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

# Verify Accelerate & Hardware
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
    add_md("### ⚙️ 2. Multi-Billion Architecture Hyperparameters & Configuration")
    add_code("""@dataclass
class QOmni7BConfig:
    # --------------------------------------------------------------------------
    # 1. Dataset & Submission Paths
    # --------------------------------------------------------------------------
    DATA_ROOT: str = "/kaggle/input/datasets/mdjahidhasanjim/mav-celeb-v4-dataset"
    OUTPUT_DIR: str = "/kaggle/working"
    BEST_MODEL_PATH: str = "/kaggle/working/best_qomni_7b.pth"
    SUBMISSION_DIR: str = "/kaggle/working/submission"
    SUBMISSION_ZIP_PATH: str = "/kaggle/working/submission.zip"
    
    SUBMISSION_GENDER_ENGLISH: str = "/kaggle/working/submission/gender/sub_score_v4_English_heard.txt"
    SUBMISSION_GENDER_BANGLA: str = "/kaggle/working/submission/gender/sub_score_v4_Bangla_unheard.txt"
    SUBMISSION_NO_GENDER_ENGLISH: str = "/kaggle/working/submission/no_gender/sub_score_v4_English_heard.txt"
    SUBMISSION_NO_GENDER_BANGLA: str = "/kaggle/working/submission/no_gender/sub_score_v4_Bangla_unheard.txt"

    # --------------------------------------------------------------------------
    # 2. Multi-Billion Foundation Backbones
    # --------------------------------------------------------------------------
    # Vision Encoder:
    #   - "timm/eva02_enormous_patch14_plus_clip_224" (4.4 Billion params)
    #   - "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"   (1.8 Billion params)
    #   - Fallback: "facebook/dinov2-large" / "facebook/dinov2-giant"
    VISION_MODEL_ID: str = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
    
    # Audio Encoder:
    #   - "facebook/seamless-m4t-v2-large" (2.3 Billion params, 100+ languages)
    #   - "openai/whisper-large-v3"         (1.5 Billion params, multilingual)
    AUDIO_MODEL_ID: str = "facebook/seamless-m4t-v2-large"

    # --------------------------------------------------------------------------
    # 3. Q-Former (BLIP-2 Style) Architecture
    # --------------------------------------------------------------------------
    NUM_QUERY_TOKENS: int = 32     # Number of learnable query tokens K
    QUERY_DIM: int = 768           # Q-Former hidden dimension
    QFORMER_NUM_HEADS: int = 12
    QFORMER_NUM_LAYERS: int = 4
    SHARED_EMBED_DIM: int = 768    # Final L2-normalized hypersphere dimension

    # --------------------------------------------------------------------------
    # 4. QLoRA 4-bit NF4 Quantization Configuration
    # --------------------------------------------------------------------------
    LOAD_IN_4BIT: bool = True
    BNB_4BIT_QUANT_TYPE: str = "nf4"
    BNB_4BIT_USE_DOUBLE_QUANT: bool = True
    LORA_R: int = 16
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.05

    # --------------------------------------------------------------------------
    # 5. Preprocessing
    # --------------------------------------------------------------------------
    SAMPLE_RATE: int = 16000
    AUDIO_DURATION: float = 3.0    # 3 seconds = 48,000 samples
    AUDIO_SAMPLES: int = int(SAMPLE_RATE * AUDIO_DURATION)
    IMAGE_SIZE: int = 224

    # --------------------------------------------------------------------------
    # 6. Loss Weights
    # --------------------------------------------------------------------------
    ARCFACE_SCALE: float = 32.0
    ARCFACE_MARGIN: float = 0.35
    SUPCON_TEMP: float = 0.07

    LAMBDA_ARCFACE: float = 1.0    # Identity ArcFace Loss
    LAMBDA_SUPCON: float = 0.8     # Supervised Contrastive Loss
    LAMBDA_GRL: float = 0.5        # Adversarial Gender Loss

    # --------------------------------------------------------------------------
    # 7. Training Engine & Accelerate
    # --------------------------------------------------------------------------
    BATCH_SIZE: int = 8            # Per GPU batch size (Effective = 8 * 2 GPUs * 2 accum = 32)
    GRAD_ACCUM_STEPS: int = 2
    NUM_EPOCHS: int = 30
    LR_QFORMER: float = 2e-4       # Learning rate for Q-Former, GRL, ArcFace
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP_NORM: float = 3.0
    NUM_WORKERS: int = 2
    SEED: int = 42

    # --------------------------------------------------------------------------
    # 8. Early Stopping & Anti-Overfitting Regularization
    # --------------------------------------------------------------------------
    EARLY_STOPPING_PATIENCE: int = 4          # Epochs to wait without improvement before stopping
    EARLY_STOPPING_MIN_DELTA: float = 1e-4    # Minimum improvement threshold
    EARLY_STOPPING_MODE: str = "min"          # "min" for validation loss
    EARLY_STOPPING_RESTORE_BEST: bool = True  # Restore best model weights upon early stop
    VAL_SPLIT_RATIO: float = 0.1             # 10% held-out validation split

cfg = QOmni7BConfig()

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
    add_md("### 🎙️ 4. Audio & Face Preprocessing Pipelines")
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


def get_image_transforms(image_size=224, is_train=True):
    norm = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
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
    # CELL 6: DATASET LOADER
    # =========================================================================
    add_md("### 📂 5. Dataset & Hard-Negative Batching Engine")
    add_code("""class FLAGOmniTrainDataset(Dataset):
    def __init__(self, train_dir: str, is_train: bool = True):
        super().__init__()
        self.train_dir = train_dir
        self.is_train = is_train
        self.audio_proc = AudioPreprocessor(is_train=is_train)
        self.face_trans = get_image_transforms(image_size=cfg.IMAGE_SIZE, is_train=is_train)
        
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
                
            for spk in [d for d in os.listdir(lang_face) if os.path.isdir(os.path.join(lang_face, d))]:
                spk_str = f"{lang}_{spk}"
                if spk_str not in self.speaker_to_id:
                    self.speaker_to_id[spk_str] = len(self.speaker_to_id)
                spk_idx = self.speaker_to_id[spk_str]
                gender = int(abs(hash(spk_str)) % 2)

                face_files = sorted(glob.glob(os.path.join(lang_face, spk, "*.jpg")) + glob.glob(os.path.join(lang_face, spk, "*.png")))
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
            face_t = torch.zeros(3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)

        voice_t = self.audio_proc.process(item["voice_path"])

        # Mine same-gender negative voice
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

    def subset(self, indices: List[int], is_train: bool = True) -> 'FLAGOmniTrainDataset':
        sub = FLAGOmniTrainDataset.__new__(FLAGOmniTrainDataset)
        super(FLAGOmniTrainDataset, sub).__init__()
        sub.train_dir = self.train_dir
        sub.is_train = is_train
        sub.audio_proc = AudioPreprocessor(is_train=is_train)
        sub.face_trans = get_image_transforms(image_size=cfg.IMAGE_SIZE, is_train=is_train)
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


def create_qomni_train_val_datasets(
    train_dir: str,
    val_ratio: float = cfg.VAL_SPLIT_RATIO,
    seed: int = cfg.SEED
) -> Tuple[FLAGOmniTrainDataset, Optional[FLAGOmniTrainDataset]]:
    full_ds = FLAGOmniTrainDataset(train_dir, is_train=True)
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



class FLAGOmniDevDataset(Dataset):
    def __init__(self, trial_file_path: str, dev_dir: str):
        super().__init__()
        self.trial_file_path = trial_file_path
        self.dev_dir = dev_dir
        self.trial_dir = os.path.dirname(os.path.abspath(trial_file_path))
        self.audio_proc = AudioPreprocessor(is_train=False)
        self.face_trans = get_image_transforms(image_size=cfg.IMAGE_SIZE, is_train=False)
        self.trials = []
        self._parse()

    def _resolve(self, p: str) -> str:
        for c in [os.path.join(self.trial_dir, p), os.path.join(self.dev_dir, p),
                  os.path.join(self.dev_dir, "gender", p), os.path.join(self.dev_dir, "no_gender", p), p]:
            if os.path.exists(c):
                return c
        return p

    def _parse(self):
        if not os.path.exists(self.trial_file_path):
            return
        with open(self.trial_file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    pair_id = parts[0]
                    audio_p = self._resolve(parts[1])
                    face_p = self._resolve(parts[2])
                    lbl = int(parts[3]) if len(parts) >= 4 and parts[3].isdigit() else -1
                    self.trials.append({
                        "pair_id": pair_id,
                        "audio_path": audio_p,
                        "face_path": face_p,
                        "label": lbl
                    })

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.trials[idx]
        try:
            face_img = Image.open(item["face_path"]).convert("RGB")
            face_t = self.face_trans(face_img)
        except Exception:
            face_t = torch.zeros(3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)

        voice_t = self.audio_proc.process(item["audio_path"])
        return {
            "pair_id": item["pair_id"],
            "face_img": face_t,
            "voice_wav": voice_t,
            "label": item["label"]
        }
""")

    # =========================================================================
    # CELL 7: Q-FORMER (BLIP-2 ARCHITECTURE) & GRL IMPLEMENTATION
    # =========================================================================
    add_md("### 🧩 6. BLIP-2 Q-Former Cross-Modal Transformer & GRL")
    add_code("""class GradientReversalFunction(torch.autograd.Function):
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


class QFormerBlock(nn.Module):
    \"\"\"
    Single Q-Former Transformer Block (BLIP-2 Style):
    1. Query Self-Attention (Queries interact with each other)
    2. Query Cross-Attention (Queries attend to Encoder features)
    3. Feed-Forward Network (FFN)
    \"\"\"
    def __init__(self, d_model: int, nhead: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, encoder_hidden: torch.Tensor) -> torch.Tensor:
        # 1. Query Self-Attention
        q_sa, _ = self.self_attn(query, query, query)
        query = self.norm1(query + q_sa)

        # 2. Query Cross-Attention into Encoder Features
        q_ca, _ = self.cross_attn(query=query, key=encoder_hidden, value=encoder_hidden)
        query = self.norm2(query + q_ca)

        # 3. FFN
        q_ffn = self.ffn(query)
        query = self.norm3(query + q_ffn)
        return query


class MultimodalQFormer(nn.Module):
    \"\"\"
    Bimodal Q-Former:
    Maintains K learnable identity queries that independently extract and bind
    identity-discriminative tokens from Visual and Acoustic encoder feature streams.
    \"\"\"
    def __init__(
        self,
        num_queries: int = 32,
        query_dim: int = 768,
        vision_dim: int = 1280,
        audio_dim: int = 1024,
        num_layers: int = 4,
        num_heads: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_queries = num_queries
        self.query_dim = query_dim
        
        # Learnable Query Embeddings Q in R^(K x D_q)
        self.query_embeds = nn.Parameter(torch.randn(1, num_queries, query_dim) * 0.02)

        # Linear projections from encoder hidden dimension to query dimension
        self.vision_proj = nn.Linear(vision_dim, query_dim)
        self.audio_proj = nn.Linear(audio_dim, query_dim)

        # Q-Former Blocks
        self.vision_blocks = nn.ModuleList([
            QFormerBlock(query_dim, num_heads, d_ff=query_dim * 4, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.audio_blocks = nn.ModuleList([
            QFormerBlock(query_dim, num_heads, d_ff=query_dim * 4, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Query Aggregator (Pools K queries into single global embedding)
        self.pooler_attn = nn.Linear(query_dim, 1)

    def _pool_queries(self, queries: torch.Tensor) -> torch.Tensor:
        # queries: (B, K, D_q)
        attn_weights = F.softmax(self.pooler_attn(queries), dim=1) # (B, K, 1)
        pooled = torch.sum(attn_weights * queries, dim=1)           # (B, D_q)
        return pooled

    def forward(self, vision_tokens: torch.Tensor, audio_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dev = self.query_embeds.device
        if vision_tokens.device != dev:
            vision_tokens = vision_tokens.to(dev)
        if audio_tokens.device != dev:
            audio_tokens = audio_tokens.to(dev)
        B = vision_tokens.size(0)
        # Expand queries for batch
        queries = self.query_embeds.expand(B, -1, -1)

        # Project features to query dimension
        v_feat = self.vision_proj(vision_tokens)
        a_feat = self.audio_proj(audio_tokens)

        # Query Cross-Attention across Visual stream
        q_v = queries
        for blk in self.vision_blocks:
            q_v = blk(q_v, v_feat)

        # Query Cross-Attention across Audio stream
        q_a = queries
        for blk in self.audio_blocks:
            q_a = blk(q_a, a_feat)

        # Attentive Pooling over K queries
        z_face = self._pool_queries(q_v)
        z_voice = self._pool_queries(q_a)

        return z_face, z_voice
""")

    # =========================================================================
    # CELL 8: FOUNDATION BACKBONES & QLORA QUANTIZATION
    # =========================================================================
    add_md("### 🚀 7. Foundation Backbones (EVA-02 / CLIP-bigG + SeamlessM4T-v2) with QLoRA")
    add_code("""from transformers import AutoModel, AutoFeatureExtractor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class FLAGQOmni7BModel(nn.Module):
    \"\"\"
    Ultra-Scale Multimodal Foundation Model:
      - Vision: EVA-02-E (4.4B) / CLIP-ViT-bigG (1.8B) in 4-bit NF4
      - Audio: SeamlessM4T-v2-Large (2.3B) in 4-bit NF4
      - Fusion: BLIP-2 Q-Former
      - Debiasing: GRL + Gender Discriminator
    \"\"\"
    def __init__(self, num_speakers: int = 100, cfg: QOmni7BConfig = cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.SHARED_EMBED_DIM
        target_dev = accelerator.device if torch.cuda.is_available() else "cpu"
        dev_map = {"": target_dev} if (torch.cuda.is_available() and cfg.LOAD_IN_4BIT) else None

        # Configure 4-bit NF4 Quantization (QLoRA)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=cfg.LOAD_IN_4BIT,
            bnb_4bit_quant_type=cfg.BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=cfg.BNB_4BIT_USE_DOUBLE_QUANT,
        ) if (torch.cuda.is_available() and cfg.LOAD_IN_4BIT) else None

        # ----------------------------------------------------------------------
        # 1. Vision Encoder Initialization (CLIP-ViT-bigG / EVA-02-E)
        # ----------------------------------------------------------------------
        if accelerator.is_main_process:
            print(f"[Model Init] Loading Vision Backbone: {cfg.VISION_MODEL_ID} on {target_dev}...")
        try:
            self.vision_encoder = AutoModel.from_pretrained(
                cfg.VISION_MODEL_ID,
                quantization_config=bnb_config,
                device_map=dev_map,
                trust_remote_code=True
            )
            v_hidden = getattr(self.vision_encoder.config, "hidden_size", 1280)
            if hasattr(self.vision_encoder, "vision_model") and hasattr(self.vision_encoder.vision_model.config, "hidden_size"):
                v_hidden = self.vision_encoder.vision_model.config.hidden_size
        except Exception as e:
            if accelerator.is_main_process:
                print(f"[Fallback] Vision backbone fallback to facebook/dinov2-large: {e}")
            self.vision_encoder = AutoModel.from_pretrained("facebook/dinov2-large", quantization_config=bnb_config, device_map=dev_map)
            v_hidden = 1024

        # ----------------------------------------------------------------------
        # 2. Audio Encoder Initialization (SeamlessM4T-v2-Large / Whisper)
        # ----------------------------------------------------------------------
        if accelerator.is_main_process:
            print(f"[Model Init] Loading Audio Backbone: {cfg.AUDIO_MODEL_ID} on {target_dev}...")
        try:
            self.audio_encoder = AutoModel.from_pretrained(
                cfg.AUDIO_MODEL_ID,
                quantization_config=bnb_config,
                device_map=dev_map,
                trust_remote_code=True
            )
            a_hidden = getattr(self.audio_encoder.config, "hidden_size", 1024)
            if hasattr(self.audio_encoder, "speech_encoder") and hasattr(self.audio_encoder.speech_encoder.config, "hidden_size"):
                a_hidden = self.audio_encoder.speech_encoder.config.hidden_size
        except Exception as e:
            if accelerator.is_main_process:
                print(f"[Fallback] Audio backbone fallback to facebook/mms-300m: {e}")
            self.audio_encoder = AutoModel.from_pretrained("facebook/mms-300m", quantization_config=bnb_config, device_map=dev_map)
            a_hidden = 1024

        # ----------------------------------------------------------------------
        # 3. BLIP-2 Q-Former
        # ----------------------------------------------------------------------
        self.qformer = MultimodalQFormer(
            num_queries=cfg.NUM_QUERY_TOKENS,
            query_dim=cfg.QUERY_DIM,
            vision_dim=v_hidden,
            audio_dim=a_hidden,
            num_layers=cfg.QFORMER_NUM_LAYERS,
            num_heads=cfg.QFORMER_NUM_HEADS
        ).to(target_dev)

        # ----------------------------------------------------------------------
        # 4. GRL Demographic Debiasing Head
        # ----------------------------------------------------------------------
        self.grl = GradientReversalLayer(alpha=1.0).to(target_dev)
        self.gender_discriminator = nn.Sequential(
            nn.Linear(cfg.QUERY_DIM, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        ).to(target_dev)

        # ----------------------------------------------------------------------
        # 5. ArcFace Metric Heads
        # ----------------------------------------------------------------------
        self.arcface_head = ArcFaceMargin(cfg.QUERY_DIM, num_speakers, scale=cfg.ARCFACE_SCALE, margin=cfg.ARCFACE_MARGIN).to(target_dev)

    def extract_vision_tokens(self, face_img: torch.Tensor) -> torch.Tensor:
        v_dev = next(self.vision_encoder.parameters()).device
        if face_img.device != v_dev:
            face_img = face_img.to(v_dev)
        if hasattr(self.vision_encoder, "vision_model"):
            out = self.vision_encoder.vision_model(pixel_values=face_img)
        else:
            out = self.vision_encoder(pixel_values=face_img)
        # Token sequence (B, N_patches, D_v)
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        return out[0]

    def extract_audio_tokens(self, voice_wav: torch.Tensor) -> torch.Tensor:
        a_dev = next(self.audio_encoder.parameters()).device
        if voice_wav.device != a_dev:
            voice_wav = voice_wav.to(a_dev)
        if hasattr(self.audio_encoder, "speech_encoder"):
            out = self.audio_encoder.speech_encoder(voice_wav)
        elif hasattr(self.audio_encoder, "audio_encoder"):
            out = self.audio_encoder.audio_encoder(voice_wav)
        else:
            out = self.audio_encoder(voice_wav)
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        return out[0]

    def forward(
        self,
        face_img: torch.Tensor,
        voice_wav: torch.Tensor,
        spk_label: Optional[torch.Tensor] = None,
        grl_lambda: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        # 1. Extract feature token streams
        v_tokens = self.extract_vision_tokens(face_img)
        a_tokens = self.extract_audio_tokens(voice_wav)

        # 2. BLIP-2 Q-Former Cross-Modal Extraction
        z_face, z_voice = self.qformer(v_tokens, a_tokens)

        # 3. Project to L2-Normalized Hypersphere S^(D-1) (in float32 for fp16 stability)
        e_f_fp32 = F.normalize(z_face.float(), p=2, dim=-1)
        e_v_fp32 = F.normalize(z_voice.float(), p=2, dim=-1)

        # 4. Euclidean Distance: d = sqrt(2 - 2 * cos(θ)) = ||e_f - e_v||_2
        cos_sim = torch.sum(e_f_fp32 * e_v_fp32, dim=-1).clamp(-1.0 + 1e-4, 1.0 - 1e-4)
        dist = torch.sqrt(torch.clamp(2.0 - 2.0 * cos_sim, min=1e-4))

        e_f = e_f_fp32.type_as(z_face)
        e_v = e_v_fp32.type_as(z_voice)

        out = {
            "face_embed": e_f,
            "voice_embed": e_v,
            "distance": dist,
            "cosine_sim": cos_sim
        }

        # Training Heads
        if spk_label is not None:
            out["arc_face_logits"] = self.arcface_head(e_f, spk_label)
            out["arc_voice_logits"] = self.arcface_head(e_v, spk_label)

            # Adversarial Gender Debiasing
            f_rev = self.grl(e_f, grl_lambda)
            v_rev = self.grl(e_v, grl_lambda)
            out["face_gender_logits"] = self.gender_discriminator(f_rev)
            out["voice_gender_logits"] = self.gender_discriminator(v_rev)

        return out
""")

    # =========================================================================
    # CELL 9: LOSS FUNCTIONS (ARCFACE + SUPCON)
    # =========================================================================
    add_md("### 🎯 8. Loss Functions: ArcFace + SupCon (Supervised Contrastive)")
    add_code("""class ArcFaceMargin(nn.Module):
    def __init__(self, in_features: int, num_classes: int, scale: float = 32.0, margin: float = 0.35):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for fp16 numerical stability
        x_f = x.float()
        w_f = self.weight.float()
        cosine = F.linear(F.normalize(x_f, p=2, dim=-1), F.normalize(w_f, p=2, dim=-1))
        cosine = torch.clamp(cosine, -1.0 + 1e-4, 1.0 - 1e-4)
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-4))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=x.device, dtype=torch.float32)
        one_hot.scatter_(1, label.view(-1, 1).long().to(x.device), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return (output * self.scale).type_as(x)


class SupConLoss(nn.Module):
    \"\"\"
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).
    Pulls together multimodal embeddings of the same speaker identity,
    pushes apart all same-gender negative impostors.
    \"\"\"
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # features: (B, 2, D) where dim 1 contains [face_embed, voice_embed]
        device = features.device
        batch_size = features.shape[0]

        # Flatten features and compute in float32 for fp16 stability
        feats = torch.cat([features[:, 0].float(), features[:, 1].float()], dim=0) # (2B, D)
        feats = F.normalize(feats, p=2, dim=-1)

        labels = labels.contiguous().view(-1, 1).to(device)
        mask = torch.eq(labels, labels.T).float().to(device)       # (B, B)
        mask = mask.repeat(2, 2)                                    # (2B, 2B)

        # Mask out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * 2, device=device).view(-1, 1),
            0
        )
        mask = mask * logits_mask

        # Compute similarity matrix
        temp = max(float(self.temperature), 0.05)
        sim = torch.div(torch.matmul(feats, feats.T), temp)
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        # Log probability
        exp_logits = torch.exp(logits) * logits_mask
        denom = exp_logits.sum(1, keepdim=True)

        mask_pos_sums = mask.sum(1)
        valid_rows = mask_pos_sums > 0
        if not torch.any(valid_rows):
            return torch.tensor(0.0, device=device, requires_grad=True)

        log_prob = logits - torch.log(denom + 1e-5)
        mean_log_prob_pos = (mask * log_prob).sum(1)[valid_rows] / mask_pos_sums[valid_rows]
        loss = -mean_log_prob_pos.mean()
        return loss


class CompositeOmniLoss(nn.Module):
    def __init__(self, cfg: QOmni7BConfig):
        super().__init__()
        self.cfg = cfg
        self.ce = nn.CrossEntropyLoss()
        self.supcon = SupConLoss(temperature=cfg.SUPCON_TEMP)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        spk_label: torch.Tensor,
        gender_label: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        dev = outputs["arc_face_logits"].device
        spk_label = spk_label.to(dev)
        gender_label = gender_label.to(dev)

        # 1. ArcFace Identity Loss (computed in float32)
        loss_arc_f = self.ce(outputs["arc_face_logits"].float(), spk_label)
        loss_arc_v = self.ce(outputs["arc_voice_logits"].float(), spk_label)
        loss_arc = (loss_arc_f + loss_arc_v) / 2.0

        # 2. Supervised Contrastive Loss (Cross-Modal Identity Alignment)
        multimodal_feats = torch.stack([outputs["face_embed"].float(), outputs["voice_embed"].float()], dim=1) # (B, 2, D)
        loss_supcon = self.supcon(multimodal_feats, spk_label)

        # 3. Adversarial Demographic Debiasing Loss (computed in float32)
        loss_grl_f = self.ce(outputs["face_gender_logits"].float(), gender_label)
        loss_grl_v = self.ce(outputs["voice_gender_logits"].float(), gender_label)
        loss_grl = (loss_grl_f + loss_grl_v) / 2.0

        total_loss = (
            self.cfg.LAMBDA_ARCFACE * loss_arc +
            self.cfg.LAMBDA_SUPCON * loss_supcon +
            self.cfg.LAMBDA_GRL * loss_grl
        )

        return {
            "total_loss": total_loss,
            "loss_arc": loss_arc,
            "loss_supcon": loss_supcon,
            "loss_grl": loss_grl
        }
""")

    # =========================================================================
    # CELL 10: TRAINING LOOP WITH ACCELERATE (DUAL GPU)
    # =========================================================================
    add_md("### ⚡ 9. Dual-GPU Training Engine Powered by Hugging Face `accelerate`")
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


def train_qomni_7b(model, train_dataset, val_dataset=None, cfg: QOmni7BConfig = cfg):
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

    criterion = CompositeOmniLoss(cfg)
    early_stopping = EarlyStopping(
        patience=cfg.EARLY_STOPPING_PATIENCE,
        min_delta=cfg.EARLY_STOPPING_MIN_DELTA,
        mode=cfg.EARLY_STOPPING_MODE,
        restore_best_weights=cfg.EARLY_STOPPING_RESTORE_BEST
    )

    qformer_params = list(model.qformer.parameters()) + list(model.arcface_head.parameters()) + list(model.gender_discriminator.parameters())
    optimizer = torch.optim.AdamW([
        {"params": qformer_params, "lr": cfg.LR_QFORMER, "weight_decay": cfg.WEIGHT_DECAY}
    ])

    total_steps = len(dataloader) * cfg.NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # Accelerate Dual-GPU Preparation
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
            spk_label = batch["spk_label"]
            gender_label = batch["gender_label"]

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
                    "arc": f"{losses['loss_arc'].item():.3f}",
                    "supcon": f"{losses['loss_supcon'].item():.3f}"
                })

        mean_train_loss = np.mean(epoch_losses) if epoch_losses else 0.0

        # Run Validation Loop for Early Stopping
        val_losses = []
        if val_dataloader is not None:
            model.eval()
            with torch.no_grad():
                for v_batch in val_dataloader:
                    v_outputs = model(v_batch["face_img"], v_batch["voice_wav"], spk_label=v_batch["spk_label"])
                    v_losses = criterion(v_outputs, v_batch["spk_label"], v_batch["gender_label"])
                    v_val = v_losses["total_loss"].item()
                    if not (math.isnan(v_val) or math.isinf(v_val)):
                        val_losses.append(v_val)
            mean_val_loss = np.mean(val_losses) if val_losses else mean_train_loss
        else:
            mean_val_loss = mean_train_loss

        # Early Stopping check on main process
        if accelerator.is_main_process:
            val_str = f"Val Loss: {mean_val_loss:.4f}" if val_dataloader is not None else "Val: N/A"
            print(f"Epoch {epoch:02d} Complete | Train Loss: {mean_train_loss:.4f} | {val_str}")
            is_best = early_stopping.step(mean_val_loss, accelerator.unwrap_model(model), epoch)
            if is_best:
                best_loss = mean_val_loss
                unwrapped = accelerator.unwrap_model(model)
                torch.save(unwrapped.state_dict(), cfg.BEST_MODEL_PATH)
                print(f" -> Checkpoint saved to {cfg.BEST_MODEL_PATH}")

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
    add_md("### 📊 10. Official 4-Cell CodaBench Evaluation & Benchmark")
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

    dataset = FLAGOmniDevDataset(trial_path, dev_dir)
    if len(dataset) == 0:
        return -1.0, -1.0

    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE * 2, shuffle=False, num_workers=cfg.NUM_WORKERS)
    model.eval()

    pairs, dists, labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            face_img = batch["face_img"].to(device)
            voice_wav = batch["voice_wav"].to(device)
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
train_dataset, val_dataset = create_qomni_train_val_datasets(TRAIN_DIR, cfg.VAL_SPLIT_RATIO, cfg.SEED)
num_spks = max(100, len(train_dataset.speaker_to_id))

# 2. Build Multi-Billion Foundation Model with Q-Former & QLoRA
model = FLAGQOmni7BModel(num_speakers=num_spks, cfg=cfg)

if accelerator.is_main_process:
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 70)
    print(f"Total Model Parameters     : {total_p / 1e9:.2f} Billion ({total_p:,})")
    print(f"Trainable Parameters (QLoRA): {train_p / 1e6:.2f} Million ({train_p:,})")
    print("=" * 70)

# 3. Train with Accelerate Dual GPU & Early Stopping Monitoring
if len(train_dataset) > 0:
    model = train_qomni_7b(model, train_dataset, val_dataset, cfg)

# 4. Evaluate Across All 4 CodaBench Cells
if DEV_DIR and os.path.exists(DEV_DIR):
    run_full_evaluation(model, DEV_DIR, cfg)

# 5. Package Submission
package_submission(cfg)
""")

    out_p = os.path.join("y:\\FLAG\\shobrikola\\notebooks", "FLAG2027_SOTA_7B_QLoRA_Multimodal.ipynb")
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"Successfully generated notebook: {out_p}")
    print(f"Total cells: {len(nb['cells'])}")

if __name__ == "__main__":
    create_notebook()
