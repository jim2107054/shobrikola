"""
FLAG 2027 Challenge: Face-Voice Association Across Languages and Gender
Dataset & Data Loading Module (dataset.py)
"""

import os
import glob
import random
from typing import List, Dict, Tuple, Optional, Any

import torch
from torch.utils.data import Dataset, Sampler, DataLoader
import torchaudio
from PIL import Image
import torchvision.transforms as transforms

from config import config


# ==============================================================================
# 1. Audio Processing & Augmentation Utility
# ==============================================================================

class AudioProcessor:
    """
    Robust audio loader, resampler to 16kHz, length normalizer, and augmentor.
    """
    def __init__(
        self,
        target_sample_rate: int = config.AUDIO_SAMPLE_RATE,
        target_num_samples: int = config.AUDIO_NUM_SAMPLES,
        is_training: bool = True,
    ):
        self.target_sample_rate = target_sample_rate
        self.target_num_samples = target_num_samples
        self.is_training = is_training
        self.resamplers: Dict[int, torchaudio.transforms.Resample] = {}

    def _get_resampler(self, orig_sr: int) -> torchaudio.transforms.Resample:
        if orig_sr not in self.resamplers:
            self.resamplers[orig_sr] = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=self.target_sample_rate
            )
        return self.resamplers[orig_sr]

    def load_and_preprocess(self, audio_path: str) -> torch.Tensor:
        """
        Loads .wav audio, resamples to 16kHz, converts to mono, normalizes,
        and cuts or pads to exact target_num_samples (48,000 samples = 3.0s).
        """
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            # Fallback for corrupted or unreadable audio: return silent waveform
            print(f"[Warning] Failed to load audio {audio_path}: {e}")
            return torch.zeros(self.target_num_samples, dtype=torch.float32)

        # 1. Convert to Mono by averaging channels if stereo/multichannel
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 2. Resample to 16,000 Hz if necessary
        if sample_rate != self.target_sample_rate:
            resampler = self._get_resampler(sample_rate)
            waveform = resampler(waveform)

        waveform = waveform.squeeze(0)  # Shape: (num_samples,)

        # 3. Audio Length Normalization (Padding or Cropping)
        num_samples = waveform.shape[0]
        if num_samples < self.target_num_samples:
            # Repeat waveform or pad with zeros
            if num_samples > 0:
                repeat_factor = (self.target_num_samples // num_samples) + 1
                waveform = waveform.repeat(repeat_factor)[:self.target_num_samples]
            else:
                waveform = torch.zeros(self.target_num_samples, dtype=torch.float32)
        elif num_samples > self.target_num_samples:
            if self.is_training:
                # Random crop during training
                max_start = num_samples - self.target_num_samples
                start = random.randint(0, max_start)
                waveform = waveform[start : start + self.target_num_samples]
            else:
                # Center crop during evaluation
                start = (num_samples - self.target_num_samples) // 2
                waveform = waveform[start : start + self.target_num_samples]

        # 4. Data Augmentation (Audio Noise, Shift, Volume) for Training
        if self.is_training:
            waveform = self._augment_audio(waveform)

        # 5. Amplitude Normalization (Zero mean, unit variance or [-1, 1] scaling)
        std = waveform.std()
        if std > 1e-6:
            waveform = (waveform - waveform.mean()) / std
        else:
            waveform = torch.zeros_like(waveform)

        return waveform.to(torch.float32)

    def _augment_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Applies random circular shift, additive noise, and volume perturbations.
        """
        # Circular time shift (up to +/- 0.5s = 8000 samples)
        if random.random() < 0.5:
            shift = random.randint(-8000, 8000)
            waveform = torch.roll(waveform, shifts=shift, dims=0)

        # Random volume perturbation
        if random.random() < 0.5:
            vol_scale = random.uniform(0.7, 1.3)
            waveform = waveform * vol_scale

        # Additive Gaussian noise (SNR perturbation)
        if random.random() < 0.4:
            noise_level = random.uniform(0.001, 0.015)
            noise = torch.randn_like(waveform) * noise_level
            waveform = waveform + noise

        return waveform


# ==============================================================================
# 2. Image Processing & Transforms
# ==============================================================================

def get_image_transforms(is_training: bool = True) -> transforms.Compose:
    """
    Standard Face Recognition transforms (112x112 resolution, ImageNet normalization).
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(
                (config.IMAGE_SIZE, config.IMAGE_SIZE),
                scale=(0.85, 1.0),
                ratio=(0.9, 1.1)
            ),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


def load_image(image_path: str, transform: transforms.Compose) -> torch.Tensor:
    """Safely loads and transforms a face image."""
    try:
        image = Image.open(image_path).convert("RGB")
        return transform(image)
    except Exception as e:
        print(f"[Warning] Failed to load image {image_path}: {e}")
        return torch.zeros((3, config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=torch.float32)


# ==============================================================================
# 3. Training Dataset with Hard Negative Mining (Same-Gender Negatives)
# ==============================================================================

class FLAGTrainDataset(Dataset):
    """
    PyTorch Dataset for MAV-Celeb v4 Training Set.
    Automatically parses directories or metadata text files.
    Enforces Hard Negative Mining: for every anchor (face, positive voice),
    it samples a same-gender negative voice from a different speaker identity.
    """
    def __init__(self, train_dir: Optional[str] = None):
        super().__init__()
        self.train_dir = train_dir or config.get_train_dir()
        self.audio_processor = AudioProcessor(is_training=True)
        self.image_transform = get_image_transforms(is_training=True)

        # Parsed dataset items
        self.samples: List[Dict[str, Any]] = []
        self.speaker_to_id: Dict[str, int] = {}
        self.id_to_speaker: Dict[int, str] = {}
        self.gender_map: Dict[str, int] = {}  # speaker_id -> 0 (Male) or 1 (Female)

        # Indexing for hard negative mining
        self.speaker_to_indices: Dict[int, List[int]] = {}
        self.gender_to_speakers: Dict[int, List[int]] = {0: [], 1: []}
        self.gender_to_indices: Dict[int, List[int]] = {0: [], 1: []}

        self._discover_and_parse()
        self._build_mining_indices()

    def _discover_and_parse(self):
        """
        Parses train_dir across diverse Kaggle directory structures:
        1. Hierarchical: faces/<lang>/<spk_id>/*.jpg and voices/<lang>/<spk_id>/*.wav
        2. Per-speaker: <spk_id>/faces/ and <spk_id>/voices/
        3. Trial / Meta text files or CSVs (e.g. train_English_faces.csv, train.txt)
        """
        if not os.path.exists(self.train_dir):
            print(f"[Warning] Training directory {self.train_dir} does not exist yet.")
            return

        # 1. Check for metadata or gender files across dataset root and train_dir
        search_dirs = [self.train_dir, os.path.dirname(self.train_dir), config.DATA_ROOT]
        meta_files = []
        for s_dir in search_dirs:
            if os.path.exists(s_dir):
                meta_files.extend(glob.glob(os.path.join(s_dir, "*meta*.txt")))
                meta_files.extend(glob.glob(os.path.join(s_dir, "*gender*.txt")))
                meta_files.extend(glob.glob(os.path.join(s_dir, "**", "*gender*.txt"), recursive=True))

        for mf in set(meta_files):
            try:
                with open(mf, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().replace(",", " ").split()
                        if len(parts) >= 2:
                            spk_id = parts[0]
                            g_str = parts[1].lower()
                            if g_str in ["f", "female", "1", "woman"]:
                                self.gender_map[spk_id] = 1
                            elif g_str in ["m", "male", "0", "man"]:
                                self.gender_map[spk_id] = 0
            except Exception as e:
                print(f"[Warning] Could not parse meta file {mf}: {e}")

        # 2. Check for Kaggle layout: faces/ and voices/ subdirectories
        faces_dir = None
        voices_dir = None
        for candidate_root in [
            self.train_dir,
            os.path.join(self.train_dir, "train_set"),
            os.path.join(config.DATA_ROOT, "train_set", "train_set", "train_set"),
            os.path.join(config.DATA_ROOT, "train_set", "train_set"),
        ]:
            if os.path.exists(candidate_root):
                f_cand = os.path.join(candidate_root, "faces")
                v_cand = os.path.join(candidate_root, "voices")
                if os.path.exists(f_cand) and os.path.exists(v_cand):
                    faces_dir = f_cand
                    voices_dir = v_cand
                    break

        if faces_dir and voices_dir:
            print(f"[Info] Found separated modality trees:")
            print(f"       Faces:  {faces_dir}")
            print(f"       Voices: {voices_dir}")

            # Collect speaker IDs under faces/<lang>/<spk_id> or faces/<spk_id>
            # and voices/<lang>/<spk_id> or voices/<spk_id>
            speaker_ids = set()

            # Check if language subdirectories exist (e.g. English, Bangla)
            face_langs = [d for d in os.listdir(faces_dir) if os.path.isdir(os.path.join(faces_dir, d))]
            for l_dir in face_langs:
                l_path = os.path.join(faces_dir, l_dir)
                for spk in os.listdir(l_path):
                    if os.path.isdir(os.path.join(l_path, spk)):
                        speaker_ids.add(spk)
                    elif any(spk.lower().endswith(ext) for ext in [".jpg", ".png", ".jpeg"]):
                        # Direct speaker directory
                        speaker_ids.add(l_dir)
                        break

            print(f"[Info] Discovered {len(speaker_ids)} speakers from faces directory.")

            for spk_id in sorted(speaker_ids):
                # Discover all images for this speaker across languages
                img_patterns = [
                    os.path.join(faces_dir, "*", spk_id, "*.jpg"),
                    os.path.join(faces_dir, "*", spk_id, "*.png"),
                    os.path.join(faces_dir, spk_id, "*.jpg"),
                    os.path.join(faces_dir, spk_id, "*.png"),
                ]
                images = []
                for pat in img_patterns:
                    images.extend(glob.glob(pat))

                # Discover all wav audios for this speaker across languages
                wav_patterns = [
                    os.path.join(voices_dir, "*", spk_id, "*.wav"),
                    os.path.join(voices_dir, spk_id, "*.wav"),
                ]
                audios = []
                for pat in wav_patterns:
                    audios.extend(glob.glob(pat))

                if not images or not audios:
                    continue

                gender = self.gender_map.get(spk_id, hash(spk_id) % 2)

                # Pair images and audios into samples
                num_pairs = max(len(images), len(audios))
                for i in range(num_pairs):
                    img_path = images[i % len(images)]
                    wav_path = audios[i % len(audios)]
                    self._add_sample(spk_id, img_path, wav_path, gender)

        # 3. Fallback: If no samples yet, check for text files or <spk_id>/faces layout
        if len(self.samples) == 0:
            train_list_files = glob.glob(os.path.join(self.train_dir, "*train*.txt"))
            for lf in train_list_files:
                if "meta" in lf or "gender" in lf:
                    continue
                try:
                    with open(lf, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().replace(",", " ").split()
                            if len(parts) >= 3:
                                spk_id = parts[0]
                                path_a, path_b = parts[1], parts[2]
                                img_path = path_a if any(path_a.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png"]) else path_b
                                wav_path = path_a if path_a.lower().endswith(".wav") else path_b
                                if not os.path.isabs(img_path):
                                    img_path = os.path.join(self.train_dir, img_path)
                                if not os.path.isabs(wav_path):
                                    wav_path = os.path.join(self.train_dir, wav_path)

                                gender = self.gender_map.get(spk_id, hash(spk_id) % 2)
                                self._add_sample(spk_id, img_path, wav_path, gender)
                except Exception as e:
                    print(f"[Notice] Could not parse {lf}: {e}")

        # 4. Fallback: Scan direct speaker directories <spk_id>/...
        if len(self.samples) == 0:
            print(f"[Info] Scanning per-speaker directories in {self.train_dir}...")
            speaker_dirs = [
                d for d in os.listdir(self.train_dir)
                if os.path.isdir(os.path.join(self.train_dir, d)) and d not in ["features"]
            ]
            for spk_id in sorted(speaker_dirs):
                spk_path = os.path.join(self.train_dir, spk_id)
                images = glob.glob(os.path.join(spk_path, "**", "*.jpg"), recursive=True) + \
                         glob.glob(os.path.join(spk_path, "**", "*.png"), recursive=True)
                audios = glob.glob(os.path.join(spk_path, "**", "*.wav"), recursive=True)

                if not images or not audios:
                    continue

                gender = self.gender_map.get(spk_id, hash(spk_id) % 2)
                num_pairs = max(len(images), len(audios))
                for i in range(num_pairs):
                    img_path = images[i % len(images)]
                    wav_path = audios[i % len(audios)]
                    self._add_sample(spk_id, img_path, wav_path, gender)

        print(f"[Dataset] Successfully loaded {len(self.samples)} train samples across {len(self.speaker_to_id)} speakers.")
        print(f"[Dataset] Gender balance: {len(self.gender_to_speakers[0])} Male, {len(self.gender_to_speakers[1])} Female speakers.")

    def _add_sample(self, spk_id: str, img_path: str, wav_path: str, gender: int):
        if spk_id not in self.speaker_to_id:
            spk_idx = len(self.speaker_to_id)
            self.speaker_to_id[spk_id] = spk_idx
            self.id_to_speaker[spk_idx] = spk_id
        else:
            spk_idx = self.speaker_to_id[spk_id]

        self.samples.append({
            "speaker_id": spk_id,
            "speaker_idx": spk_idx,
            "image_path": img_path,
            "audio_path": wav_path,
            "gender": gender,
        })

    def _build_mining_indices(self):
        """Indexes samples by speaker and gender for fast hard negative sampling."""
        for idx, sample in enumerate(self.samples):
            spk_idx = sample["speaker_idx"]
            gender = sample["gender"]

            if spk_idx not in self.speaker_to_indices:
                self.speaker_to_indices[spk_idx] = []
            self.speaker_to_indices[spk_idx].append(idx)

            if spk_idx not in self.gender_to_speakers[gender]:
                self.gender_to_speakers[gender].append(spk_idx)

            self.gender_to_indices[gender].append(idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
            - face_img: Tensor (3, 112, 112)
            - audio_wav: Tensor (48000,) [Positive face-voice match]
            - neg_audio_wav: Tensor (48000,) [Hard Negative: Same Gender, Different Speaker]
            - speaker_idx: int (anchor speaker id)
            - neg_speaker_idx: int (hard negative speaker id)
            - gender: int (0 or 1)
        """
        sample = self.samples[idx]
        spk_idx = sample["speaker_idx"]
        gender = sample["gender"]

        # 1. Load Anchor Face and Positive Audio
        face_img = load_image(sample["image_path"], self.image_transform)
        audio_wav = self.audio_processor.load_and_preprocess(sample["audio_path"])

        # 2. Hard Negative Mining: Sample audio from DIFFERENT speaker of the SAME gender
        same_gender_speakers = self.gender_to_speakers[gender]
        other_speakers = [s for s in same_gender_speakers if s != spk_idx]

        if other_speakers:
            neg_spk_idx = random.choice(other_speakers)
            neg_sample_idx = random.choice(self.speaker_to_indices[neg_spk_idx])
            neg_sample = self.samples[neg_sample_idx]
            neg_audio_wav = self.audio_processor.load_and_preprocess(neg_sample["audio_path"])
        else:
            # Fallback if only 1 speaker in that gender
            neg_spk_idx = (spk_idx + 1) % max(1, len(self.speaker_to_id))
            neg_audio_wav = torch.randn_like(audio_wav)

        return {
            "face_img": face_img,
            "audio_wav": audio_wav,
            "neg_audio_wav": neg_audio_wav,
            "speaker_idx": torch.tensor(spk_idx, dtype=torch.long),
            "neg_speaker_idx": torch.tensor(neg_spk_idx, dtype=torch.long),
            "gender": torch.tensor(gender, dtype=torch.long),
        }


# ==============================================================================
# 4. Gender-Constrained Batch Sampler (Enforcing Same-Gender Batches)
# ==============================================================================

class SameGenderBatchSampler(Sampler):
    """
    Batches are partitioned such that samples in each batch share the same gender.
    This guarantees that all in-batch negative pairs for Supervised Contrastive Loss
    are same-gender negative pairs, stripping the model's ability to use gender shortcuts.
    """
    def __init__(self, dataset: FLAGTrainDataset, batch_size: int = config.BATCH_SIZE):
        self.dataset = dataset
        self.batch_size = batch_size
        self.male_indices = list(dataset.gender_to_indices[0])
        self.female_indices = list(dataset.gender_to_indices[1])

    def __iter__(self):
        random.shuffle(self.male_indices)
        random.shuffle(self.female_indices)

        batches = []
        # Chunk male indices
        for i in range(0, len(self.male_indices), self.batch_size):
            chunk = self.male_indices[i : i + self.batch_size]
            if len(chunk) == self.batch_size:
                batches.append(chunk)

        # Chunk female indices
        for i in range(0, len(self.female_indices), self.batch_size):
            chunk = self.female_indices[i : i + self.batch_size]
            if len(chunk) == self.batch_size:
                batches.append(chunk)

        random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        return (len(self.male_indices) // self.batch_size) + (len(self.female_indices) // self.batch_size)


# ==============================================================================
# 5. Development / Evaluation Dataset (Strict CodaBench format parser)
# ==============================================================================

class FLAGDevDataset(Dataset):
    """
    PyTorch Dataset for MAV-Celeb v4 Development Set.
    Parses trial text files (e.g. dev_set/gender.txt, dev_set/no_gender.txt).
    Each line in a trial file contains: [pair_id] [path_to_audio] [path_to_face] [optional_label].
    """
    def __init__(self, trial_file_path: str, dev_dir: Optional[str] = None):
        super().__init__()
        self.trial_file_path = trial_file_path
        self.dev_dir = dev_dir or config.get_dev_dir()
        self.trial_dir = os.path.dirname(os.path.abspath(trial_file_path))
        self.audio_processor = AudioProcessor(is_training=False)
        self.image_transform = get_image_transforms(is_training=False)
        self.trials: List[Dict[str, Any]] = []

        self._parse_trial_file()

    def _resolve_path(self, rel_path: str) -> str:
        """
        Robustly resolves media paths in trial files:
        Checks trial_dir (e.g. .../gender/), dev_dir (e.g. .../dev_set/),
        and handles variations like 'English_test/faces/...' or 'faces/...'.
        """
        if os.path.isabs(rel_path) and os.path.exists(rel_path):
            return rel_path

        # 1. Relative to trial file's directory (e.g. .../dev_set/dev_set/gender/English_test/...)
        cand1 = os.path.join(self.trial_dir, rel_path)
        if os.path.exists(cand1):
            return cand1

        # 2. Relative to dev_dir
        cand2 = os.path.join(self.dev_dir, rel_path)
        if os.path.exists(cand2):
            return cand2

        # 3. Under track folder inside dev_dir (e.g. dev_dir/gender/...)
        track_name = os.path.basename(self.trial_dir)
        cand3 = os.path.join(self.dev_dir, track_name, rel_path)
        if os.path.exists(cand3):
            return cand3

        # 4. Strip leading folder prefix if already inside subfolder
        parts = rel_path.replace("\\", "/").split("/")
        if len(parts) > 1:
            cand4 = os.path.join(self.trial_dir, *parts[1:])
            if os.path.exists(cand4):
                return cand4

        return cand1

    def _parse_trial_file(self):
        """Parses the evaluation trial file with automatic path and column resolution."""
        if not os.path.exists(self.trial_file_path):
            print(f"[Warning] Trial file {self.trial_file_path} not found.")
            return

        with open(self.trial_file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.replace(",", " ").split()
                if len(parts) < 3:
                    continue

                pair_id = parts[0]
                item_a, item_b = parts[1], parts[2]

                # Identify image vs audio file paths
                if any(item_a.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                    img_path = item_a
                    wav_path = item_b
                else:
                    wav_path = item_a
                    img_path = item_b

                # Resolve relative paths
                img_path = self._resolve_path(img_path)
                wav_path = self._resolve_path(wav_path)

                # Ground truth label (1: same identity, 0: different identity, -1: test/unlabelled)
                label = -1
                if len(parts) >= 4:
                    try:
                        label = int(float(parts[3]))
                    except ValueError:
                        label = -1

                self.trials.append({
                    "pair_id": pair_id,
                    "image_path": img_path,
                    "audio_path": wav_path,
                    "label": label,
                })

        print(f"[DevDataset] Loaded {len(self.trials)} evaluation trials from {os.path.basename(self.trial_file_path)}.")

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        trial = self.trials[idx]
        face_img = load_image(trial["image_path"], self.image_transform)
        audio_wav = self.audio_processor.load_and_preprocess(trial["audio_path"])

        return {
            "pair_id": trial["pair_id"],
            "face_img": face_img,
            "audio_wav": audio_wav,
            "label": torch.tensor(trial["label"], dtype=torch.long),
        }
