"""
FLAG 2027 Challenge: Face-Voice Association Across Languages and Gender
Configuration Module (config.py)
"""

import os
from dataclasses import dataclass
import torch

@dataclass
class Config:
    # ---------------------------------------------------------
    # 1. Kaggle Hardcoded Paths
    # ---------------------------------------------------------
    DATA_ROOT: str = "/kaggle/input/mav-celeb-v4-dataset"
    TRAIN_DIR: str = "/kaggle/input/mav-celeb-v4-dataset/train_set"
    DEV_DIR: str = "/kaggle/input/mav-celeb-v4-dataset/dev_set"
    OUTPUT_DIR: str = "/kaggle/working"
    
    # Checkpoints & Submission outputs
    BEST_MODEL_PATH: str = "/kaggle/working/best_model.pth"
    LAST_CHECKPOINT_PATH: str = "/kaggle/working/last_checkpoint.pth"
    SUBMISSION_DIR: str = "/kaggle/working/submission"
    SUBMISSION_ZIP_PATH: str = "/kaggle/working/submission.zip"
    
    # 4-Cell CodaBench Submission Paths
    # gender track
    SUBMISSION_GENDER_ENGLISH: str = "/kaggle/working/submission/gender/sub_score_v4_English_heard.txt"
    SUBMISSION_GENDER_BANGLA: str = "/kaggle/working/submission/gender/sub_score_v4_Bangla_unheard.txt"
    # no_gender track
    SUBMISSION_NO_GENDER_ENGLISH: str = "/kaggle/working/submission/no_gender/sub_score_v4_English_heard.txt"
    SUBMISSION_NO_GENDER_BANGLA: str = "/kaggle/working/submission/no_gender/sub_score_v4_Bangla_unheard.txt"

    # ---------------------------------------------------------
    # 2. Audio Processing Parameters (WavLM SSL)
    # ---------------------------------------------------------
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_DURATION_SEC: float = 3.0
    AUDIO_NUM_SAMPLES: int = int(AUDIO_SAMPLE_RATE * AUDIO_DURATION_SEC)  # 48,000 samples
    WAVLM_MODEL_NAME: str = "microsoft/wavlm-base"  # or microsoft/wavlm-base-plus
    FREEZE_WAVLM_CNN: bool = True  # Keep low-level SSL feature extractor frozen for stability

    # ---------------------------------------------------------
    # 3. Image Processing Parameters (Face Encoder)
    # ---------------------------------------------------------
    IMAGE_SIZE: int = 112  # Standard face recognition resolution (112x112)
    FACE_ENCODER_BACKBONE: str = "resnet50"
    PRETRAINED_FACE: bool = True

    # ---------------------------------------------------------
    # 4. Model Architecture & Fusion
    # ---------------------------------------------------------
    EMBEDDING_DIM: int = 512
    FUSION_NUM_HEADS: int = 8
    FUSION_DROPOUT: float = 0.1
    NUM_GENDERS: int = 2  # 0: Male, 1: Female
    NUM_SPEAKERS: int = 100  # Default MAV-Celeb v4 English-Bengali split (auto-updated if needed)

    # ---------------------------------------------------------
    # 5. Gradient Reversal Layer (GRL) Parameters
    # ---------------------------------------------------------
    GRL_ALPHA: float = 10.0
    GRL_MAX_LAMBDA: float = 1.0

    # ---------------------------------------------------------
    # 6. Loss Function Weights
    # ---------------------------------------------------------
    LAMBDA_ID: float = 1.0       # Weight for Speaker Identity CrossEntropy Loss
    LAMBDA_GENDER: float = 0.5   # Weight for Adversarial Gender Loss
    LAMBDA_SUPCON: float = 0.5   # Weight for Supervised Contrastive Loss
    SUPCON_TEMPERATURE: float = 0.07

    # ---------------------------------------------------------
    # 7. Training Hyperparameters
    # ---------------------------------------------------------
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 50
    LR_HEADS: float = 1e-4        # Learning rate for projection, fusion, classifiers
    LR_BACKBONE: float = 1e-5     # Slower learning rate for pretrained WavLM & Face backbones
    WEIGHT_DECAY: float = 1e-4
    WARMUP_EPOCHS: int = 2
    GRAD_CLIP_NORM: float = 5.0
    USE_AMP: bool = True          # Automatic Mixed Precision for T4/P100 GPUs
    NUM_WORKERS: int = 2          # Safe worker count on Kaggle shared memory
    SEED: int = 42

    # ---------------------------------------------------------
    # 8. Device Configuration
    # ---------------------------------------------------------
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup_directories(cls):
        """Ensure working directory and submission directories exist."""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        os.makedirs(os.path.join(cls.SUBMISSION_DIR, "gender"), exist_ok=True)
        os.makedirs(os.path.join(cls.SUBMISSION_DIR, "no_gender"), exist_ok=True)

    @classmethod
    def get_train_dir(cls) -> str:
        """
        Dynamically detects the exact training directory across nested Kaggle folder structures:
        e.g., /kaggle/input/mav-celeb-v4-dataset/train_set/train_set/train_set
        """
        candidates = [
            os.path.join(cls.DATA_ROOT, "train_set", "train_set", "train_set"),
            os.path.join(cls.DATA_ROOT, "train_set", "train_set"),
            os.path.join(cls.DATA_ROOT, "train_set"),
            cls.TRAIN_DIR,
            "./train_set/train_set/train_set",
            "./train_set/train_set",
            "./train_set",
        ]
        for c in candidates:
            if os.path.exists(c):
                # Prefer folder containing faces and voices
                if os.path.exists(os.path.join(c, "faces")) and os.path.exists(os.path.join(c, "voices")):
                    return c
                # Check nested folder
                nested = os.path.join(c, "train_set")
                if os.path.exists(nested) and os.path.exists(os.path.join(nested, "faces")):
                    return nested

        # Fallback to first existing candidate or default
        for c in candidates:
            if os.path.exists(c):
                return c
        return cls.TRAIN_DIR

    @classmethod
    def get_dev_dir(cls) -> str:
        """
        Dynamically detects the development directory across nested Kaggle folder structures:
        e.g., /kaggle/input/mav-celeb-v4-dataset/dev_set/dev_set
        """
        candidates = [
            os.path.join(cls.DATA_ROOT, "dev_set", "dev_set"),
            os.path.join(cls.DATA_ROOT, "dev_set"),
            cls.DEV_DIR,
            "./dev_set/dev_set",
            "./dev_set",
        ]
        for c in candidates:
            if os.path.exists(c):
                # Check if gender or no_gender folder exists here
                if os.path.exists(os.path.join(c, "gender")) or os.path.exists(os.path.join(c, "no_gender")):
                    return c

        for c in candidates:
            if os.path.exists(c):
                return c
        return cls.DEV_DIR


config = Config()
