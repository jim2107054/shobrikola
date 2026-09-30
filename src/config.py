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
    DATA_ROOT: str = "/kaggle/input/datasets/mdjahidhasanjim/mav-celeb-v4-dataset"
    TRAIN_DIR: str = "/kaggle/input/datasets/mdjahidhasanjim/mav-celeb-v4-dataset/train_set/train_set/train_set"
    DEV_DIR: str = "/kaggle/input/datasets/mdjahidhasanjim/mav-celeb-v4-dataset/dev_set/dev_set"
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
    # 8. Early Stopping & Anti-Overfitting Regularization
    # ---------------------------------------------------------
    EARLY_STOPPING_PATIENCE: int = 5          # Consecutive epochs without improvement before stopping
    EARLY_STOPPING_MIN_DELTA: float = 1e-4    # Minimum delta for improvement
    EARLY_STOPPING_MODE: str = "min"          # "min" for validation loss / EER
    EARLY_STOPPING_RESTORE_BEST: bool = True  # Restore best model weights upon early stop
    VAL_SPLIT_RATIO: float = 0.1             # 10% stratified validation split to monitor generalization
    LABEL_SMOOTHING: float = 0.05            # Label smoothing in CrossEntropy to prevent overconfidence

    # ---------------------------------------------------------
    # 9. Device Configuration
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
    def print_diagnostics(cls):
        """Prints available files and directories in /kaggle/input for easy debugging."""
        print("[Diagnostic] Inspecting /kaggle/input...")
        if not os.path.exists("/kaggle/input"):
            print("  [Notice] /kaggle/input does not exist (running outside Kaggle environment).")
            return

        try:
            input_contents = os.listdir("/kaggle/input")
            print(f"  [Found {len(input_contents)} item(s) in /kaggle/input]: {input_contents}")
            for item in input_contents:
                item_path = os.path.join("/kaggle/input", item)
                if os.path.isdir(item_path):
                    print(f"  --> Dataset folder: {item}/")
                    for root, dirs, files in os.walk(item_path):
                        depth = root[len(item_path):].count(os.sep)
                        if depth <= 3:
                            indent = "      " + "  " * depth
                            print(f"{indent}{os.path.basename(root)}/ (subdirs: {dirs[:5]}, files: {files[:5]})")
        except Exception as e:
            print(f"  [Diagnostic Error] Failed to scan /kaggle/input: {e}")

    @classmethod
    def get_train_dir(cls) -> str:
        """
        Dynamically detects the exact training directory across any Kaggle input dataset name:
        Strictly prioritizes candidate paths containing 'train' and excludes 'dev' or 'test' paths.
        """
        # 1. Check predefined candidates first (most specific to least specific)
        candidates = [
            # User-specific Kaggle dataset path (3-level nesting)
            "/kaggle/input/datasets/mdjahidhasanjim/mav-celeb-v4-dataset/train_set/train_set/train_set",
            "/kaggle/input/datasets/mdjahidhasanjim/mav-celeb-v4-dataset/train_set/train_set",
            "/kaggle/input/datasets/mdjahidhasanjim/mav-celeb-v4-dataset/train_set",
            # Standard Kaggle path (3-level nesting)
            "/kaggle/input/mav-celeb-v4-dataset/train_set/train_set/train_set",
            "/kaggle/input/mav-celeb-v4-dataset/train_set/train_set",
            "/kaggle/input/mav-celeb-v4-dataset/train_set",
            # Dynamically constructed from DATA_ROOT
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
                # Check for direct faces and voices subdirectories
                if os.path.exists(os.path.join(c, "faces")) and os.path.exists(os.path.join(c, "voices")):
                    return c
                # Check 1 level down
                nested = os.path.join(c, "train_set")
                if os.path.exists(nested) and os.path.exists(os.path.join(nested, "faces")) and os.path.exists(os.path.join(nested, "voices")):
                    return nested

        # 2. Dynamic walk across /kaggle/input (STRICT: must contain 'train', NOT 'dev' or 'test')
        if os.path.exists("/kaggle/input"):
            for root, dirs, _ in os.walk("/kaggle/input"):
                root_lower = root.replace("\\", "/").lower()
                if "dev" in root_lower or "test" in root_lower:
                    continue
                if "train" in root_lower and "faces" in dirs and "voices" in dirs:
                    return root

        # 3. Check workspace directories
        for search_base in [".", ".."]:
            if os.path.exists(search_base):
                for root, dirs, _ in os.walk(search_base):
                    root_lower = root.replace("\\", "/").lower()
                    if "dev" in root_lower or "test" in root_lower:
                        continue
                    if "faces" in dirs and "voices" in dirs:
                        return root

        # Fallback to candidate that exists on disk
        for c in candidates:
            if os.path.exists(c):
                return c
        return cls.TRAIN_DIR

    @classmethod
    def get_dev_dir(cls) -> str:
        """
        Dynamically detects the development directory across any Kaggle input dataset name:
        Strictly prioritizes candidate paths containing 'dev' and excludes 'train' paths.
        """
        candidates = [
            # User-specific Kaggle dataset path
            "/kaggle/input/datasets/mdjahidhasanjim/mav-celeb-v4-dataset/dev_set/dev_set",
            "/kaggle/input/datasets/mdjahidhasanjim/mav-celeb-v4-dataset/dev_set",
            # Standard Kaggle path
            "/kaggle/input/mav-celeb-v4-dataset/dev_set/dev_set",
            "/kaggle/input/mav-celeb-v4-dataset/dev_set",
            # Dynamically constructed from DATA_ROOT
            os.path.join(cls.DATA_ROOT, "dev_set", "dev_set"),
            os.path.join(cls.DATA_ROOT, "dev_set"),
            cls.DEV_DIR,
            "./dev_set/dev_set",
            "./dev_set",
        ]
        for c in candidates:
            if os.path.exists(c):
                if os.path.exists(os.path.join(c, "gender")) or os.path.exists(os.path.join(c, "no_gender")):
                    return c

        if os.path.exists("/kaggle/input"):
            for root, dirs, files in os.walk("/kaggle/input"):
                root_lower = root.replace("\\", "/").lower()
                if "train" in root_lower:
                    continue
                if "gender" in dirs and "no_gender" in dirs:
                    return root
                if "english_test.txt" in [f.lower() for f in files] or "bangla_test.txt" in [f.lower() for f in files]:
                    parent = os.path.dirname(root)
                    if os.path.exists(os.path.join(parent, "gender")) or os.path.exists(os.path.join(parent, "no_gender")):
                        return parent
                    return root

        for search_base in [".", ".."]:
            if os.path.exists(search_base):
                for root, dirs, _ in os.walk(search_base):
                    root_lower = root.replace("\\", "/").lower()
                    if "train" in root_lower:
                        continue
                    if "gender" in dirs and "no_gender" in dirs:
                        return root

        for c in candidates:
            if os.path.exists(c):
                return c
        return cls.DEV_DIR


config = Config()
