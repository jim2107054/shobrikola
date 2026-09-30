# FLAG 2027 Challenge: Face-Voice Association Across Languages and Gender
## Winning Solution: Gender-Adversarial Contrastive Multimodal Architecture

This repository contains the complete, production-ready PyTorch codebase for the **FLAG 2027 Grand Challenge** (ICASSP 2027) on CodaBench using the **MAV-Celeb v4** dataset (English–Bengali split).

---

### Architecture Overview

```
                      Raw Audio (.wav)                Raw Face (.jpg)
                             │                               │
                      [Audio Processor]               [Face Transform]
                             │                               │
                       16kHz Waveform                   112x112 RGB
                             │                               │
                      ┌──────┴──────┐                 ┌──────┴──────┐
                      │    WavLM    │                 │  ResNet50   │
                      │ SSL Encoder │                 │   Backbone  │
                      └──────┬──────┘                 └──────┬──────┘
                             │                               │
                       Attentive Pool                   Projector
                             │                               │
                             ▼                               ▼
                      Audio Embedding                 Face Embedding
                      (512-dim, L2-norm)             (512-dim, L2-norm)
                             │                               │
                             └───────────────┬───────────────┘
                                             │
                                  ┌──────────┴──────────┐
                                  │ Bidirectional Cross-│
                                  │ Modal Attention     │
                                  │ Fusion (MHA + FFN)  │
                                  └──────────┬──────────┘
                                             │
                                      Fused Embedding
                                    (512-dim, L2-norm)
                                             │
                   ┌─────────────────────────┼─────────────────────────┐
                   │                         │                         │
                   ▼                         ▼                         ▼
            [Main Task Head]        [Gradient Reversal (GRL)]   [Pair Verification]
                   │                         │                         │
         Speaker ID Classifier       Gender Classifier         Cosine Similarity +
          (CrossEntropy Loss)       (CrossEntropy Loss)         Non-linear Logit
```

1. **Audio Encoder (Language-Agnostic)**: Pre-trained Self-Supervised Learning (SSL) model (`microsoft/wavlm-base`) with temporal attentive pooling and projection to a 512-dim L2-normalized embedding space.
2. **Face Encoder (Quality-Robust)**: Modified ResNet50 with custom projection head to a 512-dim L2-normalized embedding space.
3. **Cross-Modal Attention Fusion**: Bidirectional cross-attention (Face queries Audio, Audio queries Face) with residual LayerNorm and feed-forward networks to capture fine-grained audiovisual biometric associations.
4. **Adversarial Gender Debiasing (GRL)**: Gradient Reversal Layer with a dynamic $\lambda(p)$ schedule connected to a gender classification head. Forcing high gender error strips demographic gender shortcuts from the fused representations.
5. **Supervised Contrastive Loss (SupCon)**: Explicitly pulls same-identity face-voice pairs together and pushes apart different identities.
6. **Hard Negative Mining**: `SameGenderBatchSampler` and per-sample hard negative voice mining ensures the negative pairs share the same gender, preventing the model from exploiting gender shortcuts.

---

### Kaggle Dataset Structure (`mav-celeb-v4-dataset`)

The codebase automatically resolves and accommodates the Kaggle folder layout:

```text
mav-celeb-v4-dataset/
├── dev_set/
│   └── dev_set/
│       ├── gender/
│       │   ├── Bangla_test.txt            # Unheard cross-lingual test pairs
│       │   ├── English_test.txt           # Heard language test pairs
│       │   ├── Bangla_test/
│       │   │   ├── faces/
│       │   │   └── voices/
│       │   ├── English_test/
│       │   │   ├── faces/
│       │   │   └── voices/
│       │   └── features/
│       └── no_gender/
│           ├── Bangla_test.txt
│           ├── English_test.txt
│           ├── Bangla_test/
│           └── English_test/
└── train_set/
    └── train_set/
        ├── features/
        │   ├── faces/train_English_faces.csv
        │   └── voices/train_English_voices.csv
        └── train_set/
            ├── faces/
            │   └── English/
            │       ├── id001/*.jpg
            │       ├── id002/*.jpg
            │       └── ...
            └── voices/
                └── English/
                    ├── id001/*.wav
                    ├── id002/*.wav
                    └── ...
```

---

### Kaggle Execution Guide

In your Kaggle Notebook (with GPU T4x2 or P100 enabled):

#### Step 1: Install Dependencies
```bash
!pip install -q -r /kaggle/working/shobrikola/requirements.txt
```

#### Step 2: Run Training (Resumable)
```bash
!python /kaggle/working/shobrikola/src/train.py
```
*Note: Checkpoints (`best_model.pth` and `last_checkpoint.pth`) are saved automatically to `/kaggle/working/`. If a 12-hour session timeout occurs, re-running `train.py` will automatically pick up from the exact last saved epoch.*

#### Step 3: Run Evaluation & Generate CodaBench Submission
```bash
!python /kaggle/working/shobrikola/src/evaluate.py
```
This automatically evaluates all **4 challenge protocol cells** and generates:
- `/kaggle/working/submission/gender/sub_score_v4_English_heard.txt`
- `/kaggle/working/submission/gender/sub_score_v4_Bangla_unheard.txt`
- `/kaggle/working/submission/no_gender/sub_score_v4_English_heard.txt`
- `/kaggle/working/submission/no_gender/sub_score_v4_Bangla_unheard.txt`
- `/kaggle/working/submission.zip` (Ready for direct upload to CodaBench!)

**Official CodaBench Zip Hierarchy**:
```text
submission.zip
├── gender/
│   ├── sub_score_v4_Bangla_unheard.txt
│   └── sub_score_v4_English_heard.txt
└── no_gender/
    ├── sub_score_v4_Bangla_unheard.txt
    └── sub_score_v4_English_heard.txt
```

**Metric & Line Format**:
Each file contains one line per trial formatted as `[pair_id] [distance]`:
```text
ljAnhn41 1.162691
neHzLCeC 1.235319
```
*Where lower distance corresponds to higher likelihood of matching identity.*
