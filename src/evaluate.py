"""
FLAG 2027 Challenge: Face-Voice Association Across Languages and Gender
Evaluation, EER Calculation & CodaBench 4-Cell Submission Module (evaluate.py)
"""

import os
import sys

# Ensure local module directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import glob
import zipfile
import argparse
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from config import config
from dataset import FLAGDevDataset
from models import GenderAdversarialMultimodalModel


# ==============================================================================
# 1. EER & Evaluation Metric Computation (Distance-based)
# ==============================================================================

def compute_eer_and_auc(distance_scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    """
    Computes Equal Error Rate (EER %), Area Under Curve (AUC), and optimal distance threshold.
    In FLAG 2027, scores are Euclidean distances on normalized hypersphere:
    Lower distance = higher match probability (positive class 1).
    Higher distance = lower match probability (negative class 0).
    """
    valid_mask = (labels == 0) | (labels == 1)
    if not np.any(valid_mask):
        return -1.0, -1.0, 0.0

    valid_dists = distance_scores[valid_mask]
    valid_labels = labels[valid_mask]

    # Invert distances for ROC curve where pos_label=1 requires larger values
    similarity_proxy = -valid_dists
    fpr, tpr, neg_thresholds = roc_curve(valid_labels, similarity_proxy, pos_label=1)
    fnr = 1.0 - tpr

    # Find the threshold where FPR == FNR (EER)
    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        optimal_threshold = float(-interp1d(fpr, neg_thresholds)(eer))
    except Exception:
        eer_idx = np.nanargmin(np.absolute(fnr - fpr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
        optimal_threshold = float(-neg_thresholds[eer_idx])

    roc_auc = auc(fpr, tpr)
    return float(eer * 100.0), float(roc_auc * 100.0), optimal_threshold


# ==============================================================================
# 2. Inference Loop on Development Trials
# ==============================================================================

def run_evaluation(
    model: nn.Module,
    dev_dataset: FLAGDevDataset,
    output_txt_path: str,
    device: str
) -> Dict[str, Any]:
    """
    Runs model inference over dev_dataset, writes predictions strictly in
    CodaBench format: [pair_id] [score]
    where [score] is distance value (e.g., 1.162691).
    Computes EER / AUC if ground truth is present.
    """
    dataloader = DataLoader(
        dev_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    model.eval()
    pair_ids: List[str] = []
    distances: List[float] = []
    labels: List[int] = []

    print(f"[Eval] Running inference on {len(dev_dataset)} trial pairs...")
    with torch.no_grad():
        for batch in dataloader:
            b_pair_ids = batch["pair_id"]
            face_img = batch["face_img"].to(device, non_blocking=True)
            audio_wav = batch["audio_wav"].to(device, non_blocking=True)
            b_labels = batch["label"].cpu().numpy()

            # Forward pass through Multimodal model
            outputs = model(face_img=face_img, audio_wav=audio_wav)

            # Extract Euclidean distance on L2-normalized embedding hypersphere
            # Range: [0, 2.0], strictly matching FLAG baseline benchmark format
            batch_dists = outputs["distance"].cpu().numpy()

            pair_ids.extend(b_pair_ids)
            distances.extend(batch_dists.tolist())
            labels.extend(b_labels.tolist())

    dists_arr = np.array(distances, dtype=np.float64)
    labels_arr = np.array(labels, dtype=np.int32)

    # 1. Strictly write predictions in CodaBench format: [pair_id] [score]
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for pid, dist_val in zip(pair_ids, dists_arr):
            f.write(f"{pid} {dist_val:.6f}\n")

    print(f"[*] Saved submission file: {output_txt_path} ({len(pair_ids)} pairs)")

    # 2. Compute EER and AUC if ground-truth labels are present
    eer, roc_auc, threshold = compute_eer_and_auc(dists_arr, labels_arr)
    results = {
        "eer": eer,
        "auc": roc_auc,
        "threshold": threshold,
        "num_trials": len(pair_ids),
        "output_path": output_txt_path,
    }

    if eer >= 0.0:
        print(f"    --> EER: {eer:.2f}% | AUC: {roc_auc:.2f}% (Dist Threshold: {threshold:.4f})")
    else:
        print(f"    --> Unlabelled test split: Scores recorded for all {len(pair_ids)} pairs.")

    return results


# ==============================================================================
# 3. Discovery of 4-Cell Benchmark Trial Files (Gender & Language Split)
# ==============================================================================

def find_4cell_trial_files(dev_dir: Optional[str] = None) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Discovers trial files for the 4 protocol cells across Kaggle's nested directories:
    - gender/
        * English (heard):   English_test.txt
        * Bangla (unheard):  Bangla_test.txt
    - no_gender/
        * English (heard):   English_test.txt
        * Bangla (unheard):  Bangla_test.txt
    """
    dev_dir = dev_dir or config.get_dev_dir()
    cells: Dict[str, Dict[str, Optional[str]]] = {
        "gender": {"English_heard": None, "Bangla_unheard": None},
        "no_gender": {"English_heard": None, "Bangla_unheard": None},
    }

    if not os.path.exists(dev_dir):
        print(f"[Warning] Development directory {dev_dir} not found.")
        return cells

    all_txt = glob.glob(os.path.join(dev_dir, "**", "*.txt"), recursive=True) + \
              glob.glob(os.path.join(dev_dir, "*.txt"))
    all_txt = list(set(all_txt))

    for track in ["no_gender", "gender"]:
        for f in all_txt:
            f_norm = f.replace("\\", "/").lower()
            fname = os.path.basename(f).lower()

            # Ensure track matches correctly (avoid 'gender' matching 'no_gender')
            if track == "no_gender":
                if "no_gender" not in f_norm and "nogender" not in f_norm and "no-gender" not in f_norm:
                    continue
            else:
                # gender track
                if "no_gender" in f_norm or "nogender" in f_norm or "no-gender" in f_norm:
                    continue
                if "gender" not in f_norm:
                    continue

            # Identify English vs Bangla
            if "english" in fname and cells[track]["English_heard"] is None:
                cells[track]["English_heard"] = f
            elif "bangla" in fname and cells[track]["Bangla_unheard"] is None:
                cells[track]["Bangla_unheard"] = f

    return cells


# ==============================================================================
# 4. Packaging CodaBench 4-Cell Submission ZIP
# ==============================================================================

def package_submission_zip(
    submission_dir: str = config.SUBMISSION_DIR,
    zip_path: str = config.SUBMISSION_ZIP_PATH
):
    """
    Packs output prediction files strictly matching CodaBench FLAG 2027 requirements:
    submission.zip
    ├── gender/
    │   ├── sub_score_v4_Bangla_unheard.txt
    │   └── sub_score_v4_English_heard.txt
    └── no_gender/
        ├── sub_score_v4_Bangla_unheard.txt
        └── sub_score_v4_English_heard.txt
    """
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    added_count = 0

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for track in ["gender", "no_gender"]:
            for lang_tag in ["English_heard", "Bangla_unheard"]:
                fname = f"sub_score_v4_{lang_tag}.txt"
                fpath = os.path.join(submission_dir, track, fname)
                if os.path.exists(fpath):
                    arcname = f"{track}/{fname}"
                    zipf.write(fpath, arcname=arcname)
                    print(f"[Zip] Added {arcname} ({os.path.getsize(fpath)} bytes)")
                    added_count += 1
                else:
                    print(f"[Notice] File not found to zip: {fpath}")

    print(f"[*] Successfully packaged CodaBench ZIP ({added_count} files): {zip_path}")


# ==============================================================================
# 5. Main Evaluation Entrypoint
# ==============================================================================

def main(model_path: Optional[str] = None, dev_dir: Optional[str] = None):
    parser = argparse.ArgumentParser(description="Evaluate FLAG 2027 Model on Dev Set & Generate CodaBench Submission")
    parser.add_argument("--model_path", type=str, default=config.BEST_MODEL_PATH,
                        help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--dev_dir", type=str, default=None,
                        help="Root directory of dev_set (auto-detected if None)")
    args, _ = parser.parse_known_args()
    if model_path is not None:
        args.model_path = model_path
    if dev_dir is not None:
        args.dev_dir = dev_dir

    print("=" * 80)
    print(" FLAG 2027 Challenge: Development Set Evaluation & CodaBench Submission")
    print(" Winning Solution: Gender-Adversarial Contrastive Multimodal Architecture")
    print("=" * 80)

    config.setup_directories()
    device = config.DEVICE
    print(f"[Device] Using compute device: {device}")

    # 1. Locate and Load Model Checkpoint
    model_path = args.model_path
    if not os.path.exists(model_path):
        if os.path.exists(config.LAST_CHECKPOINT_PATH):
            model_path = config.LAST_CHECKPOINT_PATH
            print(f"[Notice] Best model not found. Using last checkpoint: {model_path}")
        else:
            input_models = glob.glob("/kaggle/input/**/best_model.pth", recursive=True) + \
                           glob.glob("/kaggle/input/**/last_checkpoint.pth", recursive=True)
            if input_models:
                model_path = input_models[0]
                print(f"[Notice] Found model in attached dataset: {model_path}")
            else:
                raise FileNotFoundError(
                    f"No trained checkpoint found at {model_path}, {config.LAST_CHECKPOINT_PATH}, or in /kaggle/input/"
                )

    print(f"[Model] Loading model weights from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    speaker_map = checkpoint.get("speaker_to_id", {})
    num_speakers = len(speaker_map) if speaker_map else config.NUM_SPEAKERS

    model = GenderAdversarialMultimodalModel(
        num_speakers=num_speakers,
        num_genders=config.NUM_GENDERS,
        embedding_dim=config.EMBEDDING_DIM
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[Model] Architecture loaded successfully ({num_speakers} speakers in training).")

    # 2. Discover 4-Cell Trial Files
    resolved_dev_dir = args.dev_dir or config.get_dev_dir()
    print(f"[Data] Resolving dev_set trials from: {resolved_dev_dir}")
    cell_trials = find_4cell_trial_files(resolved_dev_dir)

    # Submission target mapping
    cell_outputs = {
        ("gender", "English_heard"): config.SUBMISSION_GENDER_ENGLISH,
        ("gender", "Bangla_unheard"): config.SUBMISSION_GENDER_BANGLA,
        ("no_gender", "English_heard"): config.SUBMISSION_NO_GENDER_ENGLISH,
        ("no_gender", "Bangla_unheard"): config.SUBMISSION_NO_GENDER_BANGLA,
    }

    eval_summaries = {}

    # 3. Evaluate each of the 4 Cells
    for track in ["gender", "no_gender"]:
        for lang_tag in ["English_heard", "Bangla_unheard"]:
            cell_name = f"{track}/{lang_tag}"
            trial_file = cell_trials[track][lang_tag]
            out_path = cell_outputs[(track, lang_tag)]

            print(f"\n--- Evaluating Cell: {cell_name} ---")
            if trial_file and os.path.exists(trial_file):
                print(f"[Trial File] {trial_file}")
                dataset = FLAGDevDataset(trial_file_path=trial_file, dev_dir=resolved_dev_dir)
                res = run_evaluation(
                    model=model,
                    dev_dataset=dataset,
                    output_txt_path=out_path,
                    device=device
                )
                eval_summaries[cell_name] = res
            else:
                print(f"[Warning] Trial file for {cell_name} not found in {resolved_dev_dir}.")

    # 4. Package CodaBench ZIP Archive
    print("\n--- Packaging CodaBench Submission ZIP ---")
    package_submission_zip(
        submission_dir=config.SUBMISSION_DIR,
        zip_path=config.SUBMISSION_ZIP_PATH
    )

    # 5. Display Comprehensive Challenge Evaluation Summary
    print("\n" + "=" * 80)
    print(" EVALUATION SUMMARY (FLAG 2027 ICASSP Challenge Benchmark)")
    print("=" * 80)
    print(f"{'Condition Cell':<32} | {'Trials':<8} | {'EER (%)':<10} | {'AUC (%)':<10}")
    print("-" * 80)

    total_eer = []
    for cell_name, metrics in eval_summaries.items():
        eer_str = f"{metrics['eer']:.2f}%" if metrics['eer'] >= 0 else "N/A"
        auc_str = f"{metrics['auc']:.2f}%" if metrics['auc'] >= 0 else "N/A"
        if metrics['eer'] >= 0:
            total_eer.append(metrics['eer'])
        print(f"{cell_name:<32} | {metrics['num_trials']:<8} | {eer_str:<10} | {auc_str:<10}")

    if total_eer:
        mean_eer = sum(total_eer) / len(total_eer)
        print("-" * 80)
        print(f"{'Overall Mean EER (Official Metric)':<32} | {'-':<8} | {mean_eer:.2f}%     | {'-':<10}")
    print("=" * 80)
    print(f"[*] CodaBench Submission archive ready: {config.SUBMISSION_ZIP_PATH}")


if __name__ == "__main__":
    main()
