"""
FLAG 2027 Challenge: Face-Voice Association Across Languages and Gender
Evaluation, EER Calculation & CodaBench Submission Module (evaluate.py)
"""

import os
import sys

# Ensure local module directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import glob
import zipfile
import argparse
from typing import List, Tuple, Dict, Optional

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
# 1. EER & Evaluation Metric Computation
# ==============================================================================

def compute_eer_and_auc(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    """
    Computes Equal Error Rate (EER %), Area Under Curve (AUC), and optimal threshold.
    EER is the point on the ROC curve where False Acceptance Rate (FAR) equals
    False Rejection Rate (FRR = 1 - TPR).
    """
    # Filter out unlabelled test trials (label == -1)
    valid_mask = (labels == 0) | (labels == 1)
    if not np.any(valid_mask):
        return -1.0, -1.0, 0.0

    valid_scores = scores[valid_mask]
    valid_labels = labels[valid_mask]

    fpr, tpr, thresholds = roc_curve(valid_labels, valid_scores, pos_label=1)
    fnr = 1.0 - tpr

    # Find the threshold where FPR == FNR (EER)
    try:
        # Interpolate to find precise crossing point
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        optimal_threshold = float(interp1d(fpr, thresholds)(eer))
    except Exception:
        # Fallback to closest point
        eer_idx = np.nanargmin(np.absolute(fnr - fpr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
        optimal_threshold = float(thresholds[eer_idx])

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
) -> Dict[str, float]:
    """
    Runs model inference over dev_dataset, writes predictions strictly in
    CodaBench format: [pair_id] [score]
    and computes EER / AUC if ground truth is present.
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
    scores: List[float] = []
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

            # Combined verification confidence score
            # Cosine similarity + learned non-linear verification score
            cos_sim = outputs["cosine_score"].cpu().numpy()
            verif_logit = outputs["verification_logit"].cpu().numpy()

            # Robust blended score (higher = stronger face-voice association match)
            # Both terms contribute to rank accuracy across languages
            batch_scores = cos_sim + 0.3 * verif_logit

            pair_ids.extend(b_pair_ids)
            scores.extend(batch_scores.tolist())
            labels.extend(b_labels.tolist())

    scores_arr = np.array(scores, dtype=np.float64)
    labels_arr = np.array(labels, dtype=np.int32)

    # 1. Strictly write predictions in CodaBench format: [pair_id] [score]
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for pid, score in zip(pair_ids, scores_arr):
            f.write(f"{pid} {score:.6f}\n")

    print(f"[*] CodaBench submission file saved to: {output_txt_path}")

    # 2. Compute EER and AUC if labels are available
    eer, roc_auc, threshold = compute_eer_and_auc(scores_arr, labels_arr)
    results = {
        "eer": eer,
        "auc": roc_auc,
        "threshold": threshold,
        "num_trials": len(pair_ids),
    }

    if eer >= 0.0:
        print(f"    --> EER: {eer:.2f}% | AUC: {roc_auc:.2f}% (Threshold: {threshold:.4f})")
    else:
        print(f"    --> Unlabelled test split: Scores recorded for all {len(pair_ids)} pairs.")

    return results


# ==============================================================================
# 3. Discovery of Trial Files (Gender & No-Gender Subsets)
# ==============================================================================

def find_trial_files(dev_dir: str = config.DEV_DIR) -> Dict[str, Optional[str]]:
    """
    Locates the trial files for both 'gender' and 'no_gender' subsets in dev_set.
    Handles various naming conventions used in MAV-Celeb / FLAG benchmarks.
    """
    trial_files: Dict[str, Optional[str]] = {"gender": None, "no_gender": None}

    if not os.path.exists(dev_dir):
        print(f"[Warning] Dev directory {dev_dir} not found.")
        return trial_files

    all_txt = glob.glob(os.path.join(dev_dir, "**", "*.txt"), recursive=True) + \
              glob.glob(os.path.join(dev_dir, "*.txt"))

    # Remove duplicates
    all_txt = list(set(all_txt))

    # Match no_gender first
    for f in all_txt:
        fname = os.path.basename(f).lower()
        if "no_gender" in fname or "nogender" in fname or "no-gender" in fname:
            trial_files["no_gender"] = f
            break

    # Match gender next
    for f in all_txt:
        fname = os.path.basename(f).lower()
        if f != trial_files["no_gender"] and ("gender" in fname or "constrained" in fname):
            trial_files["gender"] = f
            break

    # If standard names were not explicitly matched, try folder names or fallbacks
    if trial_files["gender"] is None or trial_files["no_gender"] is None:
        for f in all_txt:
            fname = os.path.basename(f).lower()
            if "gender" in f.lower() and "no" not in f.lower() and trial_files["gender"] is None:
                trial_files["gender"] = f
            elif ("no_gender" in f.lower() or "unconstrained" in f.lower() or "standard" in f.lower()) and trial_files["no_gender"] is None:
                trial_files["no_gender"] = f

    return trial_files


# ==============================================================================
# 4. Packaging CodaBench Submission ZIP
# ==============================================================================

def package_submission_zip(
    gender_txt: str = config.SUBMISSION_GENDER_PATH,
    no_gender_txt: str = config.SUBMISSION_NO_GENDER_PATH,
    zip_path: str = config.SUBMISSION_ZIP_PATH
):
    """Packs output prediction files into a submission zip ready for CodaBench upload."""
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        if os.path.exists(gender_txt):
            zipf.write(gender_txt, arcname=os.path.basename(gender_txt))
            print(f"[Zip] Added {os.path.basename(gender_txt)} to {zip_path}")
        if os.path.exists(no_gender_txt):
            zipf.write(no_gender_txt, arcname=os.path.basename(no_gender_txt))
            print(f"[Zip] Added {os.path.basename(no_gender_txt)} to {zip_path}")

    print(f"[*] Created CodaBench submission package: {zip_path}")


# ==============================================================================
# 5. Main Evaluation Entrypoint
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate FLAG 2027 Model on Dev Set")
    parser.add_argument("--model_path", type=str, default=config.BEST_MODEL_PATH,
                        help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--dev_dir", type=str, default=config.DEV_DIR,
                        help="Root directory of dev_set")
    parser.add_argument("--gender_trials", type=str, default=None,
                        help="Explicit path to gender-constrained trial txt file")
    parser.add_argument("--no_gender_trials", type=str, default=None,
                        help="Explicit path to standard/no-gender trial txt file")
    args = parser.parse_args()

    print("=" * 80)
    print(" FLAG 2027 Challenge: Development Set Evaluation & CodaBench Submission")
    print("=" * 80)

    device = config.DEVICE
    print(f"[Device] Using device: {device}")

    # 1. Load Model Checkpoint
    model_path = args.model_path
    if not os.path.exists(model_path):
        if os.path.exists(config.LAST_CHECKPOINT_PATH):
            model_path = config.LAST_CHECKPOINT_PATH
            print(f"[Notice] Best model not found in working directory. Using last checkpoint: {model_path}")
        else:
            # Check attached notebook inputs
            input_models = glob.glob("/kaggle/input/**/best_model.pth", recursive=True) + \
                           glob.glob("/kaggle/input/**/last_checkpoint.pth", recursive=True)
            if input_models:
                model_path = input_models[0]
                print(f"[Notice] Found model in attached dataset: {model_path}")
            else:
                raise FileNotFoundError(f"No trained checkpoint found at {model_path}, {config.LAST_CHECKPOINT_PATH}, or in /kaggle/input/")

    print(f"[Model] Loading model weights from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # Determine num_speakers from checkpoint
    speaker_map = checkpoint.get("speaker_to_id", {})
    num_speakers = len(speaker_map) if speaker_map else config.NUM_SPEAKERS

    model = GenderAdversarialMultimodalModel(
        num_speakers=num_speakers,
        num_genders=config.NUM_GENDERS,
        embedding_dim=config.EMBEDDING_DIM
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("[Model] Model successfully loaded and set to evaluation mode.")

    # 2. Locate Trial Files
    trial_files = find_trial_files(args.dev_dir)
    if args.gender_trials:
        trial_files["gender"] = args.gender_trials
    if args.no_gender_trials:
        trial_files["no_gender"] = args.no_gender_trials

    print(f"[Dev Files] Gender Subset Trial File:    {trial_files['gender']}")
    print(f"[Dev Files] No-Gender Subset Trial File: {trial_files['no_gender']}")

    eval_summaries = {}

    # 3. Evaluate Gender Subset
    if trial_files["gender"] and os.path.exists(trial_files["gender"]):
        print("\n--- Evaluating Gender-Constrained Subset ---")
        gender_dataset = FLAGDevDataset(trial_files["gender"], dev_dir=args.dev_dir)
        res_gender = run_evaluation(
            model=model,
            dev_dataset=gender_dataset,
            output_txt_path=config.SUBMISSION_GENDER_PATH,
            device=device
        )
        eval_summaries["Gender-Constrained"] = res_gender
    else:
        print(f"[Warning] Gender trial file not found in {args.dev_dir}.")

    # 4. Evaluate No-Gender Subset
    if trial_files["no_gender"] and os.path.exists(trial_files["no_gender"]):
        print("\n--- Evaluating Standard (No-Gender) Subset ---")
        nogender_dataset = FLAGDevDataset(trial_files["no_gender"], dev_dir=args.dev_dir)
        res_nogender = run_evaluation(
            model=model,
            dev_dataset=nogender_dataset,
            output_txt_path=config.SUBMISSION_NO_GENDER_PATH,
            device=device
        )
        eval_summaries["Standard (No-Gender)"] = res_nogender
    else:
        print(f"[Warning] No-Gender trial file not found in {args.dev_dir}.")

    # 5. Pack CodaBench ZIP file
    print("\n--- Packaging CodaBench Submission ---")
    package_submission_zip(
        gender_txt=config.SUBMISSION_GENDER_PATH,
        no_gender_txt=config.SUBMISSION_NO_GENDER_PATH,
        zip_path=config.SUBMISSION_ZIP_PATH
    )

    # 6. Print Overall Evaluation Summary Table
    print("\n" + "=" * 80)
    print(" EVALUATION SUMMARY (FLAG 2027 Challenge)")
    print("=" * 80)
    print(f"{'Subset':<28} | {'Trials':<8} | {'EER (%)':<10} | {'AUC (%)':<10}")
    print("-" * 80)
    total_eer = []
    for subset_name, metrics in eval_summaries.items():
        eer_str = f"{metrics['eer']:.2f}%" if metrics['eer'] >= 0 else "N/A"
        auc_str = f"{metrics['auc']:.2f}%" if metrics['auc'] >= 0 else "N/A"
        if metrics['eer'] >= 0:
            total_eer.append(metrics['eer'])
        print(f"{subset_name:<28} | {metrics['num_trials']:<8} | {eer_str:<10} | {auc_str:<10}")

    if total_eer:
        mean_eer = sum(total_eer) / len(total_eer)
        print("-" * 80)
        print(f"{'Mean EER (Challenge Metric)':<28} | {'-':<8} | {mean_eer:.2f}%     | {'-':<10}")
    print("=" * 80)


if __name__ == "__main__":
    main()
