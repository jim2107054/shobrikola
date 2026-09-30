"""
FLAG 2027 Challenge: Face-Voice Association Across Languages and Gender
Training Loop & Resumable Checkpointing Module (train.py)
"""

import os
import sys
import glob

# Ensure local module directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import math
import random
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from config import config
from dataset import FLAGTrainDataset, SameGenderBatchSampler
from models import GenderAdversarialMultimodalModel
from loss import CombinedFLAGLoss


def set_seed(seed: int = config.SEED):
    """Ensures deterministic reproducibility across PyTorch, NumPy, and Python."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def compute_grl_lambda(epoch: int, total_epochs: int, alpha: float = config.GRL_ALPHA) -> float:
    """
    Dynamic schedule for Gradient Reversal Layer:
    p = epoch / total_epochs
    lambda(p) = 2 / (1 + exp(-alpha * p)) - 1
    Starts near 0 and smoothly scales up to 1.0.
    """
    p = float(epoch) / float(max(1, total_epochs))
    return float(2.0 / (1.0 + math.exp(-alpha * p)) - 1.0)


def build_optimizer_groups(model: nn.Module) -> list:
    """
    Separates parameters into differential learning rate groups:
    - Slower LR for pretrained SSL (WavLM) and Face backbones (prevents catastrophic forgetting)
    - Faster LR for cross-modal attention fusion and classification heads
    """
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "audio_encoder.wavlm" in name or "face_encoder.backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer_grouped_parameters = [
        {"params": backbone_params, "lr": config.LR_BACKBONE, "weight_decay": config.WEIGHT_DECAY},
        {"params": head_params, "lr": config.LR_HEADS, "weight_decay": config.WEIGHT_DECAY},
    ]
    return optimizer_grouped_parameters


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    checkpoint_path: str = config.LAST_CHECKPOINT_PATH,
    best_path: str = config.BEST_MODEL_PATH
):
    """Saves training checkpoint for resuming if Kaggle 12-hour session times out."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(state, checkpoint_path)
    if is_best:
        torch.save(state, best_path)
        print(f"[*] Saved new best model checkpoint to: {best_path}")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: str,
    epoch: int,
    total_epochs: int
) -> Dict[str, float]:
    """Runs one full training epoch with mixed precision, hard negatives, and GRL update."""
    model.train()

    # Update dynamic GRL alpha
    current_grl_lambda = compute_grl_lambda(epoch, total_epochs)
    model.set_grl_alpha(current_grl_lambda)

    running_total_loss = 0.0
    running_id_loss = 0.0
    running_gender_loss = 0.0
    running_supcon_loss = 0.0
    running_id_acc = 0.0
    running_gender_acc = 0.0
    num_batches = 0

    epoch_start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        face_img = batch["face_img"].to(device, non_blocking=True)
        audio_wav = batch["audio_wav"].to(device, non_blocking=True)
        neg_audio_wav = batch["neg_audio_wav"].to(device, non_blocking=True)
        speaker_idx = batch["speaker_idx"].to(device, non_blocking=True)
        neg_speaker_idx = batch["neg_speaker_idx"].to(device, non_blocking=True)
        gender = batch["gender"].to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(enabled=config.USE_AMP):
            # 1. Forward pass for Anchor Face and Positive Voice
            outputs = model(face_img=face_img, audio_wav=audio_wav)

            # 2. Extract embedding for hard negative voice (same gender)
            with torch.no_grad():
                neg_audio_emb = model.audio_encoder(neg_audio_wav)

            # 3. Compute Multi-Task Loss (Identity + Adversarial Gender + SupCon)
            total_loss, metrics = criterion(
                model_outputs=outputs,
                speaker_labels=speaker_idx,
                gender_labels=gender,
                neg_audio_emb=neg_audio_emb,
                neg_speaker_labels=neg_speaker_idx,
            )

        # 4. Backward Pass with Gradient Scaling
        scaler.scale(total_loss).backward()

        # Unscale before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP_NORM)

        scaler.step(optimizer)
        scaler.update()

        # Accumulate metrics
        running_total_loss += metrics["total_loss"]
        running_id_loss += metrics["loss_id"]
        running_gender_loss += metrics["loss_gender"]
        running_supcon_loss += metrics["loss_supcon"]
        running_id_acc += metrics["id_acc"]
        running_gender_acc += metrics["gender_acc"]
        num_batches += 1

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
            sys.stdout.write(
                f"\r[Epoch {epoch:02d}/{total_epochs:02d} | Batch {batch_idx+1:03d}/{len(dataloader):03d}] "
                f"Loss: {running_total_loss/num_batches:.4f} "
                f"(ID: {running_id_loss/num_batches:.4f}, Gen: {running_gender_loss/num_batches:.4f}, SC: {running_supcon_loss/num_batches:.4f}) | "
                f"ID Acc: {running_id_acc/num_batches*100:.1f}% | "
                f"Gen Acc: {running_gender_acc/num_batches*100:.1f}% | "
                f"GRL λ: {current_grl_lambda:.3f}"
            )
            sys.stdout.flush()

    elapsed = time.time() - epoch_start_time
    print(f" -- Epoch Time: {elapsed:.1f}s")

    avg_metrics = {
        "total_loss": running_total_loss / max(1, num_batches),
        "loss_id": running_id_loss / max(1, num_batches),
        "loss_gender": running_gender_loss / max(1, num_batches),
        "loss_supcon": running_supcon_loss / max(1, num_batches),
        "id_acc": running_id_acc / max(1, num_batches),
        "gender_acc": running_gender_acc / max(1, num_batches),
    }
    return avg_metrics


def main():
    print("=" * 80)
    print(" FLAG 2027 Challenge: Face-Voice Association Across Languages and Gender")
    print(" Winning Solution: Gender-Adversarial Contrastive Multimodal Architecture")
    print("=" * 80)

    # 1. Setup & Directories
    config.setup_directories()
    set_seed(config.SEED)
    device = config.DEVICE
    print(f"[Device] Using compute device: {device}")
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    # 2. Build Dataset & DataLoader
    resolved_train_dir = config.get_train_dir()
    print(f"[Data] Initializing Train Dataset from {resolved_train_dir}...")
    train_dataset = FLAGTrainDataset(train_dir=resolved_train_dir)

    num_speakers = len(train_dataset.speaker_to_id)
    if num_speakers == 0:
        print("[Warning] No speakers parsed from training directory. Defaulting to 100.")
        num_speakers = config.NUM_SPEAKERS
    else:
        print(f"[Data] Detected {num_speakers} unique speakers.")

    # Sampler enforcing same-gender negative sampling
    batch_sampler = SameGenderBatchSampler(train_dataset, batch_size=config.BATCH_SIZE)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    # 3. Instantiate Architecture & Criterion
    print("[Model] Initializing Multimodal Architecture (WavLM + ResNet50 + CrossAttention + GRL)...")
    model = GenderAdversarialMultimodalModel(
        num_speakers=num_speakers,
        num_genders=config.NUM_GENDERS,
        embedding_dim=config.EMBEDDING_DIM
    ).to(device)

    criterion = CombinedFLAGLoss(
        lambda_id=config.LAMBDA_ID,
        lambda_gender=config.LAMBDA_GENDER,
        lambda_supcon=config.LAMBDA_SUPCON,
        supcon_temperature=config.SUPCON_TEMPERATURE
    ).to(device)

    # 4. Optimizer, Scheduler, and GradScaler
    optimizer_groups = build_optimizer_groups(model)
    optimizer = torch.optim.AdamW(optimizer_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6
    )
    scaler = GradScaler(enabled=config.USE_AMP)

    start_epoch = 1
    best_loss = float("inf")

    # 5. Automatic Resuming from Kaggle Session Timeout or Attached Previous Versions
    resume_checkpoint = None
    if os.path.exists(config.LAST_CHECKPOINT_PATH):
        resume_checkpoint = config.LAST_CHECKPOINT_PATH
    else:
        # Search if user attached previous notebook outputs via '+ Add Data'
        input_checkpoints = glob.glob("/kaggle/input/**/last_checkpoint.pth", recursive=True) + \
                            glob.glob("/kaggle/input/**/best_model.pth", recursive=True)
        if input_checkpoints:
            resume_checkpoint = input_checkpoints[0]

    if resume_checkpoint:
        print(f"[Resume] Found checkpoint at {resume_checkpoint}. Resuming training...")
        try:
            checkpoint = torch.load(resume_checkpoint, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_loss = checkpoint.get("best_loss", float("inf"))
            print(f"[Resume] Successfully resumed from epoch {start_epoch} (Best Loss: {best_loss:.4f}).")
        except Exception as e:
            print(f"[Resume] Error loading checkpoint: {e}. Starting fresh.")
    else:
        print("[Train] No existing checkpoint found. Starting fresh training session.")

    # 6. Training Loop
    print(f"[Train] Beginning training for {config.NUM_EPOCHS} epochs...")
    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            total_epochs=config.NUM_EPOCHS
        )

        scheduler.step()

        # Check if new best model
        is_best = metrics["total_loss"] < best_loss
        if is_best:
            best_loss = metrics["total_loss"]

        # Save checkpoint after every epoch
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_loss": best_loss,
            "metrics": metrics,
            "speaker_to_id": train_dataset.speaker_to_id,
        }
        save_checkpoint(
            state=checkpoint_data,
            is_best=is_best,
            checkpoint_path=config.LAST_CHECKPOINT_PATH,
            best_path=config.BEST_MODEL_PATH
        )

        print(
            f"[Epoch {epoch:02d} Summary] Total Loss: {metrics['total_loss']:.4f} | "
            f"ID Acc: {metrics['id_acc']*100:.2f}% | "
            f"Gender Acc: {metrics['gender_acc']*100:.2f}% | "
            f"Best Loss: {best_loss:.4f}"
        )
        print("-" * 80)

    print("[*] Training completed successfully!")
    print(f"[*] Best model weights saved at: {config.BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
