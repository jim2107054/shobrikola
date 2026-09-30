"""
FLAG 2027 Challenge: Face-Voice Association Across Languages and Gender
Loss Formulation Module (loss.py)
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config


# ==============================================================================
# 1. Supervised Contrastive Loss (Cross-Modal SupCon)
# ==============================================================================

class CrossModalSupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020) adapted for
    cross-modal biometric alignment (Face-to-Voice and Voice-to-Face).
    
    Pulls face and voice representations of the same speaker identity together,
    while pushing apart representations from different speaker identities.
    Also incorporates same-gender hard negative voices into the contrastive pool.
    """
    def __init__(self, temperature: float = config.SUPCON_TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        face_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        labels: torch.Tensor,
        neg_audio_emb: Optional[torch.Tensor] = None,
        neg_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            face_emb: (Batch, Dim) - L2 normalized face embeddings
            audio_emb: (Batch, Dim) - L2 normalized positive audio embeddings
            labels: (Batch,) - Ground truth speaker identity indices
            neg_audio_emb: Optional (Batch, Dim) - Hard negative audio embeddings (same gender)
            neg_labels: Optional (Batch,) - Identity labels for hard negative audios

        Returns:
            Scalar contrastive loss
        """
        batch_size = face_emb.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=face_emb.device, requires_grad=True)

        # If hard negatives provided, concatenate into audio pool
        if neg_audio_emb is not None and neg_labels is not None:
            all_audios = torch.cat([audio_emb, neg_audio_emb], dim=0)   # (2 * B, Dim)
            all_audio_labels = torch.cat([labels, neg_labels], dim=0)   # (2 * B,)
        else:
            all_audios = audio_emb
            all_audio_labels = labels

        # Similarity matrix: (Batch, Num_Audios)
        sim_f2a = torch.matmul(face_emb, all_audios.T) / self.temperature

        # Positive mask: (Batch, Num_Audios) where labels match
        labels_row = labels.contiguous().view(-1, 1)
        labels_col = all_audio_labels.contiguous().view(1, -1)
        pos_mask = torch.eq(labels_row, labels_col).float()

        # Numerator & Denominator for log-sum-exp
        # For numerical stability, subtract max
        max_logits, _ = torch.max(sim_f2a, dim=1, keepdim=True)
        logits_f2a = sim_f2a - max_logits.detach()

        exp_logits = torch.exp(logits_f2a)
        log_prob_f2a = logits_f2a - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Mean of log-likelihood over positive pairs
        pos_counts = pos_mask.sum(dim=1)
        # Avoid division by zero
        pos_counts = torch.clamp(pos_counts, min=1.0)
        loss_f2a = - (pos_mask * log_prob_f2a).sum(dim=1) / pos_counts
        loss_f2a = loss_f2a.mean()

        # Symmetric Voice-to-Face contrastive loss (over original positive audio)
        sim_a2f = torch.matmul(audio_emb, face_emb.T) / self.temperature
        max_logits_a, _ = torch.max(sim_a2f, dim=1, keepdim=True)
        logits_a2f = sim_a2f - max_logits_a.detach()
        exp_logits_a = torch.exp(logits_a2f)
        log_prob_a2f = logits_a2f - torch.log(exp_logits_a.sum(dim=1, keepdim=True) + 1e-8)

        pos_mask_a2f = torch.eq(labels_row, labels_row.T).float()
        pos_counts_a = torch.clamp(pos_mask_a2f.sum(dim=1), min=1.0)
        loss_a2f = - (pos_mask_a2f * log_prob_a2f).sum(dim=1) / pos_counts_a
        loss_a2f = loss_a2f.mean()

        total_supcon_loss = 0.5 * (loss_f2a + loss_a2f)
        return total_supcon_loss


# ==============================================================================
# 2. Combined Winning Loss Function
# ==============================================================================

class CombinedFLAGLoss(nn.Module):
    """
    Formulation:
    Total Loss = lambda_id * L_id (CrossEntropy)
               + lambda_gender * L_gender (Adversarial CrossEntropy via GRL)
               + lambda_supcon * L_supcon (Supervised Contrastive)
    """
    def __init__(
        self,
        lambda_id: float = config.LAMBDA_ID,
        lambda_gender: float = config.LAMBDA_GENDER,
        lambda_supcon: float = config.LAMBDA_SUPCON,
        supcon_temperature: float = config.SUPCON_TEMPERATURE
    ):
        super().__init__()
        self.lambda_id = lambda_id
        self.lambda_gender = lambda_gender
        self.lambda_supcon = lambda_supcon

        label_smoothing = getattr(config, "LABEL_SMOOTHING", 0.0)
        self.id_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.gender_criterion = nn.CrossEntropyLoss()
        self.supcon_criterion = CrossModalSupConLoss(temperature=supcon_temperature)

    def forward(
        self,
        model_outputs: Dict[str, torch.Tensor],
        speaker_labels: torch.Tensor,
        gender_labels: torch.Tensor,
        neg_audio_emb: Optional[torch.Tensor] = None,
        neg_speaker_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the complete multi-task debiased loss.

        Args:
            model_outputs: Dictionary returned by GenderAdversarialMultimodalModel forward pass
            speaker_labels: (Batch,) Speaker identity indices
            gender_labels: (Batch,) Gender indices (0: Male, 1: Female)
            neg_audio_emb: Optional hard negative audio embeddings
            neg_speaker_labels: Optional hard negative speaker identity labels

        Returns:
            total_loss: Scalar torch.Tensor for backward pass
            metrics: Dictionary of individual loss components and accuracies
        """
        id_logits = model_outputs["id_logits"]
        gender_logits = model_outputs["gender_logits"]
        face_emb = model_outputs["face_emb"]
        audio_emb = model_outputs["audio_emb"]

        # 1. Main Task: Identity Classification Loss
        loss_id = self.id_criterion(id_logits, speaker_labels)

        # 2. Adversarial Gender Loss (Gradients are automatically reversed inside model via GRL)
        loss_gender = self.gender_criterion(gender_logits, gender_labels)

        # 3. Supervised Contrastive Loss with hard negative voices
        loss_supcon = self.supcon_criterion(
            face_emb=face_emb,
            audio_emb=audio_emb,
            labels=speaker_labels,
            neg_audio_emb=neg_audio_emb,
            neg_labels=neg_speaker_labels,
        )

        # 4. Total Multi-Objective Loss
        total_loss = (
            self.lambda_id * loss_id +
            self.lambda_gender * loss_gender +
            self.lambda_supcon * loss_supcon
        )

        # Calculate accuracies for monitoring
        with torch.no_grad():
            id_preds = torch.argmax(id_logits, dim=-1)
            id_acc = (id_preds == speaker_labels).float().mean().item()

            gender_preds = torch.argmax(gender_logits, dim=-1)
            gender_acc = (gender_preds == gender_labels).float().mean().item()

        metrics = {
            "total_loss": total_loss.item(),
            "loss_id": loss_id.item(),
            "loss_gender": loss_gender.item(),
            "loss_supcon": loss_supcon.item(),
            "id_acc": id_acc,
            "gender_acc": gender_acc,
        }

        return total_loss, metrics
