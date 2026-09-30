"""
FLAG 2027 Challenge: Face-Voice Association Across Languages and Gender
Multimodal Architecture Module (models.py)
"""

import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from transformers import WavLMModel, WavLMConfig

from config import config


# ==============================================================================
# 1. Gradient Reversal Layer (GRL) for Adversarial Gender Debiasing
# ==============================================================================

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer:
    Forward pass: Identity mapping (y = x).
    Backward pass: Multiplies gradients by -alpha (-lambda).
    Forces the fused feature representations to become invariant to demographic gender.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        # Gradient reversed by -alpha
        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def set_alpha(self, alpha: float):
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)


# ==============================================================================
# 2. Audio Encoder (Language-Agnostic WavLM SSL Backbone)
# ==============================================================================

class TemporalAttentivePooling(nn.Module):
    """Self-attentive pooling over temporal frames from SSL output."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Time, Dim)
        weights = self.attention(x)  # (Batch, Time, 1)
        weights = F.softmax(weights, dim=1)
        pooled = torch.sum(x * weights, dim=1)  # (Batch, Dim)
        return pooled


class AudioEncoder(nn.Module):
    """
    Extracts language-invariant voice embeddings using pre-trained WavLM.
    Features are pooled and projected into a joint 512-dim biometric space.
    """
    def __init__(
        self,
        model_name: str = config.WAVLM_MODEL_NAME,
        embedding_dim: int = config.EMBEDDING_DIM,
        freeze_feature_extractor: bool = config.FREEZE_WAVLM_CNN
    ):
        super().__init__()
        # Load WavLM backbone
        self.wavlm = WavLMModel.from_pretrained(model_name)

        if freeze_feature_extractor and hasattr(self.wavlm, "feature_extractor"):
            self.wavlm.feature_extractor._freeze_parameters()

        ssl_hidden_dim = self.wavlm.config.hidden_size  # 768 for wavlm-base

        # Temporal pooling & projection head
        self.pooling = TemporalAttentivePooling(ssl_hidden_dim)
        self.projector = nn.Sequential(
            nn.Linear(ssl_hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.PReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav: Raw waveform tensor of shape (Batch, 48000)
        Returns:
            audio_emb: L2-normalized voice embedding (Batch, 512)
        """
        # WavLM expects (Batch, Samples)
        outputs = self.wavlm(input_values=wav)
        hidden_states = outputs.last_hidden_state  # (Batch, Time, 768)

        # Attentive pooling
        pooled = self.pooling(hidden_states)       # (Batch, 768)

        # Projection to common embedding space
        projected = self.projector(pooled)         # (Batch, 512)

        # L2 Normalization for contrastive and metric learning
        normalized_emb = F.normalize(projected, p=2, dim=-1)
        return normalized_emb


# ==============================================================================
# 3. Face Encoder (Robust ResNet50 Backbone)
# ==============================================================================

class FaceEncoder(nn.Module):
    """
    Extracts facial features from 112x112 images using ResNet50.
    Outputs a 512-dim L2-normalized facial embedding.
    """
    def __init__(
        self,
        embedding_dim: int = config.EMBEDDING_DIM,
        pretrained: bool = config.PRETRAINED_FACE
    ):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Remove original 1000-class classification head
        modules = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*modules)  # Outputs (Batch, 2048, 1, 1)

        # Facial projection head
        self.projector = nn.Sequential(
            nn.Linear(2048, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.PReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Tensor of shape (Batch, 3, 112, 112)
        Returns:
            face_emb: L2-normalized face embedding (Batch, 512)
        """
        feat = self.backbone(img)                   # (Batch, 2048, 1, 1)
        feat = torch.flatten(feat, 1)               # (Batch, 2048)
        projected = self.projector(feat)            # (Batch, 512)
        normalized_emb = F.normalize(projected, p=2, dim=-1)
        return normalized_emb


# ==============================================================================
# 4. Cross-Modal Fusion via Bidirectional Multihead Attention
# ==============================================================================

class CrossModalAttentionFusion(nn.Module):
    """
    Bidirectional Cross-Attention between Face and Voice modalities.
    Face queries Voice; Voice queries Face.
    Both cross-attended representations are fused through residual LayerNorm blocks.
    """
    def __init__(
        self,
        dim: int = config.EMBEDDING_DIM,
        num_heads: int = config.FUSION_NUM_HEADS,
        dropout: float = config.FUSION_DROPOUT
    ):
        super().__init__()
        self.dim = dim

        # Cross-Attention: Face queries Audio
        self.mha_face_to_audio = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_f1 = nn.LayerNorm(dim)

        # Cross-Attention: Audio queries Face
        self.mha_audio_to_face = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_a1 = nn.LayerNorm(dim)

        # Feed-Forward Networks
        self.ffn_face = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        self.norm_f2 = nn.LayerNorm(dim)

        self.ffn_audio = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        self.norm_a2 = nn.LayerNorm(dim)

        # Joint Cross-Modal Projection
        self.joint_projector = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.BatchNorm1d(dim),
            nn.PReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )

    def forward(self, face_emb: torch.Tensor, audio_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            face_emb: (Batch, Dim)
            audio_emb: (Batch, Dim)
        Returns:
            fused_emb: (Batch, Dim) L2-normalized fused representation
        """
        # Expand sequence dimension: (Batch, 1, Dim)
        f_seq = face_emb.unsqueeze(1)
        a_seq = audio_emb.unsqueeze(1)

        # 1. Face queries Audio
        f_attended, _ = self.mha_face_to_audio(query=f_seq, key=a_seq, value=a_seq)
        f_res = self.norm_f1(f_seq + f_attended)
        f_out = self.norm_f2(f_res + self.ffn_face(f_res)).squeeze(1)  # (Batch, Dim)

        # 2. Audio queries Face
        a_attended, _ = self.mha_audio_to_face(query=a_seq, key=f_seq, value=f_seq)
        a_res = self.norm_a1(a_seq + a_attended)
        a_out = self.norm_a2(a_res + self.ffn_audio(a_res)).squeeze(1)  # (Batch, Dim)

        # 3. Concatenate and project to joint representation
        combined = torch.cat([f_out, a_out], dim=-1)  # (Batch, Dim * 2)
        fused = self.joint_projector(combined)        # (Batch, Dim)

        return F.normalize(fused, p=2, dim=-1)


# ==============================================================================
# 5. Complete Gender-Adversarial Multimodal Architecture
# ==============================================================================

class GenderAdversarialMultimodalModel(nn.Module):
    """
    Winning Multimodal Architecture for FLAG 2027 Challenge:
    1. Language-Agnostic Audio Encoder (WavLM SSL)
    2. Quality-Robust Face Encoder (ResNet50)
    3. Bidirectional Cross-Modal Attention Fusion
    4. Identity Classifier (Main Task)
    5. Gradient Reversal Layer + Adversarial Gender Classifier (Gender Debiasing)
    6. Cross-Modal Verification Scoring Head
    """
    def __init__(
        self,
        num_speakers: int = config.NUM_SPEAKERS,
        num_genders: int = config.NUM_GENDERS,
        embedding_dim: int = config.EMBEDDING_DIM
    ):
        super().__init__()
        self.num_speakers = num_speakers
        self.num_genders = num_genders
        self.embedding_dim = embedding_dim

        # 1. Encoders
        self.audio_encoder = AudioEncoder(embedding_dim=embedding_dim)
        self.face_encoder = FaceEncoder(embedding_dim=embedding_dim)

        # 2. Cross-Modal Attention Fusion
        self.fusion = CrossModalAttentionFusion(dim=embedding_dim)

        # 3. Main Task: Speaker Identity Classifier
        self.id_classifier = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(embedding_dim, num_speakers)
        )

        # 4. Adversarial Gender Debiasing: GRL + Gender Classifier
        self.grl = GradientReversal(alpha=1.0)
        self.gender_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, num_genders)
        )

        # 5. Non-linear Verification Head for pair scoring
        self.verification_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def set_grl_alpha(self, alpha: float):
        """Update the gradient reversal scale dynamically during training."""
        self.grl.set_alpha(alpha)

    def forward(
        self,
        face_img: torch.Tensor,
        audio_wav: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the entire architecture.

        Args:
            face_img: Tensor of shape (Batch, 3, 112, 112)
            audio_wav: Tensor of shape (Batch, 48000)

        Returns:
            Dict containing:
                - face_emb: (Batch, 512)
                - audio_emb: (Batch, 512)
                - fused_emb: (Batch, 512)
                - id_logits: (Batch, num_speakers)
                - gender_logits: (Batch, 2)
                - verification_score: (Batch,)
                - cosine_score: (Batch,)
        """
        # 1. Unimodal feature extraction
        face_emb = self.face_encoder(face_img)        # (B, 512)
        audio_emb = self.audio_encoder(audio_wav)     # (B, 512)

        # 2. Cross-modal attention fusion
        fused_emb = self.fusion(face_emb, audio_emb)  # (B, 512)

        # 3. Main task: Speaker Identity Classification
        id_logits = self.id_classifier(fused_emb)     # (B, num_speakers)

        # 4. Adversarial Gender Classification via GRL
        reversed_fused = self.grl(fused_emb)          # (B, 512) with reversed gradients
        gender_logits = self.gender_classifier(reversed_fused)  # (B, 2)

        # 5. Cross-modal Verification Scores
        # Direct cosine similarity between normalized face and audio embeddings
        cosine_score = torch.sum(face_emb * audio_emb, dim=-1)  # (B,)
        verification_logit = self.verification_head(fused_emb).squeeze(-1)  # (B,)
        # Euclidean distance on L2-normalized hypersphere: sqrt(2 - 2*cos) in [0, 2]
        distance = torch.sqrt(torch.clamp(2.0 - 2.0 * cosine_score, min=0.0, max=4.0))

        return {
            "face_emb": face_emb,
            "audio_emb": audio_emb,
            "fused_emb": fused_emb,
            "id_logits": id_logits,
            "gender_logits": gender_logits,
            "verification_logit": verification_logit,
            "cosine_score": cosine_score,
            "distance": distance,
        }

    def compute_similarity(
        self,
        face_img: torch.Tensor,
        audio_wav: torch.Tensor
    ) -> torch.Tensor:
        """
        Inference-time fast score computation for verification trials.
        Computes cosine similarity between extracted face and voice embeddings.
        """
        face_emb = self.face_encoder(face_img)
        audio_emb = self.audio_encoder(audio_wav)
        cosine_sim = torch.sum(face_emb * audio_emb, dim=-1)
        return cosine_sim

    def compute_distance(
        self,
        face_img: torch.Tensor,
        audio_wav: torch.Tensor
    ) -> torch.Tensor:
        """
        Inference-time metric for CodaBench evaluation:
        Computes Euclidean distance between L2-normalized face and audio embeddings.
        Lower distance = higher likelihood of matching speaker identity.
        Range: [0, 2.0], strictly matching the official FLAG benchmark evaluation format.
        """
        face_emb = self.face_encoder(face_img)
        audio_emb = self.audio_encoder(audio_wav)
        cos_sim = torch.sum(face_emb * audio_emb, dim=-1)
        dist = torch.sqrt(torch.clamp(2.0 - 2.0 * cos_sim, min=0.0, max=4.0))
        return dist


# ==============================================================================
# 6. Early Stopping Monitor for Anti-Overfitting Regularization
# ==============================================================================

class EarlyStopping:
    """
    Early Stopping monitor to prevent model overfitting.
    Monitors validation loss or validation EER.
    If the metric fails to improve by at least min_delta for `patience` consecutive epochs,
    signals early stopping and restores best model parameters.
    """
    def __init__(
        self,
        patience: int = config.EARLY_STOPPING_PATIENCE,
        min_delta: float = config.EARLY_STOPPING_MIN_DELTA,
        mode: str = config.EARLY_STOPPING_MODE,
        restore_best_weights: bool = config.EARLY_STOPPING_RESTORE_BEST,
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

        if self.mode not in ["min", "max"]:
            raise ValueError(f"EarlyStopping mode must be 'min' or 'max', got {mode}")

    def is_better(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < (self.best_score - self.min_delta)
        else:
            return score > (self.best_score + self.min_delta)

    def step(self, score: float, model: nn.Module, epoch: int) -> bool:
        """
        Updates early stopping state with the latest score.
        Returns True if a new best score was achieved, False otherwise.
        """
        if math.isnan(score) or math.isinf(score):
            if self.verbose:
                print(f"[EarlyStopping] Warning: Encountered non-finite score {score}. Ignoring.")
            return False

        if self.is_better(score):
            if self.verbose and self.best_score is not None:
                print(f"[EarlyStopping] Metric improved from {self.best_score:.4f} to {score:.4f} at epoch {epoch}. Resetting patience.")
            elif self.verbose:
                print(f"[EarlyStopping] Initial baseline metric recorded: {score:.4f} at epoch {epoch}.")
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best_weights:
                self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return True
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] Patience counter: {self.counter}/{self.patience} (Best: {self.best_score:.4f} at epoch {self.best_epoch})")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"[EarlyStopping] 🛑 Early stopping triggered! Validation performance stagnated for {self.patience} consecutive epochs.")
            return False

    def restore(self, model: nn.Module):
        """Restores model parameters to the best recorded state dict."""
        if self.restore_best_weights and self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            if self.verbose:
                print(f"[EarlyStopping] Restored best model parameters from epoch {self.best_epoch} (Score: {self.best_score:.4f}).")

