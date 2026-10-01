# 🏆 FLAG 2027 Challenge: Multi-Billion Multimodal Foundation Notebooks

This folder contains the **State-of-the-Art Multi-Billion Parameter Foundation Model Notebooks** for the **FLAG 2027 Challenge: Face-Voice Association Across Languages and Gender** (MAV-Celeb v4 dataset, English Heard & Bengali Unheard Zero-Shot).

---

## 🌟 1. Bi-MHA & Orthogonal Splitter Biometrics: `FLAG2027_SOTA_IResNet100_WavLM_BiMHA_Multimodal.ipynb`

- **Filename**: [`FLAG2027_SOTA_IResNet100_WavLM_BiMHA_Multimodal.ipynb`](file:///y:/FLAG/shobrikola/notebooks/FLAG2027_SOTA_IResNet100_WavLM_BiMHA_Multimodal.ipynb)
- **Generator**: [`build_iresnet_wavlm_notebook.py`](file:///y:/FLAG/shobrikola/notebooks/build_iresnet_wavlm_notebook.py)
- **Scale**: **~400 Million Parameters** (Extreme speed & light memory footprint).
- **Vision/Face Encoder**: **IResNet-100 (`iresnet100`, 65M Parameters, ArcFace Deep ConvNet)**.
  - *Standard 112×112 face recognition resolution with PReLU residual blocks capturing deep biometric craniofacial bone structures.*
- **Audio/Voice Encoder**: **Microsoft WavLM-Large / WavLM-Base-SV (`microsoft/wavlm-large`, 317M Parameters)**.
  - *Masked speech denoising SSL specialized for speaker biometric verification across cross-lingual shifts.*
- **Cross-Modal Fusion Mechanism**: **Bi-directional Multihead Attention (Bi-MHA)**.
  - *Face queries Audio ($F_{\text{ctx}} = \text{MHA}(Q=F, K=A, V=A)$) and Audio queries Face ($A_{\text{ctx}} = \text{MHA}(Q=A, K=F, V=F)$) with residual LayerNorm and FFN.*
- **Demographic Debiasing**: **Orthogonal Subspace Splitter + GRL**.
  - *Decomposes embeddings into Identity Subspace $z_{\text{id}} \in \mathbb{R}^{512}$ and Gender Subspace $z_{\text{gen}} \in \mathbb{R}^{256}$.*
  - *GRL actively purges gender variance from $z_{\text{id}}$, while $z_{\text{gen}}$ captures demographics.*
- **Loss Functions**:
  - **Sub-Center ArcFace Loss ($K=3$)**: Multi-prototype angular margin ($m=0.35, s=32.0$) absorbing cross-lingual intra-class variation.
  - **Orthogonal Subspace Loss**: $\mathcal{L}_{\text{orth}} = \cos^2(z_{\text{id}}, z_{\text{gen}}) \to 0$ enforcing strict geometric independence.
  - **Supervised Contrastive Loss (SupCon)**: Hard negative same-gender pull/push metric learning.

---

## 🌟 2. Flagship LLM-Reasoning Architecture: `FLAG2027_SOTA_LLM_DINOv2_WavLM_Multimodal.ipynb`

- **Filename**: [`FLAG2027_SOTA_LLM_DINOv2_WavLM_Multimodal.ipynb`](file:///y:/FLAG/shobrikola/notebooks/FLAG2027_SOTA_LLM_DINOv2_WavLM_Multimodal.ipynb)
- **Generator**: [`build_llm_multimodal_notebook.py`](file:///y:/FLAG/shobrikola/notebooks/build_llm_multimodal_notebook.py)
- **Scale**: **~3.0 Billion Parameters** (DINOv2-Giant 1.1B + WavLM-Large 317M + Qwen2.5-1.5B LLM).
- **Vision/Face Encoder**: **Meta DINOv2-Giant (`facebook/dinov2-giant`, 1.1B Parameters)**.
  - *Pure self-supervised ViT-G/14 capturing pixel-level craniofacial geometry, structural depth, and morphological invariance under blur and low light.*
- **Audio/Voice Encoder**: **Microsoft WavLM-Large (`microsoft/wavlm-large`, 317M Parameters)**.
  - *Specialized denoising speech SSL extracting language-invariant biometric vocal tract features and speaker identity.*
- **Cross-Modal Fusion Engine**: **Qwen2.5-1.5B LLM (`Qwen/Qwen2.5-1.5B` in 4-bit NF4 QLoRA)**.
  - *Modern 2026/2027 LLM multimodal reasoning backbone replacing static attention with 28-layer deep cross-modal reasoning over continuous token streams.*
- **Adversarial Demographic Debiasing**: **Gradient Reversal Layer (GRL)** + **Wasserstein Critic Network** ($W_1(P_{\text{male}}, P_{\text{female}})$).
- **Loss Functions**:
  - **AdaFace Loss**: Quality-adaptive angular margin ($m=0.4, h=0.333, s=32.0$) dynamically adjusting margin to image quality.
  - **Multimodal InfoNCE Loss**: Symmetric pull-push contrastive alignment against same-gender negative impostors.
- **Hardware Architecture**: Zero-OOM dual-GPU pipeline (`cuda:0` Vision/LLM, `cuda:1` Audio, zero-activation caching).

---

## 🌟 2. 7B Q-Former Foundation Notebook: `FLAG2027_SOTA_7B_QLoRA_Multimodal.ipynb`

- **Filename**: [`FLAG2027_SOTA_7B_QLoRA_Multimodal.ipynb`](file:///y:/FLAG/shobrikola/notebooks/FLAG2027_SOTA_7B_QLoRA_Multimodal.ipynb)
- **Scale**: **Up to 6.7 Billion Parameters** (Quantized to ~4 GB VRAM with 4-bit NF4 QLoRA).
- **Vision/Face Encoder**: **CLIP-ViT-bigG (1.8B Parameters)** / **EVA-02-E (4.4B Parameters)**.
- **Audio/Voice Encoder**: **SeamlessM4T-v2-Large (2.3B Parameters)** / **Whisper-large-v3 (1.5B Parameters)**.
- **Cross-Modal Fusion Mechanism**: **Q-Former (BLIP-2 Architecture)** ($K=32$ learnable queries, 4 transformer layers).
- **Adversarial Demographic Debiasing**: **Gradient Reversal Layer (GRL)** + 2-layer MLP **Gender Discriminator**.
- **Loss Functions**: Sub-Center ArcFace Loss + Supervised Contrastive Loss (SupCon).
- **Training Engine**: QLoRA (4-bit NF4 Quantization) + Dual-GPU Accelerated Pipeline.

---

## 🌟 3. Multilingual Foundation Notebook: `FLAG2027_SOTA_Billion_Multimodal_Foundation.ipynb`

- **Filename**: [`FLAG2027_SOTA_Billion_Multimodal_Foundation.ipynb`](file:///y:/FLAG/shobrikola/notebooks/FLAG2027_SOTA_Billion_Multimodal_Foundation.ipynb)
- **Scale**: **1+ Billion Parameters**.
- **Audio Foundation**: `facebook/mms-1b-all` (Meta Massively Multilingual Speech, 1B params, 1,400+ languages).
- **Vision Foundation**: `facebook/dinov2-large` (304M params) / `facebook/dinov2-giant` (1.1B params).
- **Modules**: Multi-Layer Softmax Pooling (48 layers) + Attentive Statistics Pooling (ASP) + Gated Cross-Modal Attention (G-MCA) + GRL + Sub-Center ArcFace.

---

## 📋 4-Cell CodaBench Evaluation Matrix

Both notebooks automatically evaluate and write predictions for all 4 official competition cells:
1. `gender / sub_score_v4_English_heard.txt`
2. `gender / sub_score_v4_Bangla_unheard.txt`
3. `no_gender / sub_score_v4_English_heard.txt`
4. `no_gender / sub_score_v4_Bangla_unheard.txt`

They compute **Equal Error Rate (EER %)**, **Area Under Curve (AUC %)**, and the **Overall Mean EER**, and package the predictions into `/kaggle/working/submission.zip` matching CodaBench verification standards.

---

## 🚀 How to Run on Kaggle

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code).
2. Click **New Notebook** -> **File** -> **Upload Notebook**.
3. Upload [`FLAG2027_SOTA_7B_QLoRA_Multimodal.ipynb`](file:///y:/FLAG/shobrikola/notebooks/FLAG2027_SOTA_7B_QLoRA_Multimodal.ipynb).
4. In the right sidebar:
   - **Accelerator**: **GPU T4 x 2** (Dual GPU with Accelerate) or **P100 / A100**.
   - **Data**: Ensure `mav-celeb-v4-dataset` is attached (`/kaggle/input/datasets/mdjahidhasanjim/mav-celeb-v4-dataset`).
5. Run All Cells.
6. Download `/kaggle/working/submission.zip` and submit directly to CodaBench!
