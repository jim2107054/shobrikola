# 🏆 FLAG 2027 Challenge: Multi-Billion Multimodal Foundation Notebooks

This folder contains the **State-of-the-Art Multi-Billion Parameter Foundation Model Notebooks** for the **FLAG 2027 Challenge: Face-Voice Association Across Languages and Gender** (MAV-Celeb v4 dataset, English Heard & Bengali Unheard Zero-Shot).

---

## 🌟 1. Flagship Ultra-SOTA Notebook: `FLAG2027_SOTA_7B_QLoRA_Multimodal.ipynb`

- **Filename**: [`FLAG2027_SOTA_7B_QLoRA_Multimodal.ipynb`](file:///y:/FLAG/shobrikola/notebooks/FLAG2027_SOTA_7B_QLoRA_Multimodal.ipynb)
- **Scale**: **Up to 6.7 Billion Parameters** (Quantized to ~4 GB VRAM with 4-bit NF4 QLoRA).
- **Vision/Face Encoder**: **EVA-02-E (4.4B Parameters)** / **CLIP-ViT-bigG (1.8B Parameters)**.
- **Audio/Voice Encoder**: **SeamlessM4T-v2-Large (2.3B Parameters)** / **Whisper-large-v3 (1.5B Parameters)**.
  - *Native multilingual representations directly covering Bengali (Bangla) and English.*
- **Cross-Modal Fusion Mechanism**: **Q-Former (BLIP-2 Architecture)**.
  - $K=32$ learnable identity queries performing cross-attention into visual patch tokens and speech acoustic frame tokens.
- **Adversarial Demographic Debiasing**: **Gradient Reversal Layer (GRL)** + 2-layer MLP **Gender Discriminator** with dynamic $\lambda(p)$ scheduling.
- **Loss Functions**:
  - **Sub-Center ArcFace Loss**: Cosine angular margin ($m=0.35, s=32.0$) on unit hypersphere for speaker identity.
  - **Supervised Contrastive Loss (SupCon)**: Multimodal identity pull-push alignment with same-gender negative impostors.
- **Training Engine**: **QLoRA (4-bit NF4 Quantization with double quantization via `bitsandbytes` & `peft`)** + **Hugging Face `accelerate` (Dual GPU support for Kaggle 2x T4)**.
- **Official Scoring Protocol**: Euclidean distance on the unit hypersphere:
  $$d = \sqrt{2 - 2 \cos(\mathbf{e}_f, \mathbf{e}_v)} = \|\mathbf{e}_f - \mathbf{e}_v\|_2 \in [0, 2.0]$$

---

## 🌟 2. Multilingual Foundation Notebook: `FLAG2027_SOTA_Billion_Multimodal_Foundation.ipynb`

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
