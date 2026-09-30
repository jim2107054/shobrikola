# 🏆 FLAG 2027 Challenge: Billion-Parameter Multimodal Notebooks

This folder contains the **SOTA Billion-Scale Multilingual Foundation Model** notebook for the **FLAG 2027 Challenge: Face-Voice Association Across Languages and Gender** (MAV-Celeb v4 dataset, English Heard & Bengali Unheard Zero-Shot).

---

## 🌟 Notebook Overview: `FLAG2027_SOTA_Billion_Multimodal_Foundation.ipynb`

- **Filename**: [`FLAG2027_SOTA_Billion_Multimodal_Foundation.ipynb`](file:///y:/FLAG/shobrikola/notebooks/FLAG2027_SOTA_Billion_Multimodal_Foundation.ipynb)
- **Scale**: **1+ Billion Parameters** across dual foundation models.
- **Audio Foundation**: `facebook/mms-1b-all` (Meta Massively Multilingual Speech, 1 Billion parameters, pre-trained on 1,400+ languages including Bengali and English).
- **Vision Foundation**: `facebook/dinov2-large` (304M params) or `facebook/dinov2-giant` (1.1B params, ViT-G/14 self-distilled).
- **Metric Head**: Sub-Center ArcFace (AAM-Softmax, margin $m=0.35$, scale $s=32.0$) on 768-dimensional L2-normalized unit hypersphere $\mathbb{S}^{D-1}$.
- **Demographic Debiasing**: Gradient Reversal Layer (GRL) + Orthogonal Subspace Loss $\mathcal{L}_{\text{orth}} = \frac{|\mathbf{z}_{\text{id}}^\top \mathbf{z}_{\text{gen}}|^2}{\|\mathbf{z}_{\text{id}}\|^2 \|\mathbf{z}_{\text{gen}}\|^2} \to 0$.
- **Zero-Shot Generalization**: Gated Bidirectional Cross-Modal Attention (G-MCA) with $\tanh(\gamma)$ residual gating.
- **Official Scoring Protocol**: Euclidean distance on the unit hypersphere:
  $$d = \sqrt{2 - 2 \cos(\mathbf{e}_f, \mathbf{e}_v)} = \|\mathbf{e}_f - \mathbf{e}_v\|_2 \in [0, 2.0]$$

---

## 📋 4-Cell CodaBench Evaluation Matrix

The notebook automatically evaluates all 4 official competition cells:
1. `gender / sub_score_v4_English_heard.txt`
2. `gender / sub_score_v4_Bangla_unheard.txt`
3. `no_gender / sub_score_v4_English_heard.txt`
4. `no_gender / sub_score_v4_Bangla_unheard.txt`

It computes Equal Error Rate (EER %), Area Under Curve (AUC %), and the Overall Mean EER, and packages them into `/kaggle/working/submission.zip` matching CodaBench verification standards.

---

## 🚀 How to Run on Kaggle

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code).
2. Click **New Notebook** -> **File** -> **Upload Notebook**.
3. Upload [`FLAG2027_SOTA_Billion_Multimodal_Foundation.ipynb`](file:///y:/FLAG/shobrikola/notebooks/FLAG2027_SOTA_Billion_Multimodal_Foundation.ipynb).
4. In the right sidebar:
   - **Accelerator**: GPU T4 x 2 or P100 (or A100 if available).
   - **Data**: Ensure `mav-celeb-v4-dataset` is attached (`/kaggle/input/datasets/mdjahidhasanjim/mav-celeb-v4-dataset`).
5. Run All Cells.
6. Download `/kaggle/working/submission.zip` and upload to CodaBench!
