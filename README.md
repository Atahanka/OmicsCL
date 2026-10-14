
# OmicsCL: Unsupervised Contrastive Learning for Cancer Subtype Discovery and Survival Stratification

OmicsCL is a modular framework for unsupervised learning of survival-relevant representations from multi-omics cancer data. It integrates gene expression, DNA methylation, and miRNA profiles using contrastive learning, enhanced with a novel survival-aware contrastive loss — enabling effective clustering and patient stratification without using subtype labels.

---

## 🚀 Paper

[![DOI](https://zenodo.org/badge/976522935.svg)](https://doi.org/10.5281/zenodo.23365346)

Read the full paper on arXiv: [OmicsCL: Unsupervised Contrastive Learning for Cancer Subtype Discovery and Survival Stratification](https://arxiv.org/abs/2505.00650)

---

## 📂 Repository Structure

```
OmicsCL/
├── data/                          # Raw multi-omics and clinical files
│   └── *.csv, *.cct, *.tsi
├── processed/                     # Preprocessed .npz files for training/validation/testing
│   ├── train.npz
│   ├── val.npz
│   └── test.npz
├── outputs/                       # Results: logs, plots, embeddings, models
│   ├── embeddings/
│   ├── models/
│   └── *.png, *.html, *.npy, *.json, *.log
├── src/                           # Source code: training, evaluation, utilities
│   ├── config.py
│   ├── contrastive_loss.py
│   ├── cox_post_eval.py
│   ├── data_loader.py
│   ├── eval.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── train.py
│   └── utils.py
├── extra_preprocess/             # Preprocessing helpers used exclusively for TCGA BRCA
│   └── *.py
├── CODEOWNERS
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🔧 Setup

```bash
# Clone and install dependencies
git clone https://github.com/Atahanka/OmicsCL.git
cd OmicsCL
pip install -r requirements.txt
```

---

## 🧪 Usage

```bash
# [Optional] Run extra preprocessing scripts for TCGA BRCA
python extra_preprocess/convert_to_csv.py
python extra_preprocess/survival_csv_edit_1.py
python extra_preprocess/survival_csv_edit_2.py
python extra_preprocess/survival_csv_edit_3.py

# Preprocess raw data
python src/preprocess.py

# Train the model
python src/train.py

# Evaluate embeddings
python src/eval.py
```

Model checkpoints and evaluation results are stored in the `outputs/` directory.

---

## 📜 License

This project is released under a **custom, permission-based license**.  
**It is NOT open-source** and **may not be reused, modified, or redistributed** in any form without **explicit written authorization** from the author.

This includes (but is not limited to):
- Using the source code or any portion of it in academic, research, or commercial projects  
- Modifying, adapting, or extending the preprocessing scripts, model architecture, or training pipeline  
- Reproducing or leveraging the trained models, embeddings, plots, or evaluation results  
- Publishing derivative works based on any part of this repository  

All contents of this repository—including but not limited to code, trained models, documentation, visualizations, evaluation results, and data processing utilities—are the exclusive intellectual property of Atahan Karagöz and are protected under international copyright and intellectual property laws.

**🔒 For complete terms, conditions, and instructions on how to request permission**, please consult the [LICENSE](./LICENSE) file.

> **© 2025 Atahan Karagöz. All rights reserved.**  
> Violations of this license may result in legal action.
