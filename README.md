
# VulSymBERT: Advancing Vulnerability Scoring through Expert-Guided Neural Symbolic AI Learning

This repository contains the official code, datasets, and supplementary material for the paper:

> **"VulSymBERT: Advancing Vulnerability Scoring through Expert-Guided Neural Symbolic AI Learning"**  

## 📌 Overview

VulSymBERT is a hybrid neuro-symbolic framework designed to enhance automated vulnerability scoring using the CVSS (Common Vulnerability Scoring System). It combines Sentence-BERT embeddings with domain expert knowledge graphs to improve parameter-wise CVSS prediction from CVE descriptions, tackling issues of trustability, interpretability, and data imbalance.

**Key Features:**
- Fine-tuned SBERT models for each CVSS vector parameter.
- Integration of symbolic knowledge from expert-defined graphs.
- Multi-stage training pipeline to improve accuracy and stability.
- Extensive evaluation against standard BERT baselines.

---

## 📁 Repository Structure

```bash
.
├── Dataset/               # Cleaned CVE datasets used for training/evaluation
├── models/                # Pretrained/fine-tuned SBERT model checkpoints 
├── figures/               # Plots and diagrams used in the paper
├── src/                   # Source code for model training and evaluation
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── requirements.txt       # Python dependencies
├── VulSymBERT.pdf         # Final version of the paper
└── README.md              # This file
