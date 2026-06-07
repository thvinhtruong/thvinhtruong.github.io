---
title: "Proposal for the Next Paper on Vision Transformer and Contrastive Learning for Aquaculture Diagnosis"
date: 2026-06-07 17:05:38 +0200
categories: []
tags: []
---


## Overview

The strongest proposal is to extend the current shrimp disease classification study into an **edge-deployable and explainable aquaculture diagnosis framework** built on self-supervised Vision Transformer representations.[web:61][web:66] This direction is stronger than a simple dataset swap because it preserves the core contribution of the current paper — ViT plus SimCLR on limited labels — while adding two practical gaps that matter for real-world deployment: low-resource inference and model interpretability.[web:55][web:32]

The proposed paper should position the current work as the teacher or baseline system: supervised ViT/EfficientNet reached about 98% accuracy and SimCLR plus ViT-Small reached about 85% on the 4-class shrimp dataset from 4,348 images.[file:1] Building on that, the new work can aim to show that self-supervised features are not only useful for classification, but can also be compressed for edge devices and audited through explainable AI methods.[web:66][web:55]

## Proposed Paper Title

**Edge-Deployable and Explainable Self-Supervised Vision Transformer for Aquaculture Disease Diagnosis**

Alternative title:

**From Contrastive Pretraining to Trustworthy Edge Inference: A Lightweight Explainable ViT for Shrimp Disease Detection**

## Core Research Question

Can a self-supervised ViT trained with SimCLR retain strong disease recognition performance after compression for edge deployment, while also producing trustworthy disease-focused explanations through transformer-specific XAI methods?[web:32][web:55]

## Main Hypothesis

SimCLR pretraining learns transferable visual representations from unlabeled aquaculture imagery, and these representations can be distilled into a lightweight student model with limited performance loss for on-device inference.[web:70][web:32] In parallel, transformer-native explanation methods such as attention rollout and GMAR can verify whether the compressed model still focuses on anatomically meaningful disease regions rather than spurious background cues.[web:35][web:55]

## Recommended Contribution Set

The paper should contribute in three layers:

1. **Self-supervised representation learning for aquaculture** using SimCLR plus ViT, motivated by the low-data advantage of SSL in domain-specific vision tasks.[web:66][web:70]
2. **Edge optimization** through quantization, pruning, or knowledge distillation into MobileViT or another lightweight student model, following recent edge-ViT compression results.[web:4][web:32]
3. **Explainability** by comparing Grad-CAM, attention rollout, and GMAR to evaluate whether the model attends to disease symptoms such as white spots or dark gill areas.[web:26][web:35][web:55]

This combination is the most publishable because many papers cover only one of these topics in isolation, while the joint formulation is still sparse in aquaculture and agricultural health monitoring.[web:29][web:32]

## Proposed Methodology

### Stage 1: Teacher Model

Use the current SimCLR plus ViT-Small pipeline as the teacher model. This preserves continuity with the current work and avoids rewriting the full training framework.[file:1] The teacher should be pretrained on all available unlabeled shrimp or aquaculture images, then fine-tuned on the 4-class labeled shrimp dataset.[web:70][web:66]

### Stage 2: Student Model for Edge Deployment

Distill the teacher into a lightweight student such as MobileViT-XS, MobileViT-S, TinyViT, or EfficientViT.[web:18][web:32] Recent agricultural IoT distillation work shows that attention plus logit distillation can preserve much of the teacher's performance while reducing compute by about 95% and memory to about 13 MB.[web:32]

### Stage 3: Compression Pipeline

Evaluate three optimization options:

- INT8 post-training quantization or quantization-aware training for deployment efficiency.[web:58][web:4]
- Structured pruning of attention heads or tokens for further latency reduction.[web:7]
- Distillation as the main route for performance-preserving compression.[web:32]

The paper does not need to include all three as equal contributions. A cleaner design is to use distillation as the main method and add INT8 quantization as the deployment step.

### Stage 4: XAI Module

Apply transformer-specific explainability methods to both teacher and student models:

- Attention Rollout as the simplest ViT-native baseline.[web:35]
- Grad-CAM adapted for transformer blocks using `pytorch-grad-cam`.[web:26]
- GMAR as the main XAI contribution because it weights attention heads by gradient importance and has shown improved interpretability over standard rollout.[web:55]

This allows the paper to answer a strong practical question: after compression, does the student still look at the same disease-relevant regions as the larger teacher?[web:55][web:35]

## Experimental Design

### Primary Task

4-class shrimp disease classification: Healthy, BG, WSSV, and WSSV_BG, using the same dataset protocol as the current study for fair comparison.[file:1]

### Secondary Task

Optional cross-domain transfer experiment: pretrain the teacher using mixed unlabeled aquaculture imagery from shrimp and fish disease datasets, then fine-tune only on shrimp labels.[web:2][web:8] This extension would strengthen the paper if additional unlabeled data are available.

### Evaluation Metrics

| Category | Metrics |
|---------|---------|
| Classification | Accuracy, macro F1, recall per class, confusion matrix |
| Edge deployment | Model size, parameters, FLOPs, latency, FPS, peak memory |
| Explainability | Qualitative saliency maps, insertion/deletion fidelity, expert inspection |
| Generalization | Performance under different lighting, farm, or background conditions |

The deployment evaluation should be performed on at least one realistic device such as Raspberry Pi 5 or Jetson Orin Nano, because edge claims are much stronger with actual hardware evidence than with FLOP-only estimates.[web:47][web:53]

## Paper Structure

A strong structure would be:

1. Introduction: practical need for low-cost and trustworthy aquaculture diagnosis.[web:29]
2. Related work: shrimp/fish disease detection, SSL in low-data vision, ViT edge compression, and XAI for transformers.[web:8][web:66][web:32][web:55]
3. Method: SimCLR teacher, distilled student, compression pipeline, XAI module.
4. Experiments: classification, deployment benchmarks, explanation comparisons.
5. Discussion: trade-off between accuracy, latency, and interpretability.
6. Conclusion: field-deployable and explainable aquaculture diagnosis.

## Why This Proposal Is Strong

This proposal is stronger than a simple domain switch for three reasons. First, it directly extends the current paper instead of abandoning it, so your prior code, results, and narrative remain useful.[file:1] Second, edge deployment addresses a real application barrier, because ViT models are often too expensive for small embedded devices without compression.[web:4][web:7] Third, XAI improves scientific trust, especially for fine-grained disease cues where users need visual evidence that the model is detecting lesions rather than background artifacts.[web:17][web:55]

## Simpler Backup Proposal

If the full edge plus XAI proposal feels too large, the best reduced-scope paper is:

**Explainable Self-Supervised Vision Transformer for Shrimp Disease Detection**

This version keeps the current SimCLR plus ViT pipeline and adds a rigorous explainability study comparing Attention Rollout, Grad-CAM, and GMAR across the 4 classes.[web:26][web:35][web:55] It is easier to finish and still gives a clear new contribution with minimal architectural change.

## Final Recommendation

The best new paper proposal is to develop a **distilled, edge-deployable, and explainable SimCLR-pretrained ViT system for shrimp disease detection**.[web:32][web:55] If time, hardware access, or implementation scope is limited, the best fallback is an **XAI-focused extension** of the current paper, because it is the fastest route to a publishable and defensible next contribution.[web:35][web:55]
