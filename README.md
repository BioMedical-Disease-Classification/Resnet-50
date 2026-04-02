 Project Overview
This repository contains the core deep learning architecture for a 3-class classification model designed to detect Normal, Bacterial Pneumonia, and Viral Pneumonia from pediatric chest X-rays.
Our approach utilizes Transfer Learning to leverage high-level visual features while fine-tuning for clinical diagnostic accuracy.

Technical Implementation 
Backbone: ResNet-50 (Pre-trained on ImageNet-1K).
Optimization Strategy: Initial residual blocks are frozen (requires_grad = False) to prevent the destruction of pre-trained feature detectors.
Custom Classification Head:Linear Layer (2048 -> 512 nodes)ReLU ActivationDropout Layer ($p=0.3$) for overfit prevention.
Final Softmax Layer (512 -> 3 nodes).

Data Engineering (Phase 1)The raw Kaggle dataset was programmatically restructured. We implemented a custom organize_data.py script to perform "filename-based surgery," splitting the generic Pneumonia folder into distinct Bacterial and Viral sub-directories. This is IMPORTANT for our 3-class diagnostic objective.
