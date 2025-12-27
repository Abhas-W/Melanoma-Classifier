# Melanoma-Classifier

## Acknowledgements 
This project was developed by the author as part of a learning exercise in deep learning with PyTorch. The image classification pipeline and model training framework were built following the excellent tutorial series by *Mike Saint-Antoine* on YouTube:
https://www.youtube.com/playlist?list=PLWVKUEZ25V975TSDG6xdjIB-5chrrUP-S

Thank you for providing an approachable introduction to convolutional neural networks and practical image classification examples.

## Overview

This repository contains a CNN-based skin lesion classification project implemented in PyTorch. Two image pre-processing pipelines are evaluated: a normalization-only baseline and a CLAHE-based contrast enhancement pipeline. Empirical evaluation showed that the normalization-only pipeline achieved superior performance.

**Disclaimer:** This project is intended strictly for educational and research purposes. It is not a certified medical device and must not be used for clinical diagnosis or decision-making.

## Dataset

The model was trained and evaluated using the publicly available melanoma skin cancer dataset from Kaggle:
https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images?resource=download

The dataset was split into training and testing subsets, with class balancing applied during training.

## Pre-processing pipeline

Two preprocessing strategies were implemented and evaluated:

### 1. Basic Pre-processing (Final Pipeline)
- Grayscale conversion
- Image resizing to 50×50
- Pixel normalization to [0, 1]

**Test Accuracy:** 84.3%

This pipeline achieved the best performance and is used in the final model.

### 2. CLAHE-based Pre-processing
- Grayscale conversion
- Image resizing to 50x50
- CLAHE-based contrast enhancement
- Pixel normalization to [0, 1]

**Test Accuracy:** 82.3%
