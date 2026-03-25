# Pet Health Risk Prediction DNN

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-red)]
[![Status](https://img.shields.io/badge/project-ready_for_GitHub-brightgreen)]

DNN-based personal project for generating synthetic pet health data, training a classifier, visualizing results, and predicting the risk level of a single pet using biometric and blood-test style inputs.

Author: YIRANG JUNG  
License: Apache License 2.0

## Overview

This repository provides a complete beginner-friendly workflow for a tabular Deep Neural Network (DNN) project.

The project does four things:

1. Generates synthetic pet health data in CSV format
2. Trains a DNN classifier with PyTorch
3. Saves model artifacts and visualization outputs
4. Predicts the risk level of an individual pet and visualizes only that pet's result

The repository is structured so it can be uploaded directly to GitHub.

## Project Goal

The main purpose of this project is not large-scale clinical deployment, but a reusable portfolio-style implementation of:

- synthetic data generation
- tabular DNN classification
- artifact saving
- prediction dashboard generation
- GitHub-ready project packaging

## Repository Structure

```text
pet-health-risk-dnn/
├─ data/
│  └─ .gitkeep
├─ artifacts/
│  └─ .gitkeep
├─ figures/
│  └─ .gitkeep
├─ 01_make_data.py
├─ 02_train_dnn.py
├─ 03_visualize_results.py
├─ 04_predict_new_pet.py
├─ requirements.txt
├─ .gitignore
├─ LICENSE
├─ NOTICE.txt
└─ README.md
```

## Features

- Generates synthetic pet health records
- Saves the generated dataset as CSV
- Trains a tabular DNN model using PyTorch
- Saves scaler, label encoder, model weights, and training history
- Produces visualization figures as PNG files
- Predicts the risk level of one pet at a time
- Displays a left-right dashboard format:
  - left: input data and final prediction
  - right: class probability graph

## Input Variables

The synthetic dataset and prediction code use the following variables:

- `age`
- `weight`
- `temperature`
- `heart_rate`
- `appetite_score`
- `activity_score`
- `wbc`
- `rbc`
- `glucose`

Target label:

- `risk_level`
  - `normal`
  - `caution`
  - `danger`

## Environment

Recommended environment:

- Windows
- Anaconda Prompt
- conda environment: `dl_env`
- Python 3.10+

## Installation

Activate your environment first:

```bash
conda activate dl_env
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

Main libraries:

- numpy
- pandas
- matplotlib
- scikit-learn
- torch
- joblib

## How to Run

Run the project in this order:

```bash
python 01_make_data.py
python 02_train_dnn.py
python 03_visualize_results.py
python 04_predict_new_pet.py
```

## Output Files

### 1. Generated dataset

```text
data/pet_health_data.csv
```

### 2. Saved training artifacts

```text
artifacts/pet_health_dnn.pt
artifacts/scaler.pkl
artifacts/label_encoder.pkl
artifacts/feature_names.pkl
artifacts/train_losses.npy
artifacts/train_accs.npy
artifacts/val_losses.npy
artifacts/val_accs.npy
artifacts/y_true.npy
artifacts/y_pred.npy
artifacts/confusion_matrix.npy
```

### 3. Visualization outputs

Examples:

```text
figures/01_class_distribution.png
figures/04_correlation_heatmap.png
figures/05_loss_curve.png
figures/06_accuracy_curve.png
figures/07_confusion_matrix.png
figures/08_prediction_dashboard.png
```

## Prediction Dashboard

The final prediction script focuses on one animal only.

Dashboard layout:

- Left panel: input values and predicted class
- Right panel: class probabilities for `normal`, `caution`, and `danger`

This design avoids confusion from overall dataset statistics when the user mainly wants the result for a single pet.

## Notes

- The dataset used here is synthetic, not clinical ground truth.
- The project is intended for learning, portfolio building, and code structure practice.
- The prediction result is a model output, not a veterinary diagnosis.

## License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.

## Notice

See `NOTICE.txt` for attribution information.

## Author

YIRANG JUNG
