# ============================================================
# 03_visualize_results.py
# Visualization export for data and training outputs
# Copyright 2026 YIRANG JUNG
# Licensed under the Apache License, Version 2.0
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt
import joblib


plt.rcParams["font.size"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["legend.fontsize"] = 20
plt.rcParams["figure.titlesize"] = 20
plt.rcParams["axes.unicode_minus"] = False

try:
    plt.rcParams["font.family"] = "Malgun Gothic"
except Exception:
    pass


def save_and_show(fig, save_path):
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def get_contrast_text_color_from_normalized(norm_value):
    return "white" if norm_value < 0.5 else "black"


def plot_class_distribution(df, figures_dir):
    counts = df["risk_level"].value_counts()
    fig = plt.figure(figsize=(10, 7))
    bars = plt.bar(counts.index, counts.values)
    plt.title("Risk Level Class Distribution")
    plt.xlabel("Risk Level")
    plt.ylabel("Count")

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 5, f"{int(h)}", ha="center", va="bottom", fontsize=20)

    save_and_show(fig, os.path.join(figures_dir, "01_class_distribution.png"))


def plot_feature_histograms(df, figures_dir):
    features = ["age", "weight", "temperature", "heart_rate", "appetite_score", "activity_score", "wbc", "rbc", "glucose"]
    for idx, feature in enumerate(features, start=1):
        fig = plt.figure(figsize=(10, 7))
        plt.hist(df[feature], bins=20)
        plt.title(f"{feature} Distribution")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        save_and_show(fig, os.path.join(figures_dir, f"02_hist_{idx:02d}_{feature}.png"))


def plot_boxplots_by_class(df, figures_dir):
    features = ["temperature", "heart_rate", "wbc", "rbc", "glucose"]
    class_order = ["normal", "caution", "danger"]

    for idx, feature in enumerate(features, start=1):
        data_to_plot = [df[df["risk_level"] == c][feature] for c in class_order]
        fig = plt.figure(figsize=(10, 7))
        plt.boxplot(data_to_plot, tick_labels=class_order)
        plt.title(f"{feature} by Risk Level")
        plt.xlabel("Risk Level")
        plt.ylabel(feature)
        save_and_show(fig, os.path.join(figures_dir, f"03_boxplot_{idx:02d}_{feature}.png"))


def plot_correlation_heatmap(df, figures_dir):
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig = plt.figure(figsize=(12, 10))
    ax = plt.gca()
    vmin, vmax = -1.0, 1.0
    im = ax.imshow(corr, interpolation="nearest", aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(im)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    plt.title("Correlation Heatmap")

    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            value = corr.iloc[i, j]
            norm_value = (value - vmin) / (vmax - vmin)
            text_color = get_contrast_text_color_from_normalized(norm_value)
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=16, color=text_color)

    save_and_show(fig, os.path.join(figures_dir, "04_correlation_heatmap.png"))


def plot_training_history(train_losses, val_losses, train_accs, val_accs, figures_dir):
    epochs_range = range(1, len(train_losses) + 1)

    fig1 = plt.figure(figsize=(10, 7))
    plt.plot(epochs_range, train_losses, marker="o", label="Train Loss")
    plt.plot(epochs_range, val_losses, marker="o", label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    save_and_show(fig1, os.path.join(figures_dir, "05_loss_curve.png"))

    fig2 = plt.figure(figsize=(10, 7))
    plt.plot(epochs_range, train_accs, marker="o", label="Train Accuracy")
    plt.plot(epochs_range, val_accs, marker="o", label="Val Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    save_and_show(fig2, os.path.join(figures_dir, "06_accuracy_curve.png"))


def plot_confusion_matrix(cm, class_names, figures_dir):
    fig = plt.figure(figsize=(8, 7))
    ax = plt.gca()
    vmin = 0
    vmax = cm.max() if cm.max() > 0 else 1
    im = ax.imshow(cm, interpolation="nearest", aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(im)

    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            norm_value = (value - vmin) / (vmax - vmin) if vmax > vmin else 0
            text_color = get_contrast_text_color_from_normalized(norm_value)
            ax.text(j, i, str(value), ha="center", va="center", fontsize=20, color=text_color)

    save_and_show(fig, os.path.join(figures_dir, "07_confusion_matrix.png"))


def main():
    base_dir = os.path.abspath(".")
    data_dir = os.path.join(base_dir, "data")
    artifact_dir = os.path.join(base_dir, "artifacts")
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "pet_health_data.csv")
    label_encoder_path = os.path.join(artifact_dir, "label_encoder.pkl")
    train_losses_path = os.path.join(artifact_dir, "train_losses.npy")
    train_accs_path = os.path.join(artifact_dir, "train_accs.npy")
    val_losses_path = os.path.join(artifact_dir, "val_losses.npy")
    val_accs_path = os.path.join(artifact_dir, "val_accs.npy")
    confusion_matrix_path = os.path.join(artifact_dir, "confusion_matrix.npy")

    required_files = [
        csv_path,
        label_encoder_path,
        train_losses_path,
        train_accs_path,
        val_losses_path,
        val_accs_path,
        confusion_matrix_path,
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Required file not found: {file_path}\nRun 01_make_data.py and 02_train_dnn.py first."
            )

    df = pd.read_csv(csv_path)
    label_encoder = joblib.load(label_encoder_path)
    train_losses = np.load(train_losses_path)
    train_accs = np.load(train_accs_path)
    val_losses = np.load(val_losses_path)
    val_accs = np.load(val_accs_path)
    cm = np.load(confusion_matrix_path)

    print("=" * 60)
    print("Visualization started")
    print(f"Figure directory: {figures_dir}")
    print("=" * 60)

    plot_class_distribution(df, figures_dir)
    plot_feature_histograms(df, figures_dir)
    plot_boxplots_by_class(df, figures_dir)
    plot_correlation_heatmap(df, figures_dir)
    plot_training_history(train_losses, val_losses, train_accs, val_accs, figures_dir)
    plot_confusion_matrix(cm, label_encoder.classes_, figures_dir)

    print("=" * 60)
    print("Visualization complete")
    print("Saved files")
    for file_name in sorted(os.listdir(figures_dir)):
        print("-", file_name)
    print("=" * 60)


if __name__ == "__main__":
    main()
