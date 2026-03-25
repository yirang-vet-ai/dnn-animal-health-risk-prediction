# ============================================================
# 04_predict_new_pet.py
# Single-pet risk prediction dashboard
# Copyright 2026 YIRANG JUNG
# Licensed under the Apache License, Version 2.0
# ============================================================

import os
import pandas as pd
import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt
import joblib
import torch
import torch.nn as nn


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


class PetHealthDNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def save_and_show(fig, save_path):
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def predict_new_pet(model, scaler, label_encoder, device, sample_dict, feature_names):
    sample_df = pd.DataFrame([sample_dict])
    sample_df = sample_df[feature_names]

    sample_scaled = scaler.transform(sample_df)
    sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(sample_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = probs.argmax()
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

    return pred_label, probs


def get_result_color(pred_label):
    if pred_label == "danger":
        return "red"
    if pred_label == "caution":
        return "orange"
    return "green"


def plot_prediction_dashboard(sample_dict, pred_label, probs, class_names, figures_dir):
    fig = plt.figure(figsize=(18, 10))

    ax_left = fig.add_subplot(1, 2, 1)
    ax_left.axis("off")
    result_color = get_result_color(pred_label)

    left_text = [
        "Input Data",
        "",
        f"- age: {sample_dict['age']}",
        f"- weight: {sample_dict['weight']}",
        f"- temperature: {sample_dict['temperature']}",
        f"- heart_rate: {sample_dict['heart_rate']}",
        f"- appetite_score: {sample_dict['appetite_score']}",
        f"- activity_score: {sample_dict['activity_score']}",
        f"- wbc: {sample_dict['wbc']}",
        f"- rbc: {sample_dict['rbc']}",
        f"- glucose: {sample_dict['glucose']}",
        "",
        "Prediction Result",
        "",
        f"Predicted Class: {pred_label}",
    ]

    ax_left.text(
        0.02,
        0.95,
        "\n".join(left_text),
        transform=ax_left.transAxes,
        va="top",
        ha="left",
        fontsize=20,
        linespacing=1.6,
    )

    ax_left.text(
        0.02,
        0.08,
        f"Final Result: {pred_label.upper()}",
        transform=ax_left.transAxes,
        va="bottom",
        ha="left",
        fontsize=26,
        color=result_color,
        fontweight="bold",
    )

    ax_right = fig.add_subplot(1, 2, 2)
    bars = ax_right.bar(class_names, probs)
    ax_right.set_title("Prediction Probabilities")
    ax_right.set_xlabel("Class")
    ax_right.set_ylabel("Probability")
    ax_right.set_ylim(0, 1.05)

    for bar, prob, cls_name in zip(bars, probs, class_names):
        ax_right.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{prob:.3f}",
            ha="center",
            va="bottom",
            fontsize=20,
        )
        if cls_name == pred_label:
            bar.set_linewidth(3)
            bar.set_edgecolor("black")

    fig.suptitle(f"Pet Health Risk Prediction: {pred_label.upper()}", fontsize=24)
    save_and_show(fig, os.path.join(figures_dir, "08_prediction_dashboard.png"))


def main():
    base_dir = os.path.abspath(".")
    artifact_dir = os.path.join(base_dir, "artifacts")
    figures_dir = os.path.join(base_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    model_path = os.path.join(artifact_dir, "pet_health_dnn.pt")
    scaler_path = os.path.join(artifact_dir, "scaler.pkl")
    label_encoder_path = os.path.join(artifact_dir, "label_encoder.pkl")
    feature_names_path = os.path.join(artifact_dir, "feature_names.pkl")

    for file_path in [model_path, scaler_path, label_encoder_path, feature_names_path]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Required file not found: {file_path}\nRun 01_make_data.py and 02_train_dnn.py first."
            )

    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)
    feature_names = joblib.load(feature_names_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PetHealthDNN(
        input_dim=len(feature_names),
        num_classes=len(label_encoder.classes_),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    print("=" * 60)
    print("Saved model loaded")
    print(f"Device: {device}")
    print("=" * 60)

    new_pet = {
        "age": 12,
        "weight": 3.2,
        "temperature": 39.5,
        "heart_rate": 150,
        "appetite_score": 2,
        "activity_score": 3,
        "wbc": 18.0,
        "rbc": 4.7,
        "glucose": 145,
    }

    print("\nInput data")
    for key, value in new_pet.items():
        print(f"- {key}: {value}")

    pred_label, pred_probs = predict_new_pet(
        model=model,
        scaler=scaler,
        label_encoder=label_encoder,
        device=device,
        sample_dict=new_pet,
        feature_names=feature_names,
    )

    print("\nPrediction result")
    print(f"Predicted class: {pred_label}")
    print("\nClass probabilities")
    for cls_name, prob in zip(label_encoder.classes_, pred_probs):
        print(f"- {cls_name}: {prob:.4f}")

    plot_prediction_dashboard(new_pet, pred_label, pred_probs, label_encoder.classes_, figures_dir)

    print("\nSaved")
    print(f"Dashboard path: {os.path.join(figures_dir, '08_prediction_dashboard.png')}")


if __name__ == "__main__":
    main()
