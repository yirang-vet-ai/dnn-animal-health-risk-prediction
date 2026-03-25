# ============================================================
# 02_train_dnn.py
# DNN training, evaluation, and artifact export
# Copyright 2026 YIRANG JUNG
# Licensed under the Apache License, Version 2.0
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class PetDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_targets)


def main():
    base_dir = os.path.abspath(".")
    data_dir = os.path.join(base_dir, "data")
    artifact_dir = os.path.join(base_dir, "artifacts")
    os.makedirs(artifact_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "pet_health_data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Data file not found: {csv_path}\nRun 01_make_data.py first."
        )

    df = pd.read_csv(csv_path)
    target_col = "risk_level"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = PetDataset(X_train_scaled, y_train)
    test_dataset = PetDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = PetHealthDNN(
        input_dim=X_train_scaled.shape[1],
        num_classes=len(label_encoder.classes_),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 30
    best_val_acc = 0.0
    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    best_model_path = os.path.join(artifact_dir, "pet_health_dnn.pt")

    print("\nTraining started")
    print("=" * 60)
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(
            f"[Epoch {epoch + 1:02d}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    print("=" * 60)
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    best_model = PetHealthDNN(
        input_dim=X_train_scaled.shape[1],
        num_classes=len(label_encoder.classes_),
    ).to(device)
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_acc, y_pred, y_true = evaluate(best_model, test_loader, criterion, device)

    print("\nFinal test performance")
    print("=" * 60)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nClassification report")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_true, y_pred)

    joblib.dump(scaler, os.path.join(artifact_dir, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(artifact_dir, "label_encoder.pkl"))
    joblib.dump(list(X.columns), os.path.join(artifact_dir, "feature_names.pkl"))

    np.save(os.path.join(artifact_dir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(artifact_dir, "train_accs.npy"), np.array(train_accs))
    np.save(os.path.join(artifact_dir, "val_losses.npy"), np.array(val_losses))
    np.save(os.path.join(artifact_dir, "val_accs.npy"), np.array(val_accs))
    np.save(os.path.join(artifact_dir, "y_true.npy"), y_true)
    np.save(os.path.join(artifact_dir, "y_pred.npy"), y_pred)
    np.save(os.path.join(artifact_dir, "confusion_matrix.npy"), cm)

    print("\nArtifacts saved")
    print(f"Model file: {best_model_path}")
    print(f"Artifact directory: {artifact_dir}")


if __name__ == "__main__":
    main()
