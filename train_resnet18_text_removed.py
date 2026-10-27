
from pathlib import Path
import copy
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "lung_xray_224_text_removed" / "lung_xray_clean_512"
OUTPUT_DIR = ROOT / "saved_models" / "lung_text_removed"
REPORT_DIR = ROOT / "reports" / "lung_text_removed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-4
NUM_WORKERS = 0  # Safe default for Windows
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 60)
print("CANCERSCAN AI - RESNET18 TEXT-REMOVED TRAINING")
print("=" * 60)
print("Dataset:", DATA_DIR)
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# VALIDATE DATASET
# ============================================================

required_splits = ["train", "val", "test"]

for split in required_splits:
    split_dir = DATA_DIR / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"Missing split folder: {split_dir}\n"
            "Expected train, val, and test directories."
        )

train_classes = sorted(
    p.name for p in (DATA_DIR / "train").iterdir() if p.is_dir()
)

if len(train_classes) != 2:
    raise ValueError(
        f"Expected two class folders, found: {train_classes}"
    )

for split in required_splits:
    split_dir = DATA_DIR / split
    split_classes = sorted(
        p.name for p in split_dir.iterdir() if p.is_dir()
    )
    if split_classes != train_classes:
        raise ValueError(
            f"Class folders differ in {split}: {split_classes}; "
            f"expected {train_classes}"
        )

print("Classes:", train_classes)


# ============================================================
# TRANSFORMS
# ============================================================
# IMPORTANT:
# Training augmentation is deliberately mild.
# Validation/test receive deterministic preprocessing only.
# Confirm normalization matches your original training setup.

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# DATASETS AND LOADERS
# ============================================================

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transform,
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=eval_transform,
)

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=eval_transform,
)

if (
    train_dataset.class_to_idx != val_dataset.class_to_idx
    or train_dataset.class_to_idx != test_dataset.class_to_idx
):
    raise ValueError("Class-to-index mappings differ across splits.")

print("\nDataset sizes:")
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))
print("Class mapping:", train_dataset.class_to_idx)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# MODEL
# ============================================================

try:
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    print("\nLoaded ImageNet pretrained ResNet18 weights.")
except Exception as exc:
    raise RuntimeError(
        "Could not load ImageNet pretrained weights. "
        "Check your internet connection or local torchvision cache. "
        "For a controlled experiment, use the same initialization "
        "strategy as your baseline.\n"
        f"Details: {exc}"
    ) from exc

model.fc = nn.Linear(model.fc.in_features, len(train_classes))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
)


# ============================================================
# TRAINING AND VALIDATION
# ============================================================

def run_epoch(loader, training):
    model.train(training)

    total_loss = 0.0
    all_targets = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = torch.argmax(logits, dim=1)

        all_targets.extend(labels.detach().cpu().numpy())
        all_predictions.extend(predictions.detach().cpu().numpy())

    mean_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_targets, all_predictions)

    return mean_loss, accuracy


best_val_loss = float("inf")
best_epoch = 0
best_state = None
history = []

start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, training=True)
    val_loss, val_acc = run_epoch(val_loader, training=False)

    scheduler.step(val_loss)

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "learning_rate": optimizer.param_groups[0]["lr"],
    })

    print(
        f"Epoch [{epoch:02d}/{EPOCHS}] | "
        f"Train loss: {train_loss:.4f} | "
        f"Train acc: {train_acc:.4f} | "
        f"Val loss: {val_loss:.4f} | "
        f"Val acc: {val_acc:.4f}"
    )

    # Save the best validation-loss checkpoint.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())

        torch.save({
            "model_state_dict": best_state,
            "classes": train_classes,
            "class_to_idx": train_dataset.class_to_idx,
            "architecture": "resnet18",
            "input_size": 224,
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        }, OUTPUT_DIR / "resnet18_text_removed_best.pth")


# Save history and best model.
history_df = pd.DataFrame(history)
history_df.to_csv(REPORT_DIR / "training_history.csv", index=False)

if best_state is None:
    raise RuntimeError("No best model checkpoint was created.")

model.load_state_dict(best_state)
model.eval()

print(f"\nBest epoch: {best_epoch}")
print(f"Best validation loss: {best_val_loss:.4f}")


# ============================================================
# TEST EVALUATION
# ============================================================

y_true = []
y_pred = []
y_prob = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(predictions.cpu().numpy())
        y_prob.extend(probabilities.cpu().numpy())

y_true = np.asarray(y_true)
y_pred = np.asarray(y_pred)
y_prob = np.asarray(y_prob)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(
    y_true, y_pred, average="macro", zero_division=0
)
recall = recall_score(
    y_true, y_pred, average="macro", zero_division=0
)
f1 = f1_score(
    y_true, y_pred, average="macro", zero_division=0
)

metrics = {
    "architecture": "ResNet18",
    "experiment": "text_removed_224",
    "best_epoch": int(best_epoch),
    "test_samples": int(len(y_true)),
    "accuracy": float(accuracy),
    "macro_precision": float(precision),
    "macro_recall": float(recall),
    "macro_f1": float(f1),
}

# ROC-AUC for binary classification.
# The positive class is the second class in ImageFolder ordering.
if len(train_classes) == 2 and len(np.unique(y_true)) == 2:
    try:
        metrics["roc_auc"] = float(
            roc_auc_score(y_true, y_prob[:, 1])
        )
    except ValueError:
        metrics["roc_auc"] = None
else:
    metrics["roc_auc"] = None

with open(REPORT_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(train_classes))),
    target_names=train_classes,
    output_dict=True,
    zero_division=0,
)

pd.DataFrame(report).transpose().to_csv(
    REPORT_DIR / "classification_report.csv"
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(train_classes))),
)

fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=train_classes,
).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("ResNet18 - Text-Removed Test Confusion Matrix")
fig.tight_layout()
fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160)
plt.close(fig)

# Training curves.
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train")
axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Validation")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(
    history_df["epoch"],
    history_df["train_accuracy"],
    label="Train",
)
axes[1].plot(
    history_df["epoch"],
    history_df["val_accuracy"],
    label="Validation",
)
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].legend()

fig.tight_layout()
fig.savefig(REPORT_DIR / "training_curves.png", dpi=160)
plt.close(fig)

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

for key, value in metrics.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=train_classes,
    zero_division=0,
))

print("\nCheckpoint:")
print(OUTPUT_DIR / "resnet18_text_removed_best.pth")

print("\nReports:")
print(REPORT_DIR)

print(f"\nTotal training time: {(time.time() - start_time) / 60:.1f} minutes")
