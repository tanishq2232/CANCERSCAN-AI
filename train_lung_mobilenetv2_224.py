
from pathlib import Path
import copy
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V2_Weights

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# ==================================================
# CONFIGURATION
# ==================================================
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = (
    BASE_DIR
    / "lung_xray_224"
    / "lung_xray_clean_512"
)

OUTPUT_DIR = BASE_DIR / "saved_models" / "lung"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = OUTPUT_DIR / "mobilenetv2_224_best.pth"
RESULTS_PATH = OUTPUT_DIR / "mobilenetv2_224_test_results.json"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 0.0001
NUM_WORKERS = 0
PATIENCE = 5

# ==================================================
# DEVICE
# ==================================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("MobileNetV2 Lung X-ray Training")
print("=" * 60)
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3),
            2,
        ),
        "GB",
    )
else:
    print("WARNING: CUDA is unavailable; training will use CPU.")

# ==================================================
# DATASET CHECKS
# ==================================================
if not DATA_DIR.is_dir():
    raise FileNotFoundError(
        f"Dataset folder not found:\n{DATA_DIR}"
    )

for split in ["train", "val", "test"]:
    if not (DATA_DIR / split).is_dir():
        raise FileNotFoundError(
            f"Missing {split} folder in {DATA_DIR}"
        )

# ==================================================
# PRETRAINED WEIGHTS AND PREPROCESSING
# ==================================================
weights = MobileNet_V2_Weights.DEFAULT

mean = weights.transforms().mean
std = weights.transforms().std

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

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

classes = train_dataset.classes

if val_dataset.classes != classes or test_dataset.classes != classes:
    raise ValueError(
        "Class names/order differ between train, validation and test."
    )

print("\nClasses:", classes)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images:", len(test_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
)

# ==================================================
# BUILD MOBILENETV2
# ==================================================
model = models.mobilenet_v2(weights=weights)

# Replace the original 1000-class ImageNet classifier.
input_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(
    input_features,
    len(classes),
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

use_amp = device.type == "cuda"

# Mixed precision can reduce GPU memory usage.
scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp,
)

print("\nArchitecture: MobileNetV2")
print("Input size:", f"{IMAGE_SIZE} x {IMAGE_SIZE}")
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("Mixed precision:", use_amp)

# ==================================================
# TRAINING AND VALIDATION
# ==================================================
best_val_loss = float("inf")
best_state = None
epochs_without_improvement = 0
history = []

for epoch in range(EPOCHS):
    start_time = time.time()

    # ---------------- TRAIN ----------------
    model.train()

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        train_loss_sum += loss.item() * batch_size
        train_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()
        train_total += batch_size

    train_loss = train_loss_sum / train_total
    train_accuracy = train_correct / train_total

    # ---------------- VALIDATE ----------------
    model.eval()

    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            val_loss_sum += loss.item() * batch_size
            val_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            val_total += batch_size

    val_loss = val_loss_sum / val_total
    val_accuracy = val_correct / val_total

    elapsed = time.time() - start_time

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f} | "
        f"Time: {elapsed:.1f}s"
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
    })

    # Save the checkpoint with the lowest validation loss.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0

        torch.save({
            "model_state_dict": best_state,
            "classes": classes,
            "image_size": IMAGE_SIZE,
            "architecture": "MobileNetV2",
            "best_val_loss": best_val_loss,
            "epoch": epoch + 1,
        }, BEST_MODEL_PATH)

        print("  Best model saved:", BEST_MODEL_PATH)

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(
            f"\nEarly stopping: validation loss did not improve "
            f"for {PATIENCE} consecutive epochs."
        )
        break

# ==================================================
# LOAD BEST CHECKPOINT
# ==================================================
if not BEST_MODEL_PATH.exists():
    raise RuntimeError("No best-model checkpoint was saved.")

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Save training history.
history_path = OUTPUT_DIR / "mobilenetv2_224_training_history.json"
history_path.write_text(
    json.dumps(history, indent=2),
    encoding="utf-8",
)

# ==================================================
# FINAL TEST EVALUATION
# ==================================================
# Test set is evaluated after selecting the best
# checkpoint using validation loss.
y_true = []
y_pred = []

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            outputs = model(images)

        predictions = outputs.argmax(dim=1).cpu().tolist()

        y_true.extend(labels.tolist())
        y_pred.extend(predictions)

test_accuracy = accuracy_score(y_true, y_pred)

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(classes))),
    target_names=classes,
    output_dict=True,
    zero_division=0,
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(classes))),
)

results = {
    "architecture": "MobileNetV2",
    "image_size": IMAGE_SIZE,
    "classes": classes,
    "train_images": len(train_dataset),
    "validation_images": len(val_dataset),
    "test_images": len(test_dataset),
    "best_epoch": checkpoint["epoch"],
    "best_validation_loss": checkpoint["best_val_loss"],
    "test_accuracy": test_accuracy,
    "classification_report": report,
    "confusion_matrix": cm.tolist(),
}

RESULTS_PATH.write_text(
    json.dumps(results, indent=2),
    encoding="utf-8",
)

print("\n" + "=" * 60)
print("FINAL TEST CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        digits=4,
        zero_division=0,
    )
)

print("Test accuracy:", f"{test_accuracy:.4f}")
print("Best epoch:", checkpoint["epoch"])
print("Best validation loss:", checkpoint["best_val_loss"])
print("\nModel saved at:", BEST_MODEL_PATH)
print("Test results saved at:", RESULTS_PATH)
print("Training history saved at:", history_path)
print("\nTraining and evaluation completed.")
