
from pathlib import Path
import copy
import csv
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "lung_xray_224" / "lung_xray_clean_512"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

SAVE_DIR = PROJECT_DIR / "saved_models" / "lung"
RESULTS_DIR = PROJECT_DIR / "lung_model_analysis" / "efficientnet_b0"

CHECKPOINT_PATH = SAVE_DIR / "efficientnet_b0_224_best.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
PATIENCE = 4
FREEZE_EPOCHS = 2
LEARNING_RATE = 0.0001
NUM_WORKERS = 0

# --------------------------------------------------
# 2. DEVICE
# --------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("EFFICIENTNET-B0 LUNG X-RAY TRAINING")
print("=" * 60)
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# --------------------------------------------------
# 3. CHECK DATASET
# --------------------------------------------------

for folder in [TRAIN_DIR, VAL_DIR]:
    if not folder.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder}"
        )

SAVE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 4. IMAGE TRANSFORMS
# --------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# --------------------------------------------------
# 5. LOAD DATASET
# --------------------------------------------------

train_dataset = datasets.ImageFolder(
    TRAIN_DIR, transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR, transform=val_transform
)

classes = train_dataset.classes

if classes != val_dataset.classes:
    raise ValueError("Train and validation class folders differ.")

if len(classes) != 2:
    raise ValueError(f"Expected 2 classes, found: {classes}")

print("\nClasses:", classes)
print("Class mapping:", train_dataset.class_to_idx)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)

# --------------------------------------------------
# 6. CLASS WEIGHTS
# --------------------------------------------------

# Helps account for unequal numbers of training images.
counts = torch.bincount(
    torch.tensor(train_dataset.targets),
    minlength=len(classes),
).float()

class_weights = counts.sum() / (len(classes) * counts)
class_weights = class_weights.to(DEVICE)

print("Training class counts:", counts.tolist())
print("Loss class weights:", class_weights.cpu().tolist())

criterion = nn.CrossEntropyLoss(weight=class_weights)

# --------------------------------------------------
# 7. LOAD PRETRAINED EFFICIENTNET-B0
# --------------------------------------------------

print("\nLoading pretrained EfficientNet-B0 weights...")

model = models.efficientnet_b0(
    weights=EfficientNet_B0_Weights.DEFAULT
)

# Replace ImageNet's 1000-class output with our 2 classes.
input_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(input_features, len(classes))

# Initially freeze feature extractor.
for parameter in model.features.parameters():
    parameter.requires_grad = False

model = model.to(DEVICE)

# Optimizer includes all parameters, including those
# that will be unfrozen later.
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=0.0001,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)

# --------------------------------------------------
# 8. TRAINING AND VALIDATION
# --------------------------------------------------

best_val_loss = float("inf")
best_epoch = 0
epochs_without_improvement = 0
history = []

start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()

    # Unfreeze the feature extractor after initial head training.
    if epoch == FREEZE_EPOCHS:
        print("\nUnfreezing EfficientNet-B0 feature extractor...")

        for parameter in model.features.parameters():
            parameter.requires_grad = True

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE * 0.1,
            weight_decay=0.0001,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

    # ---------------- TRAIN ----------------
    model.train()
    running_loss = 0.0
    train_true = []
    train_pred = []

    for images, labels in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        train_true.extend(labels.cpu().tolist())
        train_pred.extend(predictions.cpu().tolist())

    train_loss = running_loss / len(train_dataset)
    train_accuracy = accuracy_score(train_true, train_pred)

    # ---------------- VALIDATE ----------------
    model.eval()
    val_running_loss = 0.0
    val_true = []
    val_pred = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            val_true.extend(labels.cpu().tolist())
            val_pred.extend(predictions.cpu().tolist())

    val_loss = val_running_loss / len(val_dataset)
    val_accuracy = accuracy_score(val_true, val_pred)

    scheduler.step(val_loss)

    elapsed = time.time() - epoch_start

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}]"
        f"\nTrain Loss: {train_loss:.4f}"
        f" | Train Accuracy: {train_accuracy:.4f}"
        f"\nVal Loss:   {val_loss:.4f}"
        f" | Val Accuracy:   {val_accuracy:.4f}"
        f"\nLR: {optimizer.param_groups[0]['lr']:.7f}"
        f" | Time: {elapsed:.1f}s"
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "learning_rate": optimizer.param_groups[0]["lr"],
    })

    # Save the model with the lowest validation loss.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        epochs_without_improvement = 0

        torch.save(model.state_dict(), CHECKPOINT_PATH)

        with open(
            RESULTS_DIR / "class_names.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(classes, f, indent=4)

        print("Best model checkpoint saved:", CHECKPOINT_PATH)
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping after epoch {epoch + 1}.")
        break

# --------------------------------------------------
# 9. SAVE TRAINING HISTORY
# --------------------------------------------------

history_path = RESULTS_DIR / "training_history.csv"

with open(history_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=history[0].keys())
    writer.writeheader()
    writer.writerows(history)

# --------------------------------------------------
# 10. REPORT BEST VALIDATION CHECKPOINT
# --------------------------------------------------

# Reload the best checkpoint before producing the report.
model.load_state_dict(
    torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
)
model.eval()

val_true = []
val_pred = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE, non_blocking=True)
        outputs = model(images)

        val_pred.extend(outputs.argmax(dim=1).cpu().tolist())
        val_true.extend(labels.tolist())

report = classification_report(
    val_true,
    val_pred,
    labels=list(range(len(classes))),
    target_names=classes,
    digits=4,
    zero_division=0,
)

cm = confusion_matrix(
    val_true,
    val_pred,
    labels=list(range(len(classes))),
)

with open(
    RESULTS_DIR / "validation_classification_report.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write(report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(cm))

with open(
    RESULTS_DIR / "validation_confusion_matrix.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.writer(f)
    writer.writerow(["Actual/Predicted"] + classes)

    for class_name, row in zip(classes, cm):
        writer.writerow([class_name] + row.tolist())

print("\n" + "=" * 60)
print("TRAINING FINISHED")
print("=" * 60)
print("Best epoch:", best_epoch)
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Total time: {(time.time() - start_time) / 60:.1f} minutes")
print("\nBest-checkpoint validation report:\n")
print(report)
print("\nValidation confusion matrix:\n", cm)
print("\nCheckpoint:", CHECKPOINT_PATH)
print("Results:", RESULTS_DIR)

