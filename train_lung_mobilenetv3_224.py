
from pathlib import Path
import copy
import json
import time
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V3_Large_Weights
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "lung_xray_224" / "lung_xray_clean_512"
OUTPUT_DIR = BASE_DIR / "saved_models" / "lung"

BEST_MODEL_PATH = OUTPUT_DIR / "mobilenetv3_large_224_best.pth"
RESULTS_PATH = OUTPUT_DIR / "mobilenetv3_large_224_test_results.json"
HISTORY_PATH = OUTPUT_DIR / "mobilenetv3_large_224_history.json"

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 0.0001
PATIENCE = 5
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

# =========================================================
# 2. DATASET AND PREPROCESSING
# =========================================================

if not (DATA_DIR / "train").is_dir():
    raise FileNotFoundError(f"Training folder not found: {DATA_DIR / 'train'}")

weights = MobileNet_V3_Large_Weights.DEFAULT
weight_transforms = weights.transforms()

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weight_transforms.mean,
        std=weight_transforms.std,
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weight_transforms.mean,
        std=weight_transforms.std,
    ),
])

train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=eval_transform)
test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=eval_transform)

classes = train_dataset.classes

if val_dataset.classes != classes or test_dataset.classes != classes:
    raise ValueError("Class folders differ between train, validation, and test.")

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=USE_AMP
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=USE_AMP
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=USE_AMP
)

# =========================================================
# 3. LOAD PRETRAINED MOBILENETV3-LARGE
# =========================================================

model = models.mobilenet_v3_large(weights=weights)

# Replace the final classification layer for our two classes.
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, len(classes))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001,
)

scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("MobileNetV3-Large Lung X-ray Training")
print("=" * 60)
print("Device:", DEVICE)

if USE_AMP:
    print("GPU:", torch.cuda.get_device_name(0))
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU memory: {memory_gb:.1f} GB")

print("Classes:", classes)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Test images:", len(test_dataset))
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)
print("Maximum epochs:", EPOCHS)
print("Mixed precision:", USE_AMP)

# =========================================================
# 4. TRAINING AND VALIDATION
# =========================================================

best_val_loss = float("inf")
best_epoch = 0
best_state = None
epochs_without_improvement = 0
history = []

for epoch in range(EPOCHS):
    start_time = time.time()

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE, non_blocking=USE_AMP)
        labels = labels.to(DEVICE, non_blocking=USE_AMP)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_accuracy = correct / total

    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(DEVICE, non_blocking=USE_AMP)
            labels = labels.to(DEVICE, non_blocking=USE_AMP)

            with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_loss_sum += loss.item() * labels.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss_sum / val_total
    val_accuracy = val_correct / val_total
    elapsed = time.time() - start_time

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] | "
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
        "seconds": elapsed,
    })

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0

        torch.save({
            "model_state_dict": best_state,
            "classes": classes,
            "image_size": IMAGE_SIZE,
            "architecture": "MobileNetV3-Large",
            "best_val_loss": best_val_loss,
            "epoch": best_epoch,
        }, BEST_MODEL_PATH)

        print("  Saved new best checkpoint.")
    else:
        epochs_without_improvement += 1

    with open(HISTORY_PATH, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
        csv_path = OUTPUT_DIR / "mobilenetv3_large_224_history.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=history[0].keys())
    writer.writeheader()
    writer.writerows(history)

    if epochs_without_improvement >= PATIENCE:
        print(f"Early stopping after {PATIENCE} epochs without improvement.")
        break

if best_state is None:
    raise RuntimeError("No best model was saved.")

# =========================================================
# 5. TEST THE BEST CHECKPOINT
# =========================================================

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

y_true = []
y_pred = []

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(DEVICE, non_blocking=USE_AMP)

        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            outputs = model(images)

        y_pred.extend(outputs.argmax(dim=1).cpu().tolist())
        y_true.extend(labels.tolist())

test_accuracy = accuracy_score(y_true, y_pred)

report_text = classification_report(
    y_true, y_pred,
    labels=list(range(len(classes))),
    target_names=classes,
    zero_division=0,
)

report_dict = classification_report(
    y_true, y_pred,
    labels=list(range(len(classes))),
    target_names=classes,
    zero_division=0,
    output_dict=True,
)

cm = confusion_matrix(
    y_true, y_pred, labels=list(range(len(classes)))
)

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)
print("Best epoch:", best_epoch)
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(report_text)
print("Confusion matrix (rows = actual, columns = predicted):")
print(cm)

with open(RESULTS_PATH, "w", encoding="utf-8") as file:
    json.dump({
        "architecture": "MobileNetV3-Large",
        "image_size": IMAGE_SIZE,
        "classes": classes,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "test_accuracy": test_accuracy,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
    }, file, indent=2)

print("\nTraining complete.")
print("Best model:", BEST_MODEL_PATH)
print("Test results:", RESULTS_PATH)
print("Training history:", HISTORY_PATH)
