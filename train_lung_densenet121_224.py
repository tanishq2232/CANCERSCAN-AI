
from pathlib import Path
import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import DenseNet121_Weights

# ---------------- CONFIGURATION ----------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "lung_xray_224" / "lung_xray_clean_512"
OUTPUT_DIR = BASE_DIR / "saved_models" / "lung"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 0.0001
NUM_WORKERS = 0  # Windows-compatible

BEST_MODEL_PATH = OUTPUT_DIR / "densenet121_224_best.pth"

# ---------------- DEVICE ----------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ---------------- DATA ----------------

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_DIR}")

required_splits = ["train", "val", "test"]

for split in required_splits:
    if not (DATA_DIR / split).is_dir():
        raise FileNotFoundError(
            f"Missing {split} folder in {DATA_DIR}"
        )

weights = DenseNet121_Weights.DEFAULT

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std,
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std,
    ),
])

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train", transform=train_transform
)
val_dataset = datasets.ImageFolder(
    DATA_DIR / "val", transform=eval_transform
)
test_dataset = datasets.ImageFolder(
    DATA_DIR / "test", transform=eval_transform
)

classes = train_dataset.classes

if len(classes) != 2:
    raise ValueError(
        f"Expected 2 classes, found: {classes}"
    )

for name, dataset in [
    ("validation", val_dataset),
    ("test", test_dataset),
]:
    if dataset.classes != classes:
        raise ValueError(
            f"{name} classes {dataset.classes} "
            f"do not match training classes {classes}"
        )

print(f"Classes: {classes}")
print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Test images: {len(test_dataset)}")

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

# ---------------- MODEL ----------------

model = models.densenet121(weights=weights)

# Replace the 1000-class ImageNet classifier
in_features = model.classifier.in_features
model.classifier = nn.Linear(in_features, len(classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)

scaler = torch.amp.GradScaler(
    "cuda", enabled=(device.type == "cuda")
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- TRAINING ----------------

best_val_loss = float("inf")
best_state = copy.deepcopy(model.state_dict())

for epoch in range(EPOCHS):
    start = time.time()

    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * images.size(0)
        train_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()
        train_total += labels.size(0)

    train_loss /= train_total
    train_acc = train_correct / train_total

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            val_correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Time: {time.time() - start:.1f}s"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())

        torch.save({
            "model_state_dict": best_state,
            "classes": classes,
            "image_size": IMAGE_SIZE,
            "architecture": "DenseNet121",
            "best_val_loss": best_val_loss,
        }, BEST_MODEL_PATH)

        print(f"  Best model saved: {BEST_MODEL_PATH}")

# ---------------- FINAL TEST ----------------
# Test only once after selecting the best validation checkpoint.

model.load_state_dict(best_state)
model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda"),
        ):
            outputs = model(images)

        test_correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()
        test_total += labels.size(0)

test_accuracy = test_correct / test_total

print("\nTraining completed.")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Saved model: {BEST_MODEL_PATH}")
