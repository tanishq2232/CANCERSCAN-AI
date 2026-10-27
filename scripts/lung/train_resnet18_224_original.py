
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# --------------------------------------------------
# PATHS
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "lung_xray_224" / "lung_xray_clean_512"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "resnet18_224_baseline"
MODEL_DIR = PROJECT_ROOT / "saved_models" / "lung"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "resnet18_224_baseline_best.pth"
HISTORY_PATH = OUTPUT_DIR / "training_history.json"
RESULTS_PATH = OUTPUT_DIR / "test_results.json"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5
SEED = 42
NUM_WORKERS = 0  # Windows-safe

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)

    print("=" * 60)
    print("ResNet18 - 224x224 Lung X-ray Training")
    print("=" * 60)
    print("Dataset:", DATA_ROOT)
    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_dir = DATA_ROOT / "train"
    val_dir = DATA_ROOT / "val"
    test_dir = DATA_ROOT / "test"

    for folder in (train_dir, val_dir, test_dir):
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing dataset folder: {folder}")

    # ImageNet normalization; no text removal or masking.
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_transform)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_transform)

    if train_ds.classes != val_ds.classes or train_ds.classes != test_ds.classes:
        raise ValueError("Class mappings differ between splits.")

    print("\nClass mapping:", train_ds.class_to_idx)
    print(f"Train: {len(train_ds)}")
    print(f"Validation: {len(val_ds)}")
    print(f"Test: {len(test_ds)}")

    # Check exact duplicate image files across splits.
    # A duplicate indicates potential leakage.
    import hashlib

    split_files = {}
    for split_name, dataset in [
        ("train", train_ds), ("val", val_ds), ("test", test_ds)
    ]:
        for path, _ in dataset.samples:
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            split_files.setdefault(digest, set()).add(split_name)

    leakage = [splits for splits in split_files.values() if len(splits) > 1]
    print("Exact-image hashes appearing across multiple splits:", len(leakage))
    if leakage:
        print("WARNING: possible data leakage. Results may be optimistic.")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available()
    )

    # Use pretrained ImageNet weights when available locally or downloadable.
    try:
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        print("Using pretrained ImageNet weights.")
    except Exception as exc:
        print("Pretrained weights unavailable; using random initialization.")
        print("Reason:", exc)
        model = models.resnet18(weights=None)

    model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes))
    model = model.to(DEVICE)

    # Compensate for class imbalance in the training split.
    counts = np.bincount(train_ds.targets, minlength=len(train_ds.classes))
    weights = len(train_ds.targets) / (len(train_ds.classes) * counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    history = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    def run_eval(loader, collect=False):
        model.eval()
        total_loss = 0.0
        all_y, all_pred, all_prob = [], [], []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(images)
                loss = criterion(logits, labels)
                probs = torch.softmax(logits, dim=1)

                total_loss += loss.item() * labels.size(0)
                all_y.extend(labels.cpu().numpy().tolist())
                all_pred.extend(logits.argmax(1).cpu().numpy().tolist())
                all_prob.extend(probs.cpu().numpy().tolist())

        avg_loss = total_loss / len(loader.dataset)
        accuracy = accuracy_score(all_y, all_pred)

        if collect:
            return avg_loss, accuracy, np.array(all_y), np.array(all_pred), np.array(all_prob)
        return avg_loss, accuracy

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = run_eval(val_loader)
        scheduler.step(val_loss)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        HISTORY_PATH.write_text(json.dumps(history, indent=2))

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": train_ds.classes,
                "class_to_idx": train_ds.class_to_idx,
                "image_size": IMAGE_SIZE,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "best_val_loss": best_val_loss,
                "epoch": epoch,
            }, MODEL_PATH)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping: validation loss did not improve.")
            break

    # Evaluate the best validation-loss checkpoint on the test split.
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, y_true, y_pred, y_prob = run_eval(
        test_loader, collect=True
    )

    report = classification_report(
        y_true, y_pred, labels=list(range(len(test_ds.classes))),
        target_names=test_ds.classes, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(
        y_true, y_pred, labels=list(range(len(test_ds.classes)))
    )

    results = {
        "model": "ResNet18",
        "dataset": str(DATA_ROOT),
        "image_size": IMAGE_SIZE,
        "class_names": train_ds.classes,
        "train_images": len(train_ds),
        "validation_images": len(val_ds),
        "test_images": len(test_ds),
        "best_epoch": checkpoint["epoch"],
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "cross_split_exact_hash_matches": len(leakage),
        "training_seconds": round(time.time() - start_time, 2),
    }

    # AUC needs a probability column for each class; Cancer is class index 0.
    try:
        results["roc_auc_ovr"] = roc_auc_score(
            y_true, y_prob, multi_class="ovr", labels=list(range(len(test_ds.classes)))
        )
    except ValueError:
        results["roc_auc_ovr"] = None

    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print("\nTEST CLASSIFICATION REPORT")
    print(classification_report(
        y_true, y_pred, target_names=test_ds.classes, zero_division=0
    ))
    print("Confusion matrix:\n", matrix)
    print(f"Test accuracy: {test_acc:.4%}")
    print("Best checkpoint:", MODEL_PATH)
    print("Epoch history:", HISTORY_PATH)
    print("Test results:", RESULTS_PATH)
    print("Training seconds:", round(time.time() - start_time, 2))


if __name__ == "__main__":
    main()
