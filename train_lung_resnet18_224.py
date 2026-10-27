from pathlib import Path
import json
import time
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

# Project root: this script is expected to be saved in the project root.
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "lung_xray_224" / "lung_xray_clean_512"
SAVE_DIR = ROOT / "saved_models" / "lung"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
NUM_WORKERS = 0  # Windows-friendly
MODEL_PATH = SAVE_DIR / "resnet18_224_best.pth"
RESULTS_PATH = SAVE_DIR / "resnet18_224_test_results.json"
HISTORY_PATH = SAVE_DIR / "resnet18_224_training_history.json"

def main():
    if not all((DATA_DIR / split).is_dir() for split in ("train", "val", "test")):
        raise FileNotFoundError(
            f"Could not find train/val/test under {DATA_DIR}\n"
            "Check that this script is saved in your project root."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Keep the same 224x224 ImageNet normalization commonly used by pretrained models.
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=7),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
    val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=eval_transform)
    test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=eval_transform)

    classes = train_ds.classes
    if classes != val_ds.classes or classes != test_ds.classes:
        raise ValueError(
            f"Class folders differ between splits: train={classes}, "
            f"val={val_ds.classes}, test={test_ds.classes}"
        )
    if "Cancer" not in classes:
        raise ValueError(f"Expected a 'Cancer' class folder, found: {classes}")
    cancer_index = classes.index("Cancer")

    print(f"Classes (ImageFolder order): {classes}")
    print(f"Images: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print("Class counts (train):",
          {name: sum(1 for _, y in train_ds.samples if y == i)
           for i, name in enumerate(classes)})

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))

    # Transfer learning from ImageNet. The first run may download weights if not cached.
    try:
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception as exc:
        print(f"Could not load pretrained ImageNet weights ({exc}). Using random initialization.")
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    best_state = None
    history = []
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * labels.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item() * labels.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        train_loss = train_loss_sum / max(train_total, 1)
        val_loss = val_loss_sum / max(val_total, 1)
        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        scheduler.step(val_loss)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": best_state,
                "classes": classes,
                "image_size": IMAGE_SIZE,
                "best_val_loss": best_val_loss,
                "epoch": epoch,
                "architecture": "resnet18",
            }, MODEL_PATH)
            print(f"  Saved best checkpoint: {MODEL_PATH}")

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Evaluate only the best validation-loss checkpoint on the held-out test split.
    model.load_state_dict(best_state)
    model.eval()
    y_true, y_pred, cancer_scores = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1).cpu().tolist()
            y_pred.extend(predictions)
            y_true.extend(labels.tolist())
            cancer_scores.extend(probabilities[:, cancer_index].cpu().tolist())

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    # For class order ['Cancer', 'NORMAL'], TN/FP/FN/TP are interpreted
    # relative to Cancer as the positive class, not sklearn's default label order.
    cancer_binary_true = [1 if y == cancer_index else 0 for y in y_true]
    cancer_binary_pred = [1 if y == cancer_index else 0 for y in y_pred]
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(cancer_binary_true, cancer_binary_pred, zero_division=0)
    recall = recall_score(cancer_binary_true, cancer_binary_pred, zero_division=0)
    f1 = f1_score(cancer_binary_true, cancer_binary_pred, zero_division=0)
    specificity = recall_score(
        [1 - y for y in cancer_binary_true],
        [1 - y for y in cancer_binary_pred],
        zero_division=0,
    )
    try:
        auc = roc_auc_score(cancer_binary_true, cancer_scores)
    except ValueError:
        auc = None

    results = {
        "model": "ResNet18",
        "device": str(device),
        "classes": classes,
        "positive_class_for_cancer_metrics": "Cancer",
        "image_size": IMAGE_SIZE,
        "test_images": len(test_ds),
        "best_epoch": int(min(history, key=lambda x: x["val_loss"])["epoch"]),
        "accuracy": accuracy,
        "cancer_precision": precision,
        "cancer_recall_sensitivity": recall,
        "cancer_f1_score": f1,
        "specificity_normal_recall": specificity,
        "roc_auc_cancer": auc,
        "confusion_matrix_class_order": classes,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=list(range(len(classes))),
            target_names=classes, output_dict=True, zero_division=0
        ),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "note": "Research prototype only; not for clinical diagnosis."
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n===== RESNET18 TEST RESULTS =====")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Cancer precision:  {precision:.4f}")
    print(f"Cancer recall:     {recall:.4f}")
    print(f"Cancer F1-score:   {f1:.4f}")
    print(f"Specificity:       {specificity:.4f}")
    print(f"ROC-AUC:           {auc if auc is not None else 'not available'}")
    print(f"Confusion matrix (class order {classes}):\n{cm}")
    print(f"\nCheckpoint: {MODEL_PATH}")
    print(f"Test results: {RESULTS_PATH}")
    print(f"Training history: {HISTORY_PATH}")
    print("\nImportant: compare against the other models only if they use the same test images and compatible preprocessing.")

if __name__ == "__main__":
    main()
