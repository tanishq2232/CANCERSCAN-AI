
from pathlib import Path
import copy
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

# ---------------- CONFIGURATION ----------------

DATA_DIR = Path("lung_xray_224") / "lung_xray_clean_512"
SAVE_DIR = Path("saved_models") / "lung"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-4
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Check dataset folders
    for split in ["train", "val", "test"]:
        if not (DATA_DIR / split).exists():
            raise FileNotFoundError(
                f"Dataset folder not found: {DATA_DIR / split}"
            )

    # ---------------- TRANSFORMS ----------------

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

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_ds = datasets.ImageFolder(
        DATA_DIR / "train", transform=train_transform
    )
    val_ds = datasets.ImageFolder(
        DATA_DIR / "val", transform=eval_transform
    )
    test_ds = datasets.ImageFolder(
        DATA_DIR / "test", transform=eval_transform
    )

    if train_ds.classes != val_ds.classes or train_ds.classes != test_ds.classes:
        raise ValueError("Train, validation and test classes do not match.")

    print("Classes:", train_ds.classes)
    print("Class mapping:", train_ds.class_to_idx)
    print(
        f"Images: train={len(train_ds)}, "
        f"val={len(val_ds)}, test={len(test_ds)}"
    )
    print(f"Input image: {IMAGE_SIZE} x {IMAGE_SIZE} x 3")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda")
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda")
    )

    # ---------------- RESNET50 ----------------

    try:
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        print("Loaded pretrained ImageNet weights.")
    except Exception as e:
        print("Could not download pretrained weights:", e)
        print("Using randomly initialized weights.")
        model = models.resnet50(weights=None)

    # Freeze the backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace final classification layer
    model.fc = nn.Linear(
        model.fc.in_features,
        len(train_ds.classes)
    )

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.fc.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    best_state = None
    history = []
    start_time = time.time()

    # ---------------- TRAINING ----------------

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0
        train_true, train_pred = [], []

        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=DEVICE.type, enabled=use_amp
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item() * images.size(0)
            train_true.extend(labels.cpu().tolist())
            train_pred.extend(outputs.argmax(1).detach().cpu().tolist())

        train_loss = train_loss_sum / len(train_ds)
        train_acc = accuracy_score(train_true, train_pred)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_true, val_pred = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                with torch.autocast(
                    device_type=DEVICE.type, enabled=use_amp
                ):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss_sum += loss.item() * images.size(0)
                val_true.extend(labels.cpu().tolist())
                val_pred.extend(outputs.argmax(1).cpu().tolist())

        val_loss = val_loss_sum / len(val_ds)
        val_acc = accuracy_score(val_true, val_pred)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        })

        # Save best model using validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

            torch.save({
                "model_name": "ResNet50",
                "image_size": IMAGE_SIZE,
                "classes": train_ds.classes,
                "class_to_idx": train_ds.class_to_idx,
                "state_dict": best_state,
                "best_val_loss": best_val_loss,
            }, SAVE_DIR / "resnet50_224_best.pth")

            print("  Best model saved.")

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---------------- TEST EVALUATION ----------------

    model.eval()
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE, non_blocking=True)

            with torch.autocast(
                device_type=DEVICE.type, enabled=use_amp
            ):
                outputs = model(images)

            probabilities = torch.softmax(outputs.float(), dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(probabilities.argmax(1).cpu().tolist())
            y_probs.extend(probabilities.cpu().tolist())

    # Calculate metrics treating Cancer as the positive class
    if "Cancer" not in train_ds.class_to_idx:
        raise ValueError(
            "Expected a folder named 'Cancer'. Classes: "
            + str(train_ds.classes)
        )

    cancer_idx = train_ds.class_to_idx["Cancer"]

    actual_cancer = [int(y == cancer_idx) for y in y_true]
    predicted_cancer = [int(y == cancer_idx) for y in y_pred]
    cancer_probabilities = [p[cancer_idx] for p in y_probs]

    tn, fp, fn, tp = confusion_matrix(
        actual_cancer, predicted_cancer, labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    try:
        auc = roc_auc_score(actual_cancer, cancer_probabilities)
    except ValueError:
        auc = None

    results = {
        "model": "ResNet50",
        "image_size": "224x224",
        "test_images": len(test_ds),
        "classes": train_ds.classes,
        "class_to_idx": train_ds.class_to_idx,
        "accuracy": accuracy_score(y_true, y_pred),
        "cancer_precision": precision_score(
            actual_cancer, predicted_cancer, zero_division=0
        ),
        "cancer_recall_sensitivity": recall_score(
            actual_cancer, predicted_cancer, zero_division=0
        ),
        "cancer_f1": f1_score(
            actual_cancer, predicted_cancer, zero_division=0
        ),
        "specificity": specificity,
        "roc_auc": auc,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(train_ds.classes))),
            target_names=train_ds.classes,
            output_dict=True,
            zero_division=0,
        ),
        "note": "Experimental model metrics, not clinical validation.",
    }

    with open(
        SAVE_DIR / "resnet50_224_test_results.json",
        "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=2)

    with open(
        SAVE_DIR / "resnet50_224_training_history.json",
        "w", encoding="utf-8"
    ) as f:
        json.dump(history, f, indent=2)

    print("\n========== RESNET50 TEST RESULTS ==========")
    print(f"Accuracy:     {results['accuracy']:.4f}")
    print(f"Cancer precision: {results['cancer_precision']:.4f}")
    print(f"Cancer recall:    {results['cancer_recall_sensitivity']:.4f}")
    print(f"Cancer F1-score:  {results['cancer_f1']:.4f}")
    print(f"Specificity:      {results['specificity']:.4f}")
    print(f"ROC-AUC:          {results['roc_auc']}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(results["confusion_matrix"])

    print("\nSaved model:", SAVE_DIR / "resnet50_224_best.pth")
    print("Saved metrics:", SAVE_DIR / "resnet50_224_test_results.json")
    print(f"Time taken: {(time.time() - start_time) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
