
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import DenseNet121_Weights
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "lung_xray_224" / "lung_xray_clean_512"
MODEL_PATH = BASE_DIR / "saved_models" / "lung" / "densenet121_224_best.pth"
RESULT_PATH = BASE_DIR / "saved_models" / "lung" / "densenet121_224_test_results.json"

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")
if not (DATA_DIR / "test").is_dir():
    raise FileNotFoundError(f"Test folder not found: {DATA_DIR / 'test'}")

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
classes = checkpoint["classes"]
print("Checkpoint classes:", classes)

weights = DenseNet121_Weights.DEFAULT
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std,
    ),
])

test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=test_transform)

if test_dataset.classes != classes:
    raise ValueError(
        f"Class order mismatch. Checkpoint: {classes}; "
        f"test dataset: {test_dataset.classes}"
    )

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
)

model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, len(classes))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(outputs.argmax(dim=1).cpu().tolist())
        all_probabilities.extend(probabilities.cpu().tolist())

# Class 0 is Cancer, as confirmed by the checkpoint.
cancer_index = classes.index("Cancer")
normal_index = classes.index("NORMAL")

y_true = all_labels
y_pred = all_predictions
cancer_scores = [row[cancer_index] for row in all_probabilities]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(
    y_true, y_pred, pos_label=cancer_index, zero_division=0
)
recall = recall_score(
    y_true, y_pred, pos_label=cancer_index, zero_division=0
)
f1 = f1_score(
    y_true, y_pred, pos_label=cancer_index, zero_division=0
)

# Explicit class order: Cancer, NORMAL
cm = confusion_matrix(y_true, y_pred, labels=[cancer_index, normal_index])
tp, fn = int(cm[0, 0]), int(cm[0, 1])
fp, tn = int(cm[1, 0]), int(cm[1, 1])
specificity = tn / (tn + fp) if (tn + fp) else 0.0

try:
    auc = roc_auc_score(
        [1 if label == cancer_index else 0 for label in y_true],
        cancer_scores,
    )
except ValueError:
    auc = None

results = {
    "model": "DenseNet121",
    "input_size": IMAGE_SIZE,
    "classes": classes,
    "test_images": len(test_dataset),
    "accuracy": accuracy,
    "cancer_precision": precision,
    "cancer_recall_sensitivity": recall,
    "cancer_f1_score": f1,
    "normal_specificity": specificity,
    "roc_auc": auc,
    "confusion_matrix_class_order": ["Cancer", "NORMAL"],
    "confusion_matrix": cm.tolist(),
    "classification_report": classification_report(
        y_true,
        y_pred,
        labels=[cancer_index, normal_index],
        target_names=["Cancer", "NORMAL"],
        output_dict=True,
        zero_division=0,
    ),
}

RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print("\n========== DENSENET121 (224x224) TEST RESULTS ==========")
print(f"Test images           : {len(test_dataset)}")
print(f"Accuracy               : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Cancer Precision      : {precision:.4f} ({precision * 100:.2f}%)")
print(f"Cancer Recall          : {recall:.4f} ({recall * 100:.2f}%)")
print(f"Cancer F1-score        : {f1:.4f} ({f1 * 100:.2f}%)")
print(f"NORMAL Specificity     : {specificity:.4f} ({specificity * 100:.2f}%)")
print(f"ROC-AUC                : {auc:.4f}" if auc is not None else "ROC-AUC                : N/A")
print("\nConfusion Matrix (rows=true, columns=predicted)")
print("Class order: Cancer, NORMAL")
print(cm)
print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[cancer_index, normal_index],
    target_names=["Cancer", "NORMAL"],
    zero_division=0,
))
print(f"\nResults saved to: {RESULT_PATH}")
