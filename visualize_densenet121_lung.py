
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.models import DenseNet121_Weights
from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "lung_xray_224" / "lung_xray_clean_512"
TEST_DIR = DATA_DIR / "test"

CHECKPOINT_PATH = (
    BASE_DIR / "saved_models" / "lung" / "densenet121_224_best.pth"
)

OUTPUT_DIR = BASE_DIR / "lung_model_analysis" / "densenet121"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 4
NUM_WORKERS = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

if not TEST_DIR.is_dir():
    raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")

# --------------------------------------------------
# LOAD MODEL CHECKPOINT
# --------------------------------------------------
try:
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
except TypeError:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

classes = checkpoint["classes"]
print("Classes:", classes)

weights = DenseNet121_Weights.DEFAULT
preprocess = weights.transforms()

# Match training normalization.
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=preprocess.mean, std=preprocess.std),
])

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=eval_transform,
)

if test_dataset.classes != classes:
    raise ValueError(
        f"Class order mismatch: dataset={test_dataset.classes}, "
        f"checkpoint={classes}"
    )

model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, len(classes))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

print("Test images:", len(test_dataset))

# --------------------------------------------------
# 1. CLASSIFICATION REPORT
# --------------------------------------------------
y_true = []
y_pred = []

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu().tolist()

        y_true.extend(labels.tolist())
        y_pred.extend(predictions)

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(classes))),
    target_names=classes,
    digits=4,
    zero_division=0,
)

print("\nCLASSIFICATION REPORT\n")
print(report)

(OUTPUT_DIR / "classification_report.txt").write_text(
    report, encoding="utf-8"
)

report_dict = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(classes))),
    target_names=classes,
    output_dict=True,
    zero_division=0,
)

pd.DataFrame(report_dict).transpose().to_csv(
    OUTPUT_DIR / "classification_report.csv"
)

accuracy = accuracy_score(y_true, y_pred)

# --------------------------------------------------
# 2. CONFUSION MATRIX HEATMAP
# --------------------------------------------------
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(classes))),
)

pd.DataFrame(
    cm, index=classes, columns=classes
).to_csv(OUTPUT_DIR / "confusion_matrix.csv")

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes,
)
plt.title(f"DenseNet121 Confusion Matrix | Accuracy={accuracy:.4f}")
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200)
plt.close()

# --------------------------------------------------
# 3. INTERNAL CONVOLUTION AND POOLING OUTPUTS
# --------------------------------------------------
# Visualize one representative test image.
image_tensor, true_label = test_dataset[0]
batch = image_tensor.unsqueeze(0).to(device)

activations = {}
hooks = []

def capture_activation(name):
    def hook(module, inputs, output):
        activations[name] = output.detach().cpu()
    return hook

# Initial convolution, initial pooling, and later pooling.
hooks.append(
    model.features.conv0.register_forward_hook(
        capture_activation("initial_convolution")
    )
)
hooks.append(
    model.features.pool0.register_forward_hook(
        capture_activation("initial_max_pooling")
    )
)
hooks.append(
    model.features.transition1.pool.register_forward_hook(
        capture_activation("transition1_average_pooling")
    )
)
hooks.append(
    model.features.transition2.pool.register_forward_hook(
        capture_activation("transition2_average_pooling")
    )
)
hooks.append(
    model.features.transition3.pool.register_forward_hook(
        capture_activation("transition3_average_pooling")
    )
)

try:
    with torch.inference_mode():
        logits = model(batch)
        predicted_label = logits.argmax(dim=1).item()
finally:
    for hook in hooks:
        hook.remove()

# Record shapes and prediction for reproducibility.
metadata = {
    "test_image": str(test_dataset.samples[0][0]),
    "true_class": classes[true_label],
    "predicted_class": classes[predicted_label],
    "activation_shapes": {
        name: list(tensor.shape)
        for name, tensor in activations.items()
    },
}

(OUTPUT_DIR / "layer_output_shapes.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)

print("\nExample test image")
print("True class:", metadata["true_class"])
print("Predicted class:", metadata["predicted_class"])
print("\nInternal activation shapes:")

for name, tensor in activations.items():
    print(f"{name}: {tuple(tensor.shape)}")

# Save a grid of the first 8 activation channels per layer.
for name, tensor in activations.items():
    feature_maps = tensor[0]
    count = min(8, feature_maps.shape[0])

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < count:
            feature = feature_maps[i].numpy()
            ax.imshow(feature, cmap="viridis")
            ax.set_title(f"Channel {i}")

    fig.suptitle(f"DenseNet121 - {name}")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / f"{name}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

# Save a human-readable summary.
summary = {
    "architecture": "DenseNet121",
    "test_images": len(test_dataset),
    "accuracy": float(accuracy),
    "classes": classes,
    "results_directory": str(OUTPUT_DIR),
}

(OUTPUT_DIR / "summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)

print("\nAnalysis completed successfully.")
print("Accuracy:", f"{accuracy:.4f}")
print("Saved results to:", OUTPUT_DIR)
