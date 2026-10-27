
from pathlib import Path
import csv
import json

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from torchvision import datasets, transforms, models
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

TEST_DIR = (
    PROJECT_DIR
    / "lung_xray_224"
    / "lung_xray_clean_512"
    / "test"
)

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "saved_models"
    / "lung"
    / "efficientnet_b0_224_best.pth"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "lung_model_analysis"
    / "efficientnet_b0"
)

FEATURE_DIR = OUTPUT_DIR / "feature_maps"

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0
MAX_FEATURE_MAPS = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ImageNet normalization used during training
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ============================================================
# 2. CREATE OUTPUT DIRECTORIES AND CHECK PATHS
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test dataset not found:\n{TEST_DIR}"
    )

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Trained checkpoint not found:\n{CHECKPOINT_PATH}"
    )

print("=" * 65)
print("EFFICIENTNET-B0 MODEL ANALYSIS")
print("=" * 65)
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 3. LOAD TEST DATA
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform,
)

classes = test_dataset.classes

print("\nClasses:", classes)
print("Class mapping:", test_dataset.class_to_idx)
print("Test images:", len(test_dataset))

if len(classes) != 2:
    raise ValueError(
        f"Expected 2 classes, found: {classes}"
    )

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)


# ============================================================
# 4. LOAD TRAINED EFFICIENTNET-B0
# ============================================================

print("\nLoading trained EfficientNet-B0...")

model = models.efficientnet_b0(weights=None)

input_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    input_features,
    len(classes),
)

state_dict = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True,
)

model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()

print("Trained checkpoint loaded successfully.")


# ============================================================
# 5. CLASSIFICATION REPORT AND CONFUSION MATRIX
# ============================================================

print("\nEvaluating the held-out test set...")

all_true = []
all_pred = []
all_probabilities = []

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(DEVICE, non_blocking=True)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = outputs.argmax(dim=1)

        all_true.extend(labels.tolist())
        all_pred.extend(predictions.cpu().tolist())
        all_probabilities.extend(
            probabilities.cpu().tolist()
        )

accuracy = accuracy_score(all_true, all_pred)

report = classification_report(
    all_true,
    all_pred,
    labels=list(range(len(classes))),
    target_names=classes,
    digits=4,
    zero_division=0,
)

cm = confusion_matrix(
    all_true,
    all_pred,
    labels=list(range(len(classes))),
)

print("\n" + "=" * 65)
print("TEST CLASSIFICATION REPORT")
print("=" * 65)
print(report)
print("Test accuracy:", round(accuracy, 4))
print("\nConfusion matrix:\n", cm)

# Save text report
with open(
    OUTPUT_DIR / "test_classification_report.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write("EfficientNet-B0 - Test Classification Report\n\n")
    f.write(f"Test images: {len(test_dataset)}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(report)
    f.write("\n\nConfusion Matrix:\n")
    f.write(str(cm))

# Save confusion matrix as CSV
with open(
    OUTPUT_DIR / "test_confusion_matrix.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.writer(f)
    writer.writerow(["Actual / Predicted"] + classes)

    for class_name, row in zip(classes, cm):
        writer.writerow([class_name] + row.tolist())

# Save per-image predictions
with open(
    OUTPUT_DIR / "test_predictions.csv",
    "w",
    newline="",
    encoding="utf-8",
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "image_path",
        "actual_class",
        "predicted_class",
        "probability_Cancer",
        "probability_NORMAL",
        "correct",
    ])

    for i, (true_label, pred_label, probs) in enumerate(
        zip(all_true, all_pred, all_probabilities)
    ):
        image_path = test_dataset.samples[i][0]

        writer.writerow([
            image_path,
            classes[true_label],
            classes[pred_label],
            round(probs[classes.index("Cancer")], 6)
                if "Cancer" in classes else "",
            round(probs[classes.index("NORMAL")], 6)
                if "NORMAL" in classes else "",
            true_label == pred_label,
        ])


# ============================================================
# 6. CONFUSION MATRIX HEATMAP
# ============================================================

fig, ax = plt.subplots(figsize=(7, 6))

image = ax.imshow(cm, cmap="Blues")
fig.colorbar(image, ax=ax, label="Number of images")

ax.set(
    xticks=np.arange(len(classes)),
    yticks=np.arange(len(classes)),
    xticklabels=classes,
    yticklabels=classes,
    xlabel="Predicted class",
    ylabel="Actual class",
    title="EfficientNet-B0 - Test Confusion Matrix",
)

threshold = cm.max() / 2 if cm.size else 0

for row in range(cm.shape[0]):
    for col in range(cm.shape[1]):
        ax.text(
            col,
            row,
            str(cm[row, col]),
            ha="center",
            va="center",
            color="white" if cm[row, col] > threshold else "black",
            fontsize=13,
        )

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "test_confusion_matrix_heatmap.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

print("\nHeatmap saved.")


# ============================================================
# 7. CAPTURE INTERNAL CONVOLUTION AND POOLING OUTPUTS
# ============================================================

# Choose convolution layers ONLY from the feature extractor.
# This avoids accidentally selecting classifier layers.
conv_layers = [
    (name, module)
    for name, module in model.features.named_modules()
    if isinstance(module, nn.Conv2d)
]

if len(conv_layers) < 3:
    raise RuntimeError("Could not find enough convolution layers.")

selected_layers = [
    ("early_convolution", *conv_layers[0]),
    ("middle_convolution", *conv_layers[len(conv_layers) // 2]),
    ("late_convolution", *conv_layers[-1]),
]

# EfficientNet-B0 has an AdaptiveAvgPool2d layer called avgpool.
selected_layers.append(
    ("adaptive_average_pooling", "avgpool", model.avgpool)
)

captured_outputs = {}
hooks = []


def make_hook(layer_name):
    def hook_fn(module, inputs, output):
        captured_outputs[layer_name] = output.detach().cpu()
    return hook_fn


for label, layer_name, layer in selected_layers:
    hooks.append(
        layer.register_forward_hook(make_hook(label))
    )
    print(f"Hook registered: {label} -> {layer_name}")


# Use the first test image to inspect intermediate activations.
sample_image, sample_label = test_dataset[0]
sample_batch = sample_image.unsqueeze(0).to(DEVICE)

with torch.inference_mode():
    sample_logits = model(sample_batch)
    sample_probs = torch.softmax(sample_logits, dim=1)[0].cpu()

for hook in hooks:
    hook.remove()

predicted_label = int(sample_probs.argmax())

print("\nSample image:", test_dataset.samples[0][0])
print("Actual class:", classes[sample_label])
print("Predicted class:", classes[predicted_label])
print("Output probabilities:")

for class_name, probability in zip(classes, sample_probs.tolist()):
    print(f"  {class_name}: {probability:.4f}")


# ============================================================
# 8. SAVE ACTIVATION SHAPES AND FEATURE MAP IMAGES
# ============================================================

activation_shapes = {}

for label, activation in captured_outputs.items():
    activation_shapes[label] = list(activation.shape)

    # Conv outputs usually have shape [1, channels, height, width].
    # Pool output has shape [1, channels, 1, 1].
    if activation.ndim != 4:
        print("Skipping unexpected output shape:", label, activation.shape)
        continue

    feature_maps = activation[0]
    number_of_channels = feature_maps.shape[0]
    number_to_plot = min(MAX_FEATURE_MAPS, number_of_channels)

    cols = 4
    rows = int(np.ceil(number_to_plot / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(12, 3 * rows),
    )
    axes = np.array(axes).reshape(-1)

    for channel in range(number_to_plot):
        feature_map = feature_maps[channel].numpy()
        axes[channel].imshow(feature_map, cmap="viridis")
        axes[channel].set_title(f"Channel {channel}")
        axes[channel].axis("off")

    for unused in range(number_to_plot, len(axes)):
        axes[unused].axis("off")

    fig.suptitle(
        f"{label}\nActivation shape: {tuple(activation.shape)}",
        fontsize=12,
    )
    fig.tight_layout()

    fig.savefig(
        FEATURE_DIR / f"{label}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

with open(
    FEATURE_DIR / "activation_shapes.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(activation_shapes, f, indent=4)

print("\nInternal activation shapes:")
for name, shape in activation_shapes.items():
    print(f"{name}: {shape}")


# ============================================================
# 9. SAVE SUMMARY
# ============================================================

summary = {
    "model": "EfficientNet-B0",
    "checkpoint": str(CHECKPOINT_PATH),
    "evaluation_split": "test",
    "test_images": len(test_dataset),
    "classes": classes,
    "accuracy": accuracy,
    "confusion_matrix": cm.tolist(),
    "activation_shapes": activation_shapes,
    "sample_actual_class": classes[sample_label],
    "sample_predicted_class": classes[predicted_label],
    "sample_probabilities": {
        name: float(prob)
        for name, prob in zip(classes, sample_probs.tolist())
    },
}

with open(
    OUTPUT_DIR / "analysis_summary.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(summary, f, indent=4)

print("\n" + "=" * 65)
print("ANALYSIS COMPLETED")
print("=" * 65)
print("Reports and heatmap:", OUTPUT_DIR)
print("Feature maps:", FEATURE_DIR)
