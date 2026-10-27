
from pathlib import Path
import json
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "lung_xray_224" / "lung_xray_clean_512"
MODEL_DIR = BASE_DIR / "saved_models" / "lung"
OUTPUT_DIR = BASE_DIR / "lung_model_analysis"

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ============================================================
# MODEL LOADING
# ============================================================

def build_model(architecture, num_classes):
    if architecture == "MobileNetV2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes,
        )

    elif architecture == "MobileNetV3-Large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features,
            num_classes,
        )

    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return model


def load_trained_model(architecture, checkpoint_path, num_classes):
    model = build_model(architecture, num_classes)

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# INTERNAL FEATURE-MAP VISUALIZATION
# ============================================================

def visualize_internal_layers(model, architecture, sample_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    image = eval_transform(
        datasets.folder.default_loader(str(sample_path))
    ).unsqueeze(0).to(DEVICE)

    activations = {}
    hooks = []

    conv_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    ]

    if conv_layers:
        selected = [
            ("early_convolution", conv_layers[0]),
            ("middle_convolution", conv_layers[len(conv_layers) // 2]),
            ("late_convolution", conv_layers[-1]),
        ]

        seen = set()

        for label, (module_name, module) in selected:
            if id(module) in seen:
                continue
            seen.add(id(module))

            def make_hook(layer_name):
                def hook(mod, inputs, output):
                    if isinstance(output, torch.Tensor):
                        activations[layer_name] = output.detach().cpu()
                return hook

            hooks.append(
                module.register_forward_hook(make_hook(label))
            )

    pooling_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.AdaptiveAvgPool2d)
    ]

    for index, (module_name, module) in enumerate(pooling_modules):
        label = f"average_pooling_{index + 1}"

        def make_pool_hook(layer_name):
            def hook(mod, inputs, output):
                if isinstance(output, torch.Tensor):
                    activations[layer_name] = output.detach().cpu()
            return hook

        hooks.append(
            module.register_forward_hook(make_pool_hook(label))
        )

    try:
        with torch.inference_mode():
            model(image)
    finally:
        for hook in hooks:
            hook.remove()

    shapes = {}

    for layer_name, activation in activations.items():
        shapes[layer_name] = list(activation.shape)

        if activation.ndim != 4:
            continue

        feature_maps = activation[0]
        num_channels = min(16, feature_maps.shape[0])

        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        fig.suptitle(
            f"{architecture}: {layer_name}\n"
            f"Activation shape: {tuple(activation.shape)}",
            fontsize=11,
        )

        for i, ax in enumerate(axes.flat):
            ax.axis("off")

            if i < num_channels:
                feature = feature_maps[i].numpy()
                low = feature.min()
                high = feature.max()

                if high > low:
                    feature = (feature - low) / (high - low)

                ax.imshow(feature, cmap="viridis")
                ax.set_title(f"Channel {i}", fontsize=8)

        fig.tight_layout()
        fig.savefig(
            str(out_dir / f"{layer_name}.png"),
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)

    with open(
        out_dir / "activation_shapes.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(shapes, f, indent=2)

    print("Internal-layer outputs saved to:", out_dir)


# ============================================================
# EVALUATION AND CLASSIFICATION REPORT
# ============================================================

def analyze_model(architecture, checkpoint_path, test_dir, out_dir):
    print("\n" + "=" * 60)
    print("Analyzing:", architecture)
    print("=" * 60)

    if not checkpoint_path.exists():
        print("Checkpoint not found; skipping:", checkpoint_path)
        return None

    # Prepare test dataset and output folder.
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.ImageFolder(
        test_dir,
        transform=eval_transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    classes = dataset.classes

    model = load_trained_model(
        architecture,
        checkpoint_path,
        len(classes),
    )

    y_true = []
    y_pred = []

    # Run predictions on the held-out test set.
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(DEVICE)

            with torch.autocast(
                device_type=DEVICE.type,
                enabled=(DEVICE.type == "cuda"),
            ):
                outputs = model(images)

            predictions = outputs.argmax(dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(predictions.cpu().tolist())

    # Calculate evaluation metrics.
    accuracy = accuracy_score(y_true, y_pred)

    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0,
        output_dict=True,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
    )

    # Display results in the terminal.
    print(f"Test images: {len(dataset)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(report_text)
    print("Confusion matrix:")
    print(cm)

    # Save classification report as TXT.
    (out_dir / "classification_report.txt").write_text(
        f"Architecture: {architecture}\n"
        f"Accuracy: {accuracy:.6f}\n\n"
        f"{report_text}",
        encoding="utf-8",
    )

    # Save classification report as CSV.
    with open(
        out_dir / "classification_report.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "Class",
            "Precision",
            "Recall",
            "F1-score",
            "Support",
        ])

        for name, values in report_dict.items():
            if isinstance(values, dict) and "precision" in values:
                writer.writerow([
                    name,
                    values["precision"],
                    values["recall"],
                    values["f1-score"],
                    values["support"],
                ])

    # Save raw confusion matrix as CSV.
    with open(
        out_dir / "confusion_matrix.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["Actual / Predicted"] + classes)

        for name, row in zip(classes, cm):
            writer.writerow([name] + row.tolist())

    # Create and save ONE confusion-matrix heatmap.
    fig, ax = plt.subplots(figsize=(7, 6))
    heatmap = ax.imshow(cm, cmap="Blues")

    ax.set_title(f"{architecture} - Confusion Matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_xticks(range(len(classes)), classes, rotation=20)
    ax.set_yticks(range(len(classes)), classes)

    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            ax.text(
                c,
                r,
                str(cm[r, c]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[r, c] > cm.max() / 2
                    else "black"
                ),
            )

    fig.colorbar(heatmap, ax=ax)
    fig.tight_layout()

    try:
        fig.savefig(
            str(out_dir / "confusion_matrix_heatmap.png"),
            dpi=200,
            bbox_inches="tight",
        )
    finally:
        plt.close(fig)

    # Visualize internal activations using one test image.
    sample_path = next(
        (test_dir / classes[0]).glob("*"),
        None,
    )

    if sample_path is not None:
        visualize_internal_layers(
            model,
            architecture,
            sample_path,
            out_dir / "feature_maps",
        )

    # Save results in JSON.
    results = {
        "architecture": architecture,
        "test_images": len(dataset),
        "classes": classes,
        "accuracy": float(accuracy),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
    }

    with open(
        out_dir / "test_results.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f, indent=2)

    print("Results saved to:", out_dir)

    return results


# ============================================================
# RUN BOTH MODELS
# ============================================================

def main():
    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    test_dir = DATA_DIR / "test"

    if not test_dir.is_dir():
        raise FileNotFoundError(
            f"Test dataset not found: {test_dir}"
        )

    model_configs = [
        (
            "MobileNetV2",
            MODEL_DIR / "mobilenetv2_224_best.pth",
            OUTPUT_DIR / "mobilenetv2",
        ),
        (
            "MobileNetV3-Large",
            MODEL_DIR / "mobilenetv3_large_224_best.pth",
            OUTPUT_DIR / "mobilenetv3_large",
        ),
    ]

    comparison = {}

    for architecture, checkpoint, out_dir in model_configs:
        result = analyze_model(
            architecture,
            checkpoint,
            test_dir,
            out_dir,
        )

        if result is not None:
            comparison[architecture] = {
                "accuracy": result["accuracy"],
                "test_images": result["test_images"],
                "classes": result["classes"],
            }

    if comparison:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        with open(
            OUTPUT_DIR / "mobilenet_model_comparison.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(comparison, f, indent=2)

        print("\nMODEL COMPARISON")

        for name, values in comparison.items():
            print(
                f"{name}: accuracy = "
                f"{values['accuracy']:.4f}"
            )

    print("\nAnalysis finished.")
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
