
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
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# ============================================================
# PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "lung_xray_224" / "lung_xray_clean_512"
MODEL_DIR = BASE_DIR / "saved_models" / "lung"
OUTPUT_DIR = BASE_DIR / "lung_model_analysis"

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet normalization used by both pretrained architectures.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ============================================================
# MODEL DEFINITIONS
# ============================================================

def load_model(architecture, checkpoint_path, num_classes):
    if architecture == "MobileNetV2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )
    elif architecture == "DenseNet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(
            model.classifier.in_features, num_classes
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )

    # Supports checkpoints saved as dictionaries or raw state dictionaries.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def get_layer_modules(model, architecture):
    # These are named internal modules from torchvision models.
    if architecture == "MobileNetV2":
        names = {
            "initial_convolution": "features.0.0",
            "early_depthwise_convolution": "features.1.conv.0.0",
            "middle_convolution": "features.4.conv.0.0",
            "late_convolution": "features.14.conv.0.0",
            "final_convolution": "features.18.0",
            "global_average_pooling": "features",
            "classifier": "classifier",
        }
    else:
        names = {
            "initial_convolution": "features.conv0",
            "initial_max_pooling": "features.pool0",
            "early_dense_convolution": "features.denseblock1.denselayer1.conv1",
            "transition1_average_pooling": "features.transition1.pool",
            "middle_dense_convolution": "features.denseblock2.denselayer1.conv1",
            "transition2_average_pooling": "features.transition2.pool",
            "final_convolutional_features": "features.norm5",
        }

    modules = {}
    for label, module_name in names.items():
        try:
            modules[label] = model.get_submodule(module_name)
        except AttributeError:
            print(f"Skipping unavailable layer: {module_name}")

    return modules


# ============================================================
# FEATURE MAP VISUALIZATION
# ============================================================

def save_feature_maps(model, architecture, image_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    image = transform(
        datasets.folder.default_loader(str(image_path))
    ).unsqueeze(0).to(DEVICE)

    modules = get_layer_modules(model, architecture)
    activations = {}
    hooks = []

    def make_hook(layer_name):
        def hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                activations[layer_name] = output.detach().cpu()
        return hook

    for layer_name, module in modules.items():
        hooks.append(
            module.register_forward_hook(make_hook(layer_name))
        )

    try:
        with torch.inference_mode():
            _ = model(image)
    finally:
        for hook in hooks:
            hook.remove()

    # Save activation dimensions and visualize up to 16 channels per layer.
    shapes = {}

    for layer_name, activation in activations.items():
        shapes[layer_name] = list(activation.shape)

        if activation.ndim != 4:
            continue

        feature_maps = activation[0]
        channel_count = min(16, feature_maps.shape[0])

        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        fig.suptitle(
            f"{architecture}: {layer_name}\n"
            f"Activation shape: {tuple(activation.shape)}",
            fontsize=11,
        )

        for index, ax in enumerate(axes.flat):
            ax.axis("off")

            if index < channel_count:
                feature = feature_maps[index].numpy()
                low, high = feature.min(), feature.max()

                if high > low:
                    feature = (feature - low) / (high - low)

                ax.imshow(feature, cmap="viridis")
                ax.set_title(f"Channel {index}", fontsize=8)

        plt.tight_layout()
        plt.savefig(
            output_dir / f"{layer_name}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    with open(output_dir / "activation_shapes.json", "w") as f:
        json.dump(shapes, f, indent=2)

    print(f"Feature maps saved in: {output_dir}")


# ============================================================
# CLASSIFICATION REPORT AND CONFUSION MATRIX
# ============================================================

def evaluate_model(architecture, checkpoint_path, test_dir, output_dir):
    print("\n" + "=" * 60)
    print(f"Evaluating {architecture}")
    print("=" * 60)

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Skipping this model.")
        return None

    if not test_dir.exists():
        raise FileNotFoundError(f"Test folder not found: {test_dir}")

    dataset = datasets.ImageFolder(test_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    class_names = dataset.classes
    model = load_model(
        architecture, checkpoint_path, len(class_names)
    )

    y_true, y_pred = [], []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(DEVICE)
            logits = model(images)
            predictions = logits.argmax(dim=1).cpu().tolist()

            y_true.extend(labels.tolist())
            y_pred.extend(predictions)

    accuracy = accuracy_score(y_true, y_pred)
    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Test images: {len(dataset)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(report_text)

    (output_dir / "classification_report.txt").write_text(
        f"Model: {architecture}\n"
        f"Accuracy: {accuracy:.6f}\n\n{report_text}",
        encoding="utf-8",
    )

    with open(output_dir / "classification_report.csv", "w",
              newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Precision", "Recall", "F1-score", "Support"])
        for name, values in report_dict.items():
            if isinstance(values, dict) and "precision" in values:
                writer.writerow([
                    name,
                    values["precision"],
                    values["recall"],
                    values["f1-score"],
                    values["support"],
                ])

    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(len(class_names)))
    )

    # Save the raw confusion matrix.
    with open(output_dir / "confusion_matrix.csv", "w",
              newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Actual / Predicted"] + class_names)
        for name, row in zip(class_names, cm):
            writer.writerow([name] + row.tolist())

    # Draw and save the heatmap using Matplotlib.
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{architecture} - Confusion Matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_xticks(range(len(class_names)), class_names, rotation=20)
    ax.set_yticks(range(len(class_names)), class_names)

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(
                col, row, str(cm[row, col]),
                ha="center", va="center",
                color="white" if cm[row, col] > cm.max() / 2 else "black",
            )

    fig.colorbar(image, ax=ax)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix_heatmap.png", dpi=200)
    plt.close(fig)

    # Use one test image to visualize internal activations.
    sample_path = next(
        (test_dir / class_names[0]).glob("*"), None
    )
    if sample_path is not None:
        save_feature_maps(
            model,
            architecture,
            sample_path,
            output_dir / "feature_maps",
        )

    results = {
        "architecture": architecture,
        "test_images": len(dataset),
        "classes": class_names,
        "accuracy": accuracy,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
    }

    with open(output_dir / "test_results.json", "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


# ============================================================
# RUN BOTH MODELS
# ============================================================

def main():
    print("Lung X-ray model analysis")
    print(f"Device: {DEVICE}")

    test_dir = DATA_DIR / "test"

    models_to_evaluate = [
        (
            "DenseNet121",
            MODEL_DIR / "densenet121_224_best.pth",
            OUTPUT_DIR / "densenet121",
        ),
        (
            "MobileNetV2",
            MODEL_DIR / "mobilenetv2_224_best.pth",
            OUTPUT_DIR / "mobilenetv2",
        ),
    ]

    all_results = []

    for architecture, checkpoint, output_dir in models_to_evaluate:
        result = evaluate_model(
            architecture, checkpoint, test_dir, output_dir
        )
        if result is not None:
            all_results.append(result)

    if all_results:
        comparison = {
            item["architecture"]: {
                "accuracy": item["accuracy"],
                "test_images": item["test_images"],
                "classes": item["classes"],
            }
            for item in all_results
        }

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_DIR / "model_comparison.json", "w",
                  encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)

        print("\nModel comparison:")
        for name, values in comparison.items():
            print(f"{name}: accuracy = {values['accuracy']:.4f}")

    print("\nAnalysis finished.")
    print(f"Results folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
