
from pathlib import Path
import json
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "lung_xray_224" / "lung_xray_clean_512"
CHECKPOINT = BASE_DIR / "saved_models" / "lung" / "mobilenetv3_large_224_best.pth"
OUT_DIR = BASE_DIR / "lung_model_analysis" / "mobilenetv3_large"

IMAGE_SIZE = 224
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def load_model(num_classes):
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features, num_classes
    )

    try:
        checkpoint = torch.load(
            CHECKPOINT, map_location="cpu", weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(CHECKPOINT, map_location="cpu")

    state = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state)
    return model.to(DEVICE).eval()


def save_internal_outputs(model, sample_path, classes):
    feature_dir = OUT_DIR / "feature_maps"
    feature_dir.mkdir(parents=True, exist_ok=True)

    image = transform(
        datasets.folder.default_loader(str(sample_path))
    ).unsqueeze(0).to(DEVICE)

    activations = {}
    hooks = []

    conv_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    ]

    # Capture representative early, middle and late convolutions.
    selected = [
        ("early_convolution", conv_layers[0]),
        ("middle_convolution", conv_layers[len(conv_layers) // 2]),
        ("late_convolution", conv_layers[-1]),
    ]

    seen = set()

    for label, (name, layer) in selected:
        if id(layer) in seen:
            continue
        seen.add(id(layer))

        def make_hook(key):
            def hook(module, inputs, output):
                activations[key] = output.detach().cpu()
            return hook

        hooks.append(layer.register_forward_hook(make_hook(label)))

    # Capture adaptive average pooling output.
    for name, layer in model.named_modules():
        if isinstance(layer, nn.AdaptiveAvgPool2d):
            def make_pool_hook(key):
                def hook(module, inputs, output):
                    activations[key] = output.detach().cpu()
                return hook

            hooks.append(
                layer.register_forward_hook(make_pool_hook("adaptive_average_pooling"))
            )

    try:
        with torch.inference_mode():
            model(image)
    finally:
        for hook in hooks:
            hook.remove()

    shapes = {}

    for label, activation in activations.items():
        shapes[label] = list(activation.shape)

        if activation.ndim != 4:
            continue

        maps = activation[0]
        count = min(16, maps.shape[0])

        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        fig.suptitle(
            f"MobileNetV3-Large: {label}\nShape: {tuple(activation.shape)}"
        )

        for i, ax in enumerate(axes.flat):
            ax.axis("off")
            if i < count:
                feature = maps[i].numpy()
                low, high = feature.min(), feature.max()
                if high > low:
                    feature = (feature - low) / (high - low)
                ax.imshow(feature, cmap="viridis")
                ax.set_title(f"Channel {i}", fontsize=8)

        fig.tight_layout()
        fig.savefig(
            str(feature_dir / f"{label}.png"),
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

    with open(feature_dir / "activation_shapes.json", "w", encoding="utf-8") as f:
        json.dump(shapes, f, indent=2)

    print("Internal layer visualizations saved:", feature_dir)


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    test_dir = DATA_DIR / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test dataset not found: {test_dir}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = datasets.ImageFolder(test_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    classes = dataset.classes

    print("Device:", DEVICE)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Model: MobileNetV3-Large")
    print("Test images:", len(dataset))
    print("Classes:", classes)

    model = load_model(len(classes))
    y_true, y_pred = [], []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(DEVICE)
            with torch.autocast(
                device_type=DEVICE.type,
                enabled=(DEVICE.type == "cuda"),
            ):
                outputs = model(images)

            y_true.extend(labels.tolist())
            y_pred.extend(outputs.argmax(dim=1).cpu().tolist())

    accuracy = accuracy_score(y_true, y_pred)
    report_text = classification_report(
        y_true, y_pred, labels=list(range(len(classes))),
        target_names=classes, zero_division=0,
    )
    report_dict = classification_report(
        y_true, y_pred, labels=list(range(len(classes))),
        target_names=classes, zero_division=0, output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    print("\nCLASSIFICATION REPORT — MOBILENETV3-LARGE")
    print("Accuracy:", f"{accuracy:.4f}")
    print(report_text)
    print("Confusion matrix:\n", cm)

    (OUT_DIR / "classification_report.txt").write_text(
        f"Model: MobileNetV3-Large\nAccuracy: {accuracy:.6f}\n\n{report_text}",
        encoding="utf-8",
    )

    with open(OUT_DIR / "classification_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Precision", "Recall", "F1-score", "Support"])
        for name, values in report_dict.items():
            if isinstance(values, dict) and "precision" in values:
                writer.writerow([
                    name, values["precision"], values["recall"],
                    values["f1-score"], values["support"],
                ])

    with open(OUT_DIR / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Actual / Predicted"] + classes)
        for name, row in zip(classes, cm):
            writer.writerow([name] + row.tolist())

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title("MobileNetV3-Large - Confusion Matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_xticks(range(len(classes)), classes, rotation=20)
    ax.set_yticks(range(len(classes)), classes)

    for r in range(len(classes)):
        for c in range(len(classes)):
            ax.text(
                c, r, str(cm[r, c]), ha="center", va="center",
                color="white" if cm[r, c] > cm.max() / 2 else "black",
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(str(OUT_DIR / "confusion_matrix_heatmap.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    sample_path = next((test_dir / classes[0]).glob("*"), None)
    if sample_path is not None:
        save_internal_outputs(model, sample_path, classes)

    with open(OUT_DIR / "test_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "architecture": "MobileNetV3-Large",
            "test_images": len(dataset),
            "classes": classes,
            "accuracy": float(accuracy),
            "classification_report": report_dict,
            "confusion_matrix": cm.tolist(),
        }, f, indent=2)

    print("\nAnalysis complete.")
    print("All outputs:", OUT_DIR)


if __name__ == "__main__":
    main()

