
from pathlib import Path
import json
import csv
import numpy as np

RESULTS_DIR = Path("saved_models") / "lung"
OUTPUT_CSV = RESULTS_DIR / "all_models_comparison.csv"


def get_value(data, *keys):
    """Return the first available numeric metric."""
    for key in keys:
        value = data.get(key)

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)

    return None


def find_metric(data, keys):
    """Find a metric at the top level or inside a classification report."""
    value = get_value(data, *keys)

    if value is not None:
        return value

    report = data.get("classification_report", {})

    if not isinstance(report, dict):
        return None

    # Prefer the Cancer class metrics.
    for class_name in ("Cancer", "cancer"):
        class_metrics = report.get(class_name, {})

        if isinstance(class_metrics, dict):
            value = get_value(class_metrics, *keys)
            if value is not None:
                return value

    # Then check aggregate metrics.
    for section in ("binary avg", "weighted avg", "macro avg"):
        section_metrics = report.get(section, {})

        if isinstance(section_metrics, dict):
            value = get_value(section_metrics, *keys)

            if value is not None:
                return value

    return None


def calculate_from_confusion_matrix(cm, cancer_first=True):
    """
    Calculate metrics from a 2x2 confusion matrix.

    cancer_first=True:
        Label order is [Cancer, NORMAL].
        Matrix = [[TP, FN], [FP, TN]]

    cancer_first=False:
        Label order is [NORMAL, Cancer].
        Matrix = [[TN, FP], [FN, TP]]
    """
    cm = np.asarray(cm, dtype=float)

    if cm.shape != (2, 2) or not np.isfinite(cm).all():
        return {}

    if cancer_first:
        tp, fn = cm[0, 0], cm[0, 1]
        fp, tn = cm[1, 0], cm[1, 1]
    else:
        tn, fp = cm[0, 0], cm[0, 1]
        fn, tp = cm[1, 0], cm[1, 1]

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / cm.sum() if cm.sum() else 0.0

    return {
        "Accuracy": accuracy,
        "Cancer Precision": precision,
        "Cancer Recall": recall,
        "Cancer F1": f1,
        "Specificity": specificity,
    }


def get_confusion_matrix(data):
    """Read the confusion matrix from supported JSON field names."""
    for key in (
        "confusion_matrix",
        "confusion_matrix_binary_non_cancer_then_cancer",
    ):
        cm = data.get(key)

        if isinstance(cm, list):
            try:
                array = np.asarray(cm)
                if array.shape == (2, 2):
                    return array.tolist()
            except (TypeError, ValueError):
                pass

    return None


def get_class_order(data, filename):
    """
    Determine confusion-matrix label order from saved metadata.
    Falls back to known model naming only when metadata is absent.
    """
    for key in ("class_names", "classes", "class_order", "labels"):
        labels = data.get(key)

        if isinstance(labels, list) and len(labels) == 2:
            normalized = [str(label).strip().lower() for label in labels]

            if "cancer" in normalized:
                cancer_index = normalized.index("cancer")
                return cancer_index == 0

    # Known legacy ResNet50 result used [NORMAL, Cancer].
    # Other previously reported matrices used [Cancer, NORMAL].
    if "resnet50" in filename.lower():
        return False

    return True


def format_metric(value):
    return "N/A" if value is None else f"{value:.4f}"


def main():
    if not RESULTS_DIR.exists():
        print("Results folder not found:", RESULTS_DIR.resolve())
        return

    files = sorted(RESULTS_DIR.glob("*test_results.json"))

    if not files:
        print("No test-results JSON files found in", RESULTS_DIR.resolve())
        return

    rows = []

    for file in files:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            print(f"Skipping {file.name}: {error}")
            continue

        if not isinstance(data, dict):
            print(f"Skipping {file.name}: unexpected JSON format")
            continue

        model_name = data.get(
            "model",
            file.name.replace("_test_results.json", "").replace("_", " ")
        )

        metrics = {
            "Accuracy": find_metric(
                data, ("accuracy", "test_accuracy", "test_acc")
            ),
            "Cancer Precision": find_metric(
                data, ("cancer_precision", "precision", "precision_score")
            ),
            "Cancer Recall": find_metric(
                data,
                (
                    "cancer_recall_sensitivity",
                    "cancer_recall",
                    "recall",
                    "sensitivity",
                    "recall_score",
                ),
            ),
            "Cancer F1": find_metric(
                data, ("cancer_f1", "f1", "f1_score")
            ),
            "Specificity": find_metric(
                data, ("specificity", "specificity_for_non_cancer")
            ),
            "ROC-AUC": find_metric(
                data, ("roc_auc", "roc_auc_for_cancer", "auc")
            ),
        }

        cm = get_confusion_matrix(data)

        # Fill missing metrics using the confusion matrix.
        if cm is not None:
            cancer_first = get_class_order(data, file.name)
            derived = calculate_from_confusion_matrix(
                cm, cancer_first=cancer_first
            )

            for metric, value in derived.items():
                if metrics[metric] is None:
                    metrics[metric] = value

        row = {
            "Model": model_name,
            "Input Size": data.get("image_size", "Check configuration"),
            "Test Images": data.get("test_images", ""),
            **metrics,
            "Confusion Matrix": json.dumps(
                cm if cm is not None else "Not saved"
            ),
            "Results File": file.name,
        }

        rows.append(row)

    if not rows:
        print("No valid results could be read.")
        return

    columns = list(rows[0].keys())

    with OUTPUT_CSV.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

        # Format metrics consistently in the CSV.
        for row in rows:
            csv_row = row.copy()

            for metric in (
                "Accuracy",
                "Cancer Precision",
                "Cancer Recall",
                "Cancer F1",
                "Specificity",
                "ROC-AUC",
            ):
                csv_row[metric] = format_metric(row[metric])

            writer.writerow(csv_row)

    print("\n========== MODEL COMPARISON ==========\n")

    for row in rows:
        print(f"\n{row['Model']}")
        print(f"  Input Size           : {row['Input Size']}")
        print(f"  Test Images          : {row['Test Images']}")

        for metric in (
            "Accuracy",
            "Cancer Precision",
            "Cancer Recall",
            "Cancer F1",
            "Specificity",
            "ROC-AUC",
        ):
            print(f"  {metric:20}: {format_metric(row[metric])}")

        print("  Confusion Matrix     :", row["Confusion Matrix"])

    print("\nComparison CSV saved to:")
    print(OUTPUT_CSV.resolve())

    print(
        "\nNote: Missing classification metrics are calculated from the "
        "confusion matrix. ROC-AUC cannot be calculated from that matrix "
        "alone; saved prediction probabilities are required."
    )


if __name__ == "__main__":
    main()
