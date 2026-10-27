
from pathlib import Path
from datetime import datetime
import argparse
import json
import csv

import cv2
import numpy as np
from PIL import Image, ImageOps


# ============================================================
# CANCERSCAN AI - LUNG X-RAY PREPROCESSING REVIEW PIPELINE
# ============================================================
# 1. Reads image files recursively.
# 2. Preserves originals.
# 3. Resizes output to 224x224.
# 4. Detects possible text/marks near image borders.
# 5. Saves visual review previews.
# 6. Writes an audit CSV and summary JSON.
#
# IMPORTANT:
# - Candidate detection is heuristic, not definitive OCR.
# - No pixels are automatically erased.
# - Review flagged images before any removal/cropping.
# ============================================================

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
}

TARGET_SIZE = (224, 224)


def load_image(path):
    """Load an image safely and convert it to RGB."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB").copy()


def resize_image(image):
    """
    Resize to a fixed 224x224 size.

    This matches a direct Resize((224, 224)) pipeline.
    It may alter aspect ratio, so keep original files separately.
    """
    return image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)


def detect_candidate_marks(image):
    """
    Highlight possible unwanted annotations.

    Detection is deliberately limited to:
    - Small, high-contrast connected components near borders.
    - Small contour candidates near image edges.

    This is a REVIEW AID, not a reliable text detector.
    Anatomy, ribs, tubes, devices, and other meaningful content
    can also be detected. Nothing is removed.
    """
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    height, width = gray.shape
    overlay = rgb.copy()
    candidate_boxes = []

    # Analyse a border band, but never automatically crop it.
    border_x = max(8, int(width * 0.12))
    border_y = max(8, int(height * 0.12))

    border_mask = np.zeros_like(gray, dtype=np.uint8)
    border_mask[:border_y, :] = 255
    border_mask[-border_y:, :] = 255
    border_mask[:, :border_x] = 255
    border_mask[:, -border_x:] = 255

    # Local contrast can reveal lettering or marker-like objects.
    # Multiple threshold directions allow dark and bright marks.
    for polarity in ("dark", "bright"):
        if polarity == "dark":
            threshold_type = cv2.THRESH_BINARY_INV
        else:
            threshold_type = cv2.THRESH_BINARY

        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            threshold_type,
            31,
            9,
        )

        binary = cv2.bitwise_and(binary, border_mask)

        # Connect small text strokes into candidate components.
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (3, 3)
        )
        joined = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, kernel, iterations=1
        )

        contours, _ = cv2.findContours(
            joined,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            # Broad heuristic filters only; not proof of text.
            if area < 4:
                continue
            if w < 2 or h < 2:
                continue
            if w > width * 0.25 or h > height * 0.18:
                continue

            # Keep components close to a border.
            near_edge = (
                x < border_x
                or y < border_y
                or x + w > width - border_x
                or y + h > height - border_y
            )
            if not near_edge:
                continue

            # Deduplicate nearly identical candidates across thresholds.
            duplicate = False
            for old_x, old_y, old_w, old_h in candidate_boxes:
                if (
                    abs(x - old_x) < 4
                    and abs(y - old_y) < 4
                    and abs(w - old_w) < 6
                    and abs(h - old_h) < 6
                ):
                    duplicate = True
                    break

            if not duplicate:
                candidate_boxes.append((x, y, w, h))

    # Draw boxes on a copy only. The underlying image stays unchanged.
    for x, y, w, h in candidate_boxes:
        cv2.rectangle(
            overlay,
            (x, y),
            (x + w, y + h),
            (255, 40, 40),
            2,
        )

    return Image.fromarray(overlay), candidate_boxes


def make_review_preview(original_224, marked_224, source_name):
    """Create an original-vs-marked preview."""
    left = np.array(original_224.convert("RGB"))
    right = np.array(marked_224.convert("RGB"))

    gap = np.full((224, 10, 3), 40, dtype=np.uint8)
    combined = np.concatenate([left, gap, right], axis=1)

    # Add labels to the preview without changing saved model images.
    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

    cv2.putText(
        combined_bgr,
        "RESIZED ORIGINAL",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (180, 255, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        combined_bgr,
        "CANDIDATES IN RED",
        (238, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (100, 100, 255),
        1,
        cv2.LINE_AA,
    )

    return Image.fromarray(
        cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)
    )


def process_one_image(source_path, input_root, output_root):
    relative_path = source_path.relative_to(input_root)

    # Keep class folders, e.g. train/Cancer/image.jpg.
    output_original = output_root / "images_224" / relative_path
    output_preview = (
        output_root / "review_previews" / relative_path
    ).with_suffix(".jpg")

    output_original.parent.mkdir(parents=True, exist_ok=True)
    output_preview.parent.mkdir(parents=True, exist_ok=True)

    source = load_image(source_path)
    original_width, original_height = source.size

    resized = resize_image(source)
    marked, boxes = detect_candidate_marks(resized)

    # This is the unaltered resized image, not the red overlay.
    resized.save(output_original)

    preview = make_review_preview(resized, marked, source_path.name)
    preview.save(output_preview, quality=92)

    return {
        "source_file": str(relative_path),
        "source_width": original_width,
        "source_height": original_height,
        "output_width": TARGET_SIZE[0],
        "output_height": TARGET_SIZE[1],
        "candidate_count": len(boxes),
        "candidate_boxes_224": json.dumps(boxes),
        "review_status": (
            "REVIEW_REQUIRED" if boxes else "NO_CANDIDATES_DETECTED"
        ),
        "resized_image_path": str(
            output_original.relative_to(output_root)
        ),
        "preview_path": str(
            output_preview.relative_to(output_root)
        ),
        "pixels_removed": False,
    }


def main():
    parser = argparse.ArgumentParser(
        description="CancerScan AI conservative lung X-ray preprocessing"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input folder containing lung X-ray images",
    )
    parser.add_argument(
        "--output",
        default="lung_preprocessing_review",
        help="Output folder for resized images, previews, and logs",
    )
    args = parser.parse_args()

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()

    if not input_root.exists() or not input_root.is_dir():
        raise SystemExit(
            f"Input directory does not exist: {input_root}"
        )

    # Avoid recursively processing our own output directory.
    if output_root == input_root or input_root in output_root.parents:
        raise SystemExit(
            "Output must not be the input folder or a subfolder "
            "of the input folder."
        )

    output_root.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p for p in input_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not files:
        raise SystemExit(
            f"No supported images found under: {input_root}"
        )

    audit_path = output_root / "audit_log.csv"
    errors_path = output_root / "processing_errors.csv"

    results = []
    errors = []

    print(f"Input folder : {input_root}")
    print(f"Output folder: {output_root}")
    print(f"Images found : {len(files)}")
    print("Starting preprocessing review...\n")

    for index, path in enumerate(files, start=1):
        try:
            result = process_one_image(
                path,
                input_root,
                output_root,
            )
            results.append(result)

            print(
                f"[{index}/{len(files)}] {path.name} | "
                f"candidates={result['candidate_count']}"
            )

        except Exception as exc:
            errors.append({
                "source_file": str(path),
                "error": repr(exc),
            })
            print(
                f"[{index}/{len(files)}] ERROR: "
                f"{path.name}: {exc}"
            )

    # Save audit log.
    if results:
        with open(
            audit_path, "w", newline="", encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=list(results[0].keys())
            )
            writer.writeheader()
            writer.writerows(results)

    if errors:
        with open(
            errors_path, "w", newline="", encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=["source_file", "error"]
            )
            writer.writeheader()
            writer.writerows(errors)

    flagged = sum(
        row["candidate_count"] > 0 for row in results
    )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_directory": str(input_root),
        "output_directory": str(output_root),
        "images_found": len(files),
        "images_processed": len(results),
        "processing_errors": len(errors),
        "images_with_candidates": flagged,
        "images_without_candidates": len(results) - flagged,
        "target_size": list(TARGET_SIZE),
        "automatic_pixel_removal": False,
        "warning": (
            "Candidate boxes are heuristic and may include anatomy "
            "or clinically relevant structures. Review before editing."
        ),
    }

    with open(
        output_root / "summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print("\n" + "=" * 55)
    print("PREPROCESSING REVIEW COMPLETE")
    print("=" * 55)
    print(f"Processed successfully : {len(results)}")
    print(f"Images needing review  : {flagged}")
    print(f"Processing errors      : {len(errors)}")
    print(f"Resized images         : {output_root / 'images_224'}")
    print(f"Review previews        : {output_root / 'review_previews'}")
    print(f"Audit log              : {audit_path}")
    print(f"Summary                : {output_root / 'summary.json'}")

    print(
        "\nNo marks were erased. Inspect review_previews and "
        "audit_log.csv before approving any changes."
    )


if __name__ == "__main__":
    main()
