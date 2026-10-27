
from pathlib import Path
import argparse
import csv
import json

import cv2
import numpy as np


# ============================================================
# CANCERSCAN AI - X-RAY TEXT CANDIDATE DETECTOR
# ============================================================
# - Detects small possible text/marker components near edges.
# - Saves original, candidate mask, and review overlay.
# - Does NOT erase or alter any X-ray pixels.
# - Candidate detection is heuristic; manually inspect results.
# ============================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def detect_text_candidates(image):
    """
    Return:
        candidate_mask: white rectangles over candidate components
        boxes: list of (x, y, width, height)

    The input image must be a BGR image read by OpenCV.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Find the approximate image-content region, ignoring black padding.
    # This is only used to position the review region, not to crop the image.
    content = (gray > 12).astype(np.uint8) * 255
    points = cv2.findNonZero(content)

    if points is None:
        return np.zeros((h, w), np.uint8), []

    cx, cy, cw, ch = cv2.boundingRect(points)

    # Search only a narrow band around the visible image-content boundary.
    # A wide 15% border often selects shoulders, anatomy, and image padding.
    margin_x = max(3, int(cw * 0.045))
    margin_y = max(3, int(ch * 0.06))

    left = max(0, cx)
    top = max(0, cy)
    right = min(w, cx + cw)
    bottom = min(h, cy + ch)

    edge_mask = np.zeros((h, w), dtype=np.uint8)
    edge_mask[top:min(bottom, top + margin_y), left:right] = 255
    edge_mask[max(top, bottom - margin_y):bottom, left:right] = 255
    edge_mask[top:bottom, left:min(right, left + margin_x)] = 255
    edge_mask[top:bottom, max(left, right - margin_x):right] = 255

    boxes = []

    # Consider dark and bright high-contrast components.
    for threshold_mode in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            threshold_mode,
            31,
            8,
        )
        binary = cv2.bitwise_and(binary, edge_mask)

        # Do not merge nearby components into huge filled regions.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        for i in range(1, count):
            x, y, bw, bh, area = stats[i]

            # Conservative filters for small annotation-like components.
            # These filters cannot distinguish text from anatomy perfectly.
            if area < 4 or area > 250:
                continue
            if bw < 2 or bh < 3:
                continue
            if bw > max(18, int(cw * 0.06)):
                continue
            if bh > max(18, int(ch * 0.07)):
                continue

            # Require the candidate to be near the visible content boundary.
            near_content_edge = (
                x < cx + margin_x
                or x + bw > cx + cw - margin_x
                or y < cy + margin_y
                or y + bh > cy + ch - margin_y
            )
            if not near_content_edge:
                continue

            box = (int(x), int(y), int(bw), int(bh))

            # Avoid near-identical duplicate boxes from the two thresholds.
            duplicate = False
            for ox, oy, ow, oh in boxes:
                if (
                    abs(x - ox) <= 2
                    and abs(y - oy) <= 2
                    and abs(bw - ow) <= 3
                    and abs(bh - oh) <= 3
                ):
                    duplicate = True
                    break

            if not duplicate:
                boxes.append(box)

    # Candidate mask contains boxes only, not filled image regions.
    candidate_mask = np.zeros((h, w), dtype=np.uint8)

    for x, y, bw, bh in boxes:
        pad = 2
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w - 1, x + bw + pad)
        y2 = min(h - 1, y + bh + pad)

        cv2.rectangle(
            candidate_mask,
            (x1, y1),
            (x2, y2),
            255,
            thickness=1,
        )

    return candidate_mask, boxes


def create_overlay(image, candidate_mask):
    """Create a red review overlay without changing the original image."""
    overlay = image.copy()
    selected = candidate_mask > 0

    red = np.zeros_like(overlay)
    red[:, :, 2] = 255  # OpenCV uses BGR channel order.

    overlay[selected] = cv2.addWeighted(
        image, 0.35, red, 0.65, 0
    )[selected]

    return overlay


def process_image(source, out_dir):
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Cannot read image: {source}")

    mask, boxes = detect_text_candidates(image)
    overlay = create_overlay(image, mask)

    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / "original.png"), image)
    cv2.imwrite(str(out_dir / "candidate_mask.png"), mask)
    cv2.imwrite(str(out_dir / "review_overlay.png"), overlay)

    audit = {
        "source_file": str(source.resolve()),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "candidate_count": len(boxes),
        "candidate_boxes_xywh": json.dumps(boxes),
        "pixels_removed": False,
        "review_required": True,
    }

    with open(out_dir / "audit_log.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit.keys()))
        writer.writeheader()
        writer.writerow(audit)

    print(f"\nReview files saved to: {out_dir.resolve()}")
    print(f"Image dimensions: {image.shape[1]} x {image.shape[0]}")
    print(f"Candidate boxes: {len(boxes)}")
    print("No image pixels were erased.")
    print("Inspect review_overlay.png before deciding on removal.")


def main():
    parser = argparse.ArgumentParser(
        description="Detect possible X-ray alphabet/marker candidates for review."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to one X-ray image",
    )
    parser.add_argument(
        "--output",
        default="xray_text_review",
        help="Output folder for preview and audit files",
    )

    args = parser.parse_args()
    source = Path(args.input)

    if not source.is_file():
        raise SystemExit(f"Input image does not exist: {source}")

    if source.suffix.lower() not in IMAGE_EXTENSIONS:
        raise SystemExit(f"Unsupported image format: {source.suffix}")

    process_image(source, Path(args.output))


if __name__ == "__main__":
    main()
