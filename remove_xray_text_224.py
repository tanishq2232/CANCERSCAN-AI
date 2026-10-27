
from pathlib import Path
import argparse
import cv2
import numpy as np

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def remove_edge_text(image):
    """
    Conservative heuristic for small, high-contrast text near the edges.
    This is not OCR; review output because anatomy may also be selected.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Search only narrow image-edge strips.
    edge_x = max(4, int(w * 0.07))
    edge_y = max(4, int(h * 0.07))

    edge = np.zeros((h, w), np.uint8)
    edge[:edge_y, :] = 255
    edge[-edge_y:, :] = 255
    edge[:, :edge_x] = 255
    edge[:, -edge_x:] = 255

    combined = np.zeros((h, w), np.uint8)

    # Detect both dark and bright small components.
    for mode in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            mode, 21, 7
        )
        binary = cv2.bitwise_and(binary, edge)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        for i in range(1, count):
            x, y, bw, bh, area = stats[i]

            # Candidate size limits for 224x224 images.
            if not (2 <= bw <= 18 and 3 <= bh <= 18 and 3 <= area <= 180):
                continue

            # Keep only components close to an image edge.
            near_edge = (
                x < edge_x or y < edge_y
                or x + bw > w - edge_x
                or y + bh > h - edge_y
            )
            if not near_edge:
                continue

            # Slightly expand the selected component.
            pad = 2
            x1, y1 = max(0, x-pad), max(0, y-pad)
            x2, y2 = min(w, x+bw+pad), min(h, y+bh+pad)
            combined[y1:y2, x1:x2] = 255

    # Remove tiny isolated mask specks.
    combined = cv2.morphologyEx(
        combined, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    # Inpaint only detected candidates.
    cleaned = cv2.inpaint(image, combined, 3, cv2.INPAINT_TELEA)
    return cleaned, combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="224x224 dataset folder")
    parser.add_argument("--output", default="lung_xray_224_text_removed")
    args = parser.parse_args()

    source = Path(args.input).resolve()
    output = Path(args.output).resolve()

    if not source.is_dir():
        raise SystemExit(f"Input folder not found: {source}")
    if output == source or source in output.parents:
        raise SystemExit("Choose an output folder outside the input folder.")

    files = [
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]
    if not files:
        raise SystemExit("No supported images found.")

    print(f"Images found: {len(files)}")
    print(f"Saving to: {output}")

    for index, path in enumerate(files, 1):
        relative = path.relative_to(source)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)

        image = cv2.imread(str(path))
        if image is None:
            print(f"Skipping unreadable image: {path}")
            continue

        # Normalize input size to 224x224.
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        cleaned, mask = remove_edge_text(image)

        cv2.imwrite(str(destination), cleaned)

        # Save audit previews separately.
        preview_dir = output / "_review" / relative.parent
        preview_dir.mkdir(parents=True, exist_ok=True)
        stem = relative.stem

        overlay = image.copy()
        overlay[mask > 0] = (
            0.45 * overlay[mask > 0]
            + 0.55 * np.array([0, 0, 255])
        ).astype(np.uint8)

        cv2.imwrite(str(preview_dir / f"{stem}_original.png"), image)
        cv2.imwrite(str(preview_dir / f"{stem}_mask.png"), mask)
        cv2.imwrite(str(preview_dir / f"{stem}_overlay.png"), overlay)

        if index % 100 == 0 or index == len(files):
            print(f"Processed {index}/{len(files)}")

    print("\nFinished.")
    print("Cleaned images:", output)
    print("Review previews:", output / "_review")
    print("Check samples before using them for model training.")


if __name__ == "__main__":
    main()
