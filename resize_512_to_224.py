
from pathlib import Path
from PIL import Image

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

SOURCE_DIR = Path(
    r"C:\Users\tanis\OneDrive\Desktop\CANCERSCAN-AI\CANCERSCAN-AI"
)

OUTPUT_DIR = SOURCE_DIR.parent / "dataset_224"

TARGET_SIZE = (224, 224)

SUPPORTED_FORMATS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}

# --------------------------------------------------
# RESIZE IMAGES
# --------------------------------------------------

processed = 0
failed = 0

for file_path in SOURCE_DIR.rglob("*"):

    if not file_path.is_file():
        continue

    # Avoid processing generated images or output folders
    if OUTPUT_DIR == file_path or OUTPUT_DIR in file_path.parents:
        continue

    if file_path.suffix.lower() not in SUPPORTED_FORMATS:
        continue

    try:
        # Preserve relative folder structure
        relative_path = file_path.relative_to(SOURCE_DIR)
        output_path = OUTPUT_DIR / relative_path

        output_path.parent.mkdir(
            parents=True, exist_ok=True
        )

        with Image.open(file_path) as img:
            # Convert to RGB for consistent model input
            img = img.convert("RGB")

            # Resize directly from 512x512 to 224x224
            resized = img.resize(
                TARGET_SIZE,
                Image.Resampling.LANCZOS
            )

            resized.save(output_path, quality=95)

        processed += 1

        if processed % 500 == 0:
            print(f"Processed {processed} images...")

    except Exception as error:
        failed += 1
        print(f"Failed: {file_path} | {error}")

print("\nConversion completed!")
print(f"Images converted: {processed}")
print(f"Images failed: {failed}")
print(f"Output folder: {OUTPUT_DIR}")

