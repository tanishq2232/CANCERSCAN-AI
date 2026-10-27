
from pathlib import Path
from PIL import Image, ImageOps

INPUT_DIR = Path("xray_input")
OUTPUT_DIR = Path("xray_224")
SIZE = (224, 224)

def resize_xray(image_path):
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")

        # Resize proportionally without stretching
        img = ImageOps.contain(
            img,
            SIZE,
            method=Image.Resampling.LANCZOS
        )

        # Add black padding to make a square
        canvas = Image.new("RGB", SIZE, (0, 0, 0))
        x = (SIZE[0] - img.width) // 2
        y = (SIZE[1] - img.height) // 2
        canvas.paste(img, (x, y))

        output_path = OUTPUT_DIR / f"{image_path.stem}.png"
        canvas.save(output_path)

        print(f"Converted: {image_path.name} -> {SIZE[0]}x{SIZE[1]}")

def main():
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    files = [
        p for p in INPUT_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]

    if not files:
        print(f"No X-ray images found in: {INPUT_DIR.resolve()}")
        print("Copy JPG, JPEG, PNG, or BMP images into that folder.")
        return

    for path in files:
        try:
            resize_xray(path)
        except Exception as exc:
            print(f"Failed: {path.name}: {exc}")

    print(f"Finished. Check: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
