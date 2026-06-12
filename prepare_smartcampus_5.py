from pathlib import Path
import shutil

CLASSES = ["classroom", "computerroom", "library", "corridor", "office"]

SRC = Path("data/raw/Images")
DST = Path("data/mit_indoor_smartcampus_5")

DST.mkdir(parents=True, exist_ok=True)

for cls in CLASSES:
    src_dir = SRC / cls
    dst_dir = DST / cls
    dst_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {src_dir}")

    count = 0
    for img in src_dir.glob("*.jpg"):
        shutil.copy2(img, dst_dir / img.name)
        count += 1

    print(f"{cls}: {count} images")

print("DONE:", DST)
