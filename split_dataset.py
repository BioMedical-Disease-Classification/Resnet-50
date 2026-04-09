"""
split_dataset.py

Splits the raw Kaggle chest X-ray dataset into 3 classes:
  NORMAL, BACTERIAL, VIRAL

Usage:
    python split_dataset.py --src path/to/chest_xray --dst path/to/output

The Kaggle dataset structure expected:
    chest_xray/
        train/
            NORMAL/
            PNEUMONIA/   <-- contains both bacteria and virus images
        val/
            NORMAL/
            PNEUMONIA/
        test/
            NORMAL/
            PNEUMONIA/

Output structure produced:
    output/
        train/
            NORMAL/
            BACTERIAL/
            VIRAL/
        val/
            NORMAL/
            BACTERIAL/
            VIRAL/
        test/
            NORMAL/
            BACTERIAL/
            VIRAL/

Kaggle filenames for PNEUMONIA look like:
    person1_bacteria_1.jpeg  -> BACTERIAL
    person1_virus_1.jpeg     -> VIRAL
"""

import os
import shutil
import argparse
from pathlib import Path

SPLITS = ['train', 'val', 'test']
CLASSES = ['NORMAL', 'BACTERIAL', 'VIRAL']


def split_dataset(src_dir: str, dst_dir: str):
    src = Path(src_dir)
    dst = Path(dst_dir)

    # Create all output folders
    for split in SPLITS:
        for cls in CLASSES:
            (dst / split / cls).mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        split_src = src / split
        if not split_src.exists():
            print(f"Warning: '{split}' folder not found at {split_src}, skipping.")
            continue

        # --- Copy NORMAL images ---
        normal_src = split_src / 'NORMAL'
        normal_dst = dst / split / 'NORMAL'
        if normal_src.exists():
            files = list(normal_src.iterdir())
            for f in files:
                if f.suffix.lower() in ('.jpeg', '.jpg', '.png'):
                    shutil.copy2(f, normal_dst / f.name)
            print(f"[{split}] NORMAL: copied {len(files)} files")
        else:
            print(f"Warning: No NORMAL folder found in '{split}'")

        # --- Split PNEUMONIA into BACTERIAL / VIRAL by filename ---
        pneumonia_src = split_src / 'PNEUMONIA'
        if pneumonia_src.exists():
            bacterial_count = 0
            viral_count = 0
            unknown_count = 0

            for f in pneumonia_src.iterdir():
                if f.suffix.lower() not in ('.jpeg', '.jpg', '.png'):
                    continue

                name_lower = f.name.lower()
                if 'bacteria' in name_lower:
                    shutil.copy2(f, dst / split / 'BACTERIAL' / f.name)
                    bacterial_count += 1
                elif 'virus' in name_lower:
                    shutil.copy2(f, dst / split / 'VIRAL' / f.name)
                    viral_count += 1
                else:
                    # Fallback: can't determine class from filename
                    unknown_count += 1
                    print(f"  Unknown label for file: {f.name}")

            print(f"[{split}] BACTERIAL: {bacterial_count} | VIRAL: {viral_count}", end="")
            if unknown_count:
                print(f" | UNKNOWN (skipped): {unknown_count}")
            else:
                print()
        else:
            print(f"Warning: No PNEUMONIA folder found in '{split}'")

    print("\nDone! Output written to:", dst)
    print_summary(dst)


def print_summary(dst: Path):
    print("\n--- Dataset Summary ---")
    for split in SPLITS:
        print(f"\n{split}/")
        for cls in CLASSES:
            folder = dst / split / cls
            if folder.exists():
                count = len(list(folder.iterdir()))
                print(f"  {cls}: {count} images")


if __name__ == "__main__":
    split_dataset(
        src_dir=os.path.expanduser("~/Desktop/chest_xray/chest_xray"),
        dst_dir=os.path.expanduser("~/Desktop/chest_xray_split")
    )
