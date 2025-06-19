# preprocess_lfw.py — Color-safe MTCNN alignment

import os
from pathlib import Path
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision.transforms import ToPILImage
from tqdm import tqdm
import torch

# Init
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
to_pil = ToPILImage()

# Directories
split_dirs = [Path("data/train_faces"), Path("data/test_faces")]

for dir_path in split_dirs:
    print(f"\n📁 Processing {dir_path}...")
    aligned_count = 0
    skipped = 0

    for img_path in tqdm(sorted(dir_path.glob("*.jpg"))):
        try:
            image = Image.open(img_path).convert("RGB")
            aligned = mtcnn(image)

            if aligned is None:
                print(f"[!] No face detected in: {img_path.name}")
                skipped += 1
                continue

            # ✅ Proper color-preserving conversion
            aligned = aligned.clamp(0, 1)  # ensure safe range
            aligned_img = to_pil(aligned.cpu())
            aligned_img.save(img_path)

            aligned_count += 1

        except Exception as e:
            print(f"[X] Error with {img_path.name}: {e}")
            skipped += 1

    print(f"✅ {dir_path.name}: {aligned_count} aligned | {skipped} skipped")
