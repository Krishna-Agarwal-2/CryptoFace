# preprocess_retina_lfw.py

import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2

from insightface.app import FaceAnalysis

# Init
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

# Directories to process
split_dirs = [Path("data/retina_train_faces"), Path("data/retina_test_faces")]

for dir_path in split_dirs:
    print(f"\n📁 Processing {dir_path}...")
    aligned_count = 0
    skipped = 0

    for img_path in tqdm(sorted(dir_path.glob("*.jpg"))):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                raise ValueError("cv2.imread returned None")

            faces = app.get(img)
            if not faces:
                print(f"[!] No face detected in: {img_path.name}")
                skipped += 1
                continue

            # Take the largest face (safest)
            face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
            aligned = face.aligned
            if aligned is None or not hasattr(aligned, "__array_interface__"):
                print(f"[!] Aligned face missing for {img_path.name}, falling back to manual crop")
                # fallback to bbox crop
                x1, y1, x2, y2 = map(int, face.bbox)
                aligned = img[y1:y2, x1:x2]
                if aligned is None or aligned.size == 0:
                    raise ValueError("Manual crop also failed")

            # Save back
            cv2.imwrite(str(img_path), aligned) 
            aligned_count += 1

        except Exception as e:
            print(f"[X] Error with {img_path.name}: {e}")
            skipped += 1

    print(f"✅ {dir_path.name}: {aligned_count} aligned | {skipped} skipped")
