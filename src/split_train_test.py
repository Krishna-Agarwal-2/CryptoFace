# src/split_train_test.py

import shutil
from pathlib import Path

LFW_DIR = Path("data/lfw")
TRAIN_DIR = Path("data/train")
TEST_DIR = Path("data/test")

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

used_identities = 0
total_train, total_test = 0, 0

for person_dir in LFW_DIR.iterdir():
    images = list(person_dir.glob("*.jpg"))
    
    if len(images) < 2:
        continue  # skip identities with fewer than 2 images

    used_identities += 1
    images = sorted(images)

    # Image 0 → train_faces (reference)
    reference_img = images[0]
    new_name = f"{person_dir.name}_{reference_img.stem}.jpg"
    shutil.copy(reference_img, TRAIN_DIR / new_name)
    total_train += 1

    # All others → test_faces (for verification)
    for test_img in images[1:]:
        new_name = f"{person_dir.name}_{test_img.stem}.jpg"
        shutil.copy(test_img, TEST_DIR / new_name)
        total_test += 1

print(f"✅ Done. Used {used_identities} identities.")
print(f"📦 Train (reference): {total_train} | 🧪 Test (to verify): {total_test}")
