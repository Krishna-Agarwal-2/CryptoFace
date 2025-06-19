import os
import cv2
import torch
import pickle
from tqdm import tqdm
import numpy as np
import tenseal as ts
from insightface.app import FaceAnalysis

from pathlib import Path

# Set up
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📦 Using device: {device}")

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0 if device == "cuda" else -1)

# Input directory: unprocessed full images
input_dir = Path("data/train")
identities = sorted(os.listdir(input_dir))

# Encryption context
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60],
)
context.global_scale = 2**40
context.generate_galois_keys()

encrypted_db = {}

print(f"🚀 Encrypting ArcFace embeddings from: {input_dir}...\n")

for identity in tqdm(identities):
    identity_dir = input_dir / identity
    images = list(identity_dir.glob("*.jpg"))
    
    if not images:
        continue

    for image_path in images:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[!] Failed to read image: {image_path.name}")
            continue

        faces = app.get(img)

        if len(faces) == 0:
            print(f"[!] No face found in: {image_path.name}")
            continue

        face_embedding = faces[0].embedding.astype(np.float32)
        encrypted_embedding = ts.ckks_vector(context, face_embedding)

        encrypted_db[identity] = encrypted_embedding
        break  # ✅ Break only after a successful embedding

# Save encrypted DB
with open("data/encrypted_embeddings_arc.pkl", "wb") as f:
    pickle.dump(encrypted_db, f)

# Save encryption context
with open("context_arc.context", "wb") as f:
    f.write(context.serialize(save_secret_key=True))

print(f"\n✅ Done. Encrypted DB → data/encrypted_embeddings_arc.pkl")
print(f"🔐 Context saved → context_arc.context")
print(f"🧠 Total identities encrypted: {len(encrypted_db)}")
