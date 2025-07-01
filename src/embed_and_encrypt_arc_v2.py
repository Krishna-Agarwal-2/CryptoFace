# ===========================
# embed_and_encrypt_arc_v2.py
# ===========================

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
os.makedirs("data", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📦 Using device: {device}")

app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider' if device == "cuda" else 'CPUExecutionProvider'])
app.prepare(ctx_id=0 if device == "cuda" else -1)

# Input directory
input_dir = Path("data/train")
image_paths = list(input_dir.glob("*.jpg"))

# Create context WITH secret key for local encryption
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60],
)
context.global_scale = 2**40
context.generate_galois_keys()

encrypted_db = {}
seen_identities = set()

print(f"🚀 Encrypting NORMALIZED ArcFace embeddings from: {input_dir}...\n")

for image_path in tqdm(image_paths):
    identity = "_".join(image_path.stem.split("_")[:-1])
    if identity in seen_identities:
        continue

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Failed to read image: {image_path.name}")
        continue

    faces = app.get(img)
    if not faces:
        print(f"❌ No face found in: {image_path.name}")
        continue

    embedding = faces[0].embedding.astype(np.float32)
    embedding /= np.linalg.norm(embedding)

    encrypted_vector = ts.ckks_vector(context, embedding)
    encrypted_db[identity] = encrypted_vector.serialize()
    seen_identities.add(identity)

# Save encrypted DB
with open("data/encrypted_embeddings_arc_v2.pkl", "wb") as f:
    pickle.dump(encrypted_db, f)

# Save context WITHOUT secret key (for FastAPI server)
with open("context_arc_v2.context", "wb") as f:
    f.write(context.serialize(save_secret_key=False))

# Save context WITH secret key (for client decryption)
with open("context_arc_v2_full.context", "wb") as f:
    f.write(context.serialize(save_secret_key=True))

print(f"\n✅ Done. Encrypted DB → data/encrypted_embeddings_arc_v2.pkl")
print(f"🔐 Public context saved → context_arc_v2.context (no secret key)")
print(f"🔐 Private context saved → context_arc_v2_full.context (with secret key)")
print(f"🧠 Total identities encrypted: {len(encrypted_db)}")
