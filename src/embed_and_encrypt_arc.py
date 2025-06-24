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

app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider' if device == "cuda" else 'CPUExecutionProvider'])
app.prepare(ctx_id=0 if device == "cuda" else -1)

# Input directory: flat images (e.g. person1_1.jpg, person2_1.jpg)
input_dir = Path("data/train")
image_paths = list(input_dir.glob("*.jpg"))

# Encryption context
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
    # Extract identity from filename
    stem = image_path.stem
    identity = "_".join(stem.split("_")[:-1])

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

    face_embedding = faces[0].embedding.astype(np.float32)

    # ✅ Normalize embedding
    face_embedding /= np.linalg.norm(face_embedding)

    # Encrypt and serialize
    encrypted_vector = ts.ckks_vector(context, face_embedding)
    encrypted_db[identity] = encrypted_vector.serialize()
    seen_identities.add(identity)

# Save encrypted database (serialized vectors)
with open("data/encrypted_embeddings_arc.pkl", "wb") as f:
    pickle.dump(encrypted_db, f)

# Save encryption context
with open("context_arc.context", "wb") as f:
    f.write(context.serialize(save_secret_key=True))

print(f"\n✅ Done. Encrypted DB → data/encrypted_embeddings_arc.pkl")
print(f"🔐 Context saved → context_arc.context")
print(f"🧠 Total identities encrypted: {len(encrypted_db)}")
