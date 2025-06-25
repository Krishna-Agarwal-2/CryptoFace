import argparse
import os
import pickle
from pathlib import Path
from PIL import Image
import numpy as np
from insightface.app import FaceAnalysis
import tenseal as ts

# 🔐 Load encryption context
def load_context(path="context_arc.context"):
    with open(path, "rb") as f:
        return ts.context_from(f.read())

# 😈 Load target user's single encrypted embedding (1:1 version)
def load_target_embedding(name, context, db_path="data/encrypted_user_db_arc.pkl"):
    with open(db_path, "rb") as f:
        db_raw = pickle.load(f)
    if name not in db_raw:
        raise ValueError(f"Target '{name}' not found in database.")
    return ts.ckks_vector_from(context, db_raw[name])

# 🤖 Setup ArcFace
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0)

# 🧠 Get and normalize embedding
def get_arcface_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    faces = app.get(np.array(img))
    if not faces:
        return None
    return faces[0].embedding / np.linalg.norm(faces[0].embedding)

# 🔐 Encrypt embedding
def encrypt_embedding(vec, context):
    return ts.ckks_vector(context, vec.tolist())

# 🧠 Cosine similarity (approx)
def encrypted_cosine_similarity(vec1, vec2):
    dot_product = vec1.dot(vec2)
    return dot_product.decrypt()[0]

# ⚔️ Main attack function
def run_attack(target_name, folder, threshold):
    print(f"[⚔️] Running attacker test with {len(os.listdir(folder))} images...\n")
    context = load_context()
    enc_target = load_target_embedding(target_name, context)

    for filename in sorted(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        if not img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        embedding = get_arcface_embedding(img_path)
        if embedding is None:
            print(f"[!] No face found in {filename}")
            continue

        enc_vec = encrypt_embedding(embedding, context)
        score = encrypted_cosine_similarity(enc_vec, enc_target)

        if score >= threshold:
            print(f"[❌] Attack Succeeded: {filename} (Similarity: {score:.4f})")
        else:
            print(f"[✅] Attack Failed: {filename} (Similarity: {score:.4f})")

# 🏃‍♂️ CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Target user in database")
    parser.add_argument("--folder", required=True, help="Folder containing attacker images")
    parser.add_argument("--threshold", type=float, required=True, help="Threshold for success")
    args = parser.parse_args()

    run_attack(args.target, args.folder, args.threshold)
