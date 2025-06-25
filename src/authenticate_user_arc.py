# src/authenticate_user_arc.py

import argparse
import pickle
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis
import tenseal as ts
from PIL import Image

# Load encryption context
with open("context_arc.context", "rb") as f:
    context = ts.context_from(f.read())

# Setup ArcFace
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

def get_arcface_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    faces = app.get(np.array(img))
    if not faces:
        raise ValueError("No face detected.")
    return faces[0].embedding / np.linalg.norm(faces[0].embedding)

def encrypt_embedding(embedding):
    return ts.ckks_vector(context, embedding.tolist())

def cosine_similarity(enc_vec1, enc_vec2):
    return enc_vec1.dot(enc_vec2)

def authenticate_1_to_1(query_img_path, target_user, db_path, threshold=0.3):
    print(f"[🔎] Verifying if the person in the image is '{target_user}'...")

    embedding = get_arcface_embedding(query_img_path)
    enc_query = encrypt_embedding(embedding)

    with open(db_path, "rb") as f:
        db_raw = pickle.load(f)

    if target_user not in db_raw:
        print(f"[❌] Target user '{target_user}' not found in database.")
        return

    serialized_enc_vec = db_raw[target_user]
    enc_target = ts.ckks_vector_from(context, serialized_enc_vec)

    sim = cosine_similarity(enc_query, enc_target).decrypt()[0]

    print(f"[📊] Similarity score with {target_user}: {sim:.4f}")

    if sim >= threshold:
        print(f"[✅] Access Granted: Match confirmed for {target_user}")
    else:
        print(f"[❌] Access Denied: Similarity below threshold ({threshold})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="Path to query image")
    parser.add_argument("--target", required=True, help="Username to authenticate against")
    parser.add_argument("--threshold", type=float, default=0.3, help="Threshold for similarity")
    args = parser.parse_args()

    authenticate_1_to_1(args.img, args.target, "data/encrypted_user_db_arc.pkl", args.threshold)
