# src/add_user_arc.py

import pickle
from pathlib import Path
from insightface.app import FaceAnalysis
import numpy as np
import tenseal as ts
from PIL import Image
import argparse

app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

def get_arcface_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    faces = app.get(np.array(img))
    if not faces:
        raise ValueError("No face detected.")
    return faces[0].embedding / np.linalg.norm(faces[0].embedding)

def encrypt_embedding(context, embedding):
    return ts.ckks_vector(context, embedding.tolist())

def add_to_db(name, enc_embedding, db_path):
    if Path(db_path).exists():
        with open(db_path, "rb") as f:
            db = pickle.load(f)
    else:
        db = {}
    db[name] = enc_embedding.serialize()
    with open(db_path, "wb") as f:
        pickle.dump(db, f)
    print(f"[+] Added {name} to DB.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="Path to image of user")
    parser.add_argument("--name", required=True, help="Name of user")
    args = parser.parse_args()

    context_path = "context_arc.context"
    db_path = "data/encrypted_user_db_arc.pkl"

    with open(context_path, "rb") as f:
        context = ts.context_from(f.read())

    emb = get_arcface_embedding(args.img)
    enc = encrypt_embedding(context, emb)
    add_to_db(args.name, enc, db_path)
