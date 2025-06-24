# ===========================
# log_similarities_balanced_arc.py
# ===========================

import os, cv2, torch, pickle, random, csv
import numpy as np
import tenseal as ts
from tqdm import tqdm
from pathlib import Path
from insightface.app import FaceAnalysis

# Paths
TEST_DIR = Path("data/test")
DB_PATH = Path("data/encrypted_embeddings_arc.pkl")
CONTEXT_PATH = Path("context_arc.context")
CSV_OUT = Path("data/similarity_scores_arc_balanced.csv")

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📦 Using device: {device}")

# InsightFace ArcFace setup
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider' if device == "cuda" else 'CPUExecutionProvider'])
app.prepare(ctx_id=0 if device == "cuda" else -1)

# Load TenSEAL context (public key only)
with open(CONTEXT_PATH, "rb") as f:
    context = ts.context_from(f.read())

# Load encrypted database
with open(DB_PATH, "rb") as f:
    encrypted_db = pickle.load(f)

# Similarity using encrypted cosine (dot product)
def encrypted_dot_product(vec1, vec2):
    return (vec1.dot(vec2)).decrypt()[0]

# Extract identity from filename
def extract_identity(filename):
    return "_".join(filename.split("_")[:-1])

# All identities in DB
all_identities = list(encrypted_db.keys())

# CSV logging
with open(CSV_OUT, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["query_file", "ref_identity", "query_identity", "similarity", "match"])

    for img_path in tqdm(sorted(TEST_DIR.glob("*.jpg")), desc="🔐 ArcFace Authentication"):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[!] Cannot read: {img_path.name}")
                continue

            faces = app.get(img)
            if len(faces) == 0:
                print(f"[!] No face found in: {img_path.name}")
                continue

            query_identity = extract_identity(img_path.stem)

            if query_identity not in encrypted_db:
                print(f"[!] Identity not found in DB: {query_identity}")
                continue

            embedding = faces[0].embedding.astype(np.float32)
            embedding /= np.linalg.norm(embedding)  # cosine normalization

            enc_query = ts.ckks_vector(context, embedding)

            # === Genuine match ===
            enc_ref = ts.ckks_vector_from(context, encrypted_db[query_identity])
            sim_true = encrypted_dot_product(enc_query, enc_ref)
            writer.writerow([
                img_path.stem, query_identity, query_identity, sim_true, "yes"
            ])

            # === Impostor match ===
            impostors = [id_ for id_ in all_identities if id_ != query_identity]
            wrong_identity = random.choice(impostors)
            enc_wrong = ts.ckks_vector_from(context, encrypted_db[wrong_identity])
            sim_false = encrypted_dot_product(enc_query, enc_wrong)
            writer.writerow([
                img_path.stem, wrong_identity, query_identity, sim_false, "no"
            ])

        except Exception as e:
            print(f"[X] Error on {img_path.name}: {e}")

print(f"\n✅ Finished logging results → {CSV_OUT}")
