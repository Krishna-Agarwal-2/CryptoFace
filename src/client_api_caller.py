# ===========================
# client_api_caller.py
# ===========================

import cv2
import torch
import base64
import requests
import tenseal as ts
import numpy as np
from insightface.app import FaceAnalysis
from pathlib import Path

# ----------- CONFIG -----------
QUERY_IMAGE_PATH = "data/test/Abdullah_Gul_Abdullah_Gul_0004.jpg"  # change as needed
IDENTITY = "_".join(Path(QUERY_IMAGE_PATH).stem.split("_")[:-1])  # auto-extracted identity
SECRET_CONTEXT_PATH = "context_arc_v2_full.context"  # includes secret key
API_URL = "http://127.0.0.1:8000/verify"  # FastAPI endpoint
# ------------------------------

# Load private TenSEAL context
with open(SECRET_CONTEXT_PATH, "rb") as f:
    context = ts.context_from(f.read())

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load face analysis model
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider' if device == "cuda" else 'CPUExecutionProvider'])
app.prepare(ctx_id=0 if device == "cuda" else -1)

# Load query image and extract embedding
img = cv2.imread(QUERY_IMAGE_PATH)
if img is None:
    raise ValueError(f"❌ Could not read image: {QUERY_IMAGE_PATH}")

faces = app.get(img)
if not faces:
    raise ValueError(f"❌ No face detected in image: {QUERY_IMAGE_PATH}")

embedding = faces[0].embedding.astype(np.float32)
embedding /= np.linalg.norm(embedding)

# Encrypt the query embedding
enc_query = ts.ckks_vector(context, embedding)
enc_query_bytes = enc_query.serialize()
enc_query_b64 = base64.b64encode(enc_query_bytes).decode("utf-8")

# Send encrypted query to FastAPI
payload = {
    "identity": IDENTITY,
    "query_vector": enc_query_b64
}

print(f"🚀 Sending encrypted query for identity: {IDENTITY}")
response = requests.post(API_URL, json=payload)

if response.status_code != 200:
    print(f"❌ API Error: {response.status_code} - {response.json().get('detail')}")
else:
    encrypted_similarity_b64 = response.json()["encrypted_similarity"]
    encrypted_similarity_bytes = base64.b64decode(encrypted_similarity_b64.encode("utf-8"))
    enc_similarity = ts.ckks_vector_from(context, encrypted_similarity_bytes)

    # Decrypt and print similarity
    similarity = enc_similarity.decrypt()[0]
    print(f"✅ Similarity Score with '{IDENTITY}': {similarity:.4f}")
