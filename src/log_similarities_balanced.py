# ===========================
# log_similarities_balanced.py (Updated)
# ===========================

import torch, pickle, random, csv
import tenseal as ts
from pathlib import Path
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from tqdm import tqdm
import numpy as np

# Paths
TEST_DIR = Path("data/retina_test_faces")
PICKLE_FILE = Path("data/encrypted_embeddings1.pkl")
CONTEXT_PATH = Path("context1.context")
CSV_OUT = Path("data/similarity_scores1.csv")

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load context and encrypted DB
with open(CONTEXT_PATH, "rb") as f:
    context = ts.context_from(f.read())

with open(PICKLE_FILE, "rb") as f:
    encrypted_db = pickle.load(f)

# Dot product similarity function (decrypt only final result)
def encrypted_dot_product(vec1, vec2):
    return (vec1.dot(vec2)).decrypt()[0]

# Identity extraction (matches new DB format)
def extract_identity(filename):
    return "_".join(filename.split("_")[:-1])

# Get list of identities in DB
all_identities = list(encrypted_db.keys())

# CSV Logging
with open(CSV_OUT, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["query_file", "ref_identity", "query_identity", "similarity", "match"])

    for img_path in tqdm(sorted(TEST_DIR.glob("*.jpg")), desc="🔐 Authenticating"):
        try:
            # Load and preprocess test image
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = model(tensor).squeeze(0).cpu().numpy()
                emb = emb / np.linalg.norm(emb)

            enc_query = ts.ckks_vector(context, emb)
            query_identity = extract_identity(img_path.stem)

            # Skip if query identity isn't in the encrypted DB
            if query_identity not in encrypted_db:
                print(f"[!] Missing ref identity: {query_identity}")
                continue

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