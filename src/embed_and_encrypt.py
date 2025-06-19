# ===========================
# embed_and_encrypt.py
# ===========================

import torch, pickle
import tenseal as ts
from pathlib import Path
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from tqdm import tqdm
import numpy as np

# Paths
DATA_DIR = Path("data/retina_train_faces")
OUT_PATH = Path("data/encrypted_embeddings1.pkl")
CONTEXT_PATH = Path("context1.context")

# Model setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Image transform
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Create TenSEAL CKKS context
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.generate_galois_keys()
context.generate_relin_keys()
context.global_scale = 2**40

# 🔐 Save context WITH secret key (used for decryption later)
with open(CONTEXT_PATH, "wb") as f:
    f.write(context.serialize(save_secret_key=True))

# Prepare encrypted DB
encrypted_db = {}

print(f"🚀 Encrypting face embeddings from: {DATA_DIR}...\n")
for img_path in tqdm(sorted(DATA_DIR.glob("*.jpg"))):
    try:
        # Load and preprocess image
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)

        # Generate FaceNet embedding
        with torch.no_grad():
            emb = model(tensor).squeeze(0).cpu().numpy()
            emb = emb / np.linalg.norm(emb)

        # Extract identity from filename
        identity = "_".join(img_path.stem.split("_")[:-1])

        # Encrypt embedding
        encrypted_vec = ts.ckks_vector(context, emb).serialize()
        encrypted_db[identity] = encrypted_vec

    except Exception as e:
        print(f"[!] Skipping {img_path.name}: {e}")

# Save encrypted DB
with open(OUT_PATH, "wb") as f:
    pickle.dump(encrypted_db, f)

print(f"\n✅ Done. Encrypted DB → {OUT_PATH}")
print(f"🔐 Context saved → {CONTEXT_PATH}")
print(f"🧠 Total identities encrypted: {len(encrypted_db)}")
