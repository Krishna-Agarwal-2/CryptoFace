# ===========================
# fastapi_server.py
# ===========================

import base64
import pickle
import tenseal as ts
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load encrypted DB and context
with open("data/encrypted_embeddings_arc_v2.pkl", "rb") as f:
    encrypted_db = pickle.load(f)

with open("context_arc_v2.context", "rb") as f:
    context = ts.context_from(f.read())

# FastAPI app
app = FastAPI()

class QueryRequest(BaseModel):
    identity: str
    query_vector: str  # base64 string

@app.post("/verify")
def verify_identity(request: QueryRequest):
    if request.identity not in encrypted_db:
        raise HTTPException(status_code=404, detail="Identity not found in DB")

    try:
        # Load query vector from base64
        enc_query_bytes = base64.b64decode(request.query_vector.encode("utf-8"))
        enc_query = ts.ckks_vector_from(context, enc_query_bytes)

        # Load DB vector
        db_vector = ts.ckks_vector_from(context, encrypted_db[request.identity])

        # Compute encrypted dot product
        encrypted_similarity = enc_query.dot(db_vector)

        # Send encrypted similarity back
        encrypted_similarity_bytes = encrypted_similarity.serialize()
        encrypted_similarity_b64 = base64.b64encode(encrypted_similarity_bytes).decode("utf-8")

        return { "encrypted_similarity": encrypted_similarity_b64 }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Computation error: {str(e)}")
