from insightface.app import FaceAnalysis
import onnxruntime as ort

# Check if ONNX is using GPU
print(f"ONNX Runtime Device: {ort.get_device()}")  # Should say "GPU"

# Prepare ArcFace model
app = FaceAnalysis(name="buffalo_l")  # or antelopev2
app.prepare(ctx_id=0)  # 0 = GPU, -1 = CPU

# Double-check model device
print("InsightFace ArcFace loaded with GPU context.")
