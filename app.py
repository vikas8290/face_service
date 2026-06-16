import os

# Set threading limits to reduce memory footprint on CPU-only environments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Set DEEPFACE_HOME. On Render, force it to .venv to persist cached weights.
if os.environ.get("RENDER") == "true":
    os.environ["DEEPFACE_HOME"] = "/opt/render/project/src/.venv"
else:
    project_dir = os.path.dirname(os.path.abspath(__file__))
    if "DEEPFACE_HOME" not in os.environ:
        os.environ["DEEPFACE_HOME"] = project_dir

import base64
import numpy as np
from flask import Flask, request, jsonify
from deepface import DeepFace

app = Flask(__name__)

# Use Facenet by default as it is fast, accurate and uses 128-d or 512-d embeddings
MODEL_NAME = "Facenet"

# Ensure temp directory exists for decoding base64 images
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

def save_base64_to_temp(base64_str):
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    img_data = base64.b64decode(base64_str)
    temp_path = os.path.join(TEMP_DIR, f"temp_{os.urandom(8).hex()}.jpg")
    with open(temp_path, "wb") as f:
        f.write(img_data)
    return temp_path

@app.route('/represent', methods=['POST'])
def represent():
    """Extract facial embedding/descriptor from an image (file path or base64)"""
    data = request.json or {}
    img_path = data.get("img_path")
    img_base64 = data.get("img_base64")
    model = data.get("model", MODEL_NAME)

    temp_path = None
    try:
        if img_base64:
            temp_path = save_base64_to_temp(img_base64)
            path_to_process = temp_path
        elif img_path:
            path_to_process = img_path
        else:
            return jsonify({"success": False, "message": "No image provided"}), 400

        # Extract embedding using DeepFace
        # enforce_detection=True will raise an error if no face is found
        embeddings = DeepFace.represent(
            img_path=path_to_process,
            model_name=model,
            enforce_detection=True
        )

        if not embeddings:
            return jsonify({"success": False, "message": "No face detected in the image"}), 422

        # DeepFace returns a list of dicts. We take the first face detected.
        first_face = embeddings[0]
        if not isinstance(first_face, dict):
            return jsonify({"success": False, "message": "Unexpected response format from DeepFace"}), 500

        embedding = first_face["embedding"]
        facial_area = first_face["facial_area"]

        return jsonify({
            "success": True,
            "embedding": embedding,
            "facial_area": facial_area,
            "model": model
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = str(e)
        if "Face could not be detected" in error_msg:
            return jsonify({"success": False, "message": "No face detected in the image"}), 422
        return jsonify({"success": False, "message": error_msg}), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

@app.route('/verify', methods=['POST'])
def verify():
    """Verify if two images (file paths or base64) belong to the same person"""
    data = request.json or {}
    img1_path = data.get("img1_path")
    img1_base64 = data.get("img1_base64")
    img2_path = data.get("img2_path")
    img2_base64 = data.get("img2_base64")
    model = data.get("model", MODEL_NAME)

    temp_path1 = None
    temp_path2 = None

    try:
        path1 = img1_path
        if img1_base64:
            temp_path1 = save_base64_to_temp(img1_base64)
            path1 = temp_path1

        path2 = img2_path
        if img2_base64:
            temp_path2 = save_base64_to_temp(img2_base64)
            path2 = temp_path2

        if not path1 or not path2:
            return jsonify({"success": False, "message": "Both images must be provided"}), 400

        result = DeepFace.verify(
            img1_path=path1,
            img2_path=path2,
            model_name=model,
            distance_metric="cosine",
            enforce_detection=True
        )

        return jsonify({
            "success": True,
            "verified": bool(result.get("verified", False)),
            "distance": float(result.get("distance", 1.0)),
            "threshold": float(result.get("threshold", 0.4))
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = str(e)
        if "Face could not be detected" in error_msg:
            return jsonify({"success": False, "message": "Face could not be detected in one of the images"}), 422
        return jsonify({"success": False, "message": error_msg}), 500

    finally:
        for p in [temp_path1, temp_path2]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model": MODEL_NAME})

# Warm up model so it is cached in memory during startup
print(f"Preloading DeepFace model: {MODEL_NAME}...")
try:
    # Create a dummy 100x100 image for warm up
    dummy_data = np.zeros((100, 100, 3), dtype=np.uint8)
    import cv2
    dummy_path = os.path.join(TEMP_DIR, "warmup.jpg")
    cv2.imwrite(dummy_path, dummy_data)
    DeepFace.represent(img_path=dummy_path, model_name=MODEL_NAME, enforce_detection=False)
    if os.path.exists(dummy_path):
        os.remove(dummy_path)
    print("Model preloaded successfully!")
except Exception as e:
    print(f"Warning: Model preload failed: {e}")
finally:
    import gc
    gc.collect()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
