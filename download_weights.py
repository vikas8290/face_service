import os
import sys

# Set threading limits to reduce memory footprint on CPU-only environments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Set DEEPFACE_HOME to current directory before importing DeepFace if not set
project_dir = os.path.dirname(os.path.abspath(__file__))
if "DEEPFACE_HOME" not in os.environ:
    os.environ["DEEPFACE_HOME"] = project_dir

import numpy as np
import cv2
from deepface import DeepFace

print("Downloading/preloading DeepFace model weights during build time...")
# Create a dummy 100x100 image for warmup/download
dummy_data = np.zeros((100, 100, 3), dtype=np.uint8)
temp_path = os.path.join(project_dir, "warmup_build.jpg")
cv2.imwrite(temp_path, dummy_data)

try:
    # This will trigger the download and validation of model weights
    DeepFace.represent(img_path=temp_path, model_name="Facenet", enforce_detection=False)
    print("Model weights preloaded and cached successfully!")
except Exception as e:
    print(f"Error downloading/preloading model weights: {e}", file=sys.stderr)
    sys.exit(1)
finally:
    if os.path.exists(temp_path):
        os.remove(temp_path)
    import gc
    gc.collect()
