"""
DRIFTSENSE - Standalone Localization Inference

Usage:
    python inference.py <reference_image> <search_image>

Example:
    python inference.py ../eval_v5/reference/ref_000.png ../eval_v5/search/search_000.png

Output:
    Predicted center: (x, y)
"""

import os
import sys

import cv2
import torch

from model_v6 import build_model


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    "driftsense_final.pt"
)


# ---------------------------------------------------------
# Argument validation
# ---------------------------------------------------------

if len(sys.argv) != 3:
    print(
        "Usage: python inference.py "
        "<reference_image> <search_image>"
    )
    sys.exit(1)


REFERENCE_PATH = sys.argv[1]
SEARCH_PATH = sys.argv[2]


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

model = build_model().to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ---------------------------------------------------------
# Load images
# ---------------------------------------------------------

ref = cv2.imread(
    REFERENCE_PATH,
    cv2.IMREAD_GRAYSCALE
)

search = cv2.imread(
    SEARCH_PATH,
    cv2.IMREAD_GRAYSCALE
)


if ref is None:
    print(
        f"ERROR: Could not read reference image: "
        f"{REFERENCE_PATH}"
    )
    sys.exit(1)


if search is None:
    print(
        f"ERROR: Could not read search image: "
        f"{SEARCH_PATH}"
    )
    sys.exit(1)


# ---------------------------------------------------------
# Validate image dimensions
# ---------------------------------------------------------

if ref.shape != (1000, 1000):
    print(
        f"ERROR: Reference image must be 1000x1000. "
        f"Received: {ref.shape}"
    )
    sys.exit(1)


if search.shape != (1000, 1000):
    print(
        f"ERROR: Search image must be 1000x1000. "
        f"Received: {search.shape}"
    )
    sys.exit(1)


# ---------------------------------------------------------
# Convert images to tensors
# ---------------------------------------------------------

ref_t = (
    torch.from_numpy(ref)
    .float()
    .unsqueeze(0)
    .unsqueeze(0)
    / 255.0
)

search_t = (
    torch.from_numpy(search)
    .float()
    .unsqueeze(0)
    .unsqueeze(0)
    / 255.0
)


ref_t = ref_t.to(
    device,
    non_blocking=True
)

search_t = search_t.to(
    device,
    non_blocking=True
)


# ---------------------------------------------------------
# Inference
# ---------------------------------------------------------

with torch.no_grad():

    output = model(
        ref_t,
        search_t
    )

    pred_x = output.coords[0, 0].item()
    pred_y = output.coords[0, 1].item()


# ---------------------------------------------------------
# Output
# ---------------------------------------------------------

print(
    f"({pred_x:.2f}, {pred_y:.2f})"
)