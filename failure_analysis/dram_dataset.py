"""
dram_dataset.py

PyTorch Dataset wrapper for the DRAM-style synthetic pairs produced by
generate_dram_dataset.py (the team's chosen generator -- single-world
pipeline with missing vias, broken lines, blob defects, LER, vignetting,
scan-line noise, and exact ground-truth tracking through geometric
augmentation).

This file's only job: take the output folder structure that
generate_dram_dataset.py produces --
    <out_dir>/reference/ref_000.png, ref_001.png, ...
    <out_dir>/search/search_000.png, search_001.png, ...
    labels.csv  (either inside <out_dir>, or one level above it -- see
                 note below on generate_dram_dataset.py's path quirk)
and turn it into something a PyTorch model/DataLoader can consume directly.

NOTE ON labels.csv LOCATION: generate_dram_dataset.py writes labels.csv to
<out_dir>/../labels.csv specifically when out_dir's folder name is exactly
"train", and to <out_dir>/labels.csv otherwise. This loader checks both
locations automatically so it works either way without you needing to
remember which case applies.

---------------------------------------------------------------------------
WHAT THIS GIVES YOU FOR EACH SAMPLE
---------------------------------------------------------------------------
For pair index i, __getitem__ returns:
    reference_tensor : FloatTensor [1, 1000, 1000], values in [0, 1]
    search_tensor    : FloatTensor [1, 1000, 1000], values in [0, 1]
    label_tensor     : FloatTensor [2]  -> (center_x, center_y)
                        NORMALIZED to [0, 1] by dividing by IMG_SIZE, so the
                        network predicts a scale-free (x, y) fraction rather
                        than a raw pixel number (helps training stability).
    pair_id          : int (kept for debugging/traceability, not used in loss)

---------------------------------------------------------------------------
USAGE
---------------------------------------------------------------------------
    from dram_dataset import DramPairDataset
    from torch.utils.data import DataLoader

    train_ds = DramPairDataset("./train_dataset")
    val_ds   = DramPairDataset("./val_dataset")

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=2)

    for reference, search, label, pair_id in train_loader:
        # reference: [B, 1, 1000, 1000]
        # search:    [B, 1, 1000, 1000]
        # label:     [B, 2]   (normalized x, y in [0,1])
        ...

Dependencies: torch, numpy, Pillow
    pip install torch numpy pillow --break-system-packages
"""

import csv
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DramPairDataset(Dataset):
    """
    Reads labels.csv (from `dataset_dir` or its parent -- see module
    docstring) and lazily loads the corresponding reference/search PNG
    pair for each row, on demand (in __getitem__) -- NOT all at once in
    memory. This matters because 3000 pairs of two 1000x1000 images would
    otherwise use several GB of RAM if preloaded.
    """

    def __init__(self, dataset_dir, img_size=1000, normalize_labels=True):
        self.dataset_dir = dataset_dir
        self.img_size = img_size
        self.normalize_labels = normalize_labels

        # generate_dram_dataset.py sometimes writes labels.csv inside
        # dataset_dir, sometimes one level up (see module docstring) --
        # check both so this loader works regardless of which happened.
        candidate_paths = [
            os.path.join(dataset_dir, "labels.csv"),
            os.path.normpath(os.path.join(dataset_dir, "..", "labels.csv")),
        ]
        labels_path = next((p for p in candidate_paths if os.path.exists(p)), None)
        if labels_path is None:
            raise FileNotFoundError(
                f"labels.csv not found in '{dataset_dir}' or its parent folder. "
                f"Did you run generate_dram_dataset.py with --out_dir {dataset_dir}?"
            )
        self.labels_path = labels_path

        with open(labels_path, newline="") as f:
            self.rows = list(csv.DictReader(f))

        if len(self.rows) == 0:
            raise ValueError(f"labels.csv at {labels_path} is empty.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        # generate_dram_dataset.py saves images inside reference/ and
        # search/ subfolders of dataset_dir (not flat, unlike our earlier
        # generator versions).
        ref_path = os.path.join(self.dataset_dir, "reference", row["reference_file"])
        search_path = os.path.join(self.dataset_dir, "search", row["search_file"])

        # Load as grayscale, normalize pixel values from [0,255] to [0,1]
        reference = np.array(Image.open(ref_path).convert("L"), dtype=np.float32) / 255.0
        search = np.array(Image.open(search_path).convert("L"), dtype=np.float32) / 255.0

        # Add a channel dimension: [H, W] -> [1, H, W] (PyTorch conv layers
        # expect [channels, height, width])
        reference_tensor = torch.from_numpy(reference).unsqueeze(0)
        search_tensor = torch.from_numpy(search).unsqueeze(0)

        # NOTE: column names are center_x / center_y (generate_dram_dataset.py's
        # naming), not true_center_x / true_center_y (our earlier generator's
        # naming) -- this is the fix for the mismatch flagged earlier.
        true_x = float(row["center_x"])
        true_y = float(row["center_y"])
        if self.normalize_labels:
            true_x /= self.img_size
            true_y /= self.img_size

        label_tensor = torch.tensor([true_x, true_y], dtype=torch.float32)
        pair_id = int(row["pair_id"])

        return reference_tensor, search_tensor, label_tensor, pair_id


# ==========================================================================
# Quick self-test / sanity check when run directly
# ==========================================================================
if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="Sanity-check DramPairDataset")
    parser.add_argument("--dataset_dir", type=str, required=True,
                         help="Folder passed as --out_dir to generate_dram_dataset.py")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    ds = DramPairDataset(args.dataset_dir)
    print(f"Loaded dataset: {len(ds)} pairs found (labels: {ds.labels_path})")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    reference, search, label, pair_id = next(iter(loader))

    print(f"reference batch shape: {tuple(reference.shape)}")
    print(f"search batch shape:    {tuple(search.shape)}")
    print(f"label batch shape:     {tuple(label.shape)}  (normalized x, y)")
    print(f"pair_id batch:         {pair_id.tolist()}")
    print(f"reference value range: [{reference.min():.3f}, {reference.max():.3f}]")
    print(f"search value range:    [{search.min():.3f}, {search.max():.3f}]")
    print(f"label value range:     [{label.min():.3f}, {label.max():.3f}]  "
          f"(should be within [0, 1] if normalize_labels=True)")
    print("\nSanity check passed -- dataset is ready to plug into a training loop.")
