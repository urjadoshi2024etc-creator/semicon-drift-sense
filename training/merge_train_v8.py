from pathlib import Path
import csv
import shutil


BASE = Path(".")

OLD_DIR = BASE / "train_v7"
COARSE_DIR = BASE / "train_v7_coarse_pitch"
OUT_DIR = BASE / "train_v8"

OLD_COUNT = 9000
COARSE_COUNT = 4000


def check_dataset(directory):
    labels_path = directory / "labels.csv"

    with open(labels_path, newline="") as f:
        rows = list(csv.DictReader(f))

    print(f"{directory}: {len(rows)} rows")

    if not rows:
        raise RuntimeError(f"No rows found in {labels_path}")

    required = {
        "pair_id",
        "reference_file",
        "search_file",
    }

    missing = required - set(rows[0].keys())

    if missing:
        raise RuntimeError(
            f"{directory}/labels.csv missing columns: {missing}"
        )

    return rows


def copy_pair(src_dir, row, new_pair_id, out_dir):
    old_ref = src_dir / "reference" / row["reference_file"]
    old_search = src_dir / "search" / row["search_file"]

    if not old_ref.exists():
        raise FileNotFoundError(old_ref)

    if not old_search.exists():
        raise FileNotFoundError(old_search)

    new_ref_name = f"ref_{new_pair_id:05d}.png"
    new_search_name = f"search_{new_pair_id:05d}.png"

    new_ref = out_dir / "reference" / new_ref_name
    new_search = out_dir / "search" / new_search_name

    shutil.copy2(old_ref, new_ref)
    shutil.copy2(old_search, new_search)

    new_row = dict(row)

    new_row["pair_id"] = str(new_pair_id)
    new_row["reference_file"] = new_ref_name
    new_row["search_file"] = new_search_name

    return new_row


def main():

    print("=" * 80)
    print("DRIFTSENSE TRAINING DATA MERGE")
    print("=" * 80)

    # ------------------------------------------------------------
    # Check source datasets
    # ------------------------------------------------------------

    old_rows = check_dataset(OLD_DIR)
    coarse_rows = check_dataset(COARSE_DIR)

    if len(old_rows) != OLD_COUNT:
        raise RuntimeError(
            f"Expected {OLD_COUNT} rows in train_v7, "
            f"found {len(old_rows)}"
        )

    if len(coarse_rows) != COARSE_COUNT:
        raise RuntimeError(
            f"Expected {COARSE_COUNT} rows in train_v7_coarse_pitch, "
            f"found {len(coarse_rows)}"
        )

    print()
    print("Source datasets verified:")
    print(f"  train_v7              : {len(old_rows)}")
    print(f"  train_v7_coarse_pitch : {len(coarse_rows)}")
    print(f"  expected total        : {len(old_rows) + len(coarse_rows)}")

    # ------------------------------------------------------------
    # Create output directory
    # ------------------------------------------------------------

    if OUT_DIR.exists():
        raise RuntimeError(
            f"{OUT_DIR} already exists.\n"
            "Delete/rename it manually if you want to recreate it."
        )

    (OUT_DIR / "reference").mkdir(parents=True)
    (OUT_DIR / "search").mkdir(parents=True)
    (OUT_DIR / "preview").mkdir(parents=True)

    # Copy config files for traceability.
    shutil.copy2(
        OLD_DIR / "config.json",
        OUT_DIR / "config_original.json"
    )

    shutil.copy2(
        COARSE_DIR / "config.json",
        OUT_DIR / "config_coarse_pitch.json"
    )

    # ------------------------------------------------------------
    # Merge labels
    # ------------------------------------------------------------

    all_rows = []

    print()
    print("Copying original train_v7...")
    
    for i, row in enumerate(old_rows):

        # Original dataset keeps its IDs.
        new_id = i

        new_row = copy_pair(
            OLD_DIR,
            row,
            new_id,
            OUT_DIR
        )

        all_rows.append(new_row)

        if (i + 1) % 1000 == 0:
            print(f"  copied {i + 1}/{len(old_rows)}")

    print()
    print("Copying coarse-pitch supplemental dataset...")

    for i, row in enumerate(coarse_rows):

        # Supplemental dataset starts at 9000.
        new_id = OLD_COUNT + i

        new_row = copy_pair(
            COARSE_DIR,
            row,
            new_id,
            OUT_DIR
        )

        all_rows.append(new_row)

        if (i + 1) % 1000 == 0:
            print(f"  copied {i + 1}/{len(coarse_rows)}")

    # ------------------------------------------------------------
    # Write merged labels.csv
    # ------------------------------------------------------------

    labels_path = OUT_DIR / "labels.csv"

    fieldnames = list(all_rows[0].keys())

    with open(labels_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_rows)

    # ------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("VERIFYING MERGED DATASET")
    print("=" * 80)

    expected_total = OLD_COUNT + COARSE_COUNT

    if len(all_rows) != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} rows, "
            f"got {len(all_rows)}"
        )

    # Check IDs
    ids = [int(r["pair_id"]) for r in all_rows]

    expected_ids = list(range(expected_total))

    if ids != expected_ids:
        raise RuntimeError("pair_id sequence is incorrect")

    # Check files
    missing_reference = []
    missing_search = []

    for row in all_rows:

        ref = OUT_DIR / "reference" / row["reference_file"]
        search = OUT_DIR / "search" / row["search_file"]

        if not ref.exists():
            missing_reference.append(str(ref))

        if not search.exists():
            missing_search.append(str(search))

    if missing_reference:
        raise RuntimeError(
            f"Missing reference files: {missing_reference[:10]}"
        )

    if missing_search:
        raise RuntimeError(
            f"Missing search files: {missing_search[:10]}"
        )

    # Check pitch ranges
    pitch_x = [float(r["pitch_x_nm"]) for r in all_rows]
    pitch_y = [float(r["pitch_y_nm"]) for r in all_rows]

    print(f"Total pairs       : {len(all_rows)}")
    print(f"Reference images  : {len(list((OUT_DIR / 'reference').glob('*.png')))}")
    print(f"Search images     : {len(list((OUT_DIR / 'search').glob('*.png')))}")

    print()
    print(
        f"pitch_x_nm range  : "
        f"{min(pitch_x):.3f} - {max(pitch_x):.3f}"
    )

    print(
        f"pitch_y_nm range  : "
        f"{min(pitch_y):.3f} - {max(pitch_y):.3f}"
    )

    print()
    print("ID range:")
    print(f"  first : {ids[0]}")
    print(f"  last  : {ids[-1]}")

    print()
    print("MERGE SUCCESSFUL")
    print()
    print(f"Output dataset: {OUT_DIR.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()