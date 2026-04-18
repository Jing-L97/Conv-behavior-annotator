import argparse
import os
import shutil


def restructure_directories(base_path, new_dir_name, dry_run=False):
    """For every result.csv and utt.csv found under base_path,
    insert a new subdirectory (new_dir_name) between the parent folder and the CSV files.

    Example:
      Before: .../1e5/3/result.csv
      After:  .../1e5/3/<new_dir_name>/result.csv

    """
    target_files = {"result.csv", "utt.csv"}
    moved = 0
    skipped = 0

    for root, dirs, files in os.walk(base_path):
        csv_files = [f for f in files if f in target_files]
        if not csv_files:
            continue

        # Check we're not already inside a new_dir_name folder (avoid re-processing)
        if os.path.basename(root) == new_dir_name:
            print(f"  [SKIP] Already in target dir, skipping: {root}")
            skipped += len(csv_files)
            continue

        new_subdir = os.path.join(root, new_dir_name)
        for fname in csv_files:
            src = os.path.join(root, fname)
            dst_dir = new_subdir
            dst = os.path.join(dst_dir, fname)

            print(f"  {'[DRY RUN] ' if dry_run else ''}Moving:\n    {src}\n    -> {dst}")
            if not dry_run:
                os.makedirs(dst_dir, exist_ok=True)
                shutil.move(src, dst)
            moved += 1

    print(f"\nDone. {'Would move' if dry_run else 'Moved'} {moved} file(s), skipped {skipped}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insert a new subdirectory level above CSV result files.")
    parser.add_argument("base_path", help="Root path to walk (e.g. /scratch2/jliu/Feedback/results/baseline)")
    parser.add_argument("new_dir_name", help="Name of the new subdirectory to insert (e.g. 'random')")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without moving files")
    args = parser.parse_args()

    restructure_directories(args.base_path, args.new_dir_name, dry_run=args.dry_run)
