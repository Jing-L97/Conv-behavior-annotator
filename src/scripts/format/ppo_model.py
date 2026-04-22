import argparse
import os
import shutil


def restructure_directories(base_path, new_dir_name, dry_run=False):
    """For every numeric subfolder found directly under base_path,
    insert a new subdirectory (new_dir_name) between base_path and that subfolder.

    Example:
      Before: .../target_6/123/
      After:  .../target_6/<new_dir_name>/123/

    """
    moved = 0
    skipped = 0

    # List only immediate children of base_path
    try:
        entries = os.listdir(base_path)
    except FileNotFoundError:
        print(f"[ERROR] Base path not found: {base_path}")
        return

    for entry in sorted(entries):
        entry_path = os.path.join(base_path, entry)

        # Only process directories
        if not os.path.isdir(entry_path):
            continue

        # Skip if this entry is already the new_dir_name (avoid re-processing)
        if entry == new_dir_name:
            print(f"  [SKIP] Already a target dir, skipping: {entry_path}")
            skipped += 1
            continue

        new_subdir = os.path.join(base_path, new_dir_name)
        dst = os.path.join(new_subdir, entry)

        print(f"  {'[DRY RUN] ' if dry_run else ''}Moving:\n    {entry_path}\n    -> {dst}")

        if not dry_run:
            os.makedirs(new_subdir, exist_ok=True)
            shutil.move(entry_path, dst)

        moved += 1

    print(f"\nDone. {'Would move' if dry_run else 'Moved'} {moved} folder(s), skipped {skipped}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insert a new subdirectory level above numeric run folders.")
    parser.add_argument(
        "base_path",
        help=(
            "Path to the experiment variant folder whose immediate children "
            "are run-ID folders (e.g. /scratch2/jliu/Feedback/models/ppo/1e5_entropy_001_lm_loss_001_target_6)"
        ),
    )
    parser.add_argument(
        "new_dir_name",
        help="Name of the new subdirectory to insert (e.g. '3')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without moving anything",
    )
    args = parser.parse_args()
    restructure_directories(args.base_path, args.new_dir_name, dry_run=args.dry_run)
