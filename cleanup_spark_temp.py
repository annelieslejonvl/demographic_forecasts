"""
Clean up Spark temporary files and checkpoint directories.

This script removes:
- Spark checkpoint directories (spark_checkpoint_*)
- Spark temporary files
- Old metastore_db directories
- Derby logs

Usage:
    python cleanup_spark_temp.py
    python cleanup_spark_temp.py --dry-run  # Preview what would be deleted
"""
import os
import shutil
import tempfile
import argparse
from pathlib import Path


def get_size_mb(path):
    """Calculate directory size in MB."""
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total += get_size_mb(entry.path)
    except (PermissionError, FileNotFoundError):
        pass
    return total / (1024 * 1024)


def cleanup_temp_files(dry_run=False):
    """Clean up Spark temporary files."""
    temp_dir = Path(tempfile.gettempdir())

    patterns = [
        "spark_checkpoint_*",
        "spark-*",
        "blockmgr-*",
    ]

    total_size = 0
    total_count = 0

    print(f"Scanning temp directory: {temp_dir}")
    print("="*60)

    for pattern in patterns:
        for path in temp_dir.glob(pattern):
            if path.is_dir():
                size_mb = get_size_mb(str(path))
                total_size += size_mb
                total_count += 1

                print(f"  {'[DRY RUN] ' if dry_run else ''}Found: {path.name}")
                print(f"            Size: {size_mb:.2f} MB")

                if not dry_run:
                    try:
                        shutil.rmtree(path)
                        print(f"            ✓ Deleted")
                    except Exception as e:
                        print(f"            ⚠️  Error: {e}")
                print()

    # Also check current directory
    current_dir = Path.cwd()
    for item in ["metastore_db", "derby.log"]:
        path = current_dir / item
        if path.exists():
            if path.is_dir():
                size_mb = get_size_mb(str(path))
            else:
                size_mb = path.stat().st_size / (1024 * 1024)

            total_size += size_mb
            total_count += 1

            print(f"  {'[DRY RUN] ' if dry_run else ''}Found: {item} (in current directory)")
            print(f"            Size: {size_mb:.2f} MB")

            if not dry_run:
                try:
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                    print(f"            ✓ Deleted")
                except Exception as e:
                    print(f"            ⚠️  Error: {e}")
            print()

    print("="*60)
    print(f"{'Would remove' if dry_run else 'Removed'}: {total_count} items")
    print(f"Total size: {total_size:.2f} MB ({total_size/1024:.2f} GB)")

    if dry_run:
        print("\nRun without --dry-run to actually delete these files.")

    return total_count, total_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up Spark temporary files")
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview what would be deleted without actually deleting')
    args = parser.parse_args()

    print("Spark Temporary Files Cleanup")
    print("="*60)

    count, size = cleanup_temp_files(dry_run=args.dry_run)

    if count == 0:
        print("No Spark temporary files found. Everything is clean!")
    elif args.dry_run:
        print(f"\nRun 'python cleanup_spark_temp.py' to delete {count} items ({size:.2f} MB)")
    else:
        print(f"\nCleanup complete! Freed up {size:.2f} MB of disk space.")
