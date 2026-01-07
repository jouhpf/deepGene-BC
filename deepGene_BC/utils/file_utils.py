import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Union


def copy_files(
        sample_info: Union[pd.DataFrame, str, Path],
        by: str,
        source_dir: Union[str, Path],
        target_dir: Union[str, Path],
        delete: bool = False,
        verbose: bool = True
) -> None:
    """
    Select files mentioned in a CSV/DataFrame and copy them to a target directory.

    Args:
        sample_info: Path to CSV file or DataFrame containing file information
        by: Column name containing filenames to copy
        source_dir: Source directory containing files to copy
        target_dir: Target directory where files will be copied
        delete: If True, delete contents of target directory before copying
        verbose: If True, print detailed progress and summary
    """
    # Convert to Path objects
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    # Validate source directory
    if not source_dir.exists() or not source_dir.is_dir():
        raise ValueError(f"Source directory '{source_dir}' does not exist or is not a directory")

    # Read CSV if path provided
    if isinstance(sample_info, (str, Path)):
        try:
            sample_info = pd.read_csv(sample_info)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File '{sample_info}' not found") from e

    # Validate column exists
    if by not in sample_info.columns:
        raise ValueError(f"Column '{by}' not found in data. Available columns: {list(sample_info.columns)}")

    # Get unique filenames
    sample_filenames = sample_info[by].dropna().astype(str).unique().tolist()

    if not sample_filenames:
        if verbose:
            print("Warning: No filenames found in the specified column")
        return

    # Handle target directory
    if target_dir.exists() and delete:
        if verbose and any(target_dir.iterdir()):
            print(f"Cleaning target directory: {target_dir}")

        for item in target_dir.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to delete {item}: {e}")

    target_dir.mkdir(parents=True, exist_ok=True)

    # Get available files in source directory (set for O(1) lookups)
    try:
        available_files = {f.name for f in source_dir.iterdir() if f.is_file()}
    except Exception as e:
        raise RuntimeError(f"Failed to read source directory: {e}") from e

    # Copy files with progress bar
    copied = 0
    missing = 0
    errors = 0
    missing_files = []

    for sample_filename in tqdm(sample_filenames, desc="Copying files", disable=not verbose):
        if sample_filename not in available_files:
            missing += 1
            missing_files.append(sample_filename)
            continue

        source_file = source_dir / sample_filename
        target_file = target_dir / sample_filename

        try:
            shutil.copy2(source_file, target_file)
            copied += 1
        except Exception as e:
            errors += 1
            if verbose:
                print(f"Error copying '{sample_filename}': {e}")

    # Print summary if verbose
    if verbose:
        print(f"\n=== Copy Summary ===")
        print(f"Total files requested: {len(sample_filenames)}")
        print(f"Successfully copied:   {copied}")
        print(f"Missing from source:   {missing}")
        print(f"Errors during copy:    {errors}")

        if missing_files and verbose:
            print(f"\nMissing files ({len(missing_files)}):")
            for i, fname in enumerate(missing_files[:10]):  # Show first 10
                print(f"  {fname}")
            if len(missing_files) > 10:
                print(f"  ... and {len(missing_files) - 10} more")

    # If all files missing, provide more helpful error
    if copied == 0 and missing > 0 and errors == 0:
        print("\nWarning: No files were copied. Possible issues:")
        print(f"1. Check that filenames in column '{by}' match actual filenames")
        print(f"2. Verify source directory: {source_dir}")
        print(f"3. File extensions might not match (case-sensitive on some systems)")
