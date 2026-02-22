# ABOUTME: Downloads and preprocesses MNIST handwritten digit data into .npy files.
# ABOUTME: Run with `python download_data.py` to download the dataset for the ascent project.

"""
Downloads the MNIST handwritten digit dataset and saves it as numpy arrays.

This script downloads MNIST from the official source, extracts the images and
labels, normalizes pixel values to [0, 1], and saves everything as .npy files
in a data/ subdirectory.

Usage:
    python download_data.py           # Download full MNIST (digits 0-9)
    python download_data.py --subset  # Download and filter to digits 0-4 only

Output files (in data/ directory):
    train_images.npy  — Training images, shape (N, 784), float32, values in [0, 1]
    train_labels.npy  — Training labels, shape (N,), int
    test_images.npy   — Test images, shape (N, 784), float32, values in [0, 1]
    test_labels.npy   — Test labels, shape (N,), int
"""

import gzip
import os
import struct
import sys
import urllib.request

import numpy as np

# MNIST source URLs
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")


def download_file(url, filepath):
    """Download a file if it doesn't already exist."""
    if os.path.exists(filepath):
        print(f"  Already exists: {os.path.basename(filepath)}")
        return

    print(f"  Downloading: {os.path.basename(filepath)}...", end=" ", flush=True)
    urllib.request.urlretrieve(url, filepath)
    print("done")


def read_images(filepath):
    """Read MNIST image file and return as numpy array."""
    with gzip.open(filepath, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Invalid magic number: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num_images, rows * cols).astype(np.float32) / 255.0


def read_labels(filepath):
    """Read MNIST label file and return as numpy array."""
    with gzip.open(filepath, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Invalid magic number: {magic}"
        return np.frombuffer(f.read(), dtype=np.uint8).astype(int)


def filter_subset(images, labels, max_digit=4):
    """Keep only digits 0 through max_digit."""
    mask = labels <= max_digit
    return images[mask], labels[mask]


def main():
    subset_mode = "--subset" in sys.argv

    print("MNIST Data Download")
    print("=" * 40)

    # Create directories
    os.makedirs(RAW_DIR, exist_ok=True)

    # Download raw files
    print("\nDownloading MNIST files...")
    for name, filename in FILES.items():
        url = BASE_URL + filename
        filepath = os.path.join(RAW_DIR, filename)
        download_file(url, filepath)

    # Read and process
    print("\nProcessing...")
    train_images = read_images(os.path.join(RAW_DIR, FILES["train_images"]))
    train_labels = read_labels(os.path.join(RAW_DIR, FILES["train_labels"]))
    test_images = read_images(os.path.join(RAW_DIR, FILES["test_images"]))
    test_labels = read_labels(os.path.join(RAW_DIR, FILES["test_labels"]))

    if subset_mode:
        print("  Filtering to digits 0-4...")
        train_images, train_labels = filter_subset(train_images, train_labels)
        test_images, test_labels = filter_subset(test_images, test_labels)

    # Save as .npy
    print("\nSaving .npy files...")
    np.save(os.path.join(DATA_DIR, "train_images.npy"), train_images)
    np.save(os.path.join(DATA_DIR, "train_labels.npy"), train_labels)
    np.save(os.path.join(DATA_DIR, "test_images.npy"), test_images)
    np.save(os.path.join(DATA_DIR, "test_labels.npy"), test_labels)

    # Summary
    print(f"\nDone! Files saved to {DATA_DIR}/")
    print(f"  train_images.npy: {train_images.shape} (float32, values in [0, 1])")
    print(f"  train_labels.npy: {train_labels.shape} (int, digits {train_labels.min()}-{train_labels.max()})")
    print(f"  test_images.npy:  {test_images.shape}")
    print(f"  test_labels.npy:  {test_labels.shape}")

    if subset_mode:
        print(f"\n  Subset mode: only digits 0-4 ({len(train_images)} train, {len(test_images)} test)")
    else:
        print(f"\n  Full dataset: digits 0-9 ({len(train_images)} train, {len(test_images)} test)")

    print("\nUsage in your code:")
    print('  train_images = np.load("data/train_images.npy")')
    print('  train_labels = np.load("data/train_labels.npy")')


if __name__ == "__main__":
    main()
