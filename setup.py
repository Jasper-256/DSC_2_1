#!/usr/bin/env python3
"""
Download & extract the first 16 train files and first 4 test files
from the Amar-S/MOVi-MC-AC dataset on Hugging Face.
"""

import os
import tarfile
from huggingface_hub import HfApi, hf_hub_download

# — CONFIGURATION —
REPO_ID        = "Amar-S/MOVi-MC-AC"
REPO_TYPE      = "dataset"
LOCAL_TRAIN_DIR = "train"
LOCAL_TEST_DIR  = "test"
NUM_TRAIN = 4
NUM_TEST  = 4

# — MAKE OUTPUT FOLDERS —
os.makedirs(LOCAL_TRAIN_DIR, exist_ok=True)
os.makedirs(LOCAL_TEST_DIR,  exist_ok=True)

# — LIST ALL FILES IN THE DATASET REPO —
api = HfApi()
all_files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)

# — FILTER FOR .tar.gz IN train/ AND test/ —
train_files = sorted(f for f in all_files if f.startswith("train/") and f.endswith(".tar.gz"))[:NUM_TRAIN]
test_files  = sorted(f for f in all_files if f.startswith("test/")  and f.endswith(".tar.gz"))[:NUM_TEST]

def download_and_extract(remote_path: str, out_dir: str):
    # Download to cache (returns local path)
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=remote_path
    )
    # Extract
    with tarfile.open(local_path, "r:gz") as tf:
        tf.extractall(path=out_dir)
    print(f"✔ {remote_path} → {out_dir}/")

# — PROCESS train FILES —
print(f"Downloading {len(train_files)} train files …")
for fn in train_files:
    download_and_extract(fn, LOCAL_TRAIN_DIR)

# — PROCESS test FILES —
print(f"\nDownloading {len(test_files)} test files …")
for fn in test_files:
    download_and_extract(fn, LOCAL_TEST_DIR)

print("\nAll done!")
