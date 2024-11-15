# dataset_organizer.py

import os
import shutil
from concurrent.futures import ThreadPoolExecutor

# Define paths (adjusted for WSL compatibility with Windows paths)
project_dir = "/mnt/b/Xray/"
images_dir = os.path.join(project_dir, "images", "images")
train_list_path = os.path.join(project_dir, "train_val_list.txt")
test_list_path = os.path.join(project_dir, "test_list.txt")
train_dir = os.path.join(project_dir, "data", "train")
test_dir = os.path.join(project_dir, "data", "test")

# Create "train" and "test" directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Function to read filenames from a file
def read_file_list(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines()]

# Read file lists
train_files = read_file_list(train_list_path)
test_files = read_file_list(test_list_path)

# Function to move a single file (optimized for cross-filesystem)
def move_file_cross_filesystem(source_file, target_file):
    try:
        shutil.move(source_file, target_file)  # Handles cross-filesystem moves
    except FileNotFoundError:
        print(f"File not found: {source_file}")
    except Exception as e:
        print(f"Error moving {source_file} -> {target_file}: {e}")

# Batch processing function
def move_files_batch(file_list, source_dir, target_dir, batch_size=1000):
    batch = []
    for file_name in file_list:
        source_file = os.path.join(source_dir, file_name)
        target_file = os.path.join(target_dir, file_name)
        batch.append((source_file, target_file))
        if len(batch) >= batch_size:
            process_batch(batch)
            batch = []
            print(f"Processed {len(file_list)} files.")
    if batch:
        process_batch(batch)
        print(f"Processed {len(file_list)} files.")

# Function to process a batch of files in parallel
def process_batch(batch):
    with ThreadPoolExecutor() as executor: 
        executor.map(lambda files: move_file_cross_filesystem(*files), batch)

# Move files into respective directories
move_files_batch(train_files, images_dir, train_dir)
move_files_batch(test_files, images_dir, test_dir)

print("Files have been organized into 'train' and 'test' directories.")