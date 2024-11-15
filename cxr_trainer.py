# cxr_trainer.py

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.models import resnet152, ResNet152_Weights
from torchvision.models.densenet import DenseNet121_Weights
from sklearn.model_selection import train_test_split
from collections import defaultdict
from PIL import Image
from torch.amp import GradScaler, autocast
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import pandas as pd
from datetime import datetime
import time
import hashlib
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_amp = True
scaler = torch.amp.GradScaler(enabled=use_amp)

script_dir = os.path.dirname(os.path.abspath(__file__))

# Argument parser for user inputs
parser = argparse.ArgumentParser(description="Train a Chest X-ray model.")
parser.add_argument("--train_dir", type=str, default=f"{script_dir}/dataset/data/train", help="Path to training data. ")
parser.add_argument("--test_dir", type=str, default=f"{script_dir}/dataset/data/test", help="Path to test data. ")
parser.add_argument("--label_file", type=str, default=f"{script_dir}/dataset/Data_Entry_2017_v2020.csv", help="Path to label file. ")
parser.add_argument("--cache_dir", type=str, default="cache", help="Directory for caching. (default: 'cache')")
parser.add_argument("--models_dir", type=str, default=f"{script_dir}/models", help="Directory for saving models. ")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size. (default: 128)")
parser.add_argument("--num_workers", type=int, default=max(1, os.cpu_count() // 2), help="Number of parallel workers. (default: CPU cores // 2)")
parser.add_argument("--disable_cache", action="store_true", help="Disable caching. (default: False, caching is enabled by default)")
parser.add_argument("--rebuild_cache", action="store_true", help="Force cache rebuilding. (default: False)")
parser.add_argument("--num_train_images", type=int, default=None, help="Number of training images to load. Set to None to load all images. (default: None)")
parser.add_argument("--num_test_images", type=int, default=None, help="Number of test images to load. Set to None to load all images. (default: None)")
parser.add_argument("--checkpoint_interval", type=int, default=5, help="Save model checkpoints every N epochs. (default: 5)")
args = parser.parse_args()

# Caching and Preprocessing
def batch_iterable(iterable, batch_size):
    """Yield successive batches from an iterable."""
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch

def preprocess_and_save(dataset, transform, cache_dir="cache", num_workers=4, batch_size=32, enable_cache=True, rebuild_cache=False):
    """
    Preprocess and save dataset images in batches, with optional caching and multiprocessing.

    Args:
        dataset (list): List of (patient_id, image_path) pairs.
        transform (callable): Transformations to apply to the images.
        cache_dir (str): Directory to store cached preprocessed images.
        num_workers (int): Number of parallel workers for preprocessing.
        batch_size (int): Number of items to process in each batch.
        enable_cache (bool): If True, use caching; otherwise, process all files without caching.
        rebuild_cache (bool): If True, overwrite existing cache files.

    Returns:
        list: A list of (patient_id, cached_image_path or transformed_image) pairs.
    """
    if not enable_cache:
        print("Caching is disabled. Processing images in memory.")
    
    print("\nBuilding cache...")
    if enable_cache:
        os.makedirs(cache_dir, exist_ok=True)

        # Clear cache directory if rebuilding
        if rebuild_cache:
            print(f"Rebuilding cache. Clearing directory: {cache_dir}")
            for file in os.listdir(cache_dir):
                file_path = os.path.join(cache_dir, file)
                os.remove(file_path)

    def process_batch(batch):
        results = []
        for patient_id, image_path in batch:
            cache_path = os.path.join(cache_dir, f"{os.path.basename(image_path)}.pt") if enable_cache else None
            if not enable_cache or rebuild_cache or (enable_cache and not os.path.exists(cache_path)):
                try:
                    image = Image.open(image_path).convert("RGB")
                    image = transform(image)
                    if enable_cache:
                        torch.save(image, cache_path)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
            results.append((patient_id, cache_path if enable_cache else image))
        return results

    def worker(input_queue, output_queue):
        while True:
            batch = input_queue.get()
            if batch is None:  # End of queue signal
                break
            output_queue.put(process_batch(batch))

    # Create queues
    input_queue = mp.Queue()
    output_queue = mp.Queue()

    # Start worker processes
    workers = []
    for i in range(num_workers):
        print(f"Starting worker process {i+1}/{num_workers}", end="\r")
        process = mp.Process(target=worker, args=(input_queue, output_queue))
        process.start()
        workers.append(process)
    print()

    # Divide dataset into batches and add to queue
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    for i, batch in enumerate(batch_iterable(dataset, batch_size)):
        print(f"Adding batches to queue: {i+1}/{total_batches}", end="\r")
        input_queue.put(batch)
    print()

    # Signal workers to terminate
    for i in range(num_workers):
        input_queue.put(None)

    # Collect results
    preprocessed_dataset = []
    start_time = time.time()
    for i in range(total_batches):
        batch_start = time.time()
        preprocessed_dataset.extend(output_queue.get())
        batch_end = time.time()
        
        # Calculate elapsed time and remaining time
        elapsed_time = batch_end - start_time
        batches_processed = i + 1
        avg_batch_time = elapsed_time / batches_processed
        remaining_time = avg_batch_time * (total_batches - batches_processed)
        eta = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
        
        print(f"Collecting results: {batches_processed}/{total_batches}, ETA: {eta}", end="\r")
    print()

    # Wait for workers to finish
    for process in workers:
        process.join()

    print(f"Preprocessing complete. Total processed items: {len(preprocessed_dataset)}")
    return preprocessed_dataset

# Dataset Loading
def load_dataset(directory, random_selection=False, seed=None, max_total_images=None):
    if random_selection and seed is not None:
        random.seed(seed)

    patient_images = defaultdict(list)

    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".png"):
            patient_id = filename.split("_")[0]
            patient_images[patient_id].append(os.path.join(directory, filename))

    selected_images = []
    for patient_id, images in patient_images.items():
        selected_image = random.choice(images) if random_selection else images[0]
        selected_images.append((patient_id, selected_image))
        if max_total_images is not None and len(selected_images) >= max_total_images:
            break

    return selected_images

def load_labels(csv_path, conditions):
    df = pd.read_csv(csv_path)
    labels = {}
    for _, row in df.iterrows():
        image_path = row['Image Index']
        findings = row['Finding Labels'].split('|')
        label_vector = [1 if condition in findings else 0 for condition in conditions]
        labels[image_path] = label_vector
    return labels

# Dataset Class
class ChestXray14CachedDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading cached tensors and multi-label vectors.
    """
    def __init__(self, dataset, label_mapping, pathologies):
        """
        Args:
            dataset (list): List of (patient_id, cached_tensor_path) pairs.
            label_mapping (dict): Dictionary mapping image paths to label vectors.
            pathologies (list): List of pathologies for model alignment.
        """
        self.dataset = dataset
        self.label_mapping = label_mapping
        self.pathologies = pathologies

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        patient_id, tensor_path = self.dataset[idx]
        image = torch.load(tensor_path, weights_only=True)  # Set weights_only=True for security
        image_name = os.path.basename(tensor_path).replace(".pt", "")
        labels = self.label_mapping[image_name]
        label_vector = torch.tensor([labels[self.pathologies.index(p)] for p in self.pathologies], dtype=torch.float)
        return {"img": image, "lab": label_vector}

# Transforms
def get_data_transforms(image_size=224):
    return {
        "train": transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomCrop(image_size, padding=10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

# Model Preparation
def prepare_model():
    model = resnet152(weights=ResNet152_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 14),
    )
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    return model

# Training Loop with Mixed Precision
def train_model(model, train_loader, val_loader, num_epochs=45, lr=0.0005, models_dir="models", checkpoint_interval=5):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    train_losses, val_losses = [], []
    epoch_times = []  # Store times for completed epochs

    for epoch in range(num_epochs):
        start_time = time.time()  # Start timing the epoch
        train_loss, val_loss = 0.0, 0.0
        os.makedirs(models_dir, exist_ok=True)  # Ensure models directory exists

        # Training Phase
        model.train()
        for batch in train_loader:
            images, labels = batch['img'].to(device), batch['lab'].to(device)

            optimizer.zero_grad()
            with autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp): 
                outputs = model(images)
                loss = criterion(outputs, labels.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        # Validation Phase
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch['img'].to(device), batch['lab'].to(device)
                with autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels.float())
                val_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))

        

        # Calculate epoch duration and remaining time
        epoch_duration = time.time() - start_time
        epoch_times.append(epoch_duration)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_time = avg_epoch_time * (num_epochs - (epoch + 1))

        # Format remaining time as HH:MM:SS
        remaining_time_str = time.strftime('%H:%M:%S', time.gmtime(remaining_time))

        # Print epoch summary with timing and remaining time
        print(f"Epoch {epoch+1:03d}/{num_epochs:03d}, "
              f"Train Loss: {train_losses[-1]:.6f}, "
              f"Val Loss: {val_losses[-1]:.6f}, "
              f"Time: {epoch_duration:.2f} sec, "
              f"ETA: {remaining_time_str}", end=" ")
        
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = f"{models_dir}/checkpoint_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(", Checkpoint saved")
        else:
            print()

    return model

# Evaluation
def evaluate_model(model, val_loader, threshold=0.2, target_names=None):
    model.eval()

    predictions, actuals = [], []
    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch['img'].to(device), batch['lab'].to(device)
            outputs = torch.sigmoid(model(images))  # Sigmoid for probabilities
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(labels.cpu().numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Apply threshold
    binary_predictions = (predictions > threshold).astype(int)

    # Generate classification report with class names
    report = classification_report(
        actuals,
        binary_predictions,
        target_names=target_names,
        zero_division=0
    )
    print(report)

    accuracy = accuracy_score(actuals, binary_predictions)
    recall = recall_score(actuals, binary_predictions, average="micro", zero_division=0)
    precision = precision_score(actuals, binary_predictions, average="micro", zero_division=0)
    f1 = f1_score(actuals, binary_predictions, average="micro", zero_division=0)

    print(f'Accuracy: {accuracy} \nRecall: {recall} \nPrecision: {precision} \nF1: {f1}')

    auc_score = roc_auc_score(actuals, predictions, average="micro")
    print(f'ROC AUC Score: {auc_score}')

# Main Workflow
if __name__ == "__main__":
    train_dir = args.train_dir
    test_dir = args.test_dir
    labels_file = args.label_file
    cache_dir = args.cache_dir
    models_dir = args.models_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    enable_cache = not args.disable_cache  # Cache is enabled by default unless explicitly disabled
    rebuild_cache = args.rebuild_cache
    num_train_images = args.num_train_images
    num_test_images = args.num_test_images
    checkpoint_interval = args.checkpoint_interval

    # Ensure directories exist
    os.makedirs(models_dir, exist_ok=True)

    print(f"Train Directory: {train_dir}")
    print(f"Test Directory: {test_dir}")
    print(f"Cache Directory: {cache_dir}")
    print(f"Models Directory: {models_dir}")
    print(f"Batch Size: {batch_size}")
    print(f"Number of Workers: {num_workers}")
    print(f"Enable Cache: {enable_cache}")
    print(f"Checkpoint Interval: {checkpoint_interval}")


    common_pathologies = ["Infiltration", "Effusion", "Atelectasis", "Nodule",
                        "Mass", "Pneumothorax", "Consolidation", "Pleural_Thickening",
                        "Cardiomegaly", "Emphysema", "Edema", "Fibrosis", "Pneumonia", "Hernia"]
    label_mapping = load_labels(labels_file, common_pathologies)

    data_transforms = get_data_transforms() 

    train_dataset = load_dataset(train_dir, random_selection=True, seed=42, max_total_images=num_train_images)
    test_dataset = load_dataset(test_dir, random_selection=False, max_total_images=num_test_images)

    train_subset, val_subset = train_test_split(train_dataset, test_size=0.2, random_state=42)


    train_dataset = preprocess_and_save(
        train_subset,
        transform=data_transforms["train"],
        cache_dir="train_cache",
        batch_size=batch_size,
        enable_cache=enable_cache,
        rebuild_cache=rebuild_cache 
    )

    val_dataset = preprocess_and_save(
        val_subset,
        transform=data_transforms["val"],
        cache_dir="val_cache",
        batch_size=batch_size,
        enable_cache=enable_cache,
        rebuild_cache=rebuild_cache 
    )

    train_loader = DataLoader(
        ChestXray14CachedDataset(train_dataset, label_mapping, common_pathologies),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=6, persistent_workers=True
    )

    val_loader = DataLoader(
        ChestXray14CachedDataset(val_dataset, label_mapping, common_pathologies),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=6, persistent_workers=True
    )


    model = prepare_model()
    model = train_model(model, train_loader, val_loader, num_epochs=15, lr=0.0005)
    evaluate_model(model, val_loader, threshold=0.2, target_names=common_pathologies)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{models_dir}/model_{timestamp}.pth"

    print("Saving model...")
    torch.save(model.state_dict(), model_filename)
    print("Model saved successfully.")
