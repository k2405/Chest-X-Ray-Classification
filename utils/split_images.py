from PIL import Image
import os
import glob
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

def split_image_recursively(image_array, base_name, sizes, level=0, index=0, result_list=None):
    """
    Recursively split an image into progressively smaller sizes using NumPy for efficiency.

    Args:
    - image_array: NumPy array of the image to process.
    - base_name: Base name for naming purposes.
    - sizes: List of sizes to split into (e.g., [1024, 512, 256, 128, 64, 32]).
    - level: Current recursion level for naming purposes.
    - index: Index of the sub-image for naming purposes.
    - result_list: List to collect 32x32 images for later processing.
    """
    height, width = image_array.shape[:2]

    # If the current image size is smaller than the next target size, stop recursion
    if not sizes or width < sizes[0] or height < sizes[0]:
        return

    # Create sub-images using NumPy slicing based on parity of x and y
    sub_images = [
        (image_array[::2, ::2], 0),    # sub_image_index 0: y%2==0, x%2==0
        (image_array[::2, 1::2], 1),   # sub_image_index 1: y%2==0, x%2==1
        (image_array[1::2, ::2], 2),   # sub_image_index 2: y%2==1, x%2==0
        (image_array[1::2, 1::2], 3),  # sub_image_index 3: y%2==1, x%2==1
    ]

    for sub_image_array, sub_index in sub_images:
        new_height, new_width = sub_image_array.shape[:2]
        sub_image_index = index * 4 + sub_index

        # If the sub-image is the target size, add it to the result list
        if new_width == sizes[-1]:
            # Use .copy() to avoid referencing the same array
            result_list.append((sub_image_array.copy(), base_name, sub_image_index))
        else:
            # Recursively process the sub-image if sizes remain
            split_image_recursively(
                sub_image_array, base_name, sizes[1:], level=level+1, index=sub_image_index, result_list=result_list
            )

def process_image(image_path, sizes, target_size):
    """
    Process a single image: split recursively and collect target size images.

    Args:
    - image_path: Path to the input image.
    - sizes: List of sizes to split into.
    - target_size: Target size to keep (e.g., 32).
    Returns:
    - result_list: List of tuples (sub_image_array, base_name, sub_image_index).
    """
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Open the image and convert to NumPy array
    image = Image.open(image_path)
    image_array = np.array(image)

    # List to collect 32x32 images
    result_list = []

    # Start recursive splitting
    split_image_recursively(image_array, base_name, sizes, result_list=result_list)

    return result_list

def process_image_wrapper(args):
    """Wrapper function to unpack arguments for multiprocessing."""
    return process_image(*args)

def save_image_wrapper_old(args):
    """Wrapper function to save a single image. Used for multiprocessing."""
    sub_image_array, base_name, sub_image_index, output_dir, target_size = args
    sub_image = Image.fromarray(sub_image_array)
    output_filename = f"{base_name}_{target_size}_{sub_image_index}.png"
    output_path = os.path.join(output_dir, output_filename)
    sub_image.save(output_path)
    return True

def process_all_images_in_batches_old(source_dir, sizes, target_size, output_dir, num_workers=4, batch_size=1000):
    """
    Process all images in a source directory using batching and multiprocessing.

    Args:
    - source_dir: Directory containing input images.
    - sizes: List of sizes to split into.
    - target_size: Target size to keep (e.g., 32).
    - output_dir: Directory to save processed images.
    - num_workers: Number of worker processes to use.
    - batch_size: Number of images to process in each batch.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_images = glob.glob(os.path.join(source_dir, "*.png"))

    # Divide the list of images into batches
    total_images = len(all_images)
    num_batches = (total_images + batch_size - 1) // batch_size  # Ceiling division

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_images)
        batch_images = all_images[start_idx:end_idx]

        print(f"Processing batch {batch_num + 1}/{num_batches} with {len(batch_images)} images.")

        # Prepare tasks for the batch
        tasks = [
            (image_path, sizes, target_size) for image_path in batch_images
        ]

        with Pool(processes=num_workers) as pool:
            # Process images in parallel
            results = list(tqdm(
                pool.imap_unordered(process_image_wrapper, tasks),
                total=len(tasks),
                desc=f"Processing batch {batch_num + 1}/{num_batches}"
            ))

            # Flatten the list of lists
            collected_images = [item for sublist in results for item in sublist]

            # Prepare arguments for saving images in parallel
            save_tasks = [
                (sub_image_array, base_name, sub_image_index, output_dir, target_size)
                for sub_image_array, base_name, sub_image_index in collected_images
            ]

            # Save images in parallel
            list(tqdm(
                pool.imap_unordered(save_image_wrapper, save_tasks),
                total=len(save_tasks),
                desc=f"Saving batch {batch_num + 1}/{num_batches}"
            ))

        # Clear the collected_images list to free memory
        del collected_images

def save_image_wrapper(args):
    """Wrapper function to save a single image. Used for multithreading."""
    sub_image_array, base_name, sub_image_index, output_dir, target_size = args
    sub_image = Image.fromarray(sub_image_array)
    output_filename = f"{base_name}_{target_size}_{sub_image_index}.png"
    output_path = os.path.join(output_dir, output_filename)
    sub_image.save(output_path)
    return True

def process_all_images_in_batches(source_dir, sizes, target_size, output_dir, num_workers=4, batch_size=1000):
    os.makedirs(output_dir, exist_ok=True)
    all_images = glob.glob(os.path.join(source_dir, "*.png"))

    # Divide the list of images into batches
    total_images = len(all_images)
    num_batches = (total_images + batch_size - 1) // batch_size  # Ceiling division

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_images)
        batch_images = all_images[start_idx:end_idx]

        print(f"Processing batch {batch_num + 1}/{num_batches} with {len(batch_images)} images.")

        # Prepare tasks for the batch
        tasks = [
            (image_path, sizes, target_size) for image_path in batch_images
        ]

        with Pool(processes=num_workers) as pool:
            # Process images in parallel (multiprocessing)
            results = list(tqdm(
                pool.imap_unordered(process_image_wrapper, tasks),
                total=len(tasks),
                desc=f"Processing batch {batch_num + 1}/{num_batches}"
            ))

        # Flatten the list of lists
        collected_images = [item for sublist in results for item in sublist]

        # Prepare arguments for saving images
        save_tasks = [
            (sub_image_array, base_name, sub_image_index, output_dir, target_size)
            for sub_image_array, base_name, sub_image_index in collected_images
        ]

        # Save images using multithreading
        with ThreadPool(processes=num_workers * 2) as pool:
            list(tqdm(
                pool.imap_unordered(save_image_wrapper, save_tasks),
                total=len(save_tasks),
                desc=f"Saving batch {batch_num + 1}/{num_batches}"
            ))

        # Clear the collected_images list to free memory
        del collected_images

if __name__ == "__main__":
    # Define parameters
    image_sizes = [1024, 512, 256, 128, 64, 32]  # Adjust sizes according to your images
    target_size = 32
    num_workers = max(1, os.cpu_count() // 2)  # Adjust the number of worker processes here
    batch_size = 2000  # Adjust the batch size according to your memory constraints

    # Process images in the train directory
    source_dir = "/mnt/e/Xray/dataset/data/train"
    output_dir = "/mnt/e/Xray/dataset/data/train_32_split"
    process_all_images_in_batches(
        source_dir, image_sizes, target_size, output_dir,
        num_workers=num_workers, batch_size=batch_size
    )

    # Process images in the test directory
    source_dir = "/mnt/e/Xray/dataset/data/test"
    output_dir = "/mnt/e/Xray/dataset/data/test_32_split"
    process_all_images_in_batches(
        source_dir, image_sizes, target_size, output_dir,
        num_workers=num_workers, batch_size=batch_size
    )
