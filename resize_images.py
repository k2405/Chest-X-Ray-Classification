import os
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def resize_image(file_name, input_dir, output_dir, target_size):
    """
    Resizes a single image and saves it to the output directory.
    """
    input_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, file_name)
    
    # Skip if the output file already exists
    if os.path.exists(output_path):
        return False  # Indicate that the image was skipped

    try:
        with Image.open(input_path) as img:
            img = img.convert("RGB")
            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
            os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
            img_resized.save(output_path, format="PNG", optimize=True, icc_profile=None)
        return True  # Indicate that the image was successfully resized
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return False

def resize_images_parallel(input_dir, output_dir, target_size=(224, 224), max_workers=4):
    """
    Resize all .png images in a directory in parallel to the target size, 
    skipping images that already exist in the output directory.
    
    Args:
        input_dir (str): Path to the input directory containing .png images.
        output_dir (str): Path to save the resized images.
        target_size (tuple): Desired size of the output images as (width, height).
        max_workers (int): Number of threads to use for parallel processing.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
    total_images = len(all_files)

    # Use tqdm for progress tracking
    with tqdm(total=total_images, desc="Resizing images", unit="file", dynamic_ncols=True) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for file_name in all_files:
                futures.append(executor.submit(resize_image, file_name, input_dir, output_dir, target_size))
            
            # Update progress bar as tasks complete
            for future in futures:
                if future.result():  # Only update for successfully processed images
                    pbar.set_description_str("Resizing images")
                    pbar.update(1)
                else:
                    pbar.set_description_str("Skipped image")
                    pbar.update(1)

# Example usage
if __name__ == "__main__":
    script_dir = os.getcwd()
    input_directory = f"{script_dir}/dataset/data/train"
    output_directory = f"{script_dir}/dataset/data/train_512"
    target_size = (512, 512)
    max_workers = 5

    resize_images_parallel(input_directory, output_directory, target_size, max_workers)

    input_directory = f"{script_dir}/dataset/data/test"
    output_directory = f"{script_dir}/dataset/data/test_512"
    resize_images_parallel(input_directory, output_directory, target_size, max_workers)
