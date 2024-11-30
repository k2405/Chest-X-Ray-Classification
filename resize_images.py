import os
from PIL import Image

def resize_images(input_dir, output_dir, target_size=(224, 224)):
    """
    Resize all images in a directory to the target size while retaining quality,
    skipping images that already exist in the output directory.

    Args:
        input_dir (str): Path to the input directory containing images.
        output_dir (str): Path to save the resized images.
        target_size (tuple): Desired size of the output images as (width, height).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through all files in the input directory
    for root, _, files in os.walk(input_dir):
        for file_name in files:
            input_path = os.path.join(root, file_name)
            
            # Construct the output file path
            relative_path = os.path.relpath(input_path, input_dir)
            output_path = os.path.join(output_dir, relative_path)
            
            # Skip if the output file already exists
            if os.path.exists(output_path):
                print(f"Skipping (already exists): {output_path}")
                continue
            
            # Ensure the file is an image
            try:
                with Image.open(input_path) as img:
                    # Convert to RGB (if not already)
                    img = img.convert("RGB")
                    
                    # Resize the image using high-quality downsampling
                    img_resized = img.resize(target_size, Image.LANCZOS)
                    
                    # Ensure the output directory structure exists
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # Save the resized image to the output directory
                    img_resized.save(output_path, format="PNG", optimize=True, exif=img.info.get('exif'))
                    print(f"Resized and saved: {output_path}")
            except Exception as e:
                print(f"Error processing file {input_path}: {e}")

# Example usage
if __name__ == "__main__":
    input_directory = "/home/jon/projects/Xrays/dataset/data/train"
    output_directory = "/home/jon/projects/Xrays/dataset/data/train_224"
    target_size = (224, 224)

    resize_images(input_directory, output_directory, target_size)
