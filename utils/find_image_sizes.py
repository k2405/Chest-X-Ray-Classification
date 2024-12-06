import pandas as pd

# Load the dataset
file_path = "/mnt/b/Xray/dataset/Data_Entry_2017_v2020.csv"
data = pd.read_csv(file_path)

# Verify the columns
print(data.columns)

# Extract width and height from the appropriate columns
image_width = data['OriginalImage[Width']
image_height = data['Height]']

# Find the maximum and minimum image sizes
max_width = image_width.max()
max_height = image_height.max()
min_width = image_width.min()
min_height = image_height.min()

# Print the results
print(f"Maximum width: {max_width}, Maximum height: {max_height}")
print(f"Minimum width: {min_width}, Minimum height: {min_height}")
