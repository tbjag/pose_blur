import os
import random
import shutil

# Define paths
dataset_dir = "/media/Data_2/person-search/dataset/cuhk_transformed"
train_dir = "/media/Data_2/person-search/dataset/train"
test_dir = "/media/Data_2/person-search/dataset/test"

# Create train and test directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get all image and JSON paths
image_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".jpg") or f.endswith(".png")])  # Adjust extensions if needed
json_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".json")])

# Ensure each image has a corresponding JSON file
paired_files = [(img, img.replace(".jpg", ".json").replace(".png", ".json")) for img in image_files if img.replace(".jpg", ".json").replace(".png", ".json") in json_files]

# Shuffle data
random.shuffle(paired_files)

# Compute split sizes
num_train = int(len(paired_files) * 0.7)
train_set = paired_files[:num_train]
test_set = paired_files[num_train:]

# Function to move files
def move_files(file_list, destination):
    for img, json in file_list:
        shutil.move(os.path.join(dataset_dir, img), os.path.join(destination, img))
        shutil.move(os.path.join(dataset_dir, json), os.path.join(destination, json))

# Move files
move_files(train_set, train_dir)
move_files(test_set, test_dir)

print(f"Dataset split complete!")
print(f"Train: {len(train_set)} images")
print(f"Test: {len(test_set)} images")
