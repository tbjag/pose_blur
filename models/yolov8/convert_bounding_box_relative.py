import os
import json
import glob
import shutil
import random
from PIL import Image

# Base directory containing images and labels
BASE_DIR = "/media/Data_2/person-search/dataset/cuhk_transformed"
CATEGORY_ID = 0
TRAIN_RATIO = 0.7  # 70% for training, 30% for testing

def convert_xyxy_to_yolo_format(bbox, img_width, img_height):
    """
    Convert [x1, y1, x2, y2] format to YOLO [x_center, y_center, width, height] format
    All values normalized to 0-1
    """
    x1, y1, x2, y2 = bbox
    
    # Calculate width and height
    w = x2 - x1
    h = y2 - y1
    
    # Calculate center coordinates
    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height
    
    # Normalize width and height
    width = w / img_width
    height = h / img_height
    
    return [x_center, y_center, width, height]

def convert_to_yolo_format(bbox, img_width, img_height):
    """
    Convert [x, y, w, h] format to YOLO [x_center, y_center, width, height] format
    All values normalized to 0-1
    """
    x, y, w, h = bbox
    
    # Calculate center coordinates
    x_center = (x + w/2) / img_width
    y_center = (y + h/2) / img_height
    
    # Calculate normalized width and height
    width = w / img_width
    height = h / img_height
    
    return [x_center, y_center, width, height]

def split_dataset():
    """Split the dataset into train and test sets"""
    # Create directories
    for split in ['train', 'test']:
        os.makedirs(os.path.join(BASE_DIR,split), exist_ok=True)
        os.makedirs(os.path.join(BASE_DIR, split), exist_ok=True)
    
    # Get all image files from source directory
    source_imgs = glob.glob(os.path.join(BASE_DIR, "*.jpg")) + \
                  glob.glob(os.path.join(BASE_DIR, "*.jpeg")) + \
                  glob.glob(os.path.join(BASE_DIR, "*.png"))
    
    # Shuffle files for random split
    random.shuffle(source_imgs)
    
    # Calculate split index
    split_idx = int(len(source_imgs) * TRAIN_RATIO)
    train_imgs = source_imgs[:split_idx]
    test_imgs = source_imgs[split_idx:]
    
    print(f"Splitting dataset: {len(train_imgs)} for training, {len(test_imgs)} for testing")
    
    # Process train images
    for img_file in train_imgs:
        base_name = os.path.basename(img_file)
        # Copy image
        shutil.copy2(img_file, os.path.join(BASE_DIR,  "train", base_name))
        # Copy corresponding JSON if exists
        json_file = os.path.join(BASE_DIR, f"{os.path.splitext(base_name)[0]}.json")
        if os.path.exists(json_file):
            shutil.copy2(json_file, os.path.join(BASE_DIR, "train", os.path.basename(json_file)))
    
    # Process test images
    for img_file in test_imgs:
        base_name = os.path.basename(img_file)
        # Copy image
        shutil.copy2(img_file, os.path.join(BASE_DIR,  "test", base_name))
        # Copy corresponding JSON if exists
        json_file = os.path.join(BASE_DIR, f"{os.path.splitext(base_name)[0]}.json")
        if os.path.exists(json_file):
            shutil.copy2(json_file, os.path.join(BASE_DIR, "test", os.path.basename(json_file)))
    
    print("Dataset split complete!")

def process_directory(subdir):
    """Process all images and corresponding JSON files in a directory"""
    dir = os.path.join(BASE_DIR,   subdir)
    
    print(f"Processing {subdir} directory...")
    image_dir = "/media/Data_2/person-search/dataset/Image/SSM"
    # Get all image files
    img_files = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                glob.glob(os.path.join(image_dir, "*.jpeg")) + \
                glob.glob(os.path.join(image_dir, "*.png"))
    for img_file in img_files:
        base_name = os.path.basename(img_file).split('.')[0]
        json_file = os.path.join(dir, f"{base_name}.json")
        
        if not os.path.exists(json_file):
            print(f"Warning: No JSON file found for {img_file}")
            continue
        
        # Get image dimensions
        try:
            with Image.open(img_file) as img:
                img_width, img_height, = img.size
        except Exception as e:
            print(f"Error opening image {img_file}: {e}")
            continue
            
        # Load JSON data
        try:
            with open(json_file, 'r') as f:
                bboxes = json.load(f)
        except Exception as e:
            print(f"Error loading JSON {json_file}: {e}")
            continue
            
        # Create output file
        output_file = os.path.join(dir, f"{base_name}.json")
        
        yolo_bboxes = []
        for bbox in bboxes:
            yolo_bbox = convert_xyxy_to_yolo_format(bbox, img_width, img_height)
            yolo_bboxes.append([
                round(yolo_bbox[0], 6),
                round(yolo_bbox[1], 6),
                round(yolo_bbox[2], 6),
                round(yolo_bbox[3], 6)
            ])
        
        with open(output_file, 'w') as f:
            json.dump(yolo_bboxes, f)
                
        print(f"Processed {base_name}: {len(bboxes)} bounding boxes")
        # try:
        #     with open(output_file, 'w') as f:
        #         for bbox in bboxes:
        #             yolo_bbox = convert_xyxy_to_yolo_format(bbox, img_width, img_height)
        #             f.write(f"{CATEGORY_ID} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
            
        #     # Delete the JSON file after successful conversion
        #     os.remove(json_file)
        #     print(f"Processed {base_name}: {len(bboxes)} bounding boxes - JSON file deleted")
        # except Exception as e:
        #     print(f"Error writing or deleting file: {e}")

if __name__ == "__main__":
    # First split the dataset
    split_dataset()
    
    # # Create output directories
    
    # Process train and test directories
    for subdir in ["train", "test"]:
        process_directory(subdir)
    
    print("Conversion complete!")