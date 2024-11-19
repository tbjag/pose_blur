import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
import skimage.io as io
from typing import List, Tuple
import shutil
from sklearn.model_selection import train_test_split


'''
TODO
download coco dataset
unzip into data folder
find annotation folder - feed into main string
'''

class PersonImageProcessor:
    def __init__(self, standard_size: Tuple[int, int] = (512, 512), max_kernel_size: int = 21, sigma: float = 5.0):
        self.standard_size = standard_size
        self.max_kernel_size = max_kernel_size
        self.sigma = sigma
        self.transforms = T.Compose([
            T.Resize(standard_size),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
        ])
        
    def load_and_preprocess_image(self, img_path: str) -> torch.Tensor:
        if img_path.startswith('http'):
            img = Image.fromarray(io.imread(img_path))
        else:
            img = Image.open(img_path)
        return self.transforms(img)

    def scale_bbox(self, bbox: List[float], original_size: Tuple[int, int]) -> List[int]:
        orig_w, orig_h = original_size
        target_h, target_w = self.standard_size
        
        w_scale = target_w / orig_w
        h_scale = target_h / orig_h
        
        scaled_bbox = [
            int(bbox[0] * w_scale),
            int(bbox[1] * h_scale),
            int(bbox[2] * w_scale),
            int(bbox[3] * h_scale)
        ]
        return scaled_bbox

    def create_blurred_image(self, tensor: torch.Tensor, bboxes: List[List[int]]) -> torch.Tensor:
        blurred = tensor.clone()
        for bbox in bboxes:
            x, y, w, h = bbox
            region = blurred[:, y:y+h, x:x+w]
            if w >= 3 and h >= 3:  # Only blur regions that are large enough
                kernel_size = min(self.max_kernel_size, (min(w, h) // 2) * 2 + 1)
                if kernel_size >= 3:
                    padding = kernel_size // 2
                    region = region.unsqueeze(0)
                    region = torch.nn.functional.avg_pool2d(region, 3, stride=1, padding=padding)
                    blurred[:, y:y+h, x:x+w] = region.squeeze(0)
        return blurred

def prepare_coco_blur_pairs(data_dir: str = './data/coco', 
                          paired_dir: str = './data/coco_blur_pairs',
                          ann_file: str = 'val2017',
                          num_images: int = 1000):
    """Create paired dataset of original and person-blurred COCO images"""
    
    os.makedirs(paired_dir, exist_ok=True)
    
    # Initialize COCO dataset
    coco = COCO(ann_file)
    
    # Initialize processor
    processor = PersonImageProcessor()
    
    # Get images containing people
    cat_ids = coco.getCatIds(catNms=['person'])
    img_ids = coco.getImgIds(catIds=cat_ids)[:num_images]
    
    print(f"Processing {len(img_ids)} images...")
    
    for i, img_id in enumerate(img_ids):
        try:
            # Load image info and annotations
            img_info = coco.loadImgs([img_id])[0]
            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=None)
            anns = coco.loadAnns(ann_ids)
            
            # Load and preprocess image
            original_tensor = processor.load_and_preprocess_image(img_info['coco_url'])
            
            # Scale bounding boxes
            original_size = (img_info['width'], img_info['height'])
            scaled_bboxes = [processor.scale_bbox(ann['bbox'], original_size) for ann in anns]
            
            # Create blurred version
            blurred_tensor = processor.create_blurred_image(original_tensor, scaled_bboxes)
            
            # Convert tensors to PIL images
            original_image = T.ToPILImage()(original_tensor)
            blurred_image = T.ToPILImage()(blurred_tensor)
            
            # Create paired image
            paired_image = Image.new('L', (original_image.width * 2, original_image.height))
            paired_image.paste(original_image, (0, 0))
            paired_image.paste(blurred_image, (original_image.width, 0))
            
            # Save paired image
            paired_image.save(os.path.join(paired_dir, f'pair_{img_id}.png'))
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(img_ids)} images")
                
        except Exception as e:
            print(f"Error processing image {img_id}: {str(e)}")
            continue

def split_coco_dataset(paired_dir: str = './data/coco_blur_pairs', train_ratio: float = 0.8):
    """Split the paired dataset into train and test sets"""
    
    train_dir = os.path.join(paired_dir, 'train')
    test_dir = os.path.join(paired_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get all images in the paired directory
    all_images = [f for f in os.listdir(paired_dir) if f.endswith('.png')]
    print(f"Total images found: {len(all_images)}")
    
    # Split images into train and test sets
    train_images, test_images = train_test_split(all_images, 
                                               train_size=train_ratio, 
                                               random_state=42)
    
    # Move images to respective directories
    for image in train_images:
        shutil.move(os.path.join(paired_dir, image),
                   os.path.join(train_dir, image))
    
    for image in test_images:
        shutil.move(os.path.join(paired_dir, image),
                   os.path.join(test_dir, image))
    
    print(f"Dataset split complete: {len(train_images)} training images, {len(test_images)} test images")

if __name__ == '__main__':
    data_dir = './data/coco'
    paired_dir = './data/coco_blur_pairs'
    
    # Create paired dataset
    prepare_coco_blur_pairs(data_dir=data_dir, 
                          paired_dir=paired_dir,
                          subset='val2017',
                          num_images=1000)
    
    # Split into train/test sets
    split_coco_dataset(paired_dir=paired_dir, train_ratio=0.8)