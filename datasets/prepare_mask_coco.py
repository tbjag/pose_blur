import os
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter
from pycocotools.coco import COCO
from typing import List, Tuple
import json
from tqdm import tqdm
import numpy as np
import cv2

class PersonImageProcessor:
    def __init__(self, data_dir=str, standard_size: Tuple[int, int] = (512, 512), max_kernel_size: int = 21, sigma: float = 5.0):
        self.standard_size = standard_size
        self.max_kernel_size = max_kernel_size
        self.sigma = sigma
        self.transforms = T.Compose([
            T.Resize(standard_size),
            T.ToTensor(),
        ])
        self.data_dir = data_dir
        
    def load_and_preprocess_image(self, img_path: str) -> torch.Tensor:
        img_path = os.path.join(self.data_dir, img_path)
        img = Image.open(img_path).convert('RGB')
        return self.transforms(img)

    def create_segmentation_mask(self, image: Image.Image, anns: List[dict], original_size: Tuple[int, int]) -> np.ndarray:
        """
        Create a binary segmentation mask for all person instances.
        
        Args:
            image (Image.Image): Original image
            anns (List[dict]): COCO annotations for person instances
            original_size (Tuple[int, int]): Original image dimensions
        
        Returns:
            np.ndarray: Binary mask of person segmentations
        """
        # Convert PIL Image to NumPy array
        mask = np.zeros(original_size[::-1], dtype=np.uint8)
        
        for ann in anns:
            # Check if segmentation is in polygon format
            if isinstance(ann['segmentation'], list):
                # Convert polygon to binary mask
                for seg in ann['segmentation']:
                    # Reshape polygon coordinates
                    poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                    
                    # Draw filled polygon on mask
                    cv2.fillPoly(mask, [poly], 255)
        
        # Resize mask to standard size
        mask = cv2.resize(mask, self.standard_size, interpolation=cv2.INTER_NEAREST)
        return mask

    def apply_mask_blur(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """
        Apply Gaussian blur to regions defined by the mask.
        
        Args:
            image (Image.Image): Original image
            mask (np.ndarray): Binary segmentation mask
        
        Returns:
            Image.Image: Image with masked regions blurred
        """
        # Convert PIL Image to NumPy array
        img_array = np.array(image)
        
        # Create a blurred version of the entire image
        blurred = cv2.GaussianBlur(img_array, (21, 21), 0)
        
        # Create a 3-channel mask
        mask_3channel = np.stack([mask, mask, mask], axis=-1)
        
        # Blend original and blurred images using mask
        result = np.where(mask_3channel > 0, blurred, img_array).astype(np.uint8)
        
        return Image.fromarray(result)

def prepare_coco_blur_pairs(data_dir: str, 
                             paired_dir: str,
                             ann_file: str,
                             num_images: int):
    """Create paired dataset of original and person-blurred COCO images using segmentation masks"""
    
    os.makedirs(paired_dir, exist_ok=True)
    
    # Initialize COCO dataset
    coco = COCO(ann_file)
    
    # Initialize processor
    processor = PersonImageProcessor(data_dir)
    
    # Get images containing people
    cat_ids = coco.getCatIds(catNms=['person'])
    img_ids = coco.getImgIds(catIds=cat_ids)[:num_images]
    
    for img_id in tqdm(img_ids, desc=f'Processing {len(img_ids)} images...'):
        # Load image info and annotations
        img_info = coco.loadImgs([img_id])[0]
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        
        # Load and preprocess image
        original_tensor = processor.load_and_preprocess_image(img_info['file_name'])
        
        # Convert tensor to PIL image
        original_image = T.ToPILImage()(original_tensor)
        
        # Create segmentation mask
        original_size = (img_info['width'], img_info['height'])
        segmentation_mask = processor.create_segmentation_mask(original_image, anns, original_size)
        
        # Apply mask blur
        blurred_image = processor.apply_mask_blur(original_image, segmentation_mask)
        
        # Create paired image
        paired_image = Image.new('RGB', (original_image.width * 2, original_image.height))
        paired_image.paste(original_image, (0, 0))
        paired_image.paste(blurred_image, (original_image.width, 0))
        
        # Save paired image
        paired_image.save(os.path.join(paired_dir, f'{img_id}.png'))

    print('Finished processing images')

if __name__ == '__main__':
    data_dir = 'val2017'
    paired_dir = 'train/'
    ann_file = 'annotations/instances_val2017.json'
    
    # Create paired dataset
    prepare_coco_blur_pairs(data_dir=data_dir, 
                             paired_dir=paired_dir,
                             ann_file=ann_file,
                             num_images=10)