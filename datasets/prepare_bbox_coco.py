import os
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter
from pycocotools.coco import COCO
from typing import List, Tuple
import json
from tqdm import tqdm

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

    def create_blurred_image(self, image: Image.Image, bboxes: List[List[int]]) -> Image.Image:
        blurred = image.copy()

        for bbox in bboxes:
            x, y, w, h = bbox
            region = blurred.crop((x, y, x + w, y + h))
            if w >= 3 and h >= 3:  # Only blur regions that are large enough
                blurred_region = region.filter(ImageFilter.GaussianBlur(radius=5))
                blurred.paste(blurred_region, (x, y))
                
        return blurred

def prepare_coco_blur_pairs(data_dir: str, 
                          paired_dir: str,
                          ann_file: str,
                          num_images: int):
    """Create paired dataset of original and person-blurred COCO images"""
    
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
        
        # Scale bounding boxes
        original_size = (img_info['width'], img_info['height'])
        scaled_bboxes = [processor.scale_bbox(ann['bbox'], original_size) for ann in anns]
        
        # Convert tensor to PIL image
        original_image = T.ToPILImage()(original_tensor)
        blurred_image = processor.create_blurred_image(original_image, scaled_bboxes)
        
        # Create paired image
        paired_image = Image.new('RGB', (original_image.width * 2, original_image.height))
        paired_image.paste(original_image, (0, 0))

        paired_image.paste(blurred_image, (original_image.width, 0))
        
        # Save paired image
        paired_image.save(os.path.join(paired_dir, f'{img_id}.png'))

        blurred_image_bbox_path = os.path.join(paired_dir, f'{img_id}.json')
        with open(blurred_image_bbox_path, 'w') as file:
            json.dump(scaled_bboxes, file)
                

    print('finished')

if __name__ == '__main__':
    data_dir = '/media/Data_2/raw_coco/val2017'
    paired_dir = '/media/Data_2/train/'
    ann_file = '/media/Data_2/raw_coco/annotations'
    
    # Create paired dataset
    prepare_coco_blur_pairs(data_dir=data_dir, 
                          paired_dir=paired_dir,
                          ann_file='/media/Data_2/raw_coco/annotations/instances_val2017.json',
                          num_images=1000)
