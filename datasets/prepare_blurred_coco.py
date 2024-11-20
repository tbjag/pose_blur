import os
import torch
import torchvision.transforms as T
from PIL import Image
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

    def create_blurred_image(self, tensor: torch.Tensor, bboxes: List[List[int]]) -> torch.Tensor:
        blurred = tensor.clone()
        for bbox in bboxes:
            x, y, w, h = bbox
            region = blurred[:, y:y+h, x:x+w]
            if w >= 3 and h >= 3:  # Only blur regions that are large enough
                # Adjust kernel size dynamically based on region size
                kernel_size = min(self.max_kernel_size, max(3, min(w, h) // 3 * 2 + 1))
                padding = kernel_size // 2
                
                if kernel_size >= 3:
                    try:
                        region = region.unsqueeze(0)
                        region = torch.nn.functional.avg_pool2d(
                            region, 
                            kernel_size=kernel_size, 
                            stride=1, 
                            padding=padding
                        )
                        blurred[:, y:y+h, x:x+w] = region.squeeze(0)
                    except Exception as e:
                        print(f"Blurring error for bbox {bbox}: {e}")
                        continue
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
    processor = PersonImageProcessor(data_dir)
    
    # Get images containing people
    cat_ids = coco.getCatIds(catNms=['person'])
    img_ids = coco.getImgIds(catIds=cat_ids)[:num_images]
    
    for img_id in tqdm(img_ids, desc=f'Processing {len(img_ids)} images...'):
        try:
            # Load image info and annotations
            img_info = coco.loadImgs([img_id])[0]
            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=None)
            anns = coco.loadAnns(ann_ids)
            
            # Load and preprocess image
            original_tensor = processor.load_and_preprocess_image(img_info['file_name'])
            
            # Scale bounding boxes
            original_size = (img_info['width'], img_info['height'])
            scaled_bboxes = [processor.scale_bbox(ann['bbox'], original_size) for ann in anns]
            
            # Create blurred version
            blurred_tensor = processor.create_blurred_image(original_tensor, scaled_bboxes)
            
            # Convert tensors to PIL images
            blurred_image = T.ToPILImage()(blurred_tensor)
            
            # Save paired image
            blurred_image.save(os.path.join(paired_dir, f'{img_id}.png'))
            blurred_image_bbox_path = os.path.join(paired_dir, f'{img_id}.json')
            with open(blurred_image_bbox_path, 'w') as file:
                json.dump(scaled_bboxes, file)
                
        except Exception as e:
            print(f"Error processing image {img_id}: {str(e)}")
            continue

if __name__ == '__main__':
    data_dir = '/media/Data_2/raw_coco/val2017'
    paired_dir = '/media/Data_2/blurred_coco/'
    ann_file = '/media/Data_2/raw_coco/annotations'
    
    # Create paired dataset
    prepare_coco_blur_pairs(data_dir=data_dir, 
                          paired_dir=paired_dir,
                          ann_file='/media/Data_2/raw_coco/annotations/instances_val2017.json',
                          num_images=1000)
