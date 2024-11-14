import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
import skimage.io as io
from typing import List, Tuple
import math

class PersonImageProcessor:
    def __init__(self, standard_size: Tuple[int, int] = (512, 512), max_kernel_size: int = 21, sigma: float = 5.0):
        """
        Initialize the image processor
        Args:
            standard_size: Tuple of (height, width) for resizing images
            max_kernel_size: Maximum size of the Gaussian kernel (must be odd)
            sigma: Standard deviation for Gaussian kernel
        """
        self.standard_size = standard_size
        self.max_kernel_size = max_kernel_size
        self.sigma = sigma
        self.transforms = T.Compose([
            T.Resize(standard_size),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
        ])
        
    def _create_gaussian_kernel(self, kernel_size: int) -> torch.Tensor:
        """Create 2D Gaussian kernel with specified size"""
        sigma = self.sigma
        
        # Create 1D kernel coordinates
        kernel_range = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
        x, y = torch.meshgrid(kernel_range, kernel_range)
        
        # Calculate 2D Gaussian
        gaussian = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * sigma ** 2))
        gaussian = gaussian / gaussian.sum()  # Normalize
        
        # Reshape for convolution
        return gaussian.view(1, 1, kernel_size, kernel_size)

    def load_and_preprocess_image(self, img_path: str) -> torch.Tensor:
        """Load image and convert to standardized B&W tensor"""
        if img_path.startswith('http'):
            img = Image.fromarray(io.imread(img_path))
        else:
            img = Image.open(img_path)
        return self.transforms(img)

    def scale_bbox(self, bbox: List[float], original_size: Tuple[int, int]) -> List[int]:
        """Scale bounding box coordinates to match standardized image size"""
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

    def apply_gaussian_blur_to_region(self, tensor: torch.Tensor, bbox: List[int]) -> torch.Tensor:
        """Apply Gaussian blur to a specific region in the tensor"""
        x, y, w, h = bbox
        
        # Ensure coordinates are within image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, self.standard_size[1] - x)
        h = min(h, self.standard_size[0] - y)
        
        # Skip if region is too small
        if w < 3 or h < 3:
            return tensor
        
        # Calculate appropriate kernel size based on region dimensions
        kernel_size = min(
            self.max_kernel_size,
            (min(w, h) // 2) * 2 + 1  # Ensure odd number
        )
        
        # Skip if kernel would be too small
        if kernel_size < 3:
            return tensor
        
        # Create appropriate sized kernel
        gaussian_kernel = self._create_gaussian_kernel(kernel_size)
        
        # Extract region
        region = tensor[:, y:y+h, x:x+w]
        
        # Add batch dimension for convolution
        region = region.unsqueeze(0)
        
        # Calculate appropriate padding
        padding = kernel_size // 2
        padding = min(padding, min(w // 2, h // 2))  # Ensure padding isn't too large
        
        if padding > 0:
            region_padded = F.pad(region, (padding, padding, padding, padding), mode='reflect')
            
            # Apply Gaussian blur using convolution
            blurred_region = F.conv2d(region_padded, gaussian_kernel)
            
            # Remove batch dimension
            blurred_region = blurred_region.squeeze(0)
            
            # Replace original region with blurred version
            tensor[:, y:y+h, x:x+w] = blurred_region
        else:
            # For very small regions, use average pooling instead
            blurred_region = F.avg_pool2d(region, 3, stride=1, padding=1)
            tensor[:, y:y+h, x:x+w] = blurred_region.squeeze(0)
        
        return tensor

def process_coco_images(coco: COCO, img_ids: List[int], processor: PersonImageProcessor) -> List[Tuple[torch.Tensor, List[List[int]]]]:
    """Process multiple COCO images and return tensors with their scaled bounding boxes"""
    results = []
    
    for img_id in img_ids:
        # Load image info
        img_info = coco.loadImgs([img_id])[0]
        
        # Get person annotations
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=coco.getCatIds(catNms=['person']), iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        
        # Load and preprocess image
        tensor = processor.load_and_preprocess_image(img_info['coco_url'])
        
        # Scale bounding boxes
        original_size = (img_info['width'], img_info['height'])
        scaled_bboxes = [processor.scale_bbox(ann['bbox'], original_size) for ann in anns]
        
        # Apply Gaussian blur to each person region
        for bbox in scaled_bboxes:
            tensor = processor.apply_gaussian_blur_to_region(tensor, bbox)
        
        results.append((tensor, scaled_bboxes))
    
    return results

# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = PersonImageProcessor(
        standard_size=(512, 512),
        max_kernel_size=21,
        sigma=5.0
    )
    
    # Setup COCO dataset
    dataDir = 'data'
    dataType = 'val2017'
    annFile = f'{dataDir}/annotations/instances_{dataType}.json'
    coco = COCO(annFile)
    
    # Get some images with people
    cat_ids = coco.getCatIds(catNms=['person'])
    img_ids = coco.getImgIds(catIds=cat_ids)[:5]
    
    # Process images
    processed_images = process_coco_images(coco, img_ids, processor)
    
    # Visualize results
    import matplotlib.pyplot as plt
    
    for tensor, bboxes in processed_images:
        img = tensor.squeeze().numpy()
        
        plt.figure(figsize=(10, 10))
        plt.imshow(img, cmap='gray')
        
        for bbox in bboxes:
            x, y, w, h = bbox
            plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False, color='red'))
            
        plt.axis('off')
        plt.show()