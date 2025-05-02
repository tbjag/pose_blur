import os
import json
from PIL import Image, ImageDraw
import logging
import torch
import numpy as np

def tensor_to_pil(tensor, denormalize=True):
    """
    Convert PyTorch tensor to PIL Image.
    
    Args:
        tensor (torch.Tensor): Image tensor of shape (C, H, W) or (B, C, H, W)
        denormalize (bool): Whether to denormalize from [-1,1] to [0,1] range
                           or from ImageNet normalization
        
    Returns:
        PIL.Image: The converted PIL Image
    """
    # Make a copy to avoid modifying the original
    tensor = tensor.clone()
    
    # If tensor is batched (B, C, H, W), take the first image
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    
    # Move to CPU if on GPU
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Detach from computation graph
    tensor = tensor.detach()
    
    # Handle normalization
    if denormalize:
        # Check if using ImageNet normalization
        if tensor.min() < -0.2:  # Heuristic to detect ImageNet normalization
            # ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = tensor * std + mean
        else:
            # Assuming [-1,1] range normalization
            tensor = (tensor + 1) / 2.0
    
    # Ensure values are in [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and transpose
    np_img = tensor.numpy()
    
    # Transpose from (C, H, W) to (H, W, C) for PIL
    if np_img.shape[0] == 3 or np_img.shape[0] == 1:  # 3 for RGB, 1 for grayscale
        np_img = np.transpose(np_img, (1, 2, 0))
    
    # Convert to uint8
    np_img = (np_img * 255).astype(np.uint8)
    
    # If single channel, convert to correct format for PIL
    if np_img.shape[-1] == 1:
        np_img = np_img.squeeze(-1)  # Remove channel dimension if grayscale
    
    # Create PIL image
    pil_img = Image.fromarray(np_img)
    
    return pil_img


# Configuration
INPUT_DIR = "/media/Data_2/person-search/dataset/bbox_corrected_CUHK"
# INPUT_DIR = "/media/Data_2/person-search/dataset/train1"
OUTPUT_DIR = "/home/wenjun/Lab/GAN_project/tanush_pose_blur/data/output_checked"  # You can modify this if needed
COLOR = (255, 0, 0)  # Red
THICKNESS = 2

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def extract_original_image(paired_image_path):
    try:
        paired_image = Image.open(paired_image_path)
        width, height = paired_image.size
        left_half = paired_image.crop((0, 0, width // 2, height))
        return left_half
    except Exception as e:
        logging.error(f"Error extracting original image: {str(e)}")
        return None

def load_bounding_boxes(json_path):
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading bounding boxes: {str(e)}")
        return []

def draw_bounding_boxes(image, bboxes):
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    for i, bbox in enumerate(bboxes):
        try:
            x1, y1, x2, y2 = bbox
            for t in range(THICKNESS):
                draw.rectangle([(x1+t, y1+t), (x2-t, y2-t)], outline=COLOR)
            draw.text((x1, y1 - 12), f"#{i}: {x1},{y1},{x2},{y2}", fill=COLOR)
        except Exception as e:
            print(bbox)
            logging.warning(f"Error drawing bbox: {str(e)}")
    return draw_image

def process_single_image(image_name):
    if not image_name.endswith('.png'):
        img_file = f"{image_name}.png"
    else:
        img_file = image_name
        image_name = os.path.splitext(img_file)[0]

    paired_image_path = os.path.join(INPUT_DIR, img_file)
    json_path = os.path.join(INPUT_DIR, f"{image_name}.json")

    if not os.path.exists(paired_image_path):
        print(f"Image not found: {paired_image_path}")
        return

    if not os.path.exists(json_path):
        print(f"JSON not found: {json_path}")
        return

    original_image = extract_original_image(paired_image_path)
    if original_image is None:
        return

    bboxes = load_bounding_boxes(json_path)
    if not bboxes:
        print(f"No bounding boxes found in {json_path}")

    result_image = draw_bounding_boxes(original_image, bboxes)
    print(result_image)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{image_name}_checked.png")
    result_image.save(output_path)
    # print(f"Saved result with {len(bboxes)} bounding boxes to {output_path}")
    
def process_image_tensor(image_name, image, bboxes):
    image = tensor_to_pil(image)
    result_image = draw_bounding_boxes(image, bboxes)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{image_name}_transform_checked.png")
    result_image.save(output_path)
    print(f"Saved result with {len(bboxes)} bounding boxes to {output_path}")


if __name__ == "__main__":
    print("Enter the image filename (without .png):")
    image_name = "s8389"
    if image_name:
        process_single_image(image_name)
    else:
        print("Invalid input.")
