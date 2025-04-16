import os
import json
from PIL import Image, ImageDraw
import logging

# Configuration
#INPUT_DIR = "/media/Data_2/person-search/dataset/bbox_corrected_CUHK"
INPUT_DIR = "/media/Data_2/person-search/dataset/train1"
OUTPUT_DIR = "./output_checked"  # You can modify this if needed
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{image_name}_checked.png")
    result_image.save(output_path)
    print(f"Saved result with {len(bboxes)} bounding boxes to {output_path}")

if __name__ == "__main__":
    print("Enter the image filename (without .png):")
    image_name = input().strip()
    if image_name:
        process_single_image(image_name)
    else:
        print("Invalid input.")
