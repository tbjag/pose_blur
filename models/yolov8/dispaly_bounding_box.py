import cv2
import json
import os
import argparse

import cv2
import json
import os
import argparse

def draw_bounding_boxes_xyxy(image_file, json_file, output_file=None):
    """
    Draw bounding boxes on image using non-normalized [x1, y1, x2, y2] format
    from JSON file
    """
    # Load image
    image = cv2.imread(image_file)
    if image is None:
        print(f"Error: Could not load image {image_file}")
        return
    
    # Load JSON file with bounding boxes
    try:
        with open(json_file, 'r') as f:
            boxes = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return
    
    # Draw each bounding box
    for box in boxes:
        # Parse [x1, y1, x2, y2] format
        x1, y1, x2, y2 = map(int, box)
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Save or display the result
    if output_file:
        cv2.imwrite(output_file, image)
        print(f"Annotated image saved to {output_file}")
    else:
        # Display image
        cv2.imshow('Bounding Boxes', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def draw_bounding_boxes(image_file, json_file, output_file=None):
    """
    Draw bounding boxes on image using YOLO format JSON file
    
    YOLO format: [x_center, y_center, width, height] (normalized 0-1)
    """
    # Load image
    image = cv2.imread(image_file)
    if image is None:
        print(f"Error: Could not load image {image_file}")
        return
    
    height, width = image.shape[:2]
    
    # Load JSON file with bounding boxes
    try:
        with open(json_file, 'r') as f:
            boxes = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return
    
    # Draw each bounding box
    for box in boxes:
        # Parse YOLO format (x_center, y_center, width, height)
        x_center, y_center, box_width, box_height = box
        
        # Convert normalized coordinates to pixel values
        x_center *= width
        y_center *= height
        box_width *= width
        box_height *= height
        
        # Calculate top-left and bottom-right points
        x1 = int(x_center - (box_width / 2))
        y1 = int(y_center - (box_height / 2))
        x2 = int(x_center + (box_width / 2))
        y2 = int(y_center + (box_height / 2))
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Save or display the result
    if output_file:
        cv2.imwrite(output_file, image)
        print(f"Annotated image saved to {output_file}")
    else:
        # Display image
        cv2.imshow('Bounding Boxes', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw YOLO format bounding boxes on an image')
    parser.add_argument('--image', required=True, help='Path to the image file')
    parser.add_argument('--json', required=True, help='Path to the JSON file with YOLO format bounding boxes')
    parser.add_argument('--output', help='Path to save the output image (optional)')
    
    args = parser.parse_args()
    
    draw_bounding_boxes(args.image, args.json, args.output)
    
## python dispaly_bounding_box.py --image /path/to/image.jpg --json /path/to/boxes.json

#python dispaly_bounding_box.py --image /media/Data_2/person-search/dataset/Image/SSM/train/s1.jpg --json /media/Data_2/person-search/dataset/Image/SSM/train/s1.json --output result.jpg

## python dispaly_bounding_box.py --image /media/Data_2/person-search/dataset/Image/SSM/s1.jpg --json /media/Data_2/person-search/dataset/Image/SSM/s1.json --output result_1.jpg
