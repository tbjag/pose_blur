import os
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter
import scipy.io
import numpy as np
import cv2
import json
import traceback
import gc
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    filename='cuhk_processor.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CUHKProcessor:
    def __init__(self, data_dir: str, save_dir: str, standard_size=(256, 256), blur_radius=10):
        try:
            self.data_dir = data_dir
            self.save_dir = save_dir
            self.standard_size = standard_size
            self.blur_radius = blur_radius

            self.transforms = T.Compose([
                T.Resize(standard_size, interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
            ])

            # Create directories
            os.makedirs(save_dir, exist_ok=True)
            
            # Error log file path
            self.error_log = os.path.join(save_dir, "error_log.txt")
            
            # Load bounding box annotations
            self.annotation_file = os.path.join(data_dir, "annotation", "Images.mat")
            logging.info(f"Loading bounding box data from {self.annotation_file}")
            self.bbox_data = self._load_bbox_data()
            logging.info(f"Successfully loaded bounding box data for {len(self.bbox_data)} images")
            
        except Exception as e:
            logging.error(f"Initialization error: {str(e)}")
            logging.error(traceback.format_exc())
            raise

    def _load_bbox_data(self):
        """Load bounding box information from Images.mat."""
        bbox_dict = {}

        try:
            # Load MATLAB .mat file
            mat_data = scipy.io.loadmat(self.annotation_file)

            # Ensure the correct key exists
            if "Img" not in mat_data:
                available_keys = list(mat_data.keys())
                error_msg = f"'Img' key not found in .mat file! Available keys: {available_keys}"
                logging.error(error_msg)
                raise KeyError(error_msg)

            images_info = mat_data["Img"][0]  # Extract image annotations
            logging.info(f"Found {len(images_info)} image entries in annotation file")

            for img_idx, img in enumerate(images_info):
                try:
                    imname = img[0][0]  # Extract image filename (e.g., 's14859.jpg')
                    boxes_data = img[2][0]  # Extract bounding boxes

                    bbox_list = []
                    for bbox_entry in boxes_data:
                        try:
                            bbox = bbox_entry[0]  # Extract [xmin, ymin, width, height]

                            # Convert to Pix2Pix expected format [xmin, ymin, xmax, ymax]
                            bbox_list.append([
                                int(bbox[0][0]),  # xmin
                                int(bbox[0][1]),  # ymin
                                int(bbox[0][0]) + int(bbox[0][2]),  # xmax = xmin + width
                                int(bbox[0][1]) + int(bbox[0][3])   # ymax = ymin + height
                            ])
                        except Exception as e:
                            logging.warning(f"Error processing bbox for {imname}: {str(e)}")
                            continue

                    bbox_dict[imname] = bbox_list
                except Exception as e:
                    logging.warning(f"Error processing image info at index {img_idx}: {str(e)}")
                    continue

            return bbox_dict
        except Exception as e:
            logging.error(f"Failed to load bbox data: {str(e)}")
            logging.error(traceback.format_exc())
            return {}

    def _apply_blur(self, image: Image.Image, bboxes):
        """Apply Gaussian blur over detected persons in the image."""
        try:
            blurred_image = image.copy()
            img_width, img_height = image.size

            for bbox in bboxes:
                try:
                    x1, y1, x2, y2 = bbox  # Unpack bounding box
                    
                    # Validate bounding box coordinates
                    x1 = max(0, min(x1, img_width-1))
                    y1 = max(0, min(y1, img_height-1))
                    x2 = max(x1+1, min(x2, img_width))
                    y2 = max(y1+1, min(y2, img_height))
                    
                    # Check if bbox is valid
                    if x2 <= x1 or y2 <= y1:
                        logging.warning(f"Invalid bbox after validation: {bbox} -> [{x1}, {y1}, {x2}, {y2}]")
                        continue
                    
                    # Crop and blur the person region
                    cropped = image.crop((x1, y1, x2, y2)).filter(ImageFilter.GaussianBlur(self.blur_radius))
                    blurred_image.paste(cropped, (x1, y1, x2, y2))
                except Exception as e:
                    logging.warning(f"Error processing bbox {bbox}: {str(e)}")
                    continue
            
            return blurred_image
        except Exception as e:
            logging.error(f"Error in _apply_blur: {str(e)}")
            logging.error(traceback.format_exc())
            # Return original image if blurring fails
            return image

    def process_images(self):
        """Process all images in the CUHK dataset and save transformed pairs + bounding boxes."""
        try:
            image_dir = os.path.join(self.data_dir, "Image", "SSM")
            if not os.path.exists(image_dir):
                error_msg = f"Image directory not found: {image_dir}"
                logging.error(error_msg)
                raise FileNotFoundError(error_msg)

            # Get all image files
            image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            logging.info(f"Found {len(image_files)} images to process")
            
            # Check for already processed images
            processed_files = [os.path.splitext(f)[0] for f in os.listdir(self.save_dir) if f.endswith('.png')]
            logging.info(f"Found {len(processed_files)} already processed images")

            image_paths = [os.path.join(image_dir, f) for f in image_files]
            
            # Process images with error handling for each one
            for idx, img_path in enumerate(tqdm(image_paths, desc="Processing CUHK images")):
                try:
                    img_name = os.path.basename(img_path)
                    img_name_no_ext, img_ext = os.path.splitext(img_name)
                    
                    # Skip if already processed
                    if img_name_no_ext in processed_files:
                        logging.info(f"Skipping already processed image: {img_name}")
                        continue
                    
                    # Progress information
                    progress_pct = (idx+1)/len(image_paths)*100
                    logging.info(f"Processing image {idx+1}/{len(image_paths)} ({progress_pct:.1f}%): {img_name}")
                    print(f"Processing image {idx+1}/{len(image_paths)} ({progress_pct:.1f}%): {img_name}")
                    
                    # Try to load image as a test before full processing
                    try:
                        with Image.open(img_path) as test_img:
                            # Just verify image can be opened
                            w, h = test_img.size
                    except Exception as e:
                        logging.error(f"Failed to open image {img_name}: {str(e)}")
                        self._log_error(f"Corrupted image {img_name}: {str(e)}")
                        continue

                    # Load original image
                    original_image = Image.open(img_path).convert('RGB')
                    orig_w, orig_h = original_image.size
                    
                    # Get bounding boxes
                    bboxes = self.bbox_data.get(img_name, [])
                    if not bboxes:
                        logging.warning(f"No bounding boxes found for {img_name}")
                    
                    # Create blurred image
                    blurred_image = self._apply_blur(original_image, bboxes)

                    # Resize both images to standard size
                    image_resized = original_image.resize(self.standard_size, Image.BICUBIC)
                    blurred_resized = blurred_image.resize(self.standard_size, Image.BICUBIC)

                    # Scale bounding boxes to fit resized image
                    scale_x = self.standard_size[0] / orig_w
                    scale_y = self.standard_size[1] / orig_h
                    
                    scaled_bboxes = []
                    for box in bboxes:
                        try:
                            x1, y1, x2, y2 = box
                            scaled_bboxes.append([
                                int(x1 * scale_x),
                                int(y1 * scale_y),
                                int(x2 * scale_x),
                                int(y2 * scale_y)
                            ])
                        except Exception as e:
                            logging.warning(f"Error scaling bbox {box} for {img_name}: {str(e)}")
                            continue
                            
                    # Create paired image (Concatenating along width)
                    paired_image = Image.new('RGB', (self.standard_size[0] * 2, self.standard_size[1]))  
                    paired_image.paste(image_resized, (0, 0))  # Original image on the left
                    paired_image.paste(blurred_resized, (self.standard_size[0], 0))  # Blurred image on the right

                    # Save transformed pair
                    save_path = os.path.join(self.save_dir, f"{img_name_no_ext}.png")
                    paired_image.save(save_path)

                    # Save scaled bounding boxes as JSON
                    bbox_json_path = os.path.join(self.save_dir, f"{img_name_no_ext}.json")
                    with open(bbox_json_path, 'w') as f:
                        json.dump(scaled_bboxes, f)
                    
                    # Clean up memory
                    del original_image
                    del blurred_image
                    del image_resized
                    del blurred_resized
                    del paired_image
                    gc.collect()  # Force garbage collection
                    
                except MemoryError:
                    error_msg = f"Memory error processing image {img_name}"
                    logging.error(error_msg)
                    self._log_error(error_msg)
                    # Let's force garbage collection and continue
                    gc.collect()
                    continue
                except Exception as e:
                    error_msg = f"Error processing image {img_name}: {str(e)}"
                    logging.error(error_msg)
                    logging.error(traceback.format_exc())
                    self._log_error(error_msg)
                    continue

            logging.info(f"Finished processing {len(image_paths)} images. Saved to {self.save_dir}")
            print(f"Finished processing {len(image_paths)} images. Saved to {self.save_dir}")
            
        except Exception as e:
            logging.error(f"Error in process_images: {str(e)}")
            logging.error(traceback.format_exc())
            raise

    def _log_error(self, error_msg):
        """Log error to a file for later investigation."""
        try:
            with open(self.error_log, "a") as f:
                f.write(f"{error_msg}\n")
        except Exception as e:
            logging.error(f"Failed to write to error log: {str(e)}")

if __name__ == "__main__":
    try:
        data_dir = "/media/Data_2/person-search/dataset"
        save_dir = "/media/Data_2/person-search/dataset/bbox_corrected_CUHK"

        processor = CUHKProcessor(data_dir, save_dir, standard_size=(256, 256), blur_radius=10)
        processor.process_images()
    except Exception as e:
        logging.critical(f"Fatal error: {str(e)}")
        logging.critical(traceback.format_exc())
        print(f"Fatal error: {str(e)}")
        print("Check cuhk_processor.log for details")