import os
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter
import scipy.io
import numpy as np
import cv2
import json
from tqdm import tqdm

class CUHKProcessor:
    def __init__(self, data_dir: str, save_dir: str, standard_size=(256, 256), blur_radius=10):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.standard_size = standard_size
        self.blur_radius = blur_radius

        self.transforms = T.Compose([
            T.Resize(standard_size, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
        ])

        # Load bounding box annotations
        self.annotation_file = os.path.join(data_dir, "annotation", "Images.mat")
        self.bbox_data = self._load_bbox_data()

        # Create directories
        os.makedirs(save_dir, exist_ok=True)

    def _load_bbox_data(self):
        """Load bounding box information from Images.mat."""
        bbox_dict = {}

        # Load MATLAB .mat file
        mat_data = scipy.io.loadmat(self.annotation_file)

        # Ensure the correct key exists
        if "Img" not in mat_data:
            raise KeyError(f"'Img' key not found in .mat file! Available keys: {mat_data.keys()}")

        images_info = mat_data["Img"][0]  # Extract image annotations

        for img in images_info:
            imname = img[0][0]  # Extract image filename (e.g., 's14859.jpg')
            boxes_data = img[2][0]  # Extract bounding boxes

            bbox_list = []
            for bbox_entry in boxes_data:
                bbox = bbox_entry[0]  # Extract [xmin, ymin, width, height]

                # Convert to Pix2Pix expected format [xmin, ymin, xmax, ymax]
                bbox_list.append([
                    int(bbox[0][0]),  # xmin
                    int(bbox[0][1]),  # ymin
                    int(bbox[0][0]) + int(bbox[0][2]),  # xmax = xmin + width
                    int(bbox[0][1]) + int(bbox[0][3])   # ymax = ymin + height
                ])

            bbox_dict[imname] = bbox_list

        return bbox_dict

    def _apply_blur(self, image: Image.Image, bboxes):
        """Apply Gaussian blur over detected persons in the image."""
        blurred_image = image.copy()

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox  # Unpack bounding box

            # Crop and blur the person region
            cropped = image.crop((x1, y1, x2, y2)).filter(ImageFilter.GaussianBlur(self.blur_radius))
            blurred_image.paste(cropped, (x1, y1, x2, y2))
        
        return blurred_image

    def process_images(self):
        """Process all images in the CUHK dataset and save transformed pairs + bounding boxes."""
        image_dir = os.path.join(self.data_dir, "Image", "SSM")

        image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        for img_path in tqdm(image_paths, desc="Processing CUHK images"):
            img_name = os.path.basename(img_path)
            img_name_no_ext, img_ext = os.path.splitext(img_name)  # Extract filename without extension

            # Load original image
            image = Image.open(img_path).convert('RGB')

            # Get bounding boxes
            bboxes = self.bbox_data.get(img_name, [])

            # Create blurred image
            blurred_image = self._apply_blur(image, bboxes)

            # Resize both images to standard size
            image = image.resize(self.standard_size, Image.BICUBIC)
            blurred_image = blurred_image.resize(self.standard_size, Image.BICUBIC)

            # Create paired image (Concatenating along width)
            paired_image = Image.new('RGB', (self.standard_size[0] * 2, self.standard_size[1]))  
            paired_image.paste(image, (0, 0))  # Original image on the left
            paired_image.paste(blurred_image, (self.standard_size[0], 0))  # Blurred image on the right


            # Save transformed pair
            save_path = os.path.join(self.save_dir, f"{img_name_no_ext}.png")
            paired_image.save(save_path)

            # Save bounding boxes as JSON
            bbox_json_path = os.path.join(self.save_dir, f"{img_name_no_ext}.json")
            with open(bbox_json_path, 'w') as f:
                json.dump(bboxes, f)

        print(f"Finished processing {len(image_paths)} images. Saved to {self.save_dir}")

if __name__ == "__main__":
    data_dir = "/media/Data_2/person-search/dataset"
    save_dir = "/media/Data_2/person-search/dataset/cuhk_transformed_correct"

    processor = CUHKProcessor(data_dir, save_dir, standard_size=(256, 256), blur_radius=10)
    processor.process_images()
