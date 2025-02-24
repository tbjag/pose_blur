import os
import json
import numpy as np
import torch
from PIL import Image
from data.base_dataset import BaseDataset


class CuhkDataset(BaseDataset):
    """Dataset class for the preprocessed CUHK-SYSU dataset for Pix2Pix.

    This loads preprocessed image pairs (blurred + original) and bounding boxes.
    """

    def __init__(self, opt):
        """Initialize the dataset.

        Args:
            opt: Options object storing experiment flags.
        """
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.image_dir = os.path.join(opt.dataroot, "cuhk_transformed")  # Folder with preprocessed paired images
        self.image_paths = sorted([
            os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir) if f.endswith('.png')
        ])

    def __getitem__(self, index):
        """Return a preprocessed image pair (blurred, original) with bounding boxes."""
        img_path = self.image_paths[index]
        img_name = os.path.basename(img_path)
        img_name_no_ext = os.path.splitext(img_name)[0]  # Remove extension for JSON

        # Load preprocessed paired image (Blurred + Original)
        paired_image = Image.open(img_path).convert('RGB')

        # Load bounding box annotations from JSON (if available)
        json_path = os.path.join(self.image_dir, f"{img_name_no_ext}.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                bboxes = json.load(f)
        else:
            bboxes = []

        # Convert to PyTorch tensor (normalized to [0,1])
        paired_image = torch.from_numpy(np.array(paired_image)).permute(2, 0, 1).float() / 255.0

        # Split into A (blurred) and B (original)
        C, H, W = paired_image.shape
        W2 = W // 2
        A = paired_image[:, :, :W2]  # Left half: blurred
        B = paired_image[:, :, W2:]  # Right half: original

        return {
            'A': A,  
            'B': B,  
            'A_paths': img_path,
            'B_paths': img_path,
            'bbox': bboxes  # Bounding boxes for evaluation if needed
        }

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.image_paths)
