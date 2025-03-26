import os
import json
from PIL import Image
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset, make_bbox


class CuhkDataset(BaseDataset):
    """Dataset class for the preprocessed CUHK-SYSU dataset for Pix2Pix.

    This loads preprocessed AB images (left: blurred, right: original) and applies transformations.
    """

    def __init__(self, opt):
        """Initialize the dataset.

        Args:
            opt: Options object storing experiment flags.
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # Standardized naming
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # Use existing helper
        self.json_paths = sorted(make_bbox(self.dir_AB, opt.max_dataset_size))

        assert opt.load_size >= opt.crop_size, "Crop size should be smaller than load size."
        self.input_nc = opt.output_nc if opt.direction == 'BtoA' else opt.input_nc
        self.output_nc = opt.input_nc if opt.direction == 'BtoA' else opt.output_nc

    def __getitem__(self, index):
        """Return a preprocessed image pair (blurred, original) with bounding boxes."""

        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')

        # Ensure image width is even for proper splitting
        w, h = AB.size
        assert w % 2 == 0, f"[ERROR] Image width {w} is not even, cannot split into A and B."

        # Split image into A (original) and B (blurred)
        w2 = w // 2
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # Apply the same transformation to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        
        A = A_transform(A)
        B = B_transform(B)

        # Load bounding box annotations from JSON
        json_path = self.json_paths[index]
        bboxes = []
        try:
            with open(json_path, 'r') as file:
                if file.readable() and file.seek(0) or file.read(1):  # Check if file is not empty
                    file.seek(0)
                    bboxes = json.load(file)
                else:
                    print(f"[WARNING] Empty JSON file: {json_path}")
        except json.JSONDecodeError:
            print(f"[ERROR] Malformed JSON file: {json_path}")
        except Exception as e:
            print(f"[ERROR] Could not read JSON file {json_path}: {e}")

        return {
            'A': A,
            'B': B,
            'A_paths': AB_path,
            'B_paths': AB_path,
            'bbox': bboxes
        }

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.AB_paths)
