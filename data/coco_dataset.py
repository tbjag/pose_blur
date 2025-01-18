import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset, make_coco_bbox
from PIL import Image
import json

class COCODataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        self.json_paths = sorted(make_coco_bbox(self.dir_AB, opt.max_dataset_size))
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1)) # TODO remove any grayscale
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        bbox = []
        json_path = self.json_paths[index]
        try:
            with open(json_path, 'r') as file:
                if file.readable() and file.seek(0) or file.read(1):  # Check if file is not empty
                    file.seek(0)  # Reset file pointer after check
                    bbox = json.load(file)
                else:
                    print(f"Empty JSON file: {json_path}")
        except json.JSONDecodeError:
            print(f"Malformed JSON file: {json_path}")
        except Exception as e:
            print(f"Error loading JSON file {json_path}: {e}")

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path, 'bbox': bbox}

    """         with open(self.json_paths[index], 'r') as file:
            print(self.json_paths[index])
            bbox = json.load(file)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path, 'bbox': bbox} """

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
