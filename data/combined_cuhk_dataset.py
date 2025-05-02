import os
from scipy.io import loadmat
import os.path as osp
import numpy as np
import torch


import json
from PIL import Image
from data.base_dataset import BaseDataset, get_params, get_transform, build_transforms
from data.image_folder import make_dataset, make_bbox

from data.bounding_box_check import process_image_tensor, process_single_image


class CombinedCuhkDataset(BaseDataset):
    """Dataset class for the preprocessed CUHK-SYSU dataset for Pix2Pix.

    This loads preprocessed AB images (left: blurred, right: original) and applies transformations.
    """

    def __init__(self, opt, split="train"):
        """Initialize the dataset.

        Args:
            opt: Options object storing experiment flags.
        """
        BaseDataset.__init__(self, opt)
        self.root = opt.dataroot
        self.split = split
        self.annotations = self._load_annotations()
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # Standardized naming
        # self.dir_AB = os.path.join(opt.dataroot, split)  # Standardized naming

        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # Use existing helper
        self.json_paths = sorted(make_bbox(self.dir_AB, opt.max_dataset_size))
        self.transforms = build_transforms(self.split == 'train')

        assert opt.load_size >= opt.crop_size, "Crop size should be smaller than load size."
        self.input_nc = opt.output_nc if opt.direction == 'BtoA' else opt.input_nc
        self.output_nc = opt.input_nc if opt.direction == 'BtoA' else opt.output_nc

    def __getitem__(self, index):
        """Return a preprocessed image pair (blurred, original) with bounding boxes."""
        anno = self.annotations[index]
        AB = Image.open(anno["img_path"]).convert("RGB")
        AB_path = anno["img_path"]
        json_path = anno["json_path"]
        # if self.split != "train":
        #     anno = self.annotations[index]
        #     AB = Image.open(anno["img_path"]).convert("RGB")
        #     json_path = anno["json_path"]

        # else:
        #     AB_path = self.AB_paths[index]
        #     # print(f"CUHK Dataset image path {AB_path}")
        #     AB = Image.open(AB_path).convert('RGB')
        #     json_path = self.json_paths[index]

        
        
        # Ensure image width is even for proper splitting
        w, h = AB.size
        assert w % 2 == 0, f"[ERROR] Image width {w} is not even, cannot split into A and B."
        
        # Split image into A (original) and B (blurred)
        w2 = w // 2
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))
        # print(A.size)
        # Apply the same transformation to both A and B
        # transform_params = get_params(self.opt, A.size)
        # A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        # B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        
        # A = A_transform(A)
        # B = B_transform(B)

        # Load bounding box annotations from JSON
        bboxes = []
        # print(f"path of file {json_path}")
        try:
            with open(json_path, 'r') as file:
                if file.readable() and file.seek(0) or file.read(1):  # Check if file is not empty
                    file.seek(0)
                    bboxes = json.load(file)
                    bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
                else:
                    print(f"[WARNING] Empty JSON file: {json_path}")
        except json.JSONDecodeError:
            print(f"[ERROR] Malformed JSON file: {json_path}")
        except Exception as e:
            print(f"[ERROR] Could not read JSON file {json_path}: {e}")
        
        if self.split == "query":
            bboxes = anno["boxes"]
        img_name = os.path.basename(AB_path).split('.')[0]  # Get filename without extension
        pid = torch.as_tensor(anno["pids"], dtype=torch.int64) #if len(self.annotations[img_name]["pids"]) > 0 else 5555

        target = {"img_name": img_name, "boxes": bboxes, "labels": pid}
        if self.transforms is not None:
            A, _ = self.transforms(A, target)
            B, target = self.transforms(B, target)
            
        # process_single_image(img_name)
        # process_image_tensor(img_name, AB, target["boxes"])

        return {
            'A': A,
            'B': B,
            'labels':pid,
            "img_name" :img_name,
            'A_paths': AB_path,
            'B_paths': AB_path,
            'bbox': target["boxes"]
        }

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.annotations)
    
    def _load_queries(self):
        # TestG50: a test protocol, 50 gallery images per query
        protoc = loadmat(osp.join(self.root, "annotation/test/train_test/TestG50.mat"))
        protoc = protoc["TestG50"].squeeze()
        queries = []
        for item in protoc["Query"]:
            img_name = str(item["imname"][0, 0][0])
            img_key = img_name.split('.')[0]  # Get filename without extension
            roi = item["idlocate"][0, 0][0].astype(np.int32)
            roi[2:] += roi[:2]
            queries.append({
                    "img_name": img_name,
                    "img_path": osp.join(self.root, f"{img_key}.png"),
                    "json_path":osp.join(self.root, f"{img_key}.json"),

                    "boxes": roi[np.newaxis, :],
                    "pids": np.array([-100]),  # dummy pid
                })
            
        return queries

    def _load_split_img_names(self):
        """
        Load the image names for the specific split.
        """
        assert self.split in ("train", "gallery")
        # gallery images
        gallery_imgs = loadmat(osp.join(self.root, "annotation", "pool.mat"))
        gallery_imgs = gallery_imgs["pool"].squeeze()
        gallery_imgs = [str(a[0]) for a in gallery_imgs]
        if self.split == "gallery":
            return gallery_imgs
        # all images
        all_imgs = loadmat(osp.join(self.root, "annotation", "Images.mat"))
        all_imgs = all_imgs["Img"].squeeze()
        all_imgs = [str(a[0][0]) for a in all_imgs]
        # training images = all images - gallery images
        training_imgs = sorted(list(set(all_imgs) - set(gallery_imgs)))
        return training_imgs

    def _load_annotations(self):
        if self.split == "query":
            return self._load_queries()

        # load all images and build a dict from image to boxes
        all_imgs = loadmat(osp.join(self.root, "annotation", "Images.mat"))
        all_imgs = all_imgs["Img"].squeeze()
        name_to_boxes = {}
        name_to_pids = {}
        unlabeled_pid = 5555  # default pid for unlabeled people
        for img_name, _, boxes in all_imgs:
            img_name = str(img_name[0])
            boxes = np.asarray([b[0] for b in boxes[0]])
            boxes = boxes.reshape(boxes.shape[0], 4)  # (x1, y1, w, h)
            valid_index = np.where((boxes[:, 2] > 0) & (boxes[:, 3] > 0))[0]
            assert valid_index.size > 0, "Warning: {} has no valid boxes.".format(img_name)
            boxes = boxes[valid_index]
            name_to_boxes[img_name] = boxes.astype(np.int32)
            name_to_pids[img_name] = unlabeled_pid * np.ones(boxes.shape[0], dtype=np.int32)

        def set_box_pid(boxes, box, pids, pid):
            for i in range(boxes.shape[0]):
                if np.all(boxes[i] == box):
                    pids[i] = pid
                    return

        # assign a unique pid from 1 to N for each identity
        if self.split == "train":
            train = loadmat(osp.join(self.root, "annotation/test/train_test/Train.mat"))
            train = train["Train"].squeeze()
            for index, item in enumerate(train):
                scenes = item[0, 0][2].squeeze()
                for img_name, box, _ in scenes:
                    img_name = str(img_name[0])
                    box = box.squeeze().astype(np.int32)
                    set_box_pid(name_to_boxes[img_name], box, name_to_pids[img_name], index + 1)
        else:
            protoc = loadmat(osp.join(self.root, "annotation/test/train_test/TestG50.mat"))
            protoc = protoc["TestG50"].squeeze()
            for index, item in enumerate(protoc):
                # query
                im_name = str(item["Query"][0, 0][0][0])
                box = item["Query"][0, 0][1].squeeze().astype(np.int32)
                set_box_pid(name_to_boxes[im_name], box, name_to_pids[im_name], index + 1)
                # gallery
                gallery = item["Gallery"].squeeze()
                for im_name, box, _ in gallery:
                    im_name = str(im_name[0])
                    if box.size == 0:
                        break
                    box = box.squeeze().astype(np.int32)
                    set_box_pid(name_to_boxes[im_name], box, name_to_pids[im_name], index + 1)

        annotations = []
        imgs = self._load_split_img_names()
        for img_name in imgs:
            img_key = img_name.split('.')[0]  # Get filename without extension

            boxes = name_to_boxes[img_name]
            boxes[:, 2:] += boxes[:, :2]  # (x1, y1, w, h) -> (x1, y1, x2, y2)
            pids = name_to_pids[img_name]
            annotations.append({
                    "img_name": img_name,
                    "img_path": osp.join(self.root, f"{img_key}.png"),
                    "json_path":osp.join(self.root, f"{img_key}.json"),
                    "boxes": boxes,
                    "pids": pids,
                })
            
        return annotations

