"""module"""
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

#need to  import yolo as checkpoint is of yolo object
from yolov8.ultralytics import YOLO
from fileinput import filename
import torch
import torch.nn as nn
from yolov8.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    time_sync,
)
from yolov8.modules import Segment,Conv,Pose,OBB, C2f, SPPF, Concat, Detect, parse_model,initialize_weights, feature_visualization, scale_img, Conv2, DWConv, ConvTranspose, RepConv, RepVGGDW, yaml_model_load
from yolov8.loss import v8DetectionLoss, E2EDetectLoss
import yaml
from copy import deepcopy
import numpy as np
import yolov8.ops as ops
from yolov8.Results import Results #TODO Fix Result file to remove ultralytics code


from pathlib import Path
import re

__all__ = ("WNet","WNetStructure")

#To fix loading model issue make self.legacy true in Head module

#TO DO: Load in Yolo weights and test if the model is accurate. Test on the PRW dataset
#TO DO: Make the W-net
#To DO: Labels
#Fix weights slightly




def intersect_dicts(da, db, exclude=()):
    """Returns a dictionary of intersecting keys with matching shapes, excluding 'exclude' keys, using da values."""
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}

class WNetStructure(nn.Module):
    def __init__(self, nc=80, ch=3):
        super(WNetStructure, self).__init__()
        self.nc = nc
        
        # Define backbone with layer connections
        module_config = [
             # First Backbone (Downsampling)
            (Conv(ch, 16, 3, 2), -1),                # 0: P1/2
            (Conv(16, 32, 3, 2), -1),                # 1: P2/4  
            (C2f(32, 32, n=1, shortcut=True), -1),   # 2
            (Conv(32, 64, 3, 2), -1),                # 3: P3/8
            (C2f(64, 64, n=2, shortcut=True), -1),   # 4
            (Conv(64, 128, 3, 2), -1),               # 5: P4/16
            (C2f(128, 128, n=2, shortcut=True), -1), # 6
            (Conv(128, 256, 3, 2), -1),              # 7: P5/32
            (C2f(256, 256, n=1, shortcut=True), -1), # 8
            (SPPF(256, 256, k=5), -1),               # 9
            
           # First Upsampling Path
            (nn.Upsample(scale_factor=2, mode="nearest"), -1),  # 10
            (Concat(1), [-1, 6]),                     # 11
            (C2f(384, 128, n=1), -1),                # 12
            (nn.Upsample(scale_factor=2, mode="nearest"), -1),  # 13
            (Concat(1), [-1, 4]),                     # 14
            (C2f(192, 64, n=1), -1),                 # 15
            
            # Second Backbone (Mirror)
            (Conv(64, 64, 3, 2), -1),                # 16: P3/8
            (Concat(1), [-1,12]),   # 17 
            (C2f(192, 128, n=1), -1), # 18
            (Conv(128, 128, 3, 2), -1),              # 19: P5/32
            (Concat(1), [-1,9]),                     #20
            (C2f(384, 256, n=1), -1),                # 21
            (Detect(nc, [64, 128, 256]), [15, 18, 21]),  #  22 Detection head
            
            # Feature Pyramid Network, Maybe change size later
            (SPPF(64, 64, k=5), 15),                #23
            (SPPF(128, 128, k=5), 18),               #24
            (SPPF(256, 256, k=5), 21),              #25
            
             # Decoder Path (Reverse of Second Backbone)
            (nn.Upsample(scale_factor=2, mode="nearest"), -1),  # 26
            (Concat(1), [-1, 24]),                    # 27: Connect with P4 features
            (C2f(384, 256, n=2), -1),                # 28
            (nn.Upsample(scale_factor=2, mode="nearest"), -1),  # 29
            (Concat(1), [-1, 23]),                    # 30: Connect with P3 features
            
            
            # Middle Bridge (Reverse Backbone to First Backbone)
            (C2f(320, 128, n=2), -1),                 # 31
            (nn.Upsample(scale_factor=2, mode="nearest"), -1),  # 32
            (Concat(1), [-1, 2]),                     # 33: Connect with first backbone P4
            (C2f(160, 48, n=2), -1),                # 34
            (nn.Upsample(scale_factor=2, mode="nearest"), -1),  # 35
            (C2f(48, 3, n=2), -1),                # 36
            (nn.Upsample(scale_factor=2, mode="nearest"), -1),                # 37

            
        ]

        for i, (m, f) in enumerate(module_config):
            m.i, m.f = i, f
            
        self.model = nn.Sequential(*[module for module,_ in module_config])
            
    

    def forward(self, x):
        return self.model(x)
    
COCO_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
    21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella',
    26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
    31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
    41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
    66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster',
    71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
    76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}



    
class WNet(nn.Module):
    def __init__(self):
        super(WNet, self).__init__()
        self.nc = 80
        # self.yaml = config if isinstance(config, dict) else yaml_model_load(config)  # cfg dict
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # ch = self.yaml["ch"] = self.yaml.get("ch", ch)
        # ch = 3
        
        # if nc and nc != self.yaml["nc"]:
        #     print(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
        #     self.yaml["nc"] = nc  # override YAML value
            
        # self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.model = WNetStructure().model
        self.save = [2, 4, 6, 9, 12, 15, 18, 21,23,24]
        # self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.names = COCO_CLASSES
        # self.inplace = self.yaml.get("inplace", True)
        self.inplace = True
        self.end2end = getattr(self.model[-1], "end2end", False)
        m = self.model[22]  # Detect()
        m.stride = torch.tensor([ 8., 16., 32.])# forward
        self.stride = m.stride
        m.bias_init()  # only run once
        

        # Init weights, biases
        initialize_weights(self)
        
        ## figure out how to print out the layers

    def forward(self, x, *args, **kwargs):
        """
        Perform forward pass of the model for either training or inference.

        If x is a dict, calculates and returns the loss for training. Otherwise, returns predictions for inference.

        Args:
            x (torch.Tensor | dict): Input tensor for inference, or dict with image tensor and labels for training.
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            (torch.Tensor): Loss if x is a dict (training), or network predictions (inference).
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        
        return self.predict(x, *args, **kwargs)

    #Setting augment to True combines the output from all the detection heads
    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            augment (bool): Augment image during prediction, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        # print(f"testing my input predict once{x.shape}")
        objects = None
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model:
            # with open("model_weights.txt", "a+") as f:
            #     f.write(f"\nLayer: {type(m).__name__}\n")
            #     total_params = 0
                
            #     # Get state dict
            #     for name, param in m.state_dict().items():
            #         f.write(f"\nParameter: {name}\n")
            #         f.write(f"Shape: {param.shape}\n")
            #         f.write(f"Values:\n{param.detach().cpu().numpy()}\n")
            #         total_params += param.numel()
                    
            #     f.write(f"\nTotal parameters: {total_params:,}\n")
            #     f.write("-" * 80 + "\n")
               
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            # print(type(m), m.i)
            
            # if type(x) is list:
            #     # print(y)
            #     print([i.shape for i in x])
            # else:
            #     print(x.shape)
                
            if type(m) is Detect:
                objects = m(x)
                
                # print('Ran')
                y.append(x if m.i in self.save else None)  # save output
                continue
            
            x = m(x)  # run  
            y.append(x if m.i in self.save else None)  # save output
            
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return (x, objects)

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs."""
        # if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":
        #     print("WARNING ⚠️ Model does not support 'augment=True', reverting to single-scale prediction.")
        #     return self._predict_once(x)
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._predict_once(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y
    
    def load_checkpoint(self, checkpoint_path, device='cpu', verbose=False):
        """Load and transfer weights from checkpoint"""
        try:
            # 1. Load checkpoint
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            # 2. Extract state dict
            if isinstance(ckpt, dict):
                state_dict = ckpt.get('model', ckpt)
            
            # 3. Handle different model prefixes
            new_state_dict = {}
            for k, v in state_dict.state_dict().items():
                # Remove 'model.' prefix if present
                new_key = k.replace('model.','')
                # Add back model prefix to match target
                new_key = f'model.{new_key}'
                new_state_dict[new_key] = v.float()

            # with open("test_model_weights.txt","w+") as f:
            #     f.write(f"{new_state_dict.items()}")
            # 4. Load weights
            missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
            
            if verbose:
                print(f'Loaded checkpoint: {checkpoint_path}')
                print(f'Missing keys: {missing}')
                print(f'Unexpected keys: {unexpected}')
                
            return True

        except Exception as e:
            print(f'Error loading checkpoint: {str(e)}')
            return False
        
    def load(self, weights, verbose=True):
        """
        Load the weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress. Defaults to True.
        """
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(csd, strict=False)  # load
        if verbose:
            print(f"Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights")
            
    def fuse(self, verbose=True):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the
        computation efficiency.

        Returns:
            (nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepVGGDW):
                    m.fuse()
                    m.forward = m.forward_fuse
            self.info(verbose=verbose)

        return self
    
    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        Prints model information.

        Args:
            detailed (bool): if True, prints out detailed information about the model. Defaults to False
            verbose (bool): if True, prints out the model information. Defaults to False
            imgsz (int): the size of the image that the model will be trained on. Defaults to 640
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def is_fused(self, thresh=10):
        """
        Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers. Default is 10.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def loss(self, batch, preds=None):
        """
        Compute loss.

        Args:
            batch (dict): Batch to compute loss on
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        preds = self.forward(batch["img"]) if preds is None else preds
        return preds[0],self.criterion(preds[1], batch)
    
    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return E2EDetectLoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)
    
    def preprocess(self, im):
        """
        Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        # im = im.to(self.device)
        # im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im = im.float()
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0
        return im
    
    def postprocess(self, preds, img, orig_imgs,img_path):
        """Post-processes predictions and returns a list of Results objects."""
        print(preds[0].shape,[i.shape for i in preds[1]])
        preds = ops.non_max_suppression(
            preds
            # self.args.conf,
            # self.args.iou,
            # agnostic=self.args.agnostic_nms,
            # max_det=self.args.max_det,
            # classes=self.args.classes,
        )
        
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for pred, orig_img in zip(preds, orig_imgs):
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=img_path, names=self.names, boxes=pred))
        return results
    
    def inference(self, im0s,img_path):
        # Check if save_dir/ label file exists
        # Warmup model
        self.model[22].training = False
        profilers = (
            ops.Profile(device=self.device),
            ops.Profile(device=self.device),
            ops.Profile(device=self.device),
        )
        # Preprocess
        with profilers[0]:
            im = self.preprocess(im0s)

        # Inference
        with profilers[1]:
            new_image, preds = self.predict(im)
            print(type(new_image), new_image.shape)
            
            # with open("out.txt", "w+") as f:
            #     f.write(f"{preds}")
            # print(f"checking output after inference {preds[0].shape} {[i.shape for i in preds[1]]}")
            
        # Postprocess
        with profilers[2]:
            self.results = self.postprocess(preds, im, im0s,img_path)
            
        # # Visualize, save, write results
        # n = len(im0s)
        # for i in range(n):
        #     self.seen += 1
        #     self.results[i].speed = {
        #         "preprocess": profilers[0].dt * 1e3 / n,
        #         "inference": profilers[1].dt * 1e3 / n,
        #         "postprocess": profilers[2].dt * 1e3 / n,
        #     }
            

        return new_image, self.results



# Usage:
def compare_models(model1, model2, threshold=1e-6):
    """Compare weights between two models"""
    # Get state dicts
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    
    comparison = {}
    with open("my_model.txt", "w+") as f:
        f.write(str(state_dict1.items()))
    with open("loaded_model.txt", "w+") as f:
        f.write(str(state_dict2.items()))
    

    
    
    # Compare each layer
    for key in state_dict1.keys():
        if key in state_dict2:
            tensor1 = state_dict1[key]
            tensor2 = state_dict2[key]
            
            comparison[key] = {
                'shape_match': tensor1.shape == tensor2.shape,
                'shape1': tuple(tensor1.shape),
                'shape2': tuple(tensor2.shape),
                'mean_diff': (tensor1 - tensor2).abs().mean().item() if tensor1.shape == tensor2.shape else None,
                'max_diff': (tensor1 - tensor2).abs().max().item() if tensor1.shape == tensor2.shape else None
            }
    
    # Print report
    
    with open("compare_model_weights.txt","w+") as f:
        f.write("\nModel Comparison Report:")
        for key, info in comparison.items():
            if not info['shape_match'] or info['mean_diff'] > threshold:
                f.write(f"\nLayer: {key}")
                f.write(f"Shape match: {info['shape_match']}")
                f.write(f"Shape1: {info['shape1']}")
                f.write(f"Shape2: {info['shape2']}")
                if info['shape_match']:
                    f.write(f"Mean diff: {info['mean_diff']:.6f}")
                    f.write(f"Max diff: {info['max_diff']:.6f}")
    
    return comparison



if __name__ == "__main__":
    from PIL import Image
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt

    # Load the image
    image_path = 'original.jpg'
    image = Image.open(image_path).convert('RGB')

    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])

    # Apply the transformation to the image
    x = transform(image).unsqueeze(0)  # Add batch dimension
    
    
    # print(config)
    model = WNet()
    model.load_checkpoint("yolov8n.pt")
    
    
    # print(model)
    
    from ultralytics import YOLO
    
    test_model = YOLO("yolov8n.pt","detect")
    # with open("yolo_yaml.txt", "w") as f:
    #     f.write(str(test_model.model))
    # with open("yolo_struct.txt", "w") as f:
    #     f.write(str(struct.model))
    
    # comparison = compare_models(model, test_model.model)
 
    model.fuse()
    output = test_model(x, augment = False)
    print('yolo')
    print(output[0].boxes)
    for result in output:
        im = result.plot(show=True,boxes = False)
    
    new_image, results = model.inference(x, image_path)
    print("wnet")
    print(results[0].boxes)
    for result in results:
        im = result.plot(show=True,boxes = False )
        #save=True, filename="model_output_detect.jpg"

    plt.figure(figsize=(20, 10))

    plt.title('Official YOLO Detections')
    plt.axis('off')
    
    
    if isinstance(new_image, torch.Tensor):
        im2 = new_image.detach().numpy()
    im2 = im2[0,0,:,:]
    
    print(im2.shape)
    plt.imshow(im2)
    plt.title('Custom Image')
    plt.axis('off')
    plt.savefig("output_image.png")
    plt.show()
    
    
    
    