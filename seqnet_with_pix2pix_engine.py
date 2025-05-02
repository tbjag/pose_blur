import os.path as osp
from re import S
import torch

import math
import sys
from copy import deepcopy
import time

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import average_precision_score
import wandb


# Import from SeqNet
from SeqNet.utils.utils import MetricLogger, SmoothedValue, mkdir, reduce_dict, warmup_lr_scheduler
from SeqNet.utils.km import run_kuhn_munkres
from SeqNet.utils.utils import write_json, mkdir


def to_device(images, targets, device):
    images = [image.to(device) for image in images]
    for t in targets:
        t["boxes"] = t["boxes"].to(device)
        t["labels"] = t["labels"].to(device)
    return images, targets

def _compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union


def eval_detection(
    gallery_dataset, gallery_dets, det_thresh=0.5, iou_thresh=0.5, labeled_only=False
):
    """
    gallery_det (list of ndarray): n_det x [x1, y1, x2, y2, score] per image
    det_thresh (float): filter out gallery detections whose scores below this
    iou_thresh (float): treat as true positive if IoU is above this threshold
    labeled_only (bool): filter out unlabeled background people
    """
    assert len(gallery_dataset) == len(gallery_dets)
    annos = gallery_dataset.annotations

    y_true, y_score = [], []
    count_gt, count_tp = 0, 0
    for anno, det in zip(annos, gallery_dets):
        gt_boxes = anno["boxes"]
        if labeled_only:
            # exclude the unlabeled people (pid == 5555)
            inds = np.where(anno["pids"].ravel() != 5555)[0]
            if len(inds) == 0:
                continue
            gt_boxes = gt_boxes[inds]
        num_gt = gt_boxes.shape[0]

        if det != []:
            det = np.asarray(det)
            inds = np.where(det[:, 4].ravel() >= det_thresh)[0]
            det = det[inds]
            num_det = det.shape[0]
        else:
            num_det = 0
        if num_det == 0:
            count_gt += num_gt
            continue

        ious = np.zeros((num_gt, num_det), dtype=np.float32)
        for i in range(num_gt):
            for j in range(num_det):
                ious[i, j] = _compute_iou(gt_boxes[i], det[j, :4])
        tfmat = ious >= iou_thresh
        # for each det, keep only the largest iou of all the gt
        for j in range(num_det):
            largest_ind = np.argmax(ious[:, j])
            for i in range(num_gt):
                if i != largest_ind:
                    tfmat[i, j] = False
        # for each gt, keep only the largest iou of all the det
        for i in range(num_gt):
            largest_ind = np.argmax(ious[i, :])
            for j in range(num_det):
                if j != largest_ind:
                    tfmat[i, j] = False
        for j in range(num_det):
            y_score.append(det[j, -1])
            y_true.append(tfmat[:, j].any())
        count_tp += tfmat.sum()
        count_gt += num_gt

    det_rate = count_tp * 1.0 / count_gt
    ap = average_precision_score(y_true, y_score) * det_rate

    print("{} detection:".format("labeled only" if labeled_only else "all"))
    print("  recall = {:.2%}".format(det_rate))
    if not labeled_only:
        print("  ap = {:.2%}".format(ap))
    return det_rate, ap


def eval_search_cuhk(
    gallery_dataset,
    query_dataset,
    gallery_dets,
    gallery_feats,
    query_box_feats,
    query_dets,
    query_feats,
    k1=10,
    k2=3,
    det_thresh=0.5,
    cbgm=False,
    gallery_size=100,
):
    """
    gallery_dataset/query_dataset: an instance of BaseDataset
    gallery_det (list of ndarray): n_det x [x1, x2, y1, y2, score] per image
    gallery_feat (list of ndarray): n_det x D features per image
    query_feat (list of ndarray): D dimensional features per query image
    det_thresh (float): filter out gallery detections whose scores below this
    gallery_size (int): gallery size [-1, 50, 100, 500, 1000, 2000, 4000]
                        -1 for using full set
    """
    assert len(gallery_dataset) == len(gallery_dets)
    assert len(gallery_dataset) == len(gallery_feats)
    assert len(query_dataset) == len(query_box_feats)

    use_full_set = gallery_size == -1
    fname = "TestG{}".format(gallery_size if not use_full_set else 50)
    protoc = loadmat(osp.join(gallery_dataset.root, "annotation/test/train_test", fname + ".mat"))
    protoc = protoc[fname].squeeze()

    # mapping from gallery image to (det, feat)
    annos = gallery_dataset.annotations
    name_to_det_feat = {}
    for anno, det, feat in zip(annos, gallery_dets, gallery_feats):
        name = anno["img_name"]
        if len(det) != 0:
            scores = det[:, 4].ravel()
            inds = np.where(scores >= det_thresh)[0]
            if len(inds) > 0:
                name_to_det_feat[name] = (det[inds], feat[inds])

    aps = []
    accs = []
    topk = [1, 5, 10]
    ret = {"image_root": gallery_dataset.img_prefix, "results": []}
    for i in range(len(query_dataset)):
        y_true, y_score = [], []
        imgs, rois = [], []
        count_gt, count_tp = 0, 0
        # get L2-normalized feature vector
        feat_q = query_box_feats[i].ravel()
        # ignore the query image
        query_imname = str(protoc["Query"][i]["imname"][0, 0][0])
        query_roi = protoc["Query"][i]["idlocate"][0, 0][0].astype(np.int32)
        query_roi[2:] += query_roi[:2]
        query_gt = []
        tested = set([query_imname])

        name2sim = {}
        name2gt = {}
        sims = []
        imgs_cbgm = []
        # 1. Go through the gallery samples defined by the protocol
        for item in protoc["Gallery"][i].squeeze():
            gallery_imname = str(item[0][0])
            # some contain the query (gt not empty), some not
            gt = item[1][0].astype(np.int32)
            count_gt += gt.size > 0
            # compute distance between query and gallery dets
            if gallery_imname not in name_to_det_feat:
                continue
            det, feat_g = name_to_det_feat[gallery_imname]
            # no detection in this gallery, skip it
            if det.shape[0] == 0:
                continue
            # get L2-normalized feature matrix NxD
            assert feat_g.size == np.prod(feat_g.shape[:2])
            feat_g = feat_g.reshape(feat_g.shape[:2])
            # compute cosine similarities
            sim = feat_g.dot(feat_q).ravel()

            if gallery_imname in name2sim:
                continue
            name2sim[gallery_imname] = sim
            name2gt[gallery_imname] = gt
            sims.extend(list(sim))
            imgs_cbgm.extend([gallery_imname] * len(sim))
        # 2. Go through the remaining gallery images if using full set
        if use_full_set:
            # TODO: support CBGM when using full set
            for gallery_imname in gallery_dataset.imgs:
                if gallery_imname in tested:
                    continue
                if gallery_imname not in name_to_det_feat:
                    continue
                det, feat_g = name_to_det_feat[gallery_imname]
                # get L2-normalized feature matrix NxD
                assert feat_g.size == np.prod(feat_g.shape[:2])
                feat_g = feat_g.reshape(feat_g.shape[:2])
                # compute cosine similarities
                sim = feat_g.dot(feat_q).ravel()
                # guaranteed no target query in these gallery images
                label = np.zeros(len(sim), dtype=np.int32)
                y_true.extend(list(label))
                y_score.extend(list(sim))
                imgs.extend([gallery_imname] * len(sim))
                rois.extend(list(det))

        if cbgm:
            # -------- Context Bipartite Graph Matching (CBGM) ------- #
            sims = np.array(sims)
            imgs_cbgm = np.array(imgs_cbgm)
            # only process the top-k1 gallery images for efficiency
            inds = np.argsort(sims)[-k1:]
            imgs_cbgm = set(imgs_cbgm[inds])
            for img in imgs_cbgm:
                sim = name2sim[img]
                det, feat_g = name_to_det_feat[img]
                # only regard the people with top-k2 detection confidence
                # in the query image as context information
                qboxes = query_dets[i][:k2]
                qfeats = query_feats[i][:k2]
                assert (
                    query_roi - qboxes[0][:4]
                ).sum() <= 0.001, "query_roi must be the first one in pboxes"

                # build the bipartite graph and run Kuhn-Munkres (K-M) algorithm
                # to find the best match
                graph = []
                for indx_i, pfeat in enumerate(qfeats):
                    for indx_j, gfeat in enumerate(feat_g):
                        graph.append((indx_i, indx_j, (pfeat * gfeat).sum()))
                km_res, max_val = run_kuhn_munkres(graph)

                # revise the similarity between query person and its matching
                for indx_i, indx_j, _ in km_res:
                    # 0 denotes the query roi
                    if indx_i == 0:
                        sim[indx_j] = max_val
                        break
        for gallery_imname, sim in name2sim.items():
            gt = name2gt[gallery_imname]
            det, feat_g = name_to_det_feat[gallery_imname]
            # assign label for each det
            label = np.zeros(len(sim), dtype=np.int32)
            if gt.size > 0:
                w, h = gt[2], gt[3]
                gt[2:] += gt[:2]
                query_gt.append({"img": str(gallery_imname), "roi": list(map(float, list(gt)))})
                iou_thresh = min(0.5, (w * h * 1.0) / ((w + 10) * (h + 10)))
                inds = np.argsort(sim)[::-1]
                sim = sim[inds]
                det = det[inds]
                # only set the first matched det as true positive
                for j, roi in enumerate(det[:, :4]):
                    if _compute_iou(roi, gt) >= iou_thresh:
                        label[j] = 1
                        count_tp += 1
                        break
            y_true.extend(list(label))
            y_score.extend(list(sim))
            imgs.extend([gallery_imname] * len(sim))
            rois.extend(list(det))
            tested.add(gallery_imname)
        # 3. Compute AP for this query (need to scale by recall rate)
        y_score = np.asarray(y_score)
        y_true = np.asarray(y_true)
        assert count_tp <= count_gt
        recall_rate = count_tp * 1.0 / count_gt
        ap = 0 if count_tp == 0 else average_precision_score(y_true, y_score) * recall_rate
        aps.append(ap)
        inds = np.argsort(y_score)[::-1]
        y_score = y_score[inds]
        y_true = y_true[inds]
        accs.append([min(1, sum(y_true[:k])) for k in topk])
        # 4. Save result for JSON dump
        new_entry = {
            "query_img": str(query_imname),
            "query_roi": list(map(float, list(query_roi))),
            "query_gt": query_gt,
            "gallery": [],
        }
        # only record wrong results
        if int(y_true[0]):
            continue
        # only save top-10 predictions
        for k in range(10):
            new_entry["gallery"].append(
                {
                    "img": str(imgs[inds[k]]),
                    "roi": list(map(float, list(rois[inds[k]]))),
                    "score": float(y_score[k]),
                    "correct": int(y_true[k]),
                }
            )
        ret["results"].append(new_entry)

    print("search ranking:")
    print("  mAP = {:.2%}".format(np.mean(aps)))
    accs = np.mean(accs, axis=0)
    for i, k in enumerate(topk):
        print("  top-{:2d} = {:.2%}".format(k, accs[i]))

    write_json(ret, "vis/results.json")

    ret["mAP"] = np.mean(aps)
    ret["accs"] = accs
    return ret



def save_tensor_as_jpg(tensor, filepath, denormalize=True):
    """
    Save a PyTorch tensor as a JPG image.
    
    Args:
        tensor (torch.Tensor): Image tensor of shape (C, H, W) or (B, C, H, W)
        filepath (str): Path to save the image
        denormalize (bool): Whether to denormalize from [-1,1] to [0,1] range
    """
    import torch
    import numpy as np
    from PIL import Image
    
    # Make a copy of the tensor to avoid modifying the original
    img_tensor = tensor.clone().detach()
    
    # If tensor is batched (B, C, H, W), take the first image
    if len(img_tensor.shape) == 4:
        img_tensor = img_tensor[0]
    
    # Move to CPU if necessary
    if img_tensor.is_cuda:
        img_tensor = img_tensor.cpu()
    
    # Denormalize if needed (assuming the tensor is in [-1, 1] range)
    if denormalize:
        img_tensor = (img_tensor + 1) / 2.0
    
    # Clamp values to be in [0, 1]
    img_tensor = torch.clamp(img_tensor, 0, 1)
    
    # Convert to numpy and transpose from (C, H, W) to (H, W, C)
    img_np = img_tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    
    # Convert to uint8 in range [0, 255]
    img_np = (img_np * 255).astype(np.uint8)
    
    # Handle both RGB and grayscale
    if img_np.shape[2] == 1:
        img_np = img_np[:, :, 0]
    
    # Save the image
    img = Image.fromarray(img_np)
    img.save(filepath)
    print(f"Image saved to {filepath}")
    
    return filepath

def test_one_epoch(cfg, model_seqnet, modeL_pix2pix, optimizer, data_loader, device, epoch, tfboard=None,  use_wandb= False):
    model_seqnet.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    # warmup learning rate in the first epoch
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        # FIXME: min(1000, len(data_loader) - 1)
        warmup_iters = len(data_loader) - 1
        warmup_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    
    with open("test_file.txt", "w") as f:
            for i, (data) in enumerate(
                data_loader
            ):
                if i < 10080:
                    continue
                modeL_pix2pix.set_input(data)  # unpack data from data loader
                modeL_pix2pix.test()           # run inference
                images = modeL_pix2pix.fake_B
                # save_tensor_as_jpg(images, "image.jpg")
                
                targets = {"img_name": data["img_name"], "boxes": torch.as_tensor(data['bbox'], dtype=torch.float32), "labels": data["labels"]}
                
                # targets = {"img_name": data["img_name"], "boxes": data['bbox'], "labels": data["labels"]}

                targets = [targets]
                # print(images.shape)
                # exit()
                images = [images[0]]
                images, targets = to_device(images, targets, device)
                targets[0]['boxes'] =targets[0]['boxes'][0]
                targets[0]['labels'] = targets[0]['labels'][0]
                
                
                f.write(f"{targets}")

                
                try:
                    loss_dict = model_seqnet(images, targets)
                except Exception as e:
                    print(e)
                    print(targets)
                    
                losses = sum(loss for loss in loss_dict.values())
                print(f"{i} {losses}")
                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = reduce_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                loss_value = losses_reduced.item()

                if not math.isfinite(loss_value):
                    print(f"Loss is {loss_value}, stopping training")
                    print(loss_dict_reduced)
                    sys.exit(1)

                optimizer.zero_grad()
                if use_wandb:
                    wandb.log(loss_dict)
                losses.backward()
                if cfg.SOLVER.CLIP_GRADIENTS > 0:
                    clip_grad_norm_(model_seqnet.parameters(), cfg.SOLVER.CLIP_GRADIENTS)
                optimizer.step()

                if epoch == 0:
                    warmup_scheduler.step()

                metric_logger.update(loss=loss_value, **loss_dict_reduced)
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])
                if tfboard:
                    iter = epoch * len(data_loader) + i
                    for k, v in loss_dict_reduced.items():
                        tfboard.add_scalars("train", {k: v}, iter)
                avg_loss = metric_logger.meters["loss"].global_avg
            return avg_loss
        

def train_one_epoch_combined(opt,cfg, model_seqnet, modeL_pix2pix, optimizer, data_loader, device, epoch,visualizer, tfboard=None, wandb= False):
    """
    Combined training function for SeqNet and Pix2Pix
    
    Args:
        opt: Options for training
        cfg: SeqNet config
        model_seqnet: SeqNet model
        model_pix2pix: Pix2Pix model
        optimizer: Optimizer
        data_loader: Data loader
        device: Device to use
        epoch: Current epoch
        visualizer: Visualizer for Pix2Pix
        tfboard: TensorBoard SummaryWriter
        use_wandb: Whether to use wandb
        total_iters: Current total iterations (passed between epochs)
        
    Returns:
        tuple: (avg_loss, updated_total_iters)
    """

    model_seqnet.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    dataset_size = len(data_loader)
        
    
    # Initialize total_iters if not provided
    total_iters = epoch * dataset_size * opt.batch_size


    # warmup learning rate in the first epoch
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        # FIXME: min(1000, len(data_loader) - 1)
        warmup_iters = len(data_loader) - 1
        warmup_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    

    epoch_iter = 0                  # the number of training iterations in current epoch
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()    # timer for data loading per iteration
    epoch_iter = 0                  # the number of training iterations in current epoch
    visualizer.reset()              # reset the visualizer
    modeL_pix2pix.update_learning_rate()    # update learning rates
    
    for i, (data) in enumerate(
        metric_logger.log_every(data_loader, cfg.DISP_PERIOD, header)
    ):
        iter_start_time = time.time()  # timer for computation per iteration
        if total_iters % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time

        total_iters += opt.batch_size
        epoch_iter += opt.batch_size
        
        #Important here of setting input and running pix2pix model
        modeL_pix2pix.set_input(data)         # unpack data from dataset and apply preprocessing
        
        modeL_pix2pix.forward()                   # compute fake images: G(A)
        # update D
        modeL_pix2pix.set_requires_grad(modeL_pix2pix.netD, True)  # enable backprop for D
        modeL_pix2pix.optimizer_D.zero_grad()     # set D's gradients to zero
        modeL_pix2pix.backward_D()                # calculate gradients for D
        modeL_pix2pix.optimizer_D.step()          # update D's weights
        
        
        modeL_pix2pix.set_requires_grad(modeL_pix2pix.netD, False)  # D requires no gradients when optimizing G
        modeL_pix2pix.optimizer_G.zero_grad()        # set G's gradients to zero
        losses_G = modeL_pix2pix.loss_G()   # calculate loss functions, get gradients, update network weights
        images = modeL_pix2pix.fake_B
        
        #getting input for seqnet model
        targets = {"img_name": data["img_name"], "boxes": torch.as_tensor(data['bbox'], dtype=torch.float32), "labels": data["labels"]}
        targets = [targets]
        images = [images[0]]
        images, targets = to_device(images, targets, device)
        targets[0]['boxes'] =targets[0]['boxes'][0]
        targets[0]['labels'] = targets[0]['labels'][0]
        
        # running seqnet model
        loss_dict = model_seqnet(images, targets)
            
        losses_seq = sum(loss for loss in loss_dict.values())
        
        total_losses = losses_seq + losses_G

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if wandb:
            wandb.log(loss_dict)
        
        total_losses.backward()
        
        modeL_pix2pix.optimizer_G.step()             # update G's weights
        
        if cfg.SOLVER.CLIP_GRADIENTS > 0:
            clip_grad_norm_(model_seqnet.parameters(), cfg.SOLVER.CLIP_GRADIENTS)
        optimizer.step()

        if epoch == 0:
            warmup_scheduler.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
        if tfboard:
            iter = epoch * len(data_loader) + i
            for k, v in loss_dict_reduced.items():
                tfboard.add_scalars("train", {k: v}, iter)
        avg_loss = metric_logger.meters["loss"].global_avg
        
        if total_iters % opt.display_freq == 0:   # display images on visdom and save images to HTML
            save_result = total_iters % opt.update_html_freq == 0
            modeL_pix2pix.compute_visuals()
            visualizer.display_current_results(modeL_pix2pix.get_current_visuals(), epoch, save_result)

        if total_iters % opt.print_freq == 0:    # print training losses and save logging information
            losses = modeL_pix2pix.get_current_losses()
            t_comp = (time.time() - iter_start_time) / opt.batch_size
            visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
            if opt.display_id > 0:
                visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

        if total_iters % opt.save_latest_freq == 0:   # cache our latest model
            print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            modeL_pix2pix.save_networks(save_suffix)

        iter_data_time = time.time()

        
    print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

    return avg_loss


def train_one_epoch(cfg, model_seqnet, modeL_pix2pix, optimizer, data_loader, device, epoch, tfboard=None, use_wandb= False):
    model_seqnet.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    # warmup learning rate in the first epoch
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        # FIXME: min(1000, len(data_loader) - 1)
        warmup_iters = len(data_loader) - 1
        warmup_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for i, (data) in enumerate(
        metric_logger.log_every(data_loader, cfg.DISP_PERIOD, header)
    ):
        modeL_pix2pix.set_input(data)  # unpack data from data loader
        modeL_pix2pix.test()           # run inference
        images = modeL_pix2pix.fake_B
        # save_tensor_as_jpg(images, "image.jpg")
        
        targets = {"img_name": data["img_name"], "boxes": torch.as_tensor(data['bbox'], dtype=torch.float32), "labels": data["labels"]}
        
        # targets = {"img_name": data["img_name"], "boxes": data['bbox'], "labels": data["labels"]}

        targets = [targets]
        # print(images.shape)
        # exit()
        images = [images[0]]
        images, targets = to_device(images, targets, device)
        targets[0]['boxes'] =targets[0]['boxes'][0]
        targets[0]['labels'] = targets[0]['labels'][0]
        
        try:
            loss_dict = model_seqnet(images, targets)
        except Exception as e:
            print(e)
            print(targets)
            
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if use_wandb:
            wandb.log(loss_dict)
        losses.backward()
        if cfg.SOLVER.CLIP_GRADIENTS > 0:
            clip_grad_norm_(model_seqnet.parameters(), cfg.SOLVER.CLIP_GRADIENTS)
        optimizer.step()

        if epoch == 0:
            warmup_scheduler.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if tfboard:
            iter = epoch * len(data_loader) + i
            for k, v in loss_dict_reduced.items():
                tfboard.add_scalars("train", {k: v}, iter)
        avg_loss = metric_logger.meters["loss"].global_avg
    return avg_loss

@torch.no_grad()
def evaluate_performance(
    model_seqnet,model_pix2pix, gallery_loader, query_loader, device, use_gt=False, use_cache=False, use_cbgm=False
):
    """
    Args:
        use_gt (bool, optional): Whether to use GT as detection results to verify the upper
                                bound of person search performance. Defaults to False.
        use_cache (bool, optional): Whether to use the cached features. Defaults to False.
        use_cbgm (bool, optional): Whether to use Context Bipartite Graph Matching algorithm.
                                Defaults to False.
    """
    model_seqnet.eval()
    if use_cache:
        eval_cache = torch.load("data/eval_cache/eval_cache.pth")
        gallery_dets = eval_cache["gallery_dets"]
        gallery_feats = eval_cache["gallery_feats"]
        query_dets = eval_cache["query_dets"]
        query_feats = eval_cache["query_feats"]
        query_box_feats = eval_cache["query_box_feats"]
    else:
        gallery_dets, gallery_feats = [], []
        for i, (data) in tqdm(enumerate(
            gallery_loader
        ), ncols=0, total=len(gallery_loader)):
            model_pix2pix.set_input(data)  # unpack data from data loader
            model_pix2pix.test()           # run inference
            visuals = model_pix2pix.get_current_visuals()  # get image results
            images = visuals['fake_B']
            # save_tensor_as_jpg(images, "image.jpg")
            
            targets = {"img_name": data["img_name"], "boxes": torch.as_tensor(data['bbox'], dtype=torch.float32), "labels": data["labels"]}
            
            # targets = {"img_name": data["img_name"], "boxes": data['bbox'], "labels": data["labels"]}

            targets = [targets]
            # print(images.shape)
            # exit()
            images = [images[0]]
            images, targets = to_device(images, targets, device)
            targets[0]['boxes'] =targets[0]['boxes'][0]
            targets[0]['labels'] = targets[0]['labels'][0]

            if not use_gt:
                outputs = model_seqnet(images)
            else:
                boxes = targets[0]["boxes"]
                n_boxes = boxes.size(0)
                embeddings = model_seqnet(images, targets)
                outputs = [
                    {
                        "boxes": boxes,
                        "embeddings": torch.cat(embeddings),
                        "labels": torch.ones(n_boxes).to(device),
                        "scores": torch.ones(n_boxes).to(device),
                    }
                ]

            for output in outputs:
                box_w_scores = torch.cat([output["boxes"], output["scores"].unsqueeze(1)], dim=1)
                gallery_dets.append(box_w_scores.cpu().numpy())
                gallery_feats.append(output["embeddings"].cpu().numpy())

        # regarding query image as gallery to detect all people
        # i.e. query person + surrounding people (context information)
        query_dets, query_feats = [], []
        for i, (data) in tqdm(enumerate(
            query_loader
        ), ncols=0, total=len(query_loader)):
            model_pix2pix.set_input(data)  # unpack data from data loader
            model_pix2pix.test()           # run inference
            visuals = model_pix2pix.get_current_visuals()  # get image results
            images = visuals['fake_B']
            # save_tensor_as_jpg(images, "image.jpg")
            
            targets = {"img_name": data["img_name"], "boxes": torch.as_tensor(data['bbox'], dtype=torch.float32), "labels": data["labels"]}
            

            targets = [targets]
            images = [images[0]]
            images, targets = to_device(images, targets, device)
            targets[0]['boxes'] =targets[0]['boxes'][0]
            targets[0]['labels'] = targets[0]['labels'][0]

            # targets will be modified in the model, so deepcopy it
            outputs = model_seqnet(images, deepcopy(targets), query_img_as_gallery=True)

            # consistency check
            gt_box = targets[0]["boxes"].squeeze()
            assert (
                gt_box - outputs[0]["boxes"][0]
            ).sum() <= 0.001, "GT box must be the first one in the detected boxes of query image"

            for output in outputs:
                box_w_scores = torch.cat([output["boxes"], output["scores"].unsqueeze(1)], dim=1)
                query_dets.append(box_w_scores.cpu().numpy())
                query_feats.append(output["embeddings"].cpu().numpy())

        # extract the features of query boxes
        query_box_feats = []
        for i, (data) in tqdm(enumerate(
            query_loader
        ), ncols=0 , total=len(query_loader)):
            model_pix2pix.set_input(data)  # unpack data from data loader
            model_pix2pix.test()           # run inference
            visuals = model_pix2pix.get_current_visuals()  # get image results
            images = visuals['fake_B']
            # save_tensor_as_jpg(images, "image.jpg")
            
            targets = {"img_name": data["img_name"], "boxes": torch.as_tensor(data['bbox'], dtype=torch.float32), "labels": data["labels"]}
            

            targets = [targets]
            images = [images[0]]
            images, targets = to_device(images, targets, device)
            targets[0]['boxes'] =targets[0]['boxes'][0]
            targets[0]['labels'] = targets[0]['labels'][0]
            embeddings = model_seqnet(images, targets)
            assert len(embeddings) == 1, "batch size in test phase should be 1"
            query_box_feats.append(embeddings[0].cpu().numpy())

        mkdir("data/eval_cache")
        save_dict = {
            "gallery_dets": gallery_dets,
            "gallery_feats": gallery_feats,
            "query_dets": query_dets,
            "query_feats": query_feats,
            "query_box_feats": query_box_feats,
        }
        torch.save(save_dict, "data/eval_cache/eval_cache.pth")

    eval_detection(gallery_loader.dataset, gallery_dets, det_thresh=0.01)
    eval_search_func = (
        eval_search_cuhk 
    )
    eval_search_func(
        gallery_loader.dataset,
        query_loader.dataset,
        gallery_dets,
        gallery_feats,
        query_box_feats,
        query_dets,
        query_feats,
        cbgm=use_cbgm,
    )



