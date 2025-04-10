import datetime
import os.path as osp
import time
import torch


import torch

from train_seqnet_with_pix2pix import  train_one_epoch,  evaluate_performance


# Import from SeqNet
from SeqNet.defaults import get_default_cfg

from SeqNet.utils.utils import mkdir, resume_from_ckpt, save_on_master, set_random_seed
from SeqNet.models.seqnet import SeqNet

# Import from GAN
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from options.train_options import TrainOptions
from options.test_options import TestOptions



def train_seqnet(opt, model_pix2pix):
    """Train SeqNet model"""
    cfg = get_default_cfg()
    if opt.cfg_file:
        cfg.merge_from_file(opt.cfg_file)
    cfg.freeze()

    device = torch.device(cfg.DEVICE)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    print("Creating SeqNet model")
    model = SeqNet(cfg)
    model.to(device)

    print("Loading data")
    train_loader = create_dataset(opt)
    gallery_loader, query_loader = create_dataset(opt,split="gallery"),  create_dataset(opt,split="query")

    if opt.eval:
        assert opt.ckpt, "--ckpt must be specified when --eval enabled"
        resume_from_ckpt(opt.ckpt, model)
        evaluate_performance(
            model,
            gallery_loader,
            query_loader,
            device,
            use_gt=cfg.EVAL_USE_GT,
            use_cache=cfg.EVAL_USE_CACHE,
            use_cbgm=cfg.EVAL_USE_CBGM,
        )
        return

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.SGD_MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.SOLVER.LR_DECAY_MILESTONES, gamma=0.1
    )

    start_epoch = 0
    if opt.resume:
        assert opt.ckpt, "--ckpt must be specified when --resume enabled"
        start_epoch = resume_from_ckpt(opt.ckpt, model, optimizer, lr_scheduler) + 1

    print("Creating output folder")
    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)
    path = osp.join(output_dir, "config.yaml")
    with open(path, "w") as f:
        f.write(cfg.dump())
    print(f"Full config is saved to {path}")
    
    tfboard = None
    if cfg.TF_BOARD:
        from torch.utils.tensorboard import SummaryWriter
        tf_log_path = osp.join(output_dir, "tf_log")
        mkdir(tf_log_path)
        tfboard = SummaryWriter(log_dir=tf_log_path)
        print(f"TensorBoard files are saved to {tf_log_path}")

    print("Start training SeqNet")
    start_time = time.time()
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
        train_one_epoch(cfg, model, optimizer, train_loader, device, epoch, tfboard)
        lr_scheduler.step()

        if (epoch + 1) % cfg.EVAL_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS - 1:
            evaluate_performance(
                model,
                gallery_loader,
                query_loader,
                device,
                use_gt=cfg.EVAL_USE_GT,
                use_cache=cfg.EVAL_USE_CACHE,
                use_cbgm=cfg.EVAL_USE_CBGM,
            )

        if (epoch + 1) % cfg.CKPT_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS - 1:
            save_on_master(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                },
                osp.join(output_dir, f"epoch_{epoch}.pth"),
            )

    if tfboard:
        tfboard.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time {total_time_str}")

def train_gan(opt):
    """Train GAN model"""
    dataset = create_dataset(opt )  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    print("Start training GAN")
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch
        visualizer.reset()              # reset the visualizer
        model.update_learning_rate()    # update learning rates
        
        for i, data in enumerate(dataset):
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to HTML
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
            
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

def combined_train(opt):
    pass

if __name__ == '__main__':
    
    options = TrainOptions().parse()
    train_gan(options)
    
    # options = TestOptions().parse
    # options.num_threads = 0   # test code only supports num_threads = 0
    # options.batch_size = 1    # test code only supports batch_size = 1
    # options.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    # options.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    # options.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    # model_pix2pix = create_model(options)      # create a model given opt.model and other options
    # model_pix2pix.setup(options)               # regular setup: load and print networks; create schedulers
    # model_pix2pix.eval()
    
    # train_seqnet(options, model_pix2pix)

    
    
    
    
        
