import torch
from .base_model import BaseModel
from . import networks


class Pix2PixfftModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        # Add frequency loss for FFT
        parser.add_argument('--lambda_freq', type=float, default=10.0, help='weight for frequency loss')

        return parser

    def __init__(self, opt, path = None):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'G_freq','D_real', 'D_fake', 'OD']
        # self.loss_names = ['G_GAN', 'G_L1','D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if opt.netG == 'wnet':
            path = "/home/wenjun/Lab/GAN_project/tanush_pose_blur/models/yolov8n.pt"
            self.netG.module.load_checkpoint(path)
            # self.optimizer_W_Net = self.build_optimizer(
            #         model=self.netG.module,
            #     )
            # self.optimizers.append(self.optimizer_W_Net)
            # self.scaler = torch.amp.GradScaler("cuda", enabled=True)

            
        self.model_name = opt.netG
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionFreq = self.frequency_loss  # Custom frequency-domain loss for FFT
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.bounding_boxes = input['bbox'] #Check bounding boxes for FFT
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.model_name == 'wnet':
            # print(self.bounding_boxes)
            # print(torch.tensor(self.bounding_boxes))
            input = {"img":self.real_A,"batch_idx":torch.tensor([0]*len(self.bounding_boxes)),"cls":torch.tensor([0]*len(self.bounding_boxes)), "bboxes":torch.tensor(self.bounding_boxes)}
            self.fake_B, self.loss_OD= self.netG(input)  # G(A)
            # 
            self.loss_OD = self.loss_OD[0]
        else:
            
            self.fake_B = self.netG(self.real_A)  # G(A)

    def fft_image(self, img):
        fft_result = torch.fft.fft2(img)
        fft_shifted = torch.fft.fftshift(fft_result)
        magnitude = torch.abs(fft_shifted)
        return magnitude



    def create_background_mask(self, image_shape, bounding_boxes):
        mask = torch.ones(image_shape[1:], dtype=torch.float32, device=self.device)
        for box in bounding_boxes:
            x1, y1, x2, y2 = map(int, box)
            mask[y1:y2, x1:x2] = 0
        return mask.unsqueeze(0)



    def frequency_loss(self, real_img, generated_img, bounding_boxes):
        real_fft = self.fft_image(real_img)
        generated_fft = self.fft_image(generated_img)
        mask = self.create_background_mask(real_img.shape, bounding_boxes)
        high_freq_real = real_fft * mask
        high_freq_generated = generated_fft * mask
        loss = torch.mean(torch.abs(high_freq_real - high_freq_generated))
        return loss
    
    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        self.loss_G_freq = self.criterionFreq(self.real_B, self.fake_B, self.bounding_boxes) * self.opt.lambda_freq
        
        # combine loss and calculate gradients
        #self.loss_G = self.loss_G_GAN + self.loss_G_L1

        #combine all losses, including the freq loss
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_freq
        
        if self.model_name == 'wnet':
            # print(self.loss_G)
            # print([i.shape for i in self.od_loss])
            self.loss_G += self.loss_OD

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # update G's weights
