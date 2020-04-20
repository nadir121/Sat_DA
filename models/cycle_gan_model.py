import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn as nn
from torch.autograd import Variable


class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--init_weights', type=str, default='semantic model initialization ', help='directory to the pth model')
            parser.add_argument('--lambda_semantic', type=float, default=1, help='semantic loss weight')
            parser.add_argument('--lambda_pa', type=float, default=0.1, help='semantic loss weight')
            parser.add_argument('--lambda_pb', type=float, default=0.1, help='semantic loss weight')
            parser.add_argument('--lambda_d', type=float, default=1, help='descriminator loss')
            parser.add_argument('--lambda_g', type=float, default=1, help='generator loss')
            parser.add_argument('--output_stride', type=int, default=0, help='output stride (0=DLV2)')

        return parser

    #def initialize(self, opt):
     #   BaseModel.initialize(self, opt)

    def __init__(self, opt):
        """Initialize the CycleGAN class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions"""
        BaseModel.__init__(self, opt)


        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        #self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'sem_A', 'rec_sem_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'sem_B', 'rec_sem_B', 'G']
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'sem_A', 'rec_sem_A', 'D_B', 'G_B', 'cycle_B', 'sem_B', 'rec_sem_B', 'G']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:   #lys
            visual_names_A = ['real_A', 'fake_B', 'rec_A']
            visual_names_B = ['real_B', 'fake_A', 'rec_B']
            visual_names_A_feat = ['real_A_feat', 'fake_B_feat', 'rec_A_feat']############
            visual_names_B_feat = ['real_B_feat', 'fake_A_feat', 'rec_B_feat']############
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_A')
            visual_names_B.append('idt_B')
        if self.isTrain:
            self.visual_names = visual_names_A + visual_names_B
            self.visual_names_feat = visual_names_A_feat + visual_names_B_feat##############333
        else:
            if self.opt.direction == 'AtoB':
                self.visual_names = ['fake_B']
            else:
                self.visual_names = ['fake_A']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']
        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        #print("\n\n=============== GA =================\n\n")
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            #use_sigmoid = opt.no_lsgan
            use_sigmoid = False
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            self.semantic = networks.semantic(init_weights=opt.init_weights, gpu_ids=self.gpu_ids, output_stride=opt.output_stride)
            #self.interp = nn.Upsample(size=(opt.fineSize, opt.fineSize), mode='bilinear', align_corners=True)
            self.instancenorm = nn.InstanceNorm2d(7, affine=False)#classes
        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            #self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterion = networks.CrossEntropy2d().to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)        
        self.real_B = input['B' if AtoB else 'A'].to(self.device)       
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        if self.isTrain:
            self.fake_B = self.netG_A(self.real_A)
            #self.rec_A = self.netG_B(torch.cat((self.fake_B[:,0:3,:,:], self.real_A[:,3,:,:].unsqueeze(1)), 1)) #lys
            self.rec_A = self.netG_B(self.fake_B)

            self.fake_A = self.netG_B(self.real_B)
            #self.rec_B = self.netG_A(torch.cat((self.fake_A[:,0:3,:,:], self.real_B[:,3,:,:].unsqueeze(1)), 1))
            self.rec_B = self.netG_A(self.fake_A)
        else:
            if self.opt.direction == 'AtoB':
                self.fake_B = self.netG_A(self.real_A)
            else:
                self.fake_A = self.netG_B(self.real_A)
    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B) * self.opt.lambda_d

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A) * self.opt.lambda_d

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True) * self.opt.lambda_g
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True) * self.opt.lambda_g
        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # semantic loss
        self.real_A_feat, self.rec_A_feat = self.compute_semantic_feat(self.real_A), self.compute_semantic_feat(self.rec_A)
        self.real_B_feat, self.rec_B_feat = self.compute_semantic_feat(self.real_B), self.compute_semantic_feat(self.rec_B)
        self.fake_A_feat, self.fake_B_feat = self.compute_semantic_feat(self.fake_A), self.compute_semantic_feat(self.fake_B)
        self.loss_rec_sem_A = self.compute_semantic_loss(self.rec_A_feat, self.real_A_feat) * lambda_A * self.opt.lambda_semantic #lys
        self.loss_rec_sem_B = self.compute_semantic_loss(self.rec_B_feat, self.real_B_feat) * lambda_B * self.opt.lambda_semantic #lys
        
        self.loss_sem_A = self.compute_semantic_loss(self.fake_B_feat, self.real_A_feat) * self.opt.lambda_pa #lys
        self.loss_sem_B = self.compute_semantic_loss(self.fake_A_feat, self.real_B_feat) * self.opt.lambda_pb #lys  
        
        # combined loss
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_sem_A +self.loss_sem_B + self.loss_rec_sem_A + self.loss_rec_sem_B
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()

    def compute_semantic_loss(self, img_feat, target_feat):
        return torch.mean((self.instancenorm(img_feat) - self.instancenorm(target_feat)) ** 2)

    def compute_semantic_feat(self, img):
        img_vgg = self.img_preprocess(img)
        img_feat = self.semantic(img_vgg)
        return img_feat  

    def img_preprocess(self, batch):
        tensortype = type(batch.data)
        (r, g, b) = torch.chunk(batch, 3, dim=1)
        batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR
        batch = (batch + 1) * 255 * 0.5  # [-1, 1] -> [0, 255]
        mean = tensortype(batch.data.size())
        mean[:, 0, :, :] = 104.00698793
        mean[:, 1, :, :] = 116.66876762
        mean[:, 2, :, :] = 122.67891434
        batch = batch.sub(Variable(mean).to(self.device))  # subtract mean
        return batch
