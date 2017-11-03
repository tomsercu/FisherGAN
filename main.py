from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os

import models.dcgan as dcgan
import models.mlp as mlp

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--wdecay', type=float, default=0.000, help='wdecay value for Phi')
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--hiDiterStart'  , action='store_true', help='do many D iters at start')
parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
parser.add_argument('--mlp_G', action='store_true', help='use MLP for G')
parser.add_argument('--mlp_D', action='store_true', help='use MLP for D')
parser.add_argument('--G_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--D_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--rho', type=float, default=1e-6, help='Weight on the penalty term for (sigmas -1)**2')
opt = parser.parse_args()
print(opt)

if opt.experiment is None:
    opt.experiment = 'samples'
os.system('mkdir {0}'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw', 'celeba']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = int(opt.nc)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

if opt.noBN:
    netG = dcgan.DCGAN_G_nobn(opt.imageSize, nz, nc, ngf, ngpu, opt.G_extra_layers)
elif opt.mlp_G:
    netG = mlp.MLP_G(opt.imageSize, nz, nc, ngf, ngpu)
else:
    netG = dcgan.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu, opt.G_extra_layers)

netG.apply(weights_init)
if opt.netG != '': # load checkpoint if needed
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

if opt.mlp_D:
    netD = mlp.MLP_D(opt.imageSize, nz, nc, ndf, ngpu)
else:
    netD = dcgan.DCGAN_D(opt.imageSize, nz, nc, ndf, ngpu, opt.D_extra_layers)
    netD.apply(weights_init)

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
one = torch.FloatTensor([1])
mone = one * -1
alpha = torch.FloatTensor([0]) # lagrange multipliers

if opt.cuda:
    netD.cuda()
    netG.cuda()
    input = input.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    alpha = alpha.cuda()
alpha = Variable(alpha, requires_grad=True)

# setup optimizer
if opt.adam:
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999), weight_decay=opt.wdecay)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999), weight_decay=opt.wdecay)
else:
    optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lrD, weight_decay=opt.wdecay)
    optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG, weight_decay=opt.wdecay)

gen_iterations = 0
for epoch in range(opt.niter):
    data_iter = iter(dataloader)
    i = 0
    while i < len(dataloader):
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        # train the discriminator Diters times
        if opt.hiDiterStart and (gen_iterations < 25 or gen_iterations % 500 == 0):
            Diters = 100
        else:
            Diters = opt.Diters
        j = 0
        while j < Diters and i < len(dataloader):
            j += 1

            data = data_iter.next()
            i += 1

            # train with real
            real_cpu, _ = data
            netD.zero_grad()
            batch_size = real_cpu.size(0)

            if opt.cuda:
                real_cpu = real_cpu.cuda()
            input.resize_as_(real_cpu).copy_(real_cpu)
            inputv = Variable(input)

            vphi_real = netD(inputv)

            # train with fake
            noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise, volatile = True) # totally freeze netG
            fake = Variable(netG(noisev).data)
            inputv = fake

            vphi_fake = netD(inputv)
            # NOTE here f = <v,phi>   , but with modified f the below two lines are the
            # only ones that need change. E_P and E_Q refer to Expectation over real and fake.
            E_P_f,  E_Q_f  = vphi_real.mean(), vphi_fake.mean()
            E_P_f2, E_Q_f2 = (vphi_real**2).mean(), (vphi_fake**2).mean()
            constraint = (1 - (0.5*E_P_f2 + 0.5*E_Q_f2))
            # See Equation (9)
            obj_D = E_P_f - E_Q_f + alpha * constraint - opt.rho/2 * constraint**2
            # max_w min_alpha obj_D. Compute negative gradients, apply updates with negative sign.
            obj_D.backward(mone)
            optimizerD.step()
            # artisanal sgd. We minimze alpha so a <- a + lr * (-grad)
            alpha.data += opt.rho * alpha.grad.data
            alpha.grad.data.zero_()

        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():
            p.requires_grad = False # to avoid computation
        netG.zero_grad()
        # in case our last batch was the tail batch of the dataloader,
        # make sure we feed a full batch of noise
        noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        vphi_fake = netD(fake)
        obj_G = -vphi_fake.mean() # Just minimize mean difference
        obj_G.backward() # G: min_theta
        optimizerG.step()
        gen_iterations += 1

        IPM_enum  = E_P_f.data[0]  - E_Q_f.data[0]
        IPM_denom = (0.5*E_P_f2.data[0] + 0.5*E_Q_f2.data[0]) ** 0.5
        IPM_ratio = IPM_enum / IPM_denom
        print(('[%d/%d][%d/%d][%d] IPM_enum: %.4f IPM_denom: %.4f IPM_ratio: %.4f '
               'E_P_f: %.4f E_Q_f: %.4f E_P_(f^2): %.4f E_Q_(f^2): %.4f')
            % (epoch, opt.niter, i, len(dataloader), gen_iterations,
            IPM_enum, IPM_denom, IPM_ratio,
            E_P_f.data[0], E_Q_f.data[0], E_P_f2.data[0], E_Q_f2.data[0]))
        if gen_iterations % 500 == 0:
            real_cpu = real_cpu.mul(0.5).add(0.5)
            vutils.save_image(real_cpu, '{0}/real_samples.png'.format(opt.experiment))
            fake = netG(Variable(fixed_noise, volatile=True))
            fake.data = fake.data.mul(0.5).add(0.5)
            vutils.save_image(fake.data, '{0}/fake_samples_{1}.png'.format(opt.experiment, gen_iterations))

    # do checkpointing
    torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
    torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))
