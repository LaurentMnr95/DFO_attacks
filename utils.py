import torch
#from resnet import *
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from visdom import Visdom
from options_classifier import *
from argparse import ArgumentParser
from utils import *
from torch.optim import lr_scheduler
from networks import *

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.start_epoch - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize_lr, gamma=opt.step_gamma)
    elif opt.lr_policy == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, opt.milestones, gamma=opt.step_gamma, last_epoch=-1)      
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def getNetwork(args):
    if (args.net_type == 'lenet'):
        net = LeNet(args.num_classes)
        file_name = 'lenet'
    elif (args.net_type == 'vggnet'):
        net = VGG(args.depth, args.num_classes)
        file_name = 'vgg-'+str(args.depth)
    elif (args.net_type == 'resnet'):
        net = ResNet(args.depth, args.num_classes)
        file_name = 'resnet-'+str(args.depth)
    elif (args.net_type == 'wide-resnet'):
        net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, args.num_classes,sparsify_level=args.sparsify)
        file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)+'_sparse_'+str(args.sparsify)
    else:
        print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')
        sys.exit(0)

    return net, file_name

def load_data(opt,train_mode=True):
    if opt.dataset == "CIFAR10":
        if train_mode==True:
            transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
        else:
            transform = transforms.Compose([
                    transforms.ToTensor(),
                    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
        loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(os.path.join(opt.datadir,opt.dataset), train=train_mode, download=True, transform=transform),
                batch_size=opt.batch_size, shuffle=True)
        print("Loaded CIFAR 10 dataset")
    elif opt.dataset == "MNIST":
        loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(os.path.join(opt.datadir,opt.dataset), train=train_mode, download=True, transform=transform),
                batch_size=opt.batch_size, shuffle=True)
        print("Loaded MNIST dataset")
    return loader
