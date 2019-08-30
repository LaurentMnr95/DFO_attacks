import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from visdom import Visdom
from attacks.FGSM import *
from attacks.DFOattacks2 import *
from options_classifier import *
from utils import *
import torch.backends.cudnn as cudnn
from attacks import bandit_bb
import torchvision.transforms.functional as F
from torchvision import models
import torch.nn as nn
import math
from advertorch.attacks import Attack, LabelMixin
#import torch.nn.functional as F


def normalize(x, mean, std):
    y = torch.ones_like(x)
    y[:, 0, :, :] = (x[:, 0, :, :]-mean[0]) / std[0]
    y[:, 1, :, :] = (x[:, 1, :, :]-mean[1]) / std[1]
    y[:, 2, :, :] = (x[:, 2, :, :]-mean[2]) / std[2]
    return y


class norm_classifier(nn.Module):
    def __init__(self, classifier, normalize, mean, std):
        super(norm_classifier, self).__init__()
        self.classifier = classifier
        self.normalize = normalize
        self.mean = mean
        self.std = std

    def forward(self, x):
        y = self.normalize(x, self.mean, self.std)
        y = self.classifier(y)
        return y


class GradientSignAttack(Attack, LabelMixin):
    """
    One step fast gradient sign method (Goodfellow et al, 2014).
    Paper: https://arxiv.org/abs/1412.6572
    :param predict: forward pass function.
    :param loss_fn: loss function.
    :param eps: attack step size.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param targeted: indicate if this is a targeted attack.
    """

    def __init__(self, predict, loss_fn=None, eps=0.05, tile_size=20, clip_min=0.,
                 clip_max=1., targeted=False):
        """
        Create an instance of the GradientSignAttack.
        """
        super(GradientSignAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)

        self.eps = eps
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")
        self.tile_size = tile_size

    def perturb(self, x, y=None):
        """
        Given examples (x, y), returns their adversarial counterparts with
        an attack length of eps.
        :param x: input tensor.
        :param y: label tensor.
                  - if None and self.targeted=False, compute y as predicted
                    labels.
                  - if self.targeted=True, then y must be the targeted labels.
        :return: tensor containing perturbed inputs.
        """

        x, y = self._verify_and_process_inputs(x, y)
        xadv = x.requires_grad_()
        outputs = self.predict(xadv)

        loss = self.loss_fn(outputs, y)
        if self.targeted:
            loss = -loss
        loss.backward()
        grad_x = xadv.grad.detach()
        size_image = grad_x.shape[3]

        K = int(size_image/self.tile_size)
        for i in range(K):
            for j in range(K):
                i1 = i*self.tile_size
                i2 = min((i+1)*self.tile_size, size_image)
                j1 = j*self.tile_size
                j2 = min((j+1)*self.tile_size, size_image)
                grad_x[:, 0, i1:i2, j1:j2] = grad_x[:, 0, i1:i2, j1:j2].sum()
                grad_x[:, 1, i1:i2, j1:j2] = grad_x[:, 1, i1:i2, j1:j2].sum()
                grad_x[:, 2, i1:i2, j1:j2] = grad_x[:, 2, i1:i2, j1:j2].sum()

        grad_sign = grad_x.sign()

        xadv = xadv + self.eps * grad_sign

        xadv = torch.clamp(xadv, self.clip_min, self.clip_max)

        return xadv


CLASSIFIERS = {
    "inception_v3": (models.inception_v3, 299),
    "resnet50": (models.resnet50, 224),
    "vgg16_bn": (models.vgg16_bn, 224),
}

NUM_CLASSES = {
    "imagenet": 1000
}


def main(classifier="resnet50", epsilon=0.05, s=50, mode="noise", save_file="test.txt"):
    # define options
    torch.manual_seed(0)
    Classifier_u, image_size = CLASSIFIERS[classifier]

    batch_size = 50
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            "/datasets01_101/imagenet_full_size/061417/val",
            transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True)

    Classifier_u = Classifier_u(pretrained=True)
    Classifier = norm_classifier(Classifier_u, normalize, mean, std)
    Classifier = Classifier.cuda()
    Classifier = torch.nn.DataParallel(Classifier, device_ids=range(torch.cuda.device_count()))  # ,device_ids)
    cudnn.benchmark = True
    Classifier.eval()

    print("Classifier intialized")
    print(Classifier)

    # Set these to whatever you want for your gaussian filter
    kernel_size = 15
    sigma = 1

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean_ker = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
        np.exp(
        -torch.sum((xy_grid - mean_ker)**2., dim=-1) /
        (2*variance)
    )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=3, out_channels=3, padding=7,
                                kernel_size=kernel_size, groups=3, bias=False)
    gaussian_filter.weight.data = gaussian_kernel.float()
    gaussian_filter.weight.requires_grad = False

    gaussian_filter = gaussian_filter.cuda()

    upsampler = Upsample(size=(image_size, image_size))
    attack = GradientSignAttack(Classifier, tile_size=s)

    acc = 0
    pred_change = 0
    num_images = 0

    for i, data in enumerate(test_loader, 0):

        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        # normalize the inputs
        #inputs_norm = normalize(inputs,mean,std)

        # standard outputs
        outputs = Classifier(inputs)
        _, predicted = torch.max(outputs.data, 1)

        #
        if mode == "fgsm":
            inputs_adv = attack.perturb(inputs, predicted)
            outputs_adv = Classifier(inputs_adv)
            _, predicted_adv = torch.max(outputs_adv.data, 1)

        if mode == "noise":
            with torch.no_grad():
               # noisy inputs
                noise = torch.ones([inputs.shape[0], 3, s, s])*0.5
                noise = 2*torch.bernoulli(noise)-1
                noise = upsampler(noise)
                noise = noise.cuda()
                inputs_noise = inputs+noise*epsilon
                # inputs_noise = gaussian_filter(inputs_noise)
                inputs_noise = torch.clamp(inputs_noise, 0, 1)

                # noisy outputs
                outputs_noise = Classifier(inputs_noise)
                _, predicted_adv = torch.max(outputs_noise.data, 1)

        acc += (predicted == labels).double().sum().item()
        pred_change += (predicted_adv != predicted).double().sum().item()
        num_images += labels.shape[0]

    # if i == 0:
    #     torchvision.utils.save_image(inputs_adv, "images_adv.jpg", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)
    #     torchvision.utils.save_image(inputs, "images_nat.jpg", nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0)

    acc = acc/num_images
    pred_change = pred_change/num_images
    if save_file is not None:
        with open(save_file, 'a') as f:
            f.write('{} {} {} {} {}\n'.format(classifier, epsilon, s, acc, pred_change))


if __name__ == "__main__":
    main()
