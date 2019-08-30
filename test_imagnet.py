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


def main(optimizer_DFO="cGA", epsilon=0.05, prior_size=50, max_budget=1000, outfile="results"):
    # define options
    torch.manual_seed(0)
    batch_size = 1
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])
    # min_ = ((torch.zeros(3) - mean) / std).min().item()
    # max_ = ((torch.ones(3) - mean) / std).max().item()
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            "/datasets01_101/imagenet_full_size/061417/val",
            transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor()])),
        # transforms.Normalize(mean, std)])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True)

    Classifier = torchvision.models.vgg16_bn(pretrained=True)
    Classifier.cuda()
    Classifier = torch.nn.DataParallel(Classifier, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
    Classifier.eval()

    print("Classifier intialized")
    print(Classifier)

    def normalized_eval(x):
        x_copy = x.clone()
        x_copy = torch.stack([F.normalize(x_copy[i], mean, std)
                              for i in range(batch_size)])
        return Classifier(x_copy)

    correctly_classified = 0
    attack_success = 0

    num_images = 1000

    elapsed_budgets = []
    for i, data in enumerate(test_loader, 0):

        if i >= num_images:
            break
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        # zero the parameter gradients

        # forward + backward + optimize
        outputs = normalized_eval(inputs)

        _, predicted = torch.max(outputs.data, 1)
        with torch.no_grad():
            correctly_classified += (predicted == labels).double().sum().item()

        if torch.any(predicted == labels):
            if optimizer_DFO == "bandits":
                with torch.no_grad():
                    res = bandit_bb.make_adversarial_examples(inputs, labels, Classifier,
                                                              nes=False, mode="linf", epsilon=epsilon, max_queries=max_budget,
                                                              gradient_iters=1, fd_eta=0.1, image_lr=0.01, online_lr=100,
                                                              exploration=1, prior_size=prior_size,
                                                              log_progress=True)

            else:
                with torch.no_grad():
                    res = DFOattack(normalized_eval, inputs, predicted, eps=epsilon,
                                    optimizer=optimizer_DFO, budget=max_budget, prior_size=prior_size)

            done = res["success"]
            budget = res["elapsed_budget"]

            if done:
                elapsed_budgets += [budget]
                attack_success += 1

        if correctly_classified > 0:
            print("current success rate:", attack_success/correctly_classified)
            print("current average budget:", np.mean(np.array(elapsed_budgets)))
    np.savetxt('budgets_DFO/'+optimizer_DFO+"_"+str(epsilon)+'.txt', np.array(elapsed_budgets))
    med_budget = np.median(np.array(elapsed_budgets))
    mean_budget = np.mean(np.array(elapsed_budgets))
    with open(outfile, 'a') as f:
        f.write(optimizer_DFO+' '+str(epsilon)+' '+str(prior_size)+' '+str(mean_budget)+' '+str(med_budget) +
                ' '+str(correctly_classified/num_images)+' '+str(attack_success/correctly_classified)+'\n')

    print("Accuracy on test data for natural images:{}".format(correctly_classified/num_images))
    print("Attack success rate:{}".format(attack_success/correctly_classified))
    print("Accuracy {} {} {} {} {}".format(optimizer_DFO, mean_budget,
                                           med_budget, epsilon, attack_success/correctly_classified))
    print('median budget', med_budget)
    print('mean budget', mean_budget)


if __name__ == "__main__":
    main()
