import torch
from resnet import *
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

def main(optimizer_DFO="bandits",epsilon=0.05,max_budget=10000):
    # define options
    opt  = options_test()
    # opt.batch_size =100
    # defining device
    # TODO change device gestion
    # if torch.cuda.is_available():
    #     print("GPU found: device set to cuda:0")
    #     device = torch.device("cuda:{}".format(opt.gpu))
    # else:
    #     print("No GPU found: device set to cpu")
    #     device = torch.device("cpu")


    # Load inputs
    test_loader = load_data(opt, train_mode=False)
    num_images = len(test_loader.dataset)
    # Classifier  definition
    #Classifier,filename = getNetwork(opt)
    checkpoint = torch.load(opt.path_to_model)
    Classifier = checkpoint['net']

    if opt.gpu:
        Classifier.cuda()
        Classifier = torch.nn.DataParallel(Classifier,device_ids=range(torch.cuda.device_count()))#,device_ids)
        cudnn.benchmark =True
    print("Classifier intialized")
    print(Classifier)

    Classifier.eval()

    # Testing
    running_acc = 0
    running_acc_adv = 0

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
        outputs = Classifier(inputs)

        _, predicted = torch.max(outputs.data, 1)
        with torch.no_grad():
                correctly_classified += (predicted==labels).double().sum().item()


        if torch.any(predicted==labels):
            if optimizer_DFO=="bandits":
                with torch.no_grad():
                    res = bandit_bb.make_adversarial_examples(inputs, labels, Classifier,
                                    nes=False,mode="linf",epsilon=epsilon,max_queries=max_budget,
                                    gradient_iters=1,fd_eta=0.01,image_lr=0.0001,online_lr=100,
                                    exploration=0.01, prior_size = 10,
                                    log_progress=False
                                    )
            else:
                with torch.no_grad():
                    res = DFOattack(Classifier, inputs ,predicted, eps=epsilon, x_val_min=0, x_val_max=1, optimizer=optimizer_DFO, budget=max_budget)

            done =res["success"]
            budget =res["elapsed_budget"]


            if done:
                elapsed_budgets += [budget]
                attack_success +=1

        if correctly_classified>0:
            print("current success rate:",attack_success/correctly_classified)

    np.savetxt('budgets_DFO/'+optimizer_DFO+"_"+str(epsilon)+'.txt',np.array(elapsed_budgets))
    med_budget = np.median(np.array(elapsed_budgets))
    mean_budget = np.mean(np.array(elapsed_budgets))
    with open('dfo_eps_file.txt', 'a') as f:
        f.write(optimizer_DFO+' '+str(epsilon)+' '+str(mean_budget)+' '+str(med_budget)+' '+str(correctly_classified/num_images)+' '+str(attack_success/correctly_classified)+'\n')
    
    print("Accuracy on test data for natural images:{}".format(correctly_classified/num_images))
    print("Attack success rate:{}".format(attack_success/correctly_classified))
    print("Accuracy {} {} {} {} {}".format(optimizer_DFO,mean_budget, med_budget, epsilon, attack_success/correctly_classified))
    print('median budget', med_budget)
    print('mean budget', mean_budget)

if __name__ == "__main__":
    main()
   