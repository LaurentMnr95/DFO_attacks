import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from nevergrad.optimization import optimizerlib
import numpy
from scipy.special import softmax
from torch.nn.modules import Upsample


def DFOattack(net, x, y, criterion=F.cross_entropy, eps=0.1, optimizer="DE", budget=10000, prior_size=5):

    x = x.cuda()
    upsampler = Upsample(size=(224, 224))
    s = prior_size

    def convert_individual_to_image(individual):
        perturbation = torch.from_numpy(eps*individual.astype(numpy.float32))
        perturbation = perturbation.view(1, 3, s, s)
        perturbation = upsampler(perturbation)
        perturbation = perturbation.cuda()
        x_adv = x + perturbation
        x_adv = torch.clamp(x_adv, 0, 1)
        return x_adv

    def loss(abc):
        if optimizer in ['cGA', 'PBIL']:
            individual = 2*abc-1
        else:
            abc = abc.reshape((-1, 2))
            abc = softmax(abc, axis=1)
            individual = 2*(numpy.random.uniform(size=abc.shape[0]) < abc[:, 0])-1
        x_adv = convert_individual_to_image(individual)

        netx_adv = net(x_adv)
        _, predicted = torch.max(netx_adv.data, 1)
        result = (-criterion(netx_adv, y)).detach().cpu().numpy()
        return result, predicted, x_adv
    # def loss(a):
    #     a=1/(1+numpy.exp(a))
    #     individual = 2*(numpy.random.uniform(size=a.shape)<a)-1
    #     x_adv = convert_individual_to_image(individual)
    #     netx_adv = net(x_adv)
    #     _,predicted = torch.max(netx_adv.data, 1)
    #     result = (-criterion(netx_adv,y)).detach().cpu().numpy()
    #     return result, predicted,x_adv

    done = False
    if optimizer in ['cGA', 'PBIL']:
        optimizerer = optimizerlib.registry[optimizer](instrumentation=s*s*3, budget=budget)
    else:
        optimizerer = optimizerlib.registry[optimizer](instrumentation=s*s*3*2, budget=budget)

    ebudget = 0
    for u in range(budget):
        if u % 100 == 0:
            print(u, "/", budget)

        curr_value = optimizerer.ask()
        values, predicted, x_adv = loss(numpy.array(curr_value.args[0]))
        ebudget += 1
        if predicted != y:
            # print(y,predicted)
            print('win', ebudget)
            done = True
            break
        optimizerer.tell(curr_value, float(values))

    return {"image_adv": x_adv.cpu().numpy(),
            "prediction": predicted,
            "elapsed_budget": ebudget,
            "success": done}
