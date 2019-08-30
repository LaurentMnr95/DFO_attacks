import torch as ch
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.nn.modules import Upsample
import argparse
import json
import pdb



#ch.set_default_tensor_type('torch.cuda.FloatTensor')

def norm(t):
    assert len(t.shape) == 4
    norm_vec = ch.sqrt(t.pow(2).sum(dim=[1,2,3])).view(-1, 1, 1, 1)
    norm_vec += (norm_vec == 0).float()*1e-8
    return norm_vec

###
# Different optimization steps
# All take the form of func(x, g, lr)
# eg: exponentiated gradients
# l2/linf: projected gradient descent
###

def eg_step(x, g, lr):
    real_x = (x + 1)/2 # from [-1, 1] to [0, 1]
    pos = real_x*ch.exp(lr*g)
    neg = (1-real_x)*ch.exp(-lr*g)
    new_x = pos/(pos+neg)
    return new_x*2-1

def linf_step(x, g, lr):
    return x + lr*ch.sign(g)

def l2_prior_step(x, g, lr):
    new_x = x + lr*g/norm(g)
    norm_new_x = norm(new_x)
    norm_mask = (norm_new_x < 1.0).float()
    return new_x*norm_mask + (1-norm_mask)*new_x/norm_new_x

def gd_prior_step(x, g, lr):
    return x + lr*g
   
def l2_image_step(x, g, lr):
    return x + lr*g/norm(g)

##
# Projection steps for l2 and linf constraints:
# All take the form of func(new_x, old_x, epsilon)
##

def l2_proj(image, eps):
    orig = image.clone()
    def proj(new_x):
        delta = new_x - orig
        out_of_bounds_mask = (norm(delta) > eps).float()
        x = (orig + eps*delta/norm(delta))*out_of_bounds_mask
        x += new_x*(1-out_of_bounds_mask)
        return x
    return proj

def linf_proj(image, eps):
    orig = image.clone()
    def proj(new_x):
        return orig + ch.clamp(new_x - orig, -eps, eps)
    return proj

##
# Main functions
##



def make_adversarial_examples(image, true_label, model_to_fool,
                                nes=True,mode="linf",epsilon=0.04,max_queries=10000,
                                gradient_iters=50,fd_eta=0.05,image_lr=0.0001,online_lr=100,
                                exploration=1, prior_size = 50,
                                log_progress=True):


    '''
    The main process for generating adversarial examples with priors.
    '''
    # Initial setup
    batch_size = image.size(0)
    total_queries = ch.zeros(batch_size)
    upsampler = Upsample(size=(image.size(2),image.size(2)))
    prior = ch.zeros(batch_size, 3, prior_size, prior_size).cuda()


    dim = prior.nelement()/batch_size
    prior_step = gd_prior_step if mode == 'l2' else eg_step
    image_step = l2_image_step if mode == 'l2' else linf_step
    proj_maker = l2_proj if mode == 'l2' else linf_proj
    proj_step = proj_maker(image, epsilon)
    
    def normalized_eval(x):
        x_copy = x.clone()
        x_copy = ch.stack([F.normalize(x_copy[i], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) \
                        for i in range(batch_size)])
        return model_to_fool(x_copy)

    # Loss function
    criterion = ch.nn.CrossEntropyLoss(reduction='none')

    losses = criterion(normalized_eval(image), true_label)

    # Original classifications
    orig_images = image.clone()
    orig_classes = normalized_eval(image).argmax(1).cuda()
    correct_classified_mask = (orig_classes == true_label).cpu().float()
    total_ims = correct_classified_mask.cpu().sum()
    not_dones_mask = correct_classified_mask.cpu().clone()

    t = 0

    while not ch.any(total_queries > max_queries):
        t += gradient_iters*2
        if t >= max_queries:
            break
        if not nes:
            ## Updating the prior: 
            # Create noise for exporation, estimate the gradient, and take a PGD step
            exp_noise = exploration*ch.randn_like(prior)/(dim**0.5) 
            # Query deltas for finite difference estimator
            q1 = upsampler(prior + exp_noise)
            q2 = upsampler(prior - exp_noise)
            # Loss points for finite difference estimator
            l1 = criterion(normalized_eval(image + fd_eta*q1/norm(q1)), true_label) # L(prior + c*noise)
            l2 = criterion(normalized_eval(image + fd_eta*q2/norm(q2)), true_label) # L(prior - c*noise)
            # Finite differences estimate of directional derivative
            est_deriv = (l1 - l2)/(fd_eta*exploration)
            # 2-query gradient estimate
            est_grad = est_deriv.view(-1, 1, 1, 1)*exp_noise
            # Update the prior with the estimated gradient

            prior = prior_step(prior, est_grad, online_lr)

        else:
            prior = ch.zeros_like(image)
            for _ in range(gradient_iters):

                exp_noise = ch.randn_like(image)/(dim**0.5) 
                est_deriv = (criterion(normalized_eval(image + fd_eta*exp_noise), true_label) -
                         criterion(normalized_eval(image - fd_eta*exp_noise), true_label))/fd_eta
                prior += est_deriv.view(-1, 1, 1, 1)*exp_noise

            # Preserve images that are already done, 
            # Unless we are specifically measuring gradient estimation
            prior = prior*not_dones_mask.view(-1, 1, 1, 1)

        ## Update the image:
        # take a pgd step using the prior
        new_im = image_step(image, upsampler(prior*correct_classified_mask.cuda().view(-1, 1, 1, 1)), image_lr)
        image = proj_step(new_im)

        image = ch.clamp(image, 0, 1)

        ## Continue query count
        total_queries += 2*gradient_iters*not_dones_mask
        not_dones_mask = not_dones_mask*((normalized_eval(image).argmax(1) == true_label).cpu().float())

        ## Logging stuff
        new_losses = criterion(normalized_eval(image), true_label).cpu()
        success_mask = (1 - not_dones_mask)*correct_classified_mask
        num_success = success_mask.cpu().sum()
        current_success_rate = (num_success/correct_classified_mask.sum()).cpu().item()
        success_queries = ((success_mask*total_queries).sum()/num_success).cpu().item()
        not_done_loss = ((new_losses*not_dones_mask).cpu().sum()/not_dones_mask.sum()).item()
        max_curr_queries = total_queries.max().cpu().item()
        if log_progress:
            print("Queries: %d | Success rate: %f | Average queries: %f" % (max_curr_queries, current_success_rate, success_queries))

        if current_success_rate == 1.0:
            break
    if batch_size == 1:
        return {"image_adv":image.cpu().numpy(), 
                    "prediction":normalized_eval(image).argmax(1), 
                    "elapsed_budget":total_queries.cpu().numpy()[0],
                    "success":success_mask.cpu().numpy()[0] ==True}
    return {
                'average_queries': success_queries,
                'num_correctly_classified': correct_classified_mask.sum().cpu().item(),
                'success_rate': current_success_rate,
                'images_orig': orig_images.cpu().numpy(),
                'images_adv': image.cpu().numpy(),
                'all_queries': total_queries.cpu().numpy(),
                'correctly_classified': correct_classified_mask.cpu().numpy(),
                'success': success_mask.cpu().numpy()
        }

