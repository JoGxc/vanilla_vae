#!/usr/bin/env python3
# -*- coding: utf_8 -*-
"""
vanilla VAE
"""
import os
import glob
import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as nn

SAVED_MODEL = False
NORMALIZING_FLOW = False
NUM_FLOWS = 0
DTYPE = torch.FloatTensor
IMG_PATHS = glob.glob('./data/sample/*.jpg')

batch_size = 64
num_batches = int(len(IMG_PATHS)/batch_size)
num_epoch = 5000
X_dim = 64*64
h_dim = 256 # Latent dimension in question 3?
Z_dim = 100 
c = 0 #??
lr = 0.002

if NORMALIZING_FLOW:
    flow = 'norm_flow'
else:
    flow = ''

MODEL_PATH = './saved_model/' + 'model_{}_{}_{}/'.format(flow, NUM_FLOWS, Z_dim)
MODEL_FILE = MODEL_PATH + 'model'
image_path = 'out_{}_{}_{}/'.format(flow, NUM_FLOWS, Z_dim)

# =============================== preprocessing ======================================

def scale(image):
    """
    scale all the pixel between -1 and 1
    """
    min_value = np.min(image)
    max_value = np.max(image)
    scaled_image = (image - min_value)/(max_value - min_value)*2 - 1
    return scaled_image

FACES = [] 
batch_count = -1
image_count = 1
print(' [*] Loading celebrity faces...')
for image_file in tqdm(IMG_PATHS):
    # convert the image to a numpy array 
    # each pixel as a value between 0 and 256
    face = np.asarray(Image.open(image_file).convert('L'))
    scaled_face = scale(face).ravel().reshape(1, X_dim)
    if image_count % batch_size == 1:
        # start a new batch
        FACES.append(scaled_face)
        batch_count += 1
    else:
        # add current example to the array of the current batch.
        # a batch with batch size 10 and image size 64x64 look like this:
        # [[x_1.1 ,...,x_1.4096],
        #   ...
        #  [x_10.1,...,x_10.4096]] 
        cur_batch = FACES[batch_count]
        FACES[batch_count] = np.insert(cur_batch, cur_batch.shape[0], scaled_face, axis=0)
    image_count += 1 
print('Done!')

def xavier_init(size):
    """
    initialize an array of dimension size[0] x size[1].
    the elements of the array are drawn from a normal distribution
    with mean 0 and variance 2/size[0] i.e. x ~ N(0, 2/N), N = size[0]
    """
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)

if SAVED_MODEL:
    print(' [*] Loading model...')
    with open(MODEL_FILE, 'rb') as model:
        parameters = pickle.load(model)
        Wxh, bxh, Whz_mu, bhz_mu, Whz_var, bhz_var, Wzh, bzh, Whx, bhx = parameters
    print(' [*] Done!') 
else:
    Wxh = xavier_init(size=[X_dim, h_dim])
    bxh = Variable(torch.zeros(h_dim), requires_grad=True)

    Whz_mu = xavier_init(size=[h_dim, Z_dim])
    bhz_mu = Variable(torch.zeros(Z_dim), requires_grad=True)

    Whz_var = xavier_init(size=[h_dim, Z_dim])
    bhz_var = Variable(torch.zeros(Z_dim), requires_grad=True)

    Wzh = xavier_init(size=[Z_dim, h_dim])
    bzh = Variable(torch.zeros(h_dim), requires_grad=True)

    Whx = xavier_init(size=[h_dim, X_dim])
    bhx = Variable(torch.zeros(X_dim), requires_grad=True)
# =============================== Q(z|X) ======================================1

def Q(X):
    """ENCODER. 
    find the parameter of the distribution z|x ~ N(mu_z, var_z). i.e. q(z|x)
    
    h(x) = relu(xW + b)
    mu  = hU + b_mu
    var = hV + b_var
    """
    h = nn.relu(X @ Wxh + bxh.repeat(X.size(0), 1))
    mu = h @ Whz_mu + bhz_mu.repeat(h.size(0), 1)
    var = h @ Whz_var + bhz_var.repeat(h.size(0), 1)
    return mu, var

def sample_z(mu, log_var):
    """
    sample z from a gaussian 
    Input
        mu: the mean of the distribution of z|x
        log_var (= 2*log sigma): the log of the variance of the distribution of z|x
        (input the log_var instead of the var in order to ensure that the variance is positive)
    Ouput
        z: the code representing the input
    """
    eps = Variable(torch.randn(batch_size, Z_dim))
    sigma = torch.exp(log_var / 2)
    z = mu + eps * sigma
    return z

# =============================== P(X|z) ======================================

def P(z):
    """
    DECODER.
    z ~ N(mu_z, var_z)
    x = sigmoid(hW + b)
    """
    h = nn.relu(z @ Wzh + bzh.repeat(z.size(0), 1))
    logits = h @ Whx + bhx.repeat(h.size(0), 1)
    X = nn.sigmoid(logits)
    
    return X, logits

# =============================== TRAINING ====================================

params = [Wxh, bxh, Whz_mu, bhz_mu, Whz_var, bhz_var, Wzh, bzh, Whx, bhx]
solver = optim.Adam(params, lr=lr, betas=(0.9, 0.999))
print(' [*] Training started')
epoch_count = -1
for it in tqdm(range(num_epoch)):
    epoch_count += 1 
    if epoch_count == num_batches-1:
        # reset the epoch because no more batch
        epoch_count = 0
        print(' [*] All batches have been seen by the models')
    
    X = FACES[epoch_count]
    X = Variable(torch.from_numpy(X).type(DTYPE))

    # Forward (reconstruction of the image)
    z_mu, z_logvar = Q(X)
    # z dim = (batch_size, z_dim=100)
    z = sample_z(z_mu, z_logvar)
    X_sample, _ = P(z)

    # =============================== Normalizing Flow ====================================
    u, w, b, uw, muw, u_hat, zwb, f_z, psi, psi_u = [], [], [], [], [], [], [], [], [], []
    logdet_jacobian = 0

    if NORMALIZING_FLOW == True:
        for i in range(NUM_FLOWS):
            # u is an array of size (z_dim x 1)
            u.append(xavier_init(size=[Z_dim, 1]))
            w.append(xavier_init(size=[Z_dim, 1]))
            b.append(xavier_init(size=[1, 1]).repeat(z.size(0), 1)) 
            uw.append(w[i].t() @ u[i])
            
            # muw = -1 + log(1 + exp(u*w))
            muw.append(-1 + nn.softplus(uw[i])) 
            # u_i = u_i + (muw_i + u_i*w_i) * w_i / ||w||
            norm_w = w[i].pow(2).sum()
            # u_hat dim = (100, )
            # u z_dim x batch_size
            # muw batch_size x batch_size
            # uw batch_size x batch_size
            # w z_dim x batch_size
            u_hat.append(u[i] + (muw[i] - uw[i]) * w[i] / norm_w) # (z_dim, 1)
            if i == 0:
                # zwb = z * w + b; size = (16, 1)
                zwb.append(z @ w[i] + b[i]) # (batch_size, 1)
                # f_z = z + u_i*tanh(zw + b)
                f_z.append(z +  torch.tanh(zwb[i]) @ u_hat[i].t()) # (batch_size, z_dim)
            else:
                zwb.append(f_z[i-1] @ w[i] + b[i])
                f_z.append(f_z[i-1] + torch.tanh(zwb[i]) @ u_hat[i].t())

            # tanh(x)dx = 1 - tanh(x)**2
            psi.append(w[i] @ (1 - torch.tanh(zwb[i]).t() @ torch.tanh(zwb[i]))) 
            psi_u.append(psi[i].t() @ u_hat[i])
            logdet_jacobian += torch.log(torch.abs(1 + psi_u[i]))

        logits, _ = P(f_z[-1])  # add flows thing in P
    
    # Loss
    if NORMALIZING_FLOW:
        recon_loss = nn.binary_cross_entropy(logits, X, size_average=False)
        kl_loss = 0.5 * torch.sum(torch.exp(z_logvar) + z_mu**2 - 1. - z_logvar)
        loss = recon_loss + kl_loss - logdet_jacobian
        loss = loss.view(1)
    else:
        # E[log P(X|z_k)]
        recon_loss = nn.mse_loss(X_sample, X, size_average=False) 
        # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
        kl_loss = 0.5 * torch.sum(torch.exp(z_logvar) + z_mu**2 - 1. - z_logvar)
        # loss = upper bound of E[log p(x|z)]
        loss = recon_loss + kl_loss
    # Backward
    loss.backward()

    # Update
    solver.step()

    # Housekeeping
    for p in params:
        if p.grad is not None:
            data = p.grad.data
            p.grad = Variable(data.new().resize_as_(data).zero_())

    # Print and plot every now and then
    if it % 100 == 0:
        print('-------------------------')
        print('Iter: {} Loss: {:.4}'.format(it, loss.data[0]))
        
        samples, _ = P(z)
        samples = samples.data.numpy()[:16] 

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(64, 64), cmap='Greys_r')
        
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        image_file = image_path + '{}.png'.format(str(c).zfill(3)) 
        plt.savefig(image_file, bbox_inches='tight')
        c += 1
        plt.close(fig)

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
 
with open(MODEL_FILE, 'wb') as model:
    pickle.dump(params, model) 
