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
from random import random
import random as rand

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as nn

rand.seed(8)

SAVED_MODEL = True
NORMALIZING_FLOW = False
NUM_FLOWS = 0
DTYPE = torch.FloatTensor

batch_size = 1
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

IMG_PATHS = glob.glob('./data/sample/*.jpg')
MODEL_PATH = './saved_model/' + 'model_{}_{}_{}/'.format(flow, NUM_FLOWS, Z_dim)
MODEL_FILE = MODEL_PATH + 'model'
OUT_PATH = 'out5c_{}_{}_{}/'.format(flow, NUM_FLOWS, Z_dim)

# =============================== preprocessing ======================================

def scale(image):
    """
    scale all the pixel between -1 and 1
    """
    min_value = np.min(image)
    max_value = np.max(image)
    scaled_image = (image - min_value)/(max_value - min_value)*2 - 1
    return scaled_image

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
    return X

# =============================== IMAGE SAMPLING ======================================

FACES = []
print(' [*] Loading celebrity faces...')
for i in [int(random()*2000) ,1688]:
    image_file = IMG_PATHS[i]
    face = np.asarray(Image.open(image_file).convert('L'))
    scaled_face = scale(face).ravel().reshape(1, X_dim)
    scaled_face = Variable(torch.from_numpy(scaled_face).type(DTYPE))
    FACES.append(scaled_face)
print(' [*] Done!')

X1, X2 = FACES

mu1, logvar1 = Q(X1)
z1 = sample_z(mu1, logvar1)

mu2, logvar2 = Q(X2)    
z2 = sample_z(mu2, logvar2)

x1 = P(z1).data.numpy()
x2 = P(z2).data.numpy()

# =============================== z = alpha*z1 + (1 - alpha)*z2 ======================================

fig = plt.figure(figsize=(2, 11))
gs = gridspec.GridSpec(2, 11)
gs.update(wspace=0.05, hspace=0.0)

i = 0
for alpha in range(11):
    alpha = alpha/10.
    z = alpha*z1 + (1 - alpha)*z2
    X_sample = P(z)
    X_sample = X_sample.data.numpy()

    ax = plt.subplot(gs[i])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(X_sample.reshape(64, 64), cmap='Greys_r') 
    i += 1  

for alpha in range(11):
    alpha = alpha/10.

    X_sample = alpha*x1 + (1 - alpha)*x2

    ax = plt.subplot(gs[i])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(X_sample.reshape(64, 64), cmap='Greys_r') 
    i += 1

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

image_file = OUT_PATH + '5c.png'
plt.savefig(image_file, bbox_inches='tight')
plt.show()
plt.close(fig)

# =============================== z = alpha*x1 + (1 - alpha)*x2 ======================================

fig = plt.figure(figsize=(1, 11))
gs = gridspec.GridSpec(1, 11)
gs.update(wspace=0.05, hspace=0.05)

i = 0
for alpha in range(11):
    alpha = alpha/10.

    X_sample = alpha*x1 + (1 - alpha)*x2

    ax = plt.subplot(gs[i])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(X_sample.reshape(64, 64), cmap='Greys_r') 
    i += 1  

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

image_file = OUT_PATH + '5cb.png'
plt.savefig(image_file, bbox_inches='tight')
plt.show()
plt.close(fig)

#X_sample, X_sample2 = P(z1), P(z2)


