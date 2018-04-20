#!/usr/bin/env python3
"""
vanilla VAE
"""
import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as nn

DTYPE = torch.FloatTensor
IMG_PATHS = glob.glob('./data/sample/*.jpg')

batch_size = 64
num_epoch = int(len(IMG_PATHS)/batch_size)
X_dim = 64*64
h_dim = 128 # Latent dimension in question 3?
Z_dim = 100 # Reconstruction size?
c = 0 #??
lr = 0.001

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
for image_file in tqdm(IMG_PATHS):
    # convert the image to a numpy array 
    # each pixel as a value between 0 and 256
    face = np.asarray(Image.open(image_file).convert('L'))
    scaled_face = scale(face).ravel().reshape(1, 4096)
    if image_count % batch_size == 1:
        # start a new batch
        FACES.append(scaled_face)
        batch_count += 1
    else:
        # add current example to the array of the current batch.
        # a batch with batch size 10 and image size 64x64 look like this:
        # [[x_1.1 ,...,x_1.784],
        #   ...
        #  [x_10.1,...,x_10.784]] 
        cur_batch = FACES[batch_count]
        FACES[batch_count] = np.insert(cur_batch, cur_batch.shape[0], scaled_face, axis=0)
    image_count += 1 

# =============================== Q(z|X) ======================================

def xavier_init(size):
    """
    initialize an array of dimension size[0] x size[1].
    the elements of the array are drawn from a normal distribution
    with mean 0 and variance 2/size[0] i.e. x ~ N(0, 2/N), N = size[0]
    """
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)

Wxh = xavier_init(size=[X_dim, h_dim])
bxh = Variable(torch.zeros(h_dim), requires_grad=True)

Whz_mu = xavier_init(size=[h_dim, Z_dim])
bhz_mu = Variable(torch.zeros(Z_dim), requires_grad=True)

Whz_var = xavier_init(size=[h_dim, Z_dim])
bhz_var = Variable(torch.zeros(Z_dim), requires_grad=True)

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

Wzh = xavier_init(size=[Z_dim, h_dim])
bzh = Variable(torch.zeros(h_dim), requires_grad=True)

Whx = xavier_init(size=[h_dim, X_dim])
bhx = Variable(torch.zeros(X_dim), requires_grad=True)

def P(z):
    """
    DECODER.
    z ~ N(mu_z, var_z)
    x = sigmoid(hW + b)
    """
    h = nn.relu(z @ Wzh + bzh.repeat(z.size(0), 1))
    X = nn.sigmoid(h @ Whx + bhx.repeat(h.size(0), 1))
    return X

# =============================== TRAINING ====================================

params = [Wxh, bxh, Whz_mu, bhz_mu, Whz_var, bhz_var, Wzh, bzh, Whx, bhx]

solver = optim.Adam(params, lr=lr)

for it in tqdm(range(num_epoch)):
    X = FACES[it]
    X = Variable(torch.from_numpy(X).type(DTYPE))

    # Forward (reconstruction of the image)
    z_mu, z_var = Q(X)
    z = sample_z(z_mu, z_var)
    X_sample = P(z)

    # Loss
    recon_loss = nn.binary_cross_entropy(X_sample, X, size_average=False)
    kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var)
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
        print('Iter-{}; Loss: {:.4}'.format(it, loss.data[0]))

        samples = P(z).data.numpy()[:16]

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

        if not os.path.exists('out/'):
            os.makedirs('out/')

        plt.savefig('out/{}.png'.format(str(c).zfill(3)), bbox_inches='tight')
        c += 1
        plt.close(fig)
