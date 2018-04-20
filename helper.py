import os
import matplotlib.pyplot as plt
from scipy.misc import imresize 
from tqdm import tqdm

# root path depends on your computer
root = 'data/CelebFaces/'
save_root = 'data/sample/'
resize_size = 64

if not os.path.isdir(save_root):
    os.mkdir(save_root)
if not os.path.isdir(save_root):
    os.mkdir(save_root)
img_list = os.listdir(root)

one_percent = len(img_list) // 10

for i in tqdm(range(one_percent)):
    img = plt.imread(root + img_list[i])
    img = imresize(img, (resize_size, resize_size))
    plt.imsave(fname=save_root + img_list[i], arr=img)
