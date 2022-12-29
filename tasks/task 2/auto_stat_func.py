# TAKEN FROM https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac#d75c
# autoencoder using cnn
# plotting library
import matplotlib.pyplot as plt
# this module is useful to work with numerical arrays
import numpy as np
import pandas as pd
import random
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.linear_model import LogisticRegression



"""
    test_function does blah blah blah.

    :param p1: describe about parameter p1
    :param p2: describe about parameter p2
    :param p3: describe about parameter p3
    :return: describe what it returns
    """ 


### task one and two creat image from random latent vectors
"""
    create_random_img get decoder and the requir number of image to create, and create
    random image by pass to the latent vector random number

    :param decoder: the decoder after the train proccess 
    :param num_of_image: the number of image to create
    :param latent_size: the latent size of the autoencoder net

    """ 
def create_random_img(decoder, num_of_image=10, latent_size=4):
    # Set evaluation mode for the decoder
    decoder.eval()
    random_latent_vectors = []
    with torch.no_grad():  # No need to track the gradients
        for i in range(num_of_image):
            ax = plt.subplot(int(num_of_image / 5), 5, i + 1)
            if i == 0:
                ax.set_title("created random image by the decoder")
            # get random vector in range (-1,1)
            random_latent_vec = torch.tensor(-2 * np.random.rand(1, latent_size) + 1, dtype=torch.float)
            img = decoder(random_latent_vec)
            random_latent_vectors.append(random_latent_vec)
            plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

