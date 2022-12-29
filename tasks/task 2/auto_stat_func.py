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
    :param latent_size: the latent size of the autoencoder 
    :return: none
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

# The function take n image and change every digit in ther latent vector to chack what the impcat of
# every digit in the vector on the image

"""
    test_function does change every digit from latent vector in range of [-1,1] to check the image influetion from encoder.

    :param encoder: the encoder after the train proccess 
    :param decoder: the decoder after the train proccess 
    :param decoder: the test dataset 
    :param num_of_image: the number of image to create
    :param latent_size: the latent size of the autoencoder 
    :param num_of_steps: the number of steps of every  digit in range of [-1,1]
    :param device: the torch device 
    :return: none
    """ 
def latent_digit_impact(encoder, decoder,test_dataset, num_of_image=8, latent_size=4, num_of_steps=8,

                        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")):
    origin_image_list = []
    image_vec_list = []
    changed_image_list = []
    encoder.eval()
    decoder.eval()
    # plo image size
    plt.figure(figsize=(16, 4.5))
    # get the image
    targets = test_dataset.targets.numpy()
    t_idx = {i: np.where(targets == i%10)[0][0] for i in range(num_of_image)}

    with torch.no_grad():
        # save the original image and the latent vector
        for i in range(num_of_image):
            # ax = plt.subplot(latent_size,n+3,i+1)
            origin_image_list.append(test_dataset[t_idx[i]][0].unsqueeze(0).to(device))
            image_vec_list.append(encoder(origin_image_list[i]))
        # create the converted image
        for img_index in range(len(image_vec_list)):
            for cordinate in range(latent_size):
                temp_vec = image_vec_list[img_index].detach().clone()
                for step in range(num_of_steps + 1):
                    # add the offset to the curent cordinate in the original vector
                    temp_vec[0][cordinate] = -1 + step * (2 / num_of_steps)
                    changed_image = decoder(temp_vec)
                    # plot the decoder on the original vector
                    if (step == 0):
                        # the original image
                        ax = plt.subplot(latent_size, num_of_image + 3, cordinate * (num_of_steps + 3) + 1)
                        plt.imshow(origin_image_list[img_index].cpu().squeeze().numpy(), cmap='gist_gray')
                        plt.title(f"Original image")
                        # decoder on the original latent vector
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)
                        ax = plt.subplot(latent_size, num_of_image + 3, cordinate * (num_of_steps + 3) + 2)
                        plt.imshow(decoder(image_vec_list[img_index]).cpu().squeeze().numpy(), cmap='gist_gray')
                        plt.title(f"Original vector")
                    # add to the converted image to the plot
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)
                    ax = plt.subplot(latent_size, num_of_image + 3, cordinate * (num_of_steps + 3) + step + 3)
                    plt.imshow(changed_image.cpu().squeeze().numpy(), cmap='gist_gray')
                    plt.title(f"cord: {cordinate}. value: {-1 + step * (2 / num_of_steps)}")
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    # add to the the converted image to the changed list
                    changed_image_list.append(changed_image)
            plt.show()