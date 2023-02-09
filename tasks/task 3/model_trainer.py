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
from sklearn.manifold import TSNE
import seaborn as sns


import convolutional_variational_autoencoder as vae

def latent_size_stat():
    NUM_OF_EPOCH = 4
    STEP_SIZE = 0.25
    NUM_OF_STEPS = 8
    MAX_POW = 2

    ###compare the loss between the train vs test to avoid over fitting
    # creat the cvs colons
    req_col = {'Latent vector size': []}
    epoch = "Epoch"
    for i in range(NUM_OF_EPOCH):
        temp_col = epoch + " " + str(i + 1)
        req_col[temp_col] = []
    req_col["Best epoch"] = []
    req_col["Best loss"] = []
    # convert to data frae
    df = pd.DataFrame.from_dict(req_col)
    # print(df)
    # get the train and test loss for every "2 pow" latent size
    for pow in range(MAX_POW):
        lat_size = int(np.power(2, pow + 1))
        test_loss, train_loss = vae.train_model(latent_size=lat_size, num_epochs=NUM_OF_EPOCH)

        # add the test loss to data frame
        row = test_loss
        print(row)
        print(row[0])
        for i in range(len(row)):
            row[i] = row[i].item()
        row.insert(0, int(lat_size))
        for i in range(2):
            row.append(None)
        # print("new row = ",row)
        df.loc[len(df)] = row
        print(df)

    # find the nim test loss and his epooch and add them to the dataframe
    for i in df.index:
        temp = df.iloc[[i], 1: NUM_OF_EPOCH + 1].values
        print(temp)
        index = np.argmin(temp[0])
        df.iloc[[i], [NUM_OF_EPOCH + 1]] = index + 1
        df.iloc[[i], [NUM_OF_EPOCH + 2]] = temp[0][index]
        print("epoch = ", index + 1, ", min = ", temp[0][index])
    print(df)
    # save the dataframe as cvs file
    df.to_csv('statistic of latent size and epoch.csv')


latent_size_stat()

