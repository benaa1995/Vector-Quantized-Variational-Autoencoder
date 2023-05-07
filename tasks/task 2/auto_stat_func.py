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
    latent_digit_impact does change every digit from latent vector in range of [-1,1] to check the image influetion from encoder.

    :param encoder: the encoder after the train process 
    :param decoder: the decoder after the train process 
    :param test_dataset: the test dataset 
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


 # The function two image and convert one image to the second by margin the two latent vector
 # with different impact
"""
    convert_img_from_latent  Takes two images convert them to latent vector by using the decoder.
    and convert them from the first image to the second by taking the latent vector of each image
    and for each step creating a new vector composed of "p" percent of the first image and "1-p" precent 
    of the second image. And then pass the vector in the encoder and print the image
    :param encoder: the encoder after the train proccess 
    :param decoder: the decoder after the train proccess 
    :param test_dataset: the test dataset
    :param num_of_image: the number of image to create
    :param latent_size: the latent size of the autoencoder 
    :param num_of_steps: the number of steps of every  digit in range of [-1,1]
    :param device: the torch device 
    :return: none
    """ 
def convert_img_from_latent(encoder, decoder,test_dataset,num_of_image=10, latent_size=4, num_of_steps=8,
                            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")):
    origin_image_list = []
    image_vec_list = []
    print("convert_img_from_latent device = ", device)
    encoder.eval()
    decoder.eval()
    # plo image size
    plt.figure(figsize=(16, 4.5))
    # get the image
    targets = test_dataset.targets.numpy()
    t_idx = {i: np.where(targets == i%10)[0][0] for i in range(num_of_image)}
    with torch.no_grad():
        # save the requested original image and the latent vector
        for i in range(num_of_image):
            origin_image_list.append(test_dataset[t_idx[i]][0].unsqueeze(0).to(device))
            image_vec_list.append(encoder(origin_image_list[i]))
        # create the converted image
        for img_index in range(len(image_vec_list) // 2):
            temp_first_vec = image_vec_list[img_index].detach().clone()
            temp_second_vec = image_vec_list[-img_index - 1].detach().clone()
            # ax = plt.subplot(latent_size, n + 1, 12)
            # plt.imshow(origin_image_list[img_index].cpu().squeeze().numpy(), cmap='gist_gray')
            # ax.get_xaxis().set_visible(False)
            # ax.get_yaxis().set_visible(False)
            #
            # ax = plt.subplot(latent_size, n + 1, 13)
            # plt.imshow(origin_image_list[-img_index - 1].cpu().squeeze().numpy(), cmap='gist_gray')
            # ax.get_xaxis().set_visible(False)
            # ax.get_yaxis().set_visible(False)

            for step in range(num_of_steps + 1):
                t = (1 / num_of_steps) * step
                # combine the two image with impact "t" on the first ant "t-1" on the second
                temp_vec = temp_first_vec * (1 - t) + temp_second_vec * (t)

                # creat the change image by pass the vector throw the decoder
                changed_image = decoder(temp_vec)
                # add to the converted image to the plot
                ax = plt.subplot(1, num_of_image + 1, step + 1)
                plt.imshow(changed_image.cpu().squeeze().numpy(), cmap='gist_gray')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            plt.show()           



# save the latent vector in cvs and train liniar logistic on this model
"""
    convert_latent_to_cvs convert the "X" of the dataset to latent vector usin the given decoder. 
    then save all the latent vector as "X" and tag them all like as same as the original tag
    :param encoder: the encoder after the train proccess 
    :param latent_size: the latent size of the autoencoder 
    :param file_name:
    :param dataloader:
    :param test_dataset: the test dataset
    :param device: the torch device 
    :return: none
    """ 
def convert_latent_to_cvs(encoder, latent_size, file_name, dataloader, dataset, device):
    req_col = {}
    X = "X"
    for i in range(latent_size):
        temp_col = X + str(i + 1)
        req_col[temp_col] = []
    req_col["Y"] = []

    # convert to data frae
    df = pd.DataFrame.from_dict(req_col)
    print(df)
    encoder.eval()
    with torch.no_grad():  # No need to track the gradients
        # get the label
        y = dataset.targets
        y = y.cpu().detach().numpy()
        print(len(y))
        print(dataloader)
        print(len(dataloader))
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            curr_batch_size = image_batch.shape
            curr_num_of_row = curr_batch_size[0]
            # Encode data
            encoded_data = encoder(image_batch)
            bach_y = y[0:curr_num_of_row]
            y = y[curr_num_of_row:]
            encoded_data = encoded_data.cpu().detach().numpy()
            rows = np.zeros((curr_num_of_row, latent_size + 1))
            rows[:, :-1] = encoded_data
            bach_y = bach_y.reshape((-1, 1))
            rows[:, -1:] = bach_y
            # add the row to the data frame
            for row in rows:
                df.loc[len(df)] = row
        print(df)
        file_name = file_name + '.cvs'
        # save the dataframe as cvs file
        df.to_csv(file_name)

"""
    train_with_log_reg train lgistic regration on the "train_path" and print the accuracy on the train dataset
    and the test dataset
    :param test_path: the path of the test dataset, the data The should be in cvs format
    :param train_path: the path of the train dataset, the data The should be in cvs format
    :return: none
    """ 
def train_with_log_reg(test_path, train_path):
    # load the data
    # df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_train = pd.read_csv(train_path)
    x_test = df_test.to_numpy()
    x_test = x_test[:, 1:-1]
    y_test = df_test['Y'].values
    x_train = df_train.to_numpy()
    x_train = x_train[:, 1:-1]
    y_train = df_train['Y'].values

    # sklrn linear regration
    # two calasses
    clf_1 = LogisticRegression(max_iter=1000)
    clf_1.fit(x_train, y_train)
    pred_y_train = clf_1.predict(x_train)
    pred_y_test = clf_1.predict(x_test)
    accuracy_test = 0
    accuracy_train = 0
    for i in range(len(y_train)):
        if pred_y_train[i] == y_train[i]:
            accuracy_train += 1
    for i in range(len(y_test)):
        if pred_y_test[i] == y_test[i]:
            accuracy_test += 1
    print("accuracy on train = ", accuracy_train, "/", len(y_train))
    print("accuracy on test = ", accuracy_test, "/", len(y_test))



"""
    latent_size_stat call the train function on the model, for every power of 2 between 0 to "MAX_POW" latent size.
    and for every "2 power" save the test loss of every epoch between 1 to  NUM_OF_EPOCH in cvs file.
    :param train_model: the model to check statistic on
    :param target_file: the target file path
    :param num_of_epoch: the number of epoch
    :param max_pow: the maximum power of 2 latent size to check the model on 

    :return: cvs file of the statistic
    """ 
def latent_size_stat(train_model, target_file='statistic of latent size and epoch.csv',
             num_of_epoch=100, max_pow=7):
    NUM_OF_EPOCH = 100
    MAX_POW = 7

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
    # get the train and test loss for every "2 pow" latent size
    for pow in range(MAX_POW):
        lat_size = int(np.power(2, pow + 1))
        test_loss, train_loss = train_model(latent_size=lat_size, num_epochs=NUM_OF_EPOCH)

        # add the test loss to data frame
        row = test_loss
        for i in range(len(row)):
            row[i] = row[i].item()
        row.insert(0, int(lat_size))
        for i in range(2):
            row.append(None)
        df.loc[len(df)] = row
        print(df)

    # find the minimum test loss and his epooch and add them to the dataframe
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
