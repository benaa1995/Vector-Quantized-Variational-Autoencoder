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


# import seaborn as sns


def plot_ae_outputs(encoder, decoder, test_dataset, n=10,
                    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")):
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("plot_ae_outputs device = ", device)
    plt.figure(figsize=(16, 4.5))
    test = test_dataset.targets
    targets = test_dataset.targets.numpy()
    t_idx = {i: np.where(targets == i)[0][0] for i in range(n)}
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            encoded_data, _, _ = encoder(img)
            rec_img = decoder(encoded_data)
        plt.imshow(img.to(device).squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Original images')
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(rec_img.to(device).squeeze().numpy(), cmap='gist_gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Reconstructed images')
    plt.show()


def plot_ae_outputs_CIFAR10(encoder, decoder, test_dataset, n=10, classes=None,
                            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")):
    print("plot_ae_outputs_CIFAR10 device = ", device)
    plt.figure(figsize=(16, 4.5))
    test = test_dataset.targets
    targets = test_dataset.targets
    t_idx = {}
    for i in range(n):
        for j in range(len(targets)):
            if i == targets[j]:
                t_idx[i] = j
                break
    for i in range(n):

        # np_img = img.numpy()
        # plt.imshow(np.transpose(np_img, (1, 2, 0)))
        # plt.show()

        ax = plt.subplot(2, n, i + 1)
        img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            encoded_data, _, _ = encoder(img)
            rec_img = decoder(encoded_data)
        np_img = img.to(device).squeeze().numpy()
        plt.imshow(np.transpose(np_img, (1, 2, 0)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Original images')
        ax = plt.subplot(2, n, i + 1 + n)
        np_rec_img = rec_img.to(device).squeeze().numpy()
        plt.imshow(np.transpose(np_rec_img, (1, 2, 0)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n // 2:
            ax.set_title('Reconstructed images')
    plt.show()


# save the latent vector in cvs and train liniar logistic on this model
def convert_latent_to_cvs(encoder, latent_size, file_name, dataloader, device):
    req_col = {}
    X = "X"
    for i in range(latent_size):
        temp_col = X + str(i + 1)
        req_col[temp_col] = []
    req_col["Y"] = []

    # convert to data frae
    df = pd.DataFrame.from_dict(req_col)
    encoder.eval()
    with torch.no_grad():  # No need to track the gradients
        for image_batch, target_batch in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            curr_batch_size = image_batch.shape
            curr_num_of_row = curr_batch_size[0]
            # Encode data
            encoded_data, _, _ = encoder(image_batch)
            encoded_data = encoded_data.to(device).detach().numpy()
            target_batch = target_batch.to(device).detach().numpy()
            rows = np.zeros((curr_num_of_row, latent_size + 1))
            rows[:, :-1] = encoded_data
            bach_y = target_batch.reshape((-1, 1))
            rows[:, -1:] = bach_y
            # add the row to the data frame
            for row in rows:
                df.loc[len(df)] = row
        print(df)
        file_name = file_name + '.cvs'
        # save the dataframe as cvs file
        df.to_csv(file_name)


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
    print("accuracy on train = ", 100 * accuracy_train / len(y_train), "%")
    print("accuracy on test = ", 100 * accuracy_test / len(y_test), "%")
