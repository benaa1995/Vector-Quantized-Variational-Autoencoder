import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.linear_model import LogisticRegression
from torch.utils.data import random_split

from torchvision import transforms

img_idx_MNIST_adapter = lambda tar, n: {i: np.where(tar == i)[0][0] for i in range(n)}


def img_idx_CIFAR10_adapter(targets, n):
    t_idx = {}
    for i in range(n):
        for j in range(len(targets)):
            if i == targets[j]:
                t_idx[i] = j
                break
    return t_idx


plot_MNIST_adapter = lambda img: img

plot_CIFAR10_adapter = lambda img: np.transpose(img, (1, 2, 0))

targets_MNIST_adapter = lambda test_dataset: test_dataset.targets.numpy()

targets_CIFAR10_adapter = lambda test_dataset: test_dataset.targets



def plot_ae_outputs(model, test_dataset, test_loader, epoch, dir_name="plot_ae_outputs", num_of_img=10):
    model.eval()
    with torch.no_grad():
        for img, tar in test_loader:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            img = img.to(device)
            sample = img[:num_of_img]
            out, _ = model(sample)
            torchvision.utils.save_image(
                torch.cat([sample, out], 0),
                f"sample_ae{os.sep}{dir_name}{os.sep}{str(epoch + 1).zfill(5)}_{str(epoch+1).zfill(5)}.png",
                nrow=num_of_img,
                normalize=True,
                value_range=(-1, 1),
            )
            break


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


def load_data(batch_size, resize=128, data_dir = 'dataset'):
    train_dataset = torchvision.datasets.CIFAR10(data_dir, train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10(data_dir, train=False, download=True)

    train_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    train_dataset.transform = train_transform
    test_dataset.transform = test_transform


    # # ------------------------------------------------------------------------
    # # todo_d remove !!!!!
    # m = len(train_dataset)
    # smaller_train_data, val_data = random_split(train_dataset, [int(m * 0.001), int(m - m * 0.001)])
    # m = len(test_dataset)
    # smaller_test_data, val_data = random_split(test_dataset, [int(m * 0.001), int(m - m * 0.001)])
    # train_loader = torch.utils.data.DataLoader(smaller_train_data, batch_size=batch_size)
    # test_loader = torch.utils.data.DataLoader(smaller_test_data, batch_size=batch_size, shuffle=True)
    # # ---------------------------------------------------------------------------------------------
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, train_dataset, test_dataset


def load_save_checkpoint(model, epoch=-1, save=True, load_path="", save_dir=""):
    if save:
        torch.save(model.state_dict(), f"checkpoint/{save_dir}ae{str(epoch + 1).zfill(3)}.pt")
    else:
        ckpt = torch.load(os.path.join('checkpoint', load_path))
        model.load_state_dict(ckpt)
        # model = model.to(device)
    return model