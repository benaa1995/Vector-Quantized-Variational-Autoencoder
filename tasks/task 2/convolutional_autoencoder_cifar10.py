# TAKEN FROM https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac#d75c
# autoencoder using cnn
# plotting library
import argparse
import os

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

import ae_model

# writer for tnsorboard
writer = SummaryWriter(f'runs/MNIST/autoencoder_tensorboard')


def load_data(batch_size, resize=128):
    data_dir = 'dataset'

    train_dataset = torchvision.datasets.CIFAR10(data_dir, train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10(data_dir, train=False, download=True)

    train_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
    ])

    train_dataset.transform = train_transform
    test_dataset.transform = test_transform

    # batch_size = 256
    # ------------------------------------------------------------------------
    # todo remove !!!!!
    m = len(train_dataset)
    smaller_train_data, val_data = random_split(train_dataset, [int(m * 0.01), int(m - m * 0.01)])
    m = len(test_dataset)
    smaller_test_data, val_data = random_split(test_dataset, [int(m * 0.01), int(m - m * 0.01)])
    train_loader = torch.utils.data.DataLoader(smaller_train_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(smaller_test_data, batch_size=batch_size, shuffle=True)
    # ---------------------------------------------------------------------------------------------
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, train_dataset, test_dataset


def load_save_checkpoint(epoch, model=ae_model.AE(), save=True):
    if save:
        torch.save(model.state_dict(), f"checkpoint/ae{str(epoch + 1).zfill(3)}.pt")
    else:
        ckpt = torch.load(os.path.join('checkpoint', f"ae{str(epoch + 1).zfill(3)}.pt"))
        model.load_state_dict(ckpt)
        # model = model.to(device)
    return model


### Training function
def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in dataloader:  # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        # Encode data
        encoded_data = encoder(image_batch)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


### Testing function
def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad():  # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            # Encode data
            encoded_data = encoder(image_batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data


def plot_ae_outputs(model, test_dataset, test_loader, epoch, num_of_img=10):
    model.eval()
    with torch.no_grad():
        out = []
        for img, tar in test_loader:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            img = img.to(device)
            sample = img[:num_of_img]
            out ,_= model(sample)
            torchvision.utils.save_image(
                torch.cat([sample, out], 0),
                f"sample_ae/{str(epoch + 1).zfill(5)}_{str(epoch+1).zfill(5)}.png",
                nrow=num_of_img,
                normalize=True,
                value_range=(-1, 1),
            )
            break





# def plot_ae_outputs(encoder, decoder, n=10,
#                     device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")):
#     # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     print("plot_ae_outputs device = ", device)
#     plt.figure(figsize=(16, 4.5))
#     targets = test_dataset.targets.numpy()
#     t_idx = {i: np.where(targets == i)[0][0] for i in range(n)}
#     for i in range(n):
#         ax = plt.subplot(2, n, i + 1)
#         img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
#         encoder.eval()
#         decoder.eval()
#         with torch.no_grad():
#             rec_img = decoder(encoder(img))
#         plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#         if i == n // 2:
#             ax.set_title('Original images')
#         ax = plt.subplot(2, n, i + 1 + n)
#         plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#         if i == n // 2:
#             ax.set_title('Reconstructed images')
#     plt.show()


### task one and two creat image from random latent vectors
def create_random_img(decoder, n=10, latent_size=4):
    # Set evaluation mode for the decoder
    decoder.eval()
    random_latent_vectors = []
    with torch.no_grad():  # No need to track the gradients
        for i in range(n):
            ax = plt.subplot(int(n / 5), 5, i + 1)
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
def latent_digit_impact(encoder, decoder, n=8, latent_size=4, num_of_steps=8,
                        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")):
    origin_image_list = []
    image_vec_list = []
    changed_image_list = []
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("latent_digit_impact device = ", device)
    encoder.eval()
    decoder.eval()
    # plo image size
    plt.figure(figsize=(16, 4.5))
    # get the image
    targets = test_dataset.targets.numpy()
    t_idx = {i: np.where(targets == i)[0][0] for i in range(n)}

    with torch.no_grad():
        # save the original image and the latent vector
        for i in range(n):
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
                        ax = plt.subplot(latent_size, n + 3, cordinate * (num_of_steps + 3) + 1)
                        plt.imshow(origin_image_list[img_index].cpu().squeeze().numpy(), cmap='gist_gray')
                        plt.title(f"Original image")
                        # decoder on the original latent vector
                        ax = plt.subplot(latent_size, n + 3, cordinate * (num_of_steps + 3) + 2)
                        plt.imshow(decoder(image_vec_list[img_index]).cpu().squeeze().numpy(), cmap='gist_gray')
                        plt.title(f"Original vector")
                    # add to the converted image to the plot
                    ax = plt.subplot(latent_size, n + 3, cordinate * (num_of_steps + 3) + step + 3)
                    plt.imshow(changed_image.cpu().squeeze().numpy(), cmap='gist_gray')
                    plt.title(f"cord: {cordinate}. value: {-1 + step * (2 / num_of_steps)}")
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    # add to the the converted image to the changed list
                    changed_image_list.append(changed_image)
            plt.show()


# The function two image and convert one imag to the second by margin the two latent vector
# with different impact
def convert_img_from_latent(encoder, decoder, n=10, latent_size=4, num_of_steps=8,
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
    t_idx = {i: np.where(targets == i)[0][0] for i in range(n)}
    with torch.no_grad():
        # save the requested original image and the latent vector
        for i in range(n):
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
                ax = plt.subplot(1, n + 1, step + 1)
                plt.imshow(changed_image.cpu().squeeze().numpy(), cmap='gist_gray')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            plt.show()


# save the latent vector in cvs and train liniar logistic on this model
# save the latent vector in cvs and train liniar logistic on this model
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
        for image_batch, target_batch in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            curr_batch_size = image_batch.shape
            curr_num_of_row = curr_batch_size[0]
            # Encode data
            encoded_data = encoder(image_batch)
            bach_y = y[0:curr_num_of_row]
            y = y[curr_num_of_row:]
            encoded_data = encoded_data.cpu().detach().numpy()
            target_batch = target_batch.cpu().detach().numpy()
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


def train_with_TSNE(data_path, num_of_sample=1000):
    # load the data
    df = pd.read_csv(data_path)

    x = df.to_numpy()
    len_x, _ = x.shape
    if (num_of_sample > len_x):
        num_of_sample = len_x
    x = x[:num_of_sample, 1:-1]
    y = df['Y'].values.astype(int)
    y = y[:num_of_sample]

    # sklrn linear regration
    tsne = TSNE(2)
    tsne_result = tsne.fit_transform(x)
    tsne_result.shape

    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:, 0], 'tsne_2': tsne_result[:, 1], 'label': y})
    print(tsne_result_df)
    print(np.unique(y))
    fig, ax = plt.subplots(1)
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax, s=30,
                    palette=['darkgreen', 'red', 'black', 'orange', 'blue', 'cyan', 'fuchsia', 'lime', 'dimgray',
                             'brown'])

    lim = (tsne_result.min() - 5, tsne_result.max() + 5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.show()


def train_model(lr=0.001, latent_size=4, num_epochs=30):
    train_loader, test_loader, train_dataset, test_dataset = load_data(batch_size=128)

    ### Define the loss function
    loss_fn = torch.nn.MSELoss()

    ### Define an optimizer (both for the encoder and the decoder!)
    # lr= 0.001
    print("lr = ", lr)
    ### Set the random seed for reproducible results
    torch.manual_seed(0)

    ### Initialize the two networks
    # latent_size = 4
    print("latent_size = ", latent_size)
    # model = Autoencoder(encoded_space_dim=encoded_space_dim)
    model = ae_model.AE()
    params_to_optimize = [
        {'params': model.enc.parameters()},
        {'params': model.dec.parameters()}
    ]

    optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    # Move both the encoder and the decoder to the selected device
    model.to(device)

    # tensorBoard hyper-parameters
    global_step = 1
    # num_epochs = 30
    diz_loss = {'train_loss': [], 'val_loss': []}
    for epoch in range(num_epochs):
        # train_loss = train_epoch(encoder, decoder, device,
        #                          train_loader, loss_fn, optim)
        # val_loss = test_epoch(encoder, decoder, device, test_loader, loss_fn)

        train_loss = train_epoch(model.enc, model.dec, device,
                                 train_loader, loss_fn, optim)
        val_loss = test_epoch(model.enc, model.dec, device, test_loader, loss_fn)
        print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs, train_loss, val_loss))
        diz_loss['train_loss'].append(train_loss)
        diz_loss['val_loss'].append(val_loss)
        temp_name = "Loss, latent size_" + str(latent_size)
        writer.add_scalars(temp_name, {'Traning loss': train_loss, 'Test loss': val_loss}, global_step)
        writer.flush()
        global_step += 1

        # function :
        plot_ae_outputs(model, test_dataset, test_loader, epoch, num_of_img=10)
        '''if epoch%10 == 0:
        plot_ae_outputs(encoder,decoder,n=10,device)'''
    writer.add_scalar("latent size vs minimun loss", min(diz_loss["val_loss"]), latent_size)
    writer.flush()
    # plot_ae_outputs(encoder, decoder, n=10, device=device)
    # create_img(decoder,n=4)
    # create_random_img(decoder, 10, latent_size)
    # test_file_name = 'test_lat_size_' + str(latent_size)
    # convert_latent_to_cvs(encoder, latent_size, test_file_name, test_loader, test_dataset, device)
    # train_file_name = 'train_lat_size_' + str(latent_size)
    # convert_latent_to_cvs(encoder, latent_size, train_file_name, train_loader, train_dataset, device)
    # latent_digit_impact(encoder, decoder, n=8, latent_size=4, num_of_steps=8, device=device)
    # convert_img_from_latent(encoder, decoder, n=10, latent_size=latent_size, num_of_steps=8, device=device)
    return diz_loss["val_loss"], diz_loss['train_loss']


def latent_size_stat():
    NUM_OF_EPOCH = 100
    STEP_SIZE = 0.25
    NUM_OF_STEPS = 8
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
    # print(df)
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


# train_model(latent_size = 16,num_epochs=100)
# # latent_size_stat()
# #train_model(num_epochs=5)
# train_with_TSNE('test_lat_size_2.cvs', 'train_lat_size_2.cvs')
# train_with_TSNE('test_lat_size_4.cvs', 'train_lat_size_4.cvs')
# train_with_TSNE('test_lat_size_8.cvs', 'train_lat_size_8.cvs')
# train_with_TSNE('test_lat_size_16.cvs', 'train_lat_size_16.cvs')
# train_with_TSNE('test_lat_size_32.cvs', 'train_lat_size_32.cvs')
# train_with_TSNE('test_lat_size_64.cvs', 'train_lat_size_64.cvs')
# train_with_TSNE('test_lat_size_128.cvs', 'train_lat_size_128.cvs')


# def main(args):
#
#     train_loader, test_loader, train_dataset, test_dataset = load_data(batch_size=args.batch_size)
#     model = AE().to(device)
#
#     pass
def get_random_z(min_val=-1, max_val=1, channel=2, row=4, col=4):
    range_size = max_val - min_val
    rand = range_size * torch.rand(channel, row, col) + min_val
    return rand


if __name__ == '__main__':
    rand = get_random_z()
    # print(rand)
    # print(rand.size())

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    # parser.add_argument("path", type=str)

    args = parser.parse_args()

    train_model()
