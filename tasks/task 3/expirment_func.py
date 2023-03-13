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




### task one and two creat image from random latent vectors
def create_random_img(decoder, n=10, latent_size=4,
                      device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")):
    size = torch.eye(1, latent_size)
    # Set evaluation mode for the decoder
    decoder.eval()
    random_latent_vectors = []
    with torch.no_grad():  # No need to track the gradients
        for i in range(n):
            ax = plt.subplot(int(n / 5), 5, i + 1)
            if i == 0:
                plt.title("created random image by the decoder")
                # ax.set_title("created random image by the decoder")
            # get random vector in "normal(0,1)"
            p = torch.distributions.Normal(torch.zeros_like(size), torch.ones_like(size))
            random_latent_vec = p.rsample()

            # random_mu = torch.tensor(-2 * np.random.rand(1, latent_size) + 1, dtype=torch.float)
            # random_log_var = torch.tensor(-2 * np.random.rand(1, latent_size) + 1, dtype=torch.float)
            # random_latent_vec = Encoder.reparameterize(encoder,random_mu, random_log_var)
            img = decoder(random_latent_vec)
            random_latent_vectors.append(random_latent_vec)
            plt.imshow(img.to(device).squeeze().numpy(), cmap='gist_gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()


# The function take n image and change every digit in ther latent vector to chack what the impcat of
# every digit in the vector on the image
def latent_digit_impact(encoder, decoder,  test_dataset, n=8, latent_size=4, num_of_steps=8,
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
            encoded_data, _, _ = encoder(origin_image_list[i])
            image_vec_list.append(encoded_data)
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
                        plt.imshow(origin_image_list[img_index].to(device).squeeze().numpy(), cmap='gist_gray')
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)
                        #plt.title(f"Original image")
                        # decoder on the original latent vector
                        ax = plt.subplot(latent_size, n + 3, cordinate * (num_of_steps + 3) + 2)
                        plt.imshow(decoder(image_vec_list[img_index]).to(device).squeeze().numpy(), cmap='gist_gray')
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)
                        #plt.title(f"Original vector")
                    # add to the converted image to the plot
                    ax = plt.subplot(latent_size, n + 3, cordinate * (num_of_steps + 3) + step + 3)
                    plt.imshow(changed_image.to(device).squeeze().numpy(), cmap='gist_gray')
                    #plt.title(f"cord: {cordinate}. value: {-1 + step * (2 / num_of_steps)}")
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    # add to the the converted image to the changed list
                    changed_image_list.append(changed_image)
            plt.show()

# The function two image and convert one imag to the second by margin the two latent vector
# with different impact
def convert_img_from_latent(encoder, decoder, test_dataset, n=10, latent_size=4, num_of_steps=8,
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
            encoded_data, _, _ = encoder(origin_image_list[i])
            image_vec_list.append(encoded_data)
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
                plt.imshow(changed_image.to(device).squeeze().numpy(), cmap='gist_gray')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            plt.show()
            
def train_with_TSNE(data_path, num_of_sample = 1000):
    # load the data
    df = pd.read_csv(data_path)

    x = df.to_numpy()
    len_x, _ = x.shape
    if(num_of_sample>len_x):
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
