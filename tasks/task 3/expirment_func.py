import os
from itertools import product

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

import halper_func as hf


### task one and two creat image from random latent vectors
# def create_random_img(decoder, device, plot_adapter, cmap=None, n=10, latent_size=4):
#     size = torch.eye(1, latent_size)
#     # Set evaluation mode for the decoder
#     decoder.eval()
#     random_latent_vectors = []
#     with torch.no_grad():  # No need to track the gradients
#         for i in range(n):
#             ax = plt.subplot(int(n / 5), 5, i + 1)
#             if i == 0:
#                 plt.title("created random image by the decoder")
#
#             # get random vector in "normal(0,1)" distribution
#             p = torch.distributions.Normal(torch.zeros_like(size), torch.ones_like(size))
#             random_latent_vec = p.rsample()
#             random_latent_vec.to(device)
#             img = decoder(random_latent_vec)
#             random_latent_vectors.append(random_latent_vec)
#             np_img = img.cpu().squeeze().numpy()
#             print("img.cpu().squeeze().numpy()")
#             plt.imshow(plot_adapter(np_img), cmap=cmap)
#             ax.get_xaxis().set_visible(False)
#             ax.get_yaxis().set_visible(False)
#         plt.show()

def create_random_img(decoder, device, dir_name="random_Z_1",epoch=-1, num_of_img=10, latent_size=(2, 4, 4)):

    # Set evaluation mode for the decoder
    decoder.eval()
    with torch.no_grad():  # No need to track the gradients
        sample_size = (num_of_img, ) + latent_size
        size = torch.zeros(sample_size)
        p = torch.distributions.Normal(torch.zeros_like(size), torch.ones_like(size))
        random_latent_vec = p.rsample()
        random_latent_vec = random_latent_vec.to(device)

        out = decoder(random_latent_vec)
        torchvision.utils.save_image(
            out,
            f"experiment_vae{os.sep}{dir_name}{os.sep}{str(epoch + 1).zfill(5)}.png",
            nrow=num_of_img,
            normalize=True,
            value_range=(-1, 1),
        )


# # The function take n image and change every digit in ther latent vector to chack what the impcat of
# # every digit in the vector on the image
# def latent_digit_impact(encoder, decoder, device, test_dataset, targets_adapter, img_idx_adapter, plot_adapter,
#                         cmap=None, n=8, latent_size=4, num_of_steps=8):
#     origin_image_list = []
#     image_vec_list = []
#     changed_image_list = []
#
#     encoder.eval()
#     decoder.eval()
#     # plo image size
#     plt.figure(figsize=(16, 4.5))
#     # get the image
#     targets = targets_adapter(test_dataset)
#     t_idx = img_idx_adapter(targets, n)
#
#     with torch.no_grad():
#         # save the original image and the latent vector
#         for i in range(n):
#             # ax = plt.subplot(latent_size,n+3,i+1)
#             origin_image_list.append(test_dataset[t_idx[i]][0].unsqueeze(0).to(device))
#             encoded_data, _, _ = encoder(origin_image_list[i])
#             image_vec_list.append(encoded_data)
#         # create the converted image
#         for img_index in range(len(image_vec_list)):
#             for cordinate in range(latent_size):
#                 temp_vec = image_vec_list[img_index].detach().clone()
#                 for step in range(num_of_steps + 1):
#                     # add the offset to the curent cordinate in the original vector
#                     temp_vec[0][cordinate] = -1 + step * (2 / num_of_steps)
#                     changed_image = decoder(temp_vec)
#                     # plot the decoder on the original vector
#                     if (step == 0):
#                         # the original image
#                         ax = plt.subplot(latent_size, n + 3, cordinate * (num_of_steps + 3) + 1)
#                         np_curr_origin_image = origin_image_list[img_index].to(device).squeeze().numpy()
#                         plt.imshow(plot_adapter(np_curr_origin_image), cmap=cmap)
#                         ax.get_xaxis().set_visible(False)
#                         ax.get_yaxis().set_visible(False)
#                         # plt.title(f"Original image")
#                         # decoder on the original latent vector
#                         ax = plt.subplot(latent_size, n + 3, cordinate * (num_of_steps + 3) + 2)
#                         np_curr_rec_image = decoder(image_vec_list[img_index]).to(device).squeeze().numpy()
#                         plt.imshow(plot_adapter(np_curr_rec_image), cmap=cmap)
#                         ax.get_xaxis().set_visible(False)
#                         ax.get_yaxis().set_visible(False)
#                         # plt.title(f"Original vector")
#                     # add to the converted image to the plot
#                     ax = plt.subplot(latent_size, n + 3, cordinate * (num_of_steps + 3) + step + 3)
#                     np_changed_image = changed_image.to(device).squeeze().numpy()
#                     plt.imshow(plot_adapter(np_changed_image), cmap=cmap)
#                     # plt.title(f"cord: {cordinate}. value: {-1 + step * (2 / num_of_steps)}")
#                     ax.get_xaxis().set_visible(False)
#                     ax.get_yaxis().set_visible(False)
#                     # add to the converted image to the changed list
#                     changed_image_list.append(changed_image)
#             plt.show()



# The function take n image and change every digit in ther latent vector to chack what the impcat of
# every digit in the vector on the image
def latent_digit_impact(model, device, test_loader, label=0, max_pix_in_file=-1, dir_name="latent_digit_impact", epoch=-1,
                        num_of_steps=8):
    # create dir for the current epoch
    dir_path = f"experiment_vae{os.sep}{dir_name}{os.sep}epoch_{str(epoch + 1)}"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    max_img_in_file = max_pix_in_file * (num_of_steps + 2)
    start_pix = 0
    end_pix = 0

    model.eval()
    with torch.no_grad():
        tar_image = None
        changed_image_list = []
        # find image by target
        for batch, tar in test_loader:
            for  i, img in enumerate(batch):
                if label == int(tar[i]):
                    tar_image = img
                    break
            if tar_image is not None:
                break
        # conver the img from 3D to 4D
        tar_image = tar_image.unsqueeze(0)
        # save the "Z" of the target image
        tar_image = tar_image.to(device)
        tar_Z, _, _ = model.enc(tar_image)
        # convert every coordinate of the "Z" and save the reconstruct X
        for ch, row, col in product(range(tar_Z.size(1)),range(tar_Z.size(2)),range(tar_Z.size(3))):
            # append the original image
            changed_image_list.append(tar_image)
            end_pix += 1
            # append the reconstruct image
            changed_image_list.append(model.dec(tar_Z))

            temp_vec = tar_Z.detach().clone()

            for step in range(num_of_steps + 1):
                # add the offset to the curent cordinate in the original vector
                temp_vec[0][ch][row][col] = -1 + step * (2 / num_of_steps)
                changed_image = model.dec(temp_vec)
                changed_image_list.append(changed_image)
            if len(changed_image_list) >= max_img_in_file - 1:
                torchvision.utils.save_image(
                    torch.cat(changed_image_list, 0),
                    os.path.join(dir_path, f"pixels_{str(start_pix)}-{str(end_pix-1)}.png"),
                    nrow=num_of_steps+3,
                    normalize=True,
                    value_range=(-1, 1),
                )
                start_pix = end_pix
                changed_image_list = []


# # The function two image and convert one imag to the second by margin the two latent vector
# # with different impact
# def convert_img_from_latent(encoder, decoder, device, test_dataset, targets_adapter, img_idx_adapter, plot_adapter,
#                             cmap=None, n=10, latent_size=4, num_of_steps=8):
#     origin_image_list = []
#     image_vec_list = []
#     encoder.eval()
#     decoder.eval()
#     # plo image size
#     plt.figure(figsize=(16, 4.5))
#     # get the image
#     targets = targets_adapter(test_dataset)
#     t_idx = img_idx_adapter(targets, n)
#     with torch.no_grad():
#         # save the requested original image and the latent vector
#         for i in range(n):
#             origin_image_list.append(test_dataset[t_idx[i]][0].unsqueeze(0).to(device))
#             encoded_data, _, _ = encoder(origin_image_list[i])
#             image_vec_list.append(encoded_data)
#         # create the converted image
#         for img_index in range(len(image_vec_list) // 2):
#             temp_first_vec = image_vec_list[img_index].detach().clone()
#             temp_second_vec = image_vec_list[-img_index - 1].detach().clone()
#
#             for step in range(num_of_steps + 1):
#                 t = (1 / num_of_steps) * step
#                 # combine the two image with impact "t" on the first ant "t-1" on the second
#                 temp_vec = temp_first_vec * (1 - t) + temp_second_vec * (t)
#                 # creat the change image by pass the vector throw the decoder
#                 changed_image = decoder(temp_vec)
#                 # add to the converted image to the plot
#                 ax = plt.subplot(1, n + 1, step + 1)
#                 np_changed_image = changed_image.to(device).squeeze().numpy()
#                 plt.imshow(plot_adapter(np_changed_image), cmap=cmap)
#                 ax.get_xaxis().set_visible(False)
#                 ax.get_yaxis().set_visible(False)
#             plt.show()


# The function two image and convert one imag to the second by margin the two latent vector
# with different impact
def convert_img_from_latent(model, device, test_loader, label_1=0, label_2=1, dir_name="convert_img_from_latent",
                            epoch=-1, num_of_steps=8):
    origin_image = {"img_1":None,"img_2":None}
    origin_Z = {"Z_1":None,"Z_2":None}
    output_list = []
    model.eval()
    # find images by label
    with torch.no_grad():
        # find image by target
        for batch, tar in test_loader:
            for i, img in enumerate(batch):
                if label_1 == int(tar[i]):
                    origin_image["img_1"] = img.unsqueeze(0).to(device)
                if label_2 == int(tar[i]):
                    origin_image["img_2"] = img.unsqueeze(0).to(device)
                if origin_image["img_1"] is not None and origin_image["img_2"] is not None:
                    break
            if origin_image["img_1"] is not None and origin_image["img_2"] is not None:
                break
        # create the "Z" for each image
        origin_Z["Z_1"], _, _ = model.enc(origin_image["img_1"])
        origin_Z["Z_2"], _, _ = model.enc(origin_image["img_2"])
        # add the first original image to the output list
        output_list.append(origin_image["img_1"])

        for step in range(num_of_steps + 1):
            temp_first_vec = origin_Z["Z_1"].detach().clone()
            temp_second_vec = origin_Z["Z_2"].detach().clone()
            t = (1 / num_of_steps) * step
            # combine the two image with impact "t" on the first ant "t-1" on the second
            temp_vec = temp_first_vec * (1 - t) + temp_second_vec * (t)
            # creat the change image by pass the vector throw the decoder
            changed_image = model.dec(temp_vec)
            # add the changed image to the output list
            output_list.append(changed_image)

        # add the second original image to the output list
        output_list.append(origin_image["img_2"])
        # save the images
        torchvision.utils.save_image(
            torch.cat(output_list, 0),
            f"experiment_vae{os.sep}{dir_name}{os.sep}{str(epoch + 1).zfill(5)}.png",
            nrow=num_of_steps+3,
            normalize=True,
            value_range=(-1, 1),
        )






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
