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
# import vae_model
# import vae_model_ch1 as vae_model
import vae_model_ch1_original_size as vae_model
# from sklearn.manifold import TSNE
# import seaborn as sns
# import matplotlib.animation as animation


# import seaborn as sns

import halper_func as hf
import expirment_func as ef

# writer for tnsorboard
writer = SummaryWriter(f'runs/CIFAR10_beta8_ch1/autoencoder_tensorboard')

data_dir = 'dataset_CIFAR10_vgg16_bn'


def vae_loss(image_batch, decoded_data, loss_fn, z_log_var, z_mean, beta_exp=0):
    # a = torch.randn(4, 4)
    # b = torch.sum(a, 1)
    # c = torch.arange(3* 4 * 5 * 6).view(3, 4, 5, 6)
    # d = torch.ones(3, 4, 5, 6)
    # e = torch.sum(c, (3, 2, 1))
    # f = torch.sum(d, (3, 2, 1))
    #
    # pow_log_var = torch.exp(z_log_var)
    # pow_mean = z_mean ** 2
    # kl_1 = 1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var)
    # z_log_var_lv = 1 + z_log_var
    # temp_1 = 1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var)
    # temp_2 = torch.sum(temp_1, axis=(3, 2, 1))
    # temp_2 = -0.5 * temp_2
    kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var),
                              axis=(3, 2, 1))  # sum over latent dimension
    batchsize = kl_div.size(0)
    kl_div = kl_div.mean()  # average over batch dimension

    # pixelwise = loss_fn(decoded_data, image_batch, reduction='none')
    pixelwise = loss_fn(decoded_data, image_batch)
    # pixelwise = pixelwise.view(batchsize, -1).sum(axis=1)  # sum over pixels
    # pixelwise = pixelwise.mean()  # average over batch dimension

    # beta_exp = 12
    beta = 2 ** beta_exp
    # kl_div_temp = beta * kl_div
    # todo remove!!
    beta = 0.0005
    # Evaluate loss
    loss = pixelwise + beta * kl_div
    return loss


#
# def vae_loss(image_batch, decoded_data, loss_fn, z_log_var, z_mean, beta_exp=0):
#     kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var), axis=1)  # sum over latent dimension
#     batchsize = kl_div.size(0)
#     kl_div = kl_div.mean()  # average over batch dimension
#
#     pixelwise = loss_fn(decoded_data, image_batch, reduction='none')
#     pixelwise = pixelwise.view(batchsize, -1).sum(axis=1)  # sum over pixels
#     pixelwise = pixelwise.mean()  # average over batch dimension
#
#     beta = 2 ** beta_exp
#
#     # Evaluate loss
#     loss = pixelwise + beta * kl_div
#     return loss


### Training function
def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer, args, beta_exp=0):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in dataloader:
        # print(image_batch.shape)
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        # Encode data
        encoded_data, z_mean, z_log_var = encoder(image_batch)
        # Decode data
        decoded_data = decoder(encoded_data)

        # Evaluate loss
        loss = vae_loss(image_batch, decoded_data, loss_fn, z_log_var, z_mean, beta_exp=beta_exp)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().to(device))  # .numpy()
    tensor_train_loss = torch.stack(train_loss)
    return torch.mean(tensor_train_loss)


### Testing function
def test_epoch(encoder, decoder, device, dataloader, loss_fn, beta_exp=0):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad():  # No need to track the gradients
        # Define the lists to store the loss for each batch
        val_loss = []
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            # Encode data
            encoded_data, z_mean, z_log_var = encoder(image_batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Append the network loss to the list
            val_loss.append(
                vae_loss(image_batch, decoded_data, loss_fn, z_log_var, z_mean, beta_exp=beta_exp).to(device))
        # Create a single tensor with all the values in the lists
        val_loss = torch.stack(val_loss)
        # Evaluate global loss
        val_loss = torch.mean(val_loss)
    return val_loss.data


def train_model(model, device, loss_function, train_loader, test_loader, test_dataset, optim, args, latent_size=4,
                num_epochs=30, curr_epoch=0, beta_exp=0):
    # tensorBoard hyper-parameters
    global_step = 1
    diz_loss = {'train_loss': [], 'val_loss': []}
    for epoch in range(curr_epoch, num_epochs):
        train_loss = train_epoch(model.enc, model.dec, device, train_loader, loss_function, optim, args=args,
                                 beta_exp=beta_exp)
        val_loss = test_epoch(model.enc, model.dec, device, test_loader, loss_function, beta_exp=beta_exp)
        print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs, train_loss, val_loss))
        diz_loss['train_loss'].append(train_loss)
        diz_loss['val_loss'].append(val_loss)
        temp_name = "Loss, latent_ =" + str(latent_size)
        writer.add_scalars(temp_name, {'Traning loss': train_loss, 'Test loss': val_loss}, global_step)
        writer.flush()
        global_step += 1

        # save the current checkpoint
        hf.load_save_checkpoint(model, epoch=epoch, save_path=args.save_ckp_dir)

        hf.plot_ae_outputs(model, test_loader, epoch, args.recon_dir, num_of_img=10)
        # ef.create_random_img(model.dec, device, epoch=epoch, latent_size=(16, 16, 16))
        ef.create_random_img(model.dec, device, dir_name=args.plt_random_dir, epoch=epoch, latent_size=(1, 16, 16))
        # ef.latent_digit_impact(model, device, test_loader, dir_name=args.digit_impact_dir, label=2,
        #                   max_pix_in_file=16*8, epoch=epoch)
        ef.convert_img_from_latent(model, device, test_loader, dir_name=args.convert_img_dir, label_1=0, label_2=1, epoch=epoch)

    writer.add_scalar("latent_size vs minimun loss", min(diz_loss["val_loss"]), latent_size)
    writer.flush()
    return diz_loss["val_loss"], diz_loss['train_loss']


def main(args):
    # print(args)
    # lat_size_list = [6]  # 4, 5, 6, 7, 8
    # for pow_lat_size in lat_size_list:
    #     black_img = np.ones((32, 32))
    #     plt.imshow(black_img)
    #     plt.show()
    #     lat_size = 2 ** pow_lat_size
    #     VAE = vae_model.VAE()
    #     train_loader, test_loader, train_dataset, test_dataset = load_data(10, resize=128)
    #     train_model(VAE, train_loader, test_loader, test_dataset, latent_size=lat_size,
    #                 num_epochs=100, path_for_model_weights='vgg16_lat_64_weights_epoch_1.pth')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')
    # load the dataset
    train_loader, test_loader, train_dataset, test_dataset = hf.load_data(batch_size=args.batch_size, resize=args.size)
    # load the model
    model = vae_model.VAE().to(device)
    if args.load_checkpoint_path is not None:
        model = hf.load_save_checkpoint(model, save=None, load_path=args.load_checkpoint_path)

    ## Define the loss function
    # loss_function = F.mse_loss
    ### Define the loss function
    loss_function = torch.nn.MSELoss()

    ### Define an optimizer (both for the encoder and the decoder!)

    ### Set the random seed for reproducible results
    torch.manual_seed(0)

    ### Initialize the two networks
    params_to_optimize = [
        {'params': model.enc.parameters()},
        {'params': model.dec.parameters()}
    ]

    optim = torch.optim.Adam(params_to_optimize, lr=args.lr, weight_decay=1e-05)

    # Move both the encoder and the decoder to the selected device
    model.to(device)

    train_model(model, device, loss_function, train_loader, test_loader, test_dataset, optim, args=args,
                num_epochs=args.epoch,
                beta_exp=args.beta_exp, curr_epoch=args.curr_epoch)

    ef.latent_digit_impact(model, device, test_loader, label=2, max_pix_in_file=16 * 8, epoch=args.epoch - 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta_exp", type=int, default=0)
    parser.add_argument("--plt_random_dir", type=str, default="random_Z_ch1_0005")
    parser.add_argument("--digit_impact_dir", type=str, default="latent_digit_impact_ch1_0005")
    parser.add_argument("--convert_img_dir", type=str, default="convert_img_from_latent_ch1_0005")
    parser.add_argument("--recon_dir", type=str, default="ch_1_b_0005")
    parser.add_argument("--save_ckp_dir", type=str, default="ckp_1_0005")

    parser.add_argument("--load_checkpoint_path", type=str, default=None)
    parser.add_argument("--curr_epoch", type=int, default=0)

    args = parser.parse_args()
    main(args)
