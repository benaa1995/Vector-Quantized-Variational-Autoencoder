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
import vae_model

# from sklearn.manifold import TSNE
# import seaborn as sns
# import matplotlib.animation as animation


# import seaborn as sns

import halper_func as hf
import expirment_func as ef

# writer for tnsorboard
writer = SummaryWriter(f'runs/CIFAR10_vgg16_bn/autoencoder_tensorboard')

data_dir = 'dataset_CIFAR10_vgg16_bn'


def load_data(batch_size, resize=128, data_dir='dataset'):
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

    # ------------------------------------------------------------------------
    # todo remove !!!!!
    m = len(train_dataset)
    smaller_train_data, val_data = random_split(train_dataset, [int(m * 0.001), int(m - m * 0.001)])
    m = len(test_dataset)
    smaller_test_data, val_data = random_split(test_dataset, [int(m * 0.005), int(m - m * 0.005)])
    train_loader = torch.utils.data.DataLoader(smaller_train_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(smaller_test_data, batch_size=batch_size, shuffle=True)
    # ---------------------------------------------------------------------------------------------
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, train_dataset, test_dataset


def load_save_checkpoint(epoch, model=vae_model.VAE(), save=True, load_path="", save_dir=""):
    if save:
        torch.save(model.state_dict(), f"checkpoint/{save_dir}vae{str(epoch + 1).zfill(3)}.pt")
    else:
        ckpt = torch.load(os.path.join('checkpoint', load_path))
        model.load_state_dict(ckpt)
        # model = model.to(device)
    return model


def vae_loss(image_batch, decoded_data, loss_fn, z_log_var, z_mean, beta_exp=8):
    a = torch.randn(4, 4)
    b = torch.sum(a, 1)
    c = torch.arange(3* 4 * 5 * 6).view(3, 4, 5, 6)
    d = torch.ones(3, 4, 5, 6)
    e = torch.sum(c, (3, 2, 1))
    f = torch.sum(d, (3, 2, 1))

    pow_log_var = torch.exp(z_log_var)
    pow_mean = z_mean ** 2
    kl_1 = 1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var)
    z_log_var_lv = 1 + z_log_var
    kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var), axis=(3, 2, 1))  # sum over latent dimension
    batchsize = kl_div.size(0)
    kl_div = kl_div.mean()  # average over batch dimension

    pixelwise = loss_fn(decoded_data, image_batch, reduction='none')
    pixelwise = pixelwise.view(batchsize, -1).sum(axis=1)  # sum over pixels
    pixelwise = pixelwise.mean()  # average over batch dimension

    # beta_exp = 12
    beta = 2 ** beta_exp
    # kl_div_temp = beta * kl_div

    # Evaluate loss
    loss = pixelwise + beta * kl_div
    return loss


### Training function
def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer, beta_exp=0):
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


def train_model(VAE, train_loader, test_loader, test_dataset, lr=0.001, latent_size=4, num_epochs=30, beta_exp=0,
                save_weights=True, load_weights=False,
                path_for_model_weights='model_weights_epoch_1.pth'):
    ### Define the loss function
    loss_fn = F.mse_loss

    ### Define an optimizer (both for the encoder and the decoder!)
    # lr= 0.001
    print("lr = ", lr)
    ### Set the random seed for reproducible results
    torch.manual_seed(0)

    ### Initialize the two networks
    # latent_size = 4
    print("latent_size = ", latent_size)
    # encoder = Vgg16_Encoder(encoded_space_dim=latent_size, fc2_input_dim=128)
    # decoder = Vgg16_Decoder(encoded_space_dim=latent_size, fc2_input_dim=128)
    params_to_optimize = [
        {'params': VAE.enc.parameters()},
        {'params': VAE.dec.parameters()}
    ]

    optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")
    print(f'Selected device: {device}')

    # Move both the encoder and the decoder to the selected device
    VAE.to(device)

    # if load_weights:
    #     encoder_path_for_model_weights = './weights/encoder_' + path_for_model_weights
    #     decoder_path_for_model_weights = './weights/decoder_' + path_for_model_weights
    #     encoder.load_state_dict(torch.load(encoder_path_for_model_weights))
    #     decoder.load_state_dict(torch.load(decoder_path_for_model_weights))

    # tensorBoard hyper-parameters
    global_step = 1
    # num_epochs = 30
    diz_loss = {'train_loss': [], 'val_loss': []}
    for epoch in range(num_epochs):
        train_loss = train_epoch(VAE.enc, VAE.dec, device,
                                 train_loader, loss_fn, optim, beta_exp=beta_exp)
        val_loss = test_epoch(VAE.enc, VAE.dec, device, test_loader, loss_fn, beta_exp=beta_exp)
        print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs, train_loss, val_loss))
        diz_loss['train_loss'].append(train_loss)
        diz_loss['val_loss'].append(val_loss)
        temp_name = "Loss, latent_ =" + str(latent_size)
        writer.add_scalars(temp_name, {'Traning loss': train_loss, 'Test loss': val_loss}, global_step)
        writer.flush()
        global_step += 1
        # # save the current model weights
        # if save_weights:
        #     splited = path_for_model_weights.split('_')
        #     splited_2 = splited[-1].split('.')
        #     splited_2[0] = str(epoch)
        #     splited_2 = tuple(splited_2)
        #     joined_2 = '.'.join(splited_2)
        #     splited[-1] = joined_2
        #     splited = tuple(splited)
        #     joined = '_'.join(splited)
        #     encoder_path_for_model_weights = './encoder_'+joined
        #     decoder_path_for_model_weights = './decoder_'+joined

        # torch.save(encoder.state_dict(), encoder_path_for_model_weights)
        # torch.save(decoder.state_dict(), decoder_path_for_model_weights)

        # model.load_state_dict(torch.load(path_for_model_weights))

        # hf.plot_ae_outputs(VAE.enc, VAE.dec, test_dataset, device, hf.targets_CIFAR10_adapter,
        #                    hf.img_idx_CIFAR10_adapter, hf.plot_CIFAR10_adapter)
        # save the current checkpoint
        load_save_checkpoint(epoch, model=VAE)

        hf.convert_latent_to_cvs(VAE.enc, 4, "kkkkk", test_loader, device)

        # hf.plot_ae_outputs(VAE, test_loader, epoch, "cifar10_1", num_of_img=10)
        # ef.create_random_img(VAE.dec, device, epoch=epoch)
        # ef.latent_digit_impact(VAE, device, test_loader, label=2, epoch=epoch)
        # ef.convert_img_from_latent(VAE, device, test_loader, label_1=0, label_2=1, epoch=epoch)
        print("ooops")

    writer.add_scalar("latent_size vs minimun loss", min(diz_loss["val_loss"]), latent_size)
    writer.flush()
    return diz_loss["val_loss"], diz_loss['train_loss']


def main(args):
    print(args)
    lat_size_list = [6]  # 4, 5, 6, 7, 8
    for pow_lat_size in lat_size_list:
        black_img = np.ones((32, 32))
        plt.imshow(black_img)
        plt.show()
        lat_size = 2 ** pow_lat_size
        VAE = vae_model.VAE()
        train_loader, test_loader, train_dataset, test_dataset = load_data(10, resize=128)
        train_model(VAE, train_loader, test_loader, test_dataset, latent_size=lat_size,
                    num_epochs=100, path_for_model_weights='vgg16_lat_64_weights_epoch_1.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--plt_output_dir", type=str, default="random_Z")
    parser.add_argument("--digit_impact_dir", type=str, default="latent_digit_impact")
    parser.add_argument("--convert_img_dir", type=str, default="convert_img_from_latent")

    args = parser.parse_args()
    main(args)
