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



from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.animation as animation




# import seaborn as sns

import halper_func as hf
import expirment_func as ef

# writer for tnsorboard
writer = SummaryWriter(f'runs/CIFAR10_vgg16_bn/autoencoder_tensorboard')

data_dir = 'dataset_CIFAR10_vgg16_bn'

train_dataset = torchvision.datasets.CIFAR10(data_dir, train=True, download=True)
test_dataset = torchvision.datasets.CIFAR10(data_dir, train=False, download=True)

train_transform = transforms.Compose([
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset.transform = train_transform
test_dataset.transform = test_transform

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

m = len(train_dataset)
print("len(train_dataset)= ", m, "len(test_dataset)", len(test_dataset))
print(test_dataset.targets)
# train_data, val_data = random_split(train_dataset, [int(m - m * 0.995), int(m * 0.995)])
batch_size = 100

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
# valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


class Vgg16_Encoder(nn.Module):

    def _init_(self, encoded_space_dim, fc2_input_dim):
        super()._init_()

        ### Convolutional section
        self.features_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True), )

        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)

        self.features_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True), )

        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)

        self.features_3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True), )

        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)

        self.features_4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section

        self.encoder_lin = nn.Sequential(
            nn.Linear(4 * 4 * 512, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 1000),
            nn.ReLU(True),

        )
        self.z_mean = nn.Linear(1000, encoded_space_dim)
        self.z_log_var = nn.Linear(1000, encoded_space_dim)

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(z_mu.get_device())
        z = z_mu + eps * torch.exp(z_log_var / 2.)
        return z

    def forward(self, x):
        x = self.features_1(x)
        x, indices_pool_1 = self.pool_1(x)
        # print(indices_pool_1)
        x = self.features_2(x)
        x, indices_pool_2 = self.pool_2(x)
        x = self.features_3(x)
        x, indices_pool_3 = self.pool_3(x)
        x = self.features_4(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded_data = self.reparameterize(z_mean, z_log_var)

        indices = [indices_pool_1, indices_pool_2, indices_pool_3]
        return encoded_data, z_mean, z_log_var, indices





class Vgg16_Decoder(nn.Module):

    def _init_(self, encoded_space_dim, fc2_input_dim):
        super()._init_()

        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4 * 4 * 512)
        )

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(512, 4, 4))

        self.features_4 = nn.Sequential(
            # nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True), )

        self.un_pool_3 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)

        self.features_3 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True), )

        self.un_pool_2 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)

        self.features_2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True), )

        self.un_pool_1 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)

        self.features_1 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

    def forward(self, x, indices):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.features_4(x)
        x = self.un_pool_3(x, indices[2])
        x = self.features_3(x)
        x = self.un_pool_2(x, indices[1])
        x = self.features_2(x)
        x = self.un_pool_1(x, indices[0])
        x = self.features_1(x)
        x = torch.sigmoid(x)
        return x


def vae_loss(image_batch, decoded_data, loss_fn, z_log_var, z_mean, beta_exp=0):
    kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var), axis=1)  # sum over latent dimension
    batchsize = kl_div.size(0)
    kl_div = kl_div.mean()  # average over batch dimension

    pixelwise = loss_fn(decoded_data, image_batch, reduction='none')
    pixelwise = pixelwise.view(batchsize, -1).sum(axis=1)  # sum over pixels
    pixelwise = pixelwise.mean()  # average over batch dimension

    beta = 2 ** beta_exp

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
    for image_batch, _ in dataloader:  # with "_" we just ignore the labels (the second element of the dataloader tuple)
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
            val_loss.append(vae_loss(image_batch, decoded_data, loss_fn, z_log_var, z_mean, beta_exp=beta_exp).to(device))
        # Create a single tensor with all the values in the lists
        val_loss = torch.stack(val_loss)
        # Evaluate global loss
        val_loss = torch.mean(val_loss)
    return val_loss.data


def train_model(lr=0.001, latent_size=4, num_epochs=30, beta_exp=0, save_weights=True, load_weights = False,
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
    # model = Autoencoder(encoded_space_dim=encoded_space_dim)
    encoder = Vgg16_Encoder(encoded_space_dim=latent_size, fc2_input_dim=128)
    decoder = Vgg16_Decoder(encoded_space_dim=latent_size, fc2_input_dim=128)
    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]

    optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")
    print(f'Selected device: {device}')

    # Move both the encoder and the decoder to the selected device
    encoder.to(device)
    decoder.to(device)

    indices=[]
    # tensorBoard hyper-parameters
    global_step = 1
    # num_epochs = 30
    diz_loss = {'train_loss': [], 'val_loss': []}
    for epoch in range(num_epochs):

        if load_weights:
            encoder_path_for_model_weights = './weights/encoder_' + path_for_model_weights
            decoder_path_for_model_weights = './weights/decoder_' + path_for_model_weights
            encoder.load_state_dict(torch.load(encoder_path_for_model_weights))
            decoder.load_state_dict(torch.load(decoder_path_for_model_weights))

        train_loss, curr_indices = train_epoch(encoder, decoder, device,
                                 train_loader, loss_fn, optim, beta_exp=beta_exp)
        if epoch == 0:
            indices = curr_indices
        val_loss = test_epoch(encoder, decoder, device, test_loader, loss_fn, beta_exp=beta_exp)
        print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs, train_loss, val_loss))
        diz_loss['train_loss'].append(train_loss)
        diz_loss['val_loss'].append(val_loss)
        temp_name = "Loss, latent_ =" + str(latent_size)
        writer.add_scalars(temp_name, {'Traning loss': train_loss, 'Test loss': val_loss}, global_step)
        writer.flush()
        global_step += 1
        # save the current model weights
        if save_weights:
            splited = path_for_model_weights.split('_')
            splited_2 = splited[-1].split('.')
            splited_2[0] = str(epoch)
            splited_2 = tuple(splited_2)
            joined_2 = '.'.join(splited_2)
            splited[-1] = joined_2
            splited = tuple(splited)
            joined = '_'.join(splited)
            encoder_path_for_model_weights = './encoder_'+joined
            decoder_path_for_model_weights = './decoder_'+joined

            torch.save(encoder.state_dict(), encoder_path_for_model_weights)
            torch.save(decoder.state_dict(), decoder_path_for_model_weights)


        # model.load_state_dict(torch.load(path_for_model_weights))

        hf.plot_ae_outputs(encoder, decoder, test_dataset, device, hf.targets_CIFAR10_adapter,
                           hf.img_idx_CIFAR10_adapter, hf.plot_CIFAR10_adapter)

        # hf.convert_latent_to_cvs(encoder, latent_size, 'test_CIFAR10', test_loader, device)
        # ef.create_random_img(decoder, device, hf.plot_CIFAR10_adapter, latent_size=latent_size)



        # if epoch % 10 == 0:
        # hf.plot_ae_outputs_CIFAR10(encoder, decoder, test_dataset, classes=classes, n=10, device=device)
    #     ef.create_random_img(decoder, device, hf.plot_CIFAR10_adapter, latent_size=latent_size)
    #     ef.latent_digit_impact(encoder, decoder, device, test_dataset, hf.targets_CIFAR10_adapter,hf.img_idx_CIFAR10_adapter,
    #                            hf.plot_CIFAR10_adapter,latent_size=latent_size)
    #
    #     ef.convert_img_from_latent(encoder, decoder, device, test_dataset, hf.targets_CIFAR10_adapter,
    #                                hf.img_idx_CIFAR10_adapter, hf.plot_CIFAR10_adapter)
    #
    # converted_file_name = 'test_vgg16_test' + str(latent_size)
    # hf.convert_latent_to_cvs(encoder, latent_size, converted_file_name, test_loader, device)
    # converted_file_name = 'test_vgg16_train' + str(latent_size)
    # hf.convert_latent_to_cvs(encoder, latent_size, converted_file_name, train_loader, device)
    writer.add_scalar("latent_size vs minimun loss", min(diz_loss["val_loss"]), latent_size)
    writer.flush()
    return diz_loss["val_loss"], diz_loss['train_loss']


def main():
    lat_size_list = [6]  # 4, 5, 6, 7, 8
    for pow_lat_size in lat_size_list:
        black_img = np.ones((32, 32))
        plt.imshow(black_img)
        plt.show()
        lat_size = 2 ** pow_lat_size
        train_model(latent_size=lat_size, num_epochs=100, path_for_model_weights='vgg16_lat_64_weights_epoch_1.pth')

    # ef.train_with_TSNE('test_CIFAR10.cvs')
    #
    # beta_exp_list = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
    # for beta_exp in beta_exp_list:
    #     train_model(latent_size=16, num_epochs=5, beta_exp=beta_exp)
    #
    # train_model(latent_size=64, num_epochs=10, beta_exp=-6)

    # ef.train_with_TSNE('test_CIFAR10.cvs')
    # latent_size_stat()
    # train_model(num_epochs=5)
    # train_with_TSNE('test_lat_size_2.cvs', 'train_lat_size_2.cvs')
    # train_with_TSNE('test_lat_size_4.cvs', 'train_lat_size_4.cvs')
    # train_with_TSNE('test_lat_size_8.cvs', 'train_lat_size_8.cvs')
    # train_with_TSNE('test_lat_size_16.cvs', 'train_lat_size_16.cvs')
    # train_with_TSNE('test_lat_size_32.cvs', 'train_lat_size_32.cvs')
    # train_with_TSNE('test_lat_size_64.cvs', 'train_lat_size_64.cvs')
    # train_with_TSNE('test_lat_size_128.cvs', 'train_lat_size_128.cvs')


if __name__ == '__main__':
    main()


