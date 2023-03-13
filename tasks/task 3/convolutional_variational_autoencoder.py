# TAKEN FROM https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac#d75c
# autoencoder using cnn
import numpy as np

import torchvision
from torchvision import transforms

import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import halper_func as hf
import expirment_func as ef

# writer for tnsorboard
writer = SummaryWriter(f'runs/MNIST/autoencoder_tensorboard')

data_dir = 'dataset'

train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True)

train_transform = transforms.Compose([
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset.transform = train_transform
test_dataset.transform = test_transform

m = len(train_dataset)
print("len(train_dataset)= ", m, "len(test_dataset)", len(test_dataset))
print(test_dataset.targets)
train_data, val_data = random_split(train_dataset, [int(m - m * 0.2), int(m * 0.2)])
batch_size = 256

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


class Encoder(nn.Module):

    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )

        self.z_mean = nn.Linear(128, encoded_space_dim)
        self.z_log_var = nn.Linear(128, encoded_space_dim)

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            # nn.Linear(128, encoded_space_dim)
        )

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1))  # .to(z_mu.get_device())
        z = z_mu + eps * torch.exp(z_log_var / 2.)
        return z

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded_data = self.reparameterize(z_mean, z_log_var)

        return encoded_data, z_mean, z_log_var


class Decoder(nn.Module):

    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(32, 3, 3))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3,
                               stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2,
                               padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
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
        train_loss.append(loss.detach().to(device).numpy())

    return np.mean(train_loss)


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


def train_model(lr=0.001, latent_size=4, num_epochs=30, beta_exp=0):
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
    encoder = Encoder(encoded_space_dim=latent_size, fc2_input_dim=128)
    decoder = Decoder(encoded_space_dim=latent_size, fc2_input_dim=128)
    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]

    optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    print(f'Selected device: {device}')

    # Move both the encoder and the decoder to the selected device
    encoder.to(device)
    decoder.to(device)

    # tensorBoard hyper-parameters
    global_step = 1
    # num_epochs = 30
    diz_loss = {'train_loss': [], 'val_loss': []}
    for epoch in range(num_epochs):
        train_loss = train_epoch(encoder, decoder, device,
                                 train_loader, loss_fn, optim, beta_exp=beta_exp)
        val_loss = test_epoch(encoder, decoder, device, test_loader, loss_fn, beta_exp=beta_exp)
        print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs, train_loss, val_loss))
        diz_loss['train_loss'].append(train_loss)
        diz_loss['val_loss'].append(val_loss)
        temp_name = "Loss, beta_exp =" + str(beta_exp)
        writer.add_scalars(temp_name, {'Traning loss': train_loss, 'Test loss': val_loss}, global_step)
        writer.flush()
        global_step += 1

        if epoch % 10 == 0:
            hf.plot_ae_outputs(encoder, decoder, test_dataset, device, hf.targets_MNIST_adapter,
                               hf.img_idx_MNIST_adapter, hf.plot_MNIST_adapter, cmap='gist_gray')
            ef.create_random_img(decoder, device, hf.plot_MNIST_adapter, cmap='gist_gray', latent_size=latent_size)
            # ef.latent_digit_impact(encoder, decoder, device, test_dataset, hf.targets_MNIST_adapter,
            #                        hf.img_idx_MNIST_adapter, hf.plot_MNIST_adapter, cmap='gist_gray',
            #                        latent_size=latent_size)

            ef.convert_img_from_latent(encoder, decoder, device, test_dataset, hf.targets_MNIST_adapter,
                                       hf.img_idx_MNIST_adapter, hf.plot_MNIST_adapter, cmap='gist_gray',
                                       latent_size=latent_size)
    converted_file_name =  'test_beta_exp_test' + str(beta_exp)
    hf.convert_latent_to_cvs(encoder, latent_size, converted_file_name, test_loader, device)
    writer.add_scalar("beta_exp vs minimun loss", min(diz_loss["val_loss"]), beta_exp)
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


def main():
    #
    # beta_exp_list = [-4,-3,-2,-1,0,1,2,3,4,5,6,7]
    # for beta_exp in beta_exp_list:
    #     train_model(latent_size=16, num_epochs=5,beta_exp=beta_exp)

    ef.train_with_TSNE('test_lat_size_16.cvs')
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
