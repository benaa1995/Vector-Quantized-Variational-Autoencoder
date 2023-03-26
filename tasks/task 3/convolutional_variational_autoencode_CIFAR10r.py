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
writer = SummaryWriter(f'runs/CIFAR10/autoencoder_tensorboard')

data_dir = 'dataset_CIFAR10'

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
# train_data, val_data = random_split(train_dataset, [int(m - m * 0.2), int(m * 0.2)])
batch_size = 30

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
# valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)




class Vgg16_Encoder(nn.Module):

    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()

        ### Convolutional section
        self.features = nn.Sequential(
          nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          nn.BatchNorm2d(64),
          nn.ReLU(inplace=True),
          nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          nn.BatchNorm2d(64),
          nn.ReLU(inplace=True),
          nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          nn.BatchNorm2d(128),
          nn.ReLU(inplace=True),
          nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
          nn.BatchNorm2d(128),
          nn.ReLU(inplace=True),
          nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          nn.BatchNorm2d(256),
          nn.ReLU(inplace=True),
          nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          nn.BatchNorm2d(256),
          nn.ReLU(inplace=True),
          nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
          nn.BatchNorm2d(256),
          nn.ReLU(inplace=True),
          nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          nn.BatchNorm2d(512),
          nn.ReLU(inplace=True),
          nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          nn.BatchNorm2d(512),
          nn.ReLU(inplace=True),
          nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
          nn.BatchNorm2d(512),
          nn.ReLU(inplace=True),
          nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          nn.BatchNorm2d(512),
          nn.ReLU(inplace=True),
          nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          nn.BatchNorm2d(512),
          nn.ReLU(inplace=True),
          nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
          nn.BatchNorm2d(512),
          nn.ReLU(inplace=True),
  )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section

        self.encoder_lin = nn.Sequential(
            nn.Linear(4 * 4 * 512, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 1000),
            nn.ReLU(True)
        )
        self.z_mean = nn.Linear(1000, encoded_space_dim)
        self.z_log_var = nn.Linear(1000, encoded_space_dim)

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1))  # .to(z_mu.get_device())
        z = z_mu + eps * torch.exp(z_log_var / 2.)
        return z

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded_data = self.reparameterize(z_mean, z_log_var)
        return encoded_data, z_mean, z_log_var


class Vgg16_Decoder(nn.Module):

    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4 * 4 * 512)
        )

        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(512, 4, 4))

        self.features = nn.Sequential(
            # nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(2, 2), output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.features(x)
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
    encoder = Vgg16_Encoder(encoded_space_dim=latent_size, fc2_input_dim=128)
    decoder = Vgg16_Decoder(encoded_space_dim=latent_size, fc2_input_dim=128)
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

        hf.plot_ae_outputs(encoder, decoder, test_dataset, device, hf.targets_CIFAR10_adapter,
                           hf.img_idx_CIFAR10_adapter, hf.plot_CIFAR10_adapter)
        # hf.convert_latent_to_cvs(encoder, latent_size, 'test_CIFAR10', test_loader, device)
        # ef.create_random_img(decoder, device, hf.plot_CIFAR10_adapter, latent_size=latent_size)

        # ef.latent_digit_impact(encoder, decoder, device, test_dataset, hf.targets_CIFAR10_adapter,hf.img_idx_CIFAR10_adapter,
        #                        hf.plot_CIFAR10_adapter,latent_size=latent_size)

        # ef.convert_img_from_latent(encoder, decoder, device, test_dataset, hf.targets_CIFAR10_adapter,
        #                            hf.img_idx_CIFAR10_adapter, hf.plot_CIFAR10_adapter)

        # if epoch % 10 == 0:
        # hf.plot_ae_outputs_CIFAR10(encoder, decoder, test_dataset, classes=classes, n=10, device=device)
        # ef.create_random_img(decoder, n=10, latent_size=latent_size)
        # ef.latent_digit_impact(encoder, decoder,  test_dataset, latent_size=latent_size)
        # ef.convert_img_from_latent(encoder, decoder, test_dataset, latent_size=latent_size)
    # converted_file_name = 'test_beta_exp_test' + str(beta_exp)
    # hf.convert_latent_to_cvs(encoder, latent_size, converted_file_name, test_loader, device)
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



    # fig = plt.figure()
    # axis = plt.axes(xlim=(-50, 50),
    #                 ylim=(-50, 50))
    #
    # curr = axis.plot()
    #
    # def init():
    #     curr.set_data([], [])
    #     return curr

    # initializing empty values
    # for x and y co-ordinates
    xdata, ydata = [], []

    # animation function
    # def animate(i):
    #     print(i)
    #     num_of_sample = 1000
    #     data_path = origin_data_path+str(i)
    #     # load the data
    #     df = pd.read_csv(data_path)
    #
    #     x = df.to_numpy()
    #     len_x, _ = x.shape
    #     if (num_of_sample > len_x):
    #         num_of_sample = len_x
    #     x = x[:num_of_sample, 1:-1]
    #     y = df['Y'].values.astype(int)
    #     y = y[:num_of_sample]
    #
    #     # sklrn linear regration
    #     tsne = TSNE(2)
    #     tsne_result = tsne.fit_transform(x)
    #     tsne_result.shape

        # tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:, 0], 'tsne_2': tsne_result[:, 1], 'label': y})
        # print(tsne_result_df)
        # print(np.unique(y))
        # fig,  = plt.subplots(1)
        # sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax, s=30,
        #                 palette=['darkgreen', 'red', 'black', 'orange', 'blue', 'cyan', 'fuchsia', 'lime', 'dimgray',
        #                          'brown'])

        # lim = (tsne_result.min() - 5, tsne_result.max() + 5)
        # ax.set_xlim(lim)
        # ax.set_ylim(lim)
        # ax.set_aspect('equal')
        # ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        # curr.set_data(xdata, ydata)
        # return fig







        # # t is a parameter which varies
        # # with the frame number
        # t = 0.1 * i
        #
        # # x, y values to be plotted
        # x = t * np.sin(t)
        # y = t * np.cos(t)
        #
        # # appending values to the previously
        # # empty x and y data holders
        # xdata.append(x)
        # ydata.append(y)
        # curr.set_data(xdata, ydata)
        #
        # return curr,

    # # calling the animation function
    # anim = animation.FuncAnimation(fig, animate,
    #                                init_func=init,
    #                                frames=500,
    #                                interval=20,
    #                                blit=True)
    #
    # # saves the animation in our desktop
    # anim.save('growingCoil.mp4', writer='ffmpeg', fps=30)





    #
    # fig, ax = plt.subplots(1)
    # ax.set_aspect('equal')
    # ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)




    #
    # beta_exp_list = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
    # for beta_exp in beta_exp_list:
    #     train_model(latent_size=16, num_epochs=5, beta_exp=beta_exp)
    #
    train_model(latent_size=64, num_epochs=10, beta_exp=-6)

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


