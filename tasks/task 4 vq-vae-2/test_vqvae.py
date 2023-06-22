import argparse
import os

import torch
from torchvision.utils import save_image


import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from torchvision import datasets, transforms, utils


from vqvae import VQVAE
from pixelsnail_mnist import PixelSNAIL

import expirment_func as ef
import halper_func as hf

# def load_data(batch_size, resize=128, data_path='dataset'):
#     train_dataset = torchvision.datasets.CIFAR10(data_path, train=True, download=True)
#     test_dataset = torchvision.datasets.CIFAR10(data_path, train=False, download=True)
#
#     train_transform = transforms.Compose([
#         transforms.Resize(resize),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
#     ])
#
#     test_transform = transforms.Compose([
#         transforms.Resize(resize),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
#     ])
#
#     train_dataset.transform = train_transform
#     test_dataset.transform = test_transform
#
#     # # ------------------------------------------------------------------------
#     # # todo remove !!!!!
#     # m = len(train_dataset)
#     # smaller_train_data, val_data = random_split(train_dataset, [int(m * 0.001), int(m - m * 0.001)])
#     # m = len(test_dataset)
#     # smaller_test_data, val_data = random_split(test_dataset, [int(m * 0.005), int(m - m * 0.005)])
#     # train_loader = torch.utils.data.DataLoader(smaller_train_data, batch_size=batch_size)
#     # test_loader = torch.utils.data.DataLoader(smaller_test_data, batch_size=batch_size, shuffle=True)
#     # # ---------------------------------------------------------------------------------------------
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
#
#     return train_loader, test_loader, train_dataset, test_dataset


def load_data(batch_size, resize=128, data_dir='dataset'):
    train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True)

    train_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    train_dataset.transform = train_transform
    test_dataset.transform = test_transform

    # # ------------------------------------------------------------------------
    # # todo remove !!!!!
    # m = len(train_dataset)
    # smaller_train_data, val_data = random_split(train_dataset, [int(m * 0.001), int(m - m * 0.001)])
    # m = len(test_dataset)
    # smaller_test_data, val_data = random_split(test_dataset, [int(m * 0.005), int(m - m * 0.005)])
    # train_loader = torch.utils.data.DataLoader(smaller_train_data, batch_size=batch_size)
    # test_loader = torch.utils.data.DataLoader(smaller_test_data, batch_size=batch_size, shuffle=True)
    # # ---------------------------------------------------------------------------------------------
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    test_loader_2 = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    return train_loader, test_loader, train_dataset, test_dataset, test_loader_2



def load_model(model, checkpoint, device):
    ckpt = torch.load(os.path.join('checkpoint', checkpoint))

    if 'args' in ckpt:
        args = ckpt['args']

    if model == 'vqvae':
        model = VQVAE()

    elif model == 'pixelsnail_top':
        model = PixelSNAIL(
            [2, 2],
            32,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            attention=False,
            dropout=args.dropout,
            n_cond_res_block=args.n_cond_res_block,
            cond_res_channel=args.n_res_channel,
            # n_out_res_block=args.n_out_res_block,
        )

    elif model == 'pixelsnail_bottom':
        model = PixelSNAIL(
            [4, 4],
            32,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            attention=False,
            dropout=args.dropout,
            n_cond_res_block=args.n_cond_res_block,
            cond_res_channel=args.n_res_channel,
        )

    if 'model' in ckpt:
        ckpt = ckpt['model']

    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    return model



def main(args):
    device = 'cuda'
    train_loader, test_loader, train_dataset, test_dataset, test_loader_2 = load_data(args.size, resize=32, data_dir=args.path)

    model_vqvae = load_model('vqvae', args.vqvae, device)
    # model_top = load_model('pixelsnail_top', args.top, device)
    # model_bottom = load_model('pixelsnail_bottom', args.bottom, device)
    print("blabla")
    for i in range(10):
        ef.latent_digit_impact(model_vqvae, device, test_loader, label=i, epoch=i-1, codebook_size=args.codebook_size)

    # ef.convert_img_from_latent(model_vqvae, device, test_loader_2, label_1=4, label_2=5, dir_name="convert_img_from_latent",
    #                         epoch=9, num_of_steps=8)
    # hf.convert_latent_to_cvs(model_vqvae.encode, latent_size=16*16, file_name="train_16", dataloader=train_loader, device=device)
    # hf.convert_latent_to_cvs(model_vqvae.encode, latent_size=16*16, file_name="test_16",
    #                          dataloader=test_loader, device=device)


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--codebook_size', type=int, default=32)
    parser.add_argument('--path', type=str, default="cifar10")
    parser.add_argument('--vqvae', type=str)
    parser.add_argument('--top', type=str)
    parser.add_argument('--bottom', type=str)
    parser.add_argument('--temp', type=float, default=1.0)
    # parser.add_argument('filename', type=str)

    args = parser.parse_args()
    print(args)



    main(args)