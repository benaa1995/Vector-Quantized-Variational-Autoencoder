import argparse
import sys
import os

import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from vqvae import VQVAE
# from vqvae_mnist import VQVAE
from scheduler import CycleScheduler
import distributed as dist


def load_data(batch_size, resize=128, data_path='dataset', dataset="MNIST"):
    if dataset == "MNIST":
        train_dataset = torchvision.datasets.MNIST(data_path, train=True, download=True)
        test_dataset = torchvision.datasets.MNIST(data_path, train=False, download=True)

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
    else:
        train_dataset = torchvision.datasets.CIFAR10(data_path, train=True, download=True)
        test_dataset = torchvision.datasets.CIFAR10(data_path, train=False, download=True)

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

    return train_loader, test_loader, train_dataset, test_dataset


def train(epoch, loader, model, optimizer, scheduler, device, args):
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0

    for i, (img, label) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        comm = dist.all_gather(comm)

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"lr: {lr:.5f}"
                )
            )

            if i == 1:
                model.eval()

                sample = img[:sample_size]

                with torch.no_grad():
                    out, _ = model(sample)

                utils.save_image(
                    torch.cat([sample, out], 0),
                    f"sample/vqvae{args.plt_dir}/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                    nrow=sample_size,
                    normalize=True,
                    value_range=(-1, 1),
                )

                model.train()


def main(args):
    device = "cuda"

    args.distributed = dist.get_world_size() > 1

    train_loader, test_loader, train_dataset, test_dataset = load_data(args.size, data_path=args.path, resize=args.size)


    # # dataset = datasets.ImageFolder(args.path, transform=transform)
    # sampler = dist.data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
    # loader = DataLoader(
    #     train_dataset, batch_size=128 // args.n_gpu, sampler=sampler, num_workers=2
    # )

    model = VQVAE().to(device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(train_loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

    for i in range(args.epoch):
        train(i, train_loader, model, optimizer, scheduler, device, args)

        if dist.is_primary():
            torch.save(model.state_dict(), f"checkpoint/vqvae{args.ckp_dir}/vqvae{str(i + 1).zfill(3)}.pt")


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("path", type=str)

    parser.add_argument("--ckp_dir", type=str, default="")
    parser.add_argument("--plt_dir", type=str, default="")

    args = parser.parse_args()

    print("args = ", args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
