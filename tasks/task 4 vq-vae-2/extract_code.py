import argparse
import pickle
import os
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import lmdb
from tqdm import tqdm

from dataset import ImageFileDataset, CodeRow
from vqvae import VQVAE

# compress the dataset to the top and bottom quantize data (with the index of the codebook vector)
# for the cnn model

def extract(lmdb_env, loader, model, device):
    index = 0

    with lmdb_env.begin(write=True) as txn:
        pbar = tqdm(loader)

        for img,  filename in pbar:
            img = img.to(device)

            _, _, _, id_t, id_b = model.encode(img)
            id_t = id_t.detach().cpu().numpy()
            id_b = id_b.detach().cpu().numpy()

            for file, top, bottom in zip(filename, id_t, id_b):
                row = CodeRow(top=top, bottom=bottom, filename=file)
                txn.put(str(index).encode('utf-8'), pickle.dumps(row))
                index += 1
                pbar.set_description(f'inserted: {index}')

        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))

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



if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('path', type=str)

    args = parser.parse_args()

    device = 'cuda'

    train_loader, test_loader, train_dataset, test_dataset = load_data(args.size, resize=128, data_dir=args.path)






    model = VQVAE()
    model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)
    model.eval()
    # TODO change the map size
    map_size = 100 * 1024 * 1024 * 1024

    env = lmdb.open(args.name, map_size=map_size)

    extract(env, train_loader, model, device)
