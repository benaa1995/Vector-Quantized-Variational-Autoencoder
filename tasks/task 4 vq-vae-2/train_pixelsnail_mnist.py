import argparse
import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from apex import amp

except ImportError:
    amp = None

from dataset import LMDBDataset
from pixelsnail_mnist import PixelSNAIL
from scheduler import CycleScheduler


def train(args, epoch, loader, model, optimizer, scheduler, device):
    loader = tqdm(loader)

    criterion = nn.CrossEntropyLoss()

    for i, (top, bottom, label) in enumerate(loader):
        model.zero_grad()

        top = top.to(device)


        if args.hier == 'top':
            target = top
            curr_batch_size = label.size()[0]
            label = torch.reshape(label, (curr_batch_size, 1, 1))
            # label = torch.reshape(label, (curr_batch_size, 1))
            # new_label = torch.cat((label, label), 1)
            # new_label_1 = torch.reshape(new_label, (curr_batch_size, 1,2))
            # new_label_2 = torch.cat((new_label_1, new_label_1), 1)
            out, _ = model(top, condition=label)
            # out, _ = model(top, condition=label)
            # out, _ = model(top)
            # print("print(out ",out.size)

        elif args.hier == 'bottom':
            bottom = bottom.to(device)
            target = bottom
            out, _ = model(bottom, condition=top)

        loss = criterion(out, target)
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        _, pred = out.max(1)
        correct = (pred == target).float()
        accuracy = correct.sum() / target.numel()

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1}; loss: {loss.item():.5f}; '
                f'acc: {accuracy:.5f}; lr: {lr:.5f}'
            )
        )


class PixelTransform:
    def __init__(self):
        pass

    def __call__(self, input):
        ar = np.array(input)

        return torch.from_numpy(ar).long()


if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--hier', type=str, default='top')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--channel', type=int, default=128)
    parser.add_argument('--n_res_block', type=int, default=4)
    parser.add_argument('--n_res_channel', type=int, default=128)
    parser.add_argument('--n_out_res_block', type=int, default=0)
    parser.add_argument('--n_cond_res_block', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--amp', type=str, default='O0')
    parser.add_argument('--sched', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('path', type=str)

    parser.add_argument('--load_ps_ckpt', type=str)
    parser.add_argument('--curr_epoch', type=int, default=0)

    args = parser.parse_args()

    print("\n\n--------------------------\n", args, "\n-------------------------------------------")

    device = 'cuda'

    dataset = LMDBDataset(args.path)
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True
    )

    ckpt = {}
    curr_epoch = args.curr_epoch
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)
        args = ckpt['args']

    # TODO try the pixelsnail initialization over Mnist
    # model = PixelSNAIL([28, 28], 256, 128, 5, 2, 4, 128)
    # model = model.to(device)

    if args.hier == 'top':
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

    elif args.hier == 'bottom':
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

    # if args.load_ps_ckpt is not None:
    #     cur_ckpt = torch.load(args.load_ps_ckpt)
    #     model = PixelSNAIL.load_state_dict(cur_ckpt)

    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if amp is not None:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp)

    model = nn.DataParallel(model)
    model = model.to(device)

    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
        )
    d = range(curr_epoch, args.epoch)
    for i in range(curr_epoch, args.epoch):
        train(args, i, loader, model, optimizer, scheduler, device)
        torch.save(
            {'model': model.module.state_dict(), 'args': args},
            f'checkpoint/{args.hier}_label_cond_pixelsnail_mnist/pixelsnail_{args.hier}_{str(i + 1).zfill(3)}.pt',
        )
