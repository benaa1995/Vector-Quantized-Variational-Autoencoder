import argparse
import os

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from vqvae import VQVAE
from pixelsnail_mnist import PixelSNAIL


@torch.no_grad()
def sample_model(model, device, batch, size, temperature, condition=None):
    row = torch.zeros(batch, *size, dtype=torch.int64).to(device)
    cache = {}

    for i in tqdm(range(size[0])):
        for j in range(size[1]):
            out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            sample = torch.multinomial(prob, 1).squeeze(-1)
            row[:, i, j] = sample

    return row


def load_model(model, checkpoint, device, label=None):
    ckpt = torch.load(os.path.join('checkpoint', checkpoint))

    
    if 'args' in ckpt:
        args = ckpt['args']

    if model == 'vqvae':
        model = VQVAE()

    elif model == 'pixelsnail_top':
        if label is not  None:
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
            # model = PixelSNAIL(
            #     [2, 2],
            #     32,
            #     args.channel,
            #     5,
            #     4,
            #     args.n_res_block,
            #     args.n_res_channel,
            #     # attention=False,
            #     dropout=args.dropout,
            #     n_cond_res_block=args.n_cond_res_block,
            #     cond_res_channel=args.n_res_channel,
            #     n_out_res_block=args.n_out_res_block,
            # )
        else:
            model = PixelSNAIL(
                [2, 2],
                32,
                args.channel,
                5,
                4,
                args.n_res_block,
                args.n_res_channel,
                dropout=args.dropout,
                n_out_res_block=args.n_out_res_block,
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


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--vqvae', type=str)
    parser.add_argument('--top', type=str)
    parser.add_argument('--bottom', type=str)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('filename', type=str)

    parser.add_argument('--label', type=int)
    # parser.add_argument('--condition', type=int)

    args = parser.parse_args()
    print(args)
    model_vqvae = load_model('vqvae', args.vqvae, device)
    model_top = load_model('pixelsnail_top', args.top, device, label=args.label)
    model_bottom = load_model('pixelsnail_bottom', args.bottom, device)
    if args.label is not None:
        top_cond = torch.full((args.batch, 1, 1), args.label)
        top_sample = sample_model(model_top, device, args.batch, [2, 2], args.temp, condition=top_cond)
    else:
        top_sample = sample_model(model_top, device, args.batch, [2, 2], args.temp)
    bottom_sample = sample_model(
        model_bottom, device, args.batch, [4, 4], args.temp, condition=top_sample
    )
    # print("\n\n----------------------------------------------------------")
    # print(top_sample)
    # print("----------------------------------------------------------\n\n")
    # print("\n\n----------------------------------------------------------")
    # print(bottom_sample)
    # print("----------------------------------------------------------\n\n")
    decoded_sample = model_vqvae.decode_code(top_sample, bottom_sample)
    decoded_sample = decoded_sample.clamp(-1, 1)

    save_image(decoded_sample, args.filename, normalize=True, range=(-1, 1))