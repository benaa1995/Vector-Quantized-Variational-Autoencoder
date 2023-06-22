import torch
from torch import nn


# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel):
        super().__init__()

        blocks = [
            # 3 -> 8
            nn.Conv2d(in_channel, channel // 8, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 8 -> 16
            nn.Conv2d(channel // 8, channel // 4, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 16 -> 32
            nn.Conv2d(channel // 4, channel // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # channel  -->  channel * 2
            # 32 -> 64
            nn.Conv2d(channel // 2, channel, 3, padding=1),
        ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        # 64 ->32
        blocks.append(nn.Conv2d(channel, channel // 2, 3, padding=1))
        blocks.append(nn.ReLU(inplace=True))
        # 32 -> 16
        blocks.append(nn.Conv2d(channel // 2, channel // 4, 3, padding=1))
        blocks.append(nn.ReLU(inplace=True))
        # 16 -> 8
        blocks.append(nn.Conv2d(channel // 4, channel // 8, 3, padding=1))
        blocks.append(nn.ReLU(inplace=True))
        # 8 -> 4
        blocks.append(nn.Conv2d(channel // 8, channel // 16, 3, padding=1))
        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

        # adding the mean and log_var layers for the vae normal distribution

        self.z_mean = nn.Conv2d(channel // 16, 1, 3, padding=1)
        self.z_log_var = nn.Conv2d(channel // 16, 1, 3, padding=1)

    def reparameterize(self, z_mu, z_log_var):
        # print(f"z_mu.size() = {z_mu.size()}")
        # print(
        #     f"z_mu.size(0) = {z_mu.size(0)}, z_mu.size(1) = {z_mu.size(1)}, z_mu.size(2) = {z_mu.size(2)}, z_mu.size(3) = {z_mu.size(3)}")
        # print(f"z_log_var.get_device() = {z_log_var.get_device()}")
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        eps = torch.randn(z_mu.size(0), z_mu.size(1), z_mu.size(2), z_mu.size(3)).to(device)
        z = z_mu + eps * torch.exp(z_log_var / 2.)
        return z

    def forward(self, input):
        input = self.blocks(input)
        z_mean, z_log_var = self.z_mean(input), self.z_log_var(input)
        encoded_data = self.reparameterize(z_mean, z_log_var)
        return encoded_data, z_mean, z_log_var


class Decoder(nn.Module):

    def __init__(
            self, in_channel, out_channel, channel, n_res_block, n_res_channel):
        super().__init__()

        blocks = []
        # 1 -> 4
        blocks.append(nn.Conv2d(in_channel, channel // 16, 3, padding=1))
        blocks.append(nn.ReLU(inplace=True))
        # 4 -> 8
        blocks.append(nn.Conv2d(channel // 16, channel // 8, 3, padding=1))
        blocks.append(nn.ReLU(inplace=True))
        # 8 ->32
        blocks.append(nn.Conv2d(channel // 8, channel // 2, 3, padding=1))
        blocks.append(nn.ReLU(inplace=True))
        # 32 -> 64
        blocks.append(nn.Conv2d(channel // 2, channel, 3, padding=1))
        blocks.append(nn.ReLU(inplace=True))

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        blocks.extend(
            [
                # 64 -> 32
                nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                # 32 -> 16
                nn.Conv2d(channel // 2, channel // 4, 3, padding=1),
                nn.ReLU(inplace=True),
                # 16 -> 8
                nn.ConvTranspose2d(channel // 4, channel // 8, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                # 8 -> 3
                nn.ConvTranspose2d(channel // 8, out_channel, 4, stride=2, padding=1),
            ]
        )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        # print("decoder input = ", input.size())
        dec_output = self.blocks(input)
        # print("decoder output = ", dec_output.size())
        return dec_output


class VAE(nn.Module):
    def __init__(
            self,
            in_channel=3,
            channel=64,
            n_res_block=2,
            n_res_channel=32,
            embed_dim=1,
    ):
        super().__init__()

        self.enc = Encoder(in_channel, channel, n_res_block, n_res_channel)

        self.dec = Decoder(embed_dim, in_channel, channel, n_res_block, n_res_channel)

    def forward(self, input):
        enc, z_mean, z_log_var = self.enc(input)
        rec_input = self.dec(enc)

        return rec_input, 1