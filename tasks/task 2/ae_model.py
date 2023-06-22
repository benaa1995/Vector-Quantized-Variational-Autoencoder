import torch
from torch import nn
from torch.nn import functional as F


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
            # 3 -> 16
            nn.Conv2d(in_channel, channel // 8, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 16 -> 32
            nn.Conv2d(channel // 8, channel // 4, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 32 -> 64
            nn.Conv2d(channel // 4, channel // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # channel  -->  channel * 2
            # 64 -> 128
            nn.Conv2d(channel // 2, channel, 3, padding=1),
        ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        # channels - 128 ->64
        blocks.append(nn.Conv2d(channel, channel // 2, 3, padding=1))
        blocks.append(nn.ReLU(inplace=True))
        # channels - 64 -> 32
        blocks.append(nn.Conv2d(channel // 2, channel // 4, 3, padding=1))
        blocks.append(nn.ReLU(inplace=True))
        # channels - 32 -> 16
        blocks.append(nn.Conv2d(channel // 4, channel // 8, 3, padding=1))
        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        # print("encoder input = ", input.size())
        enc_output = self.blocks(input)
        # print("encoder output = ",enc_output.size())
        return enc_output


class Decoder(nn.Module):
    def __init__(
            self, in_channel, out_channel, channel, n_res_block, n_res_channel
    ):
        super().__init__()

        blocks = []
        # if stride == 4:
        #     # 32 -> 16
        #     blocks.append(nn.Conv2d(in_channel, channel // 8, 3, padding=1))
        #     blocks.append(nn.ReLU(inplace=True))

        # 16 -> 16
        blocks.append(nn.Conv2d(in_channel, channel // 8, 3, padding=1))
        blocks.append(nn.ReLU(inplace=True))
        # 16 ->64
        blocks.append(nn.Conv2d(channel // 8, channel // 2, 3, padding=1))
        blocks.append(nn.ReLU(inplace=True))
        # 64 -> 128
        blocks.append(nn.Conv2d(channel // 2, channel, 3, padding=1))
        blocks.append(nn.ReLU(inplace=True))

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        blocks.extend(
            [
                # 128 -> 64
                nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                # 64 -> 32
                nn.Conv2d(channel // 2, channel // 4, 3, padding=1),
                nn.ReLU(inplace=True),
                # 32 -> 16
                nn.ConvTranspose2d(channel // 4, channel // 8, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                # 16 -> 3
                nn.ConvTranspose2d(channel // 8, out_channel, 4, stride=2, padding=1),
            ]
        )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        # print("decoder input = ", input.size())
        dec_output = self.blocks(input)
        # print("decoder output = ", dec_output.size())
        return dec_output


class AE(nn.Module):
    def __init__(
            self,
            in_channel=3,
            channel=128,
            n_res_block=2,
            n_res_channel=32,
            embed_dim=16,
    ):
        super().__init__()

        self.enc = Encoder(in_channel, channel, n_res_block, n_res_channel)
        self.dec = Decoder(embed_dim, in_channel, channel, n_res_block, n_res_channel)

    def forward(self, input):
        enc = self.enc(input)
        # print("\nenc size = ",enc.size())
        rec_input = self.dec(enc)
        # print(self.enc)

        return rec_input, 1

