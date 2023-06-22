import torch
from torch import nn
from torch.nn import functional as F

import distributed as dist_fn


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


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        # print("Quantize - ",input.size())
        flatten = input.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


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
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                # nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                # nn.ReLU(inplace=True),
                # nn.Conv2d(channel, channel, 3, padding=1),
                # 3 -> 16
                nn.Conv2d(in_channel, channel // 8, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                # 16 -> 32
                nn.Conv2d(channel // 8, channel//4, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                # ------------------------------------------------------------------------
                # 32 -> 64
                nn.Conv2d(channel//4, channel//2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # nn.Conv2d(channel * 2, channel * 4, 4, stride=2, padding=1),
                # nn.ReLU(inplace=True),
                #
                # # # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                # # nn.Conv2d(channel * 4, channel * 8, 4, stride=2, padding=1),
                # # nn.ReLU(inplace=True),
                # # nn.Conv2d(channel * 8, channel * 4, 3, padding=1),
                # # nn.ReLU(inplace=True),
                # # # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                # nn.Conv2d(channel * 4, channel * 2, 3, padding=1),
                # nn.ReLU(inplace=True),
                # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # channel  -->  channel * 2
                # 64 -> 128
                nn.Conv2d(channel//2, channel, 3, padding=1),
                # ------------------------------------------------------------------------
                # nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                # 16 -> 32
                nn.Conv2d(in_channel // 8, channel // 4, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                # 32 -> 64
                nn.Conv2d(channel // 4, channel//2, 3, padding=1),
                nn.ReLU(inplace=True),
                # 64 -> 128
                nn.Conv2d(channel // 2, channel, 3, padding=1),
                nn.ReLU(inplace=True),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        # if stride == 4 or stride == 2:
        #     # 128 ->64
        #     blocks.append(nn.Conv2d(channel, channel // 2, 3, padding=1))
        #     blocks.append(nn.ReLU(inplace=True))
        #     # 64 -> 32
        #     blocks.append(nn.Conv2d(channel // 2, channel // 4, 3, padding=1))
        #     blocks.append(nn.ReLU(inplace=True))
        #     # 32 -> 16
        #     blocks.append(nn.Conv2d(channel // 4, channel // 8, 3, padding=1))
        #     blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        # print("encoder input = ", input.size())
        enc_output = self.blocks(input)
        # print("encoder output = ",enc_output.size())
        return enc_output


class Decoder(nn.Module):
   # embed_dim=16, embed_dim, channel=128,
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]
        # blocks = []
        # if stride == 4:
        #     # 32 -> 16
        #     blocks.append(nn.Conv2d(in_channel, channel // 8, 3, padding=1))
        #     blocks.append(nn.ReLU(inplace=True))
        # if stride == 2:
        #     # 16 -> 16
        #     blocks.append(nn.Conv2d(in_channel, channel // 8, 3, padding=1))
        #     blocks.append(nn.ReLU(inplace=True))
        # # 16 ->64
        # blocks.append(nn.Conv2d(channel // 8, channel // 2, 3, padding=1))
        # blocks.append(nn.ReLU(inplace=True))
        # # 64 -> 128
        # blocks.append(nn.Conv2d(channel // 2, channel, 3, padding=1))
        # blocks.append(nn.ReLU(inplace=True))


        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        # if stride == 4:
        #     blocks.extend(
        #         [
        #             # ------------------------------------------------------------------------------------------
        #             # 128 -> 64
        #             nn.ConvTranspose2d(channel, channel//2, 4, stride=2, padding=1),
        #             nn.ReLU(inplace=True),
        #             # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #             # nn.Conv2d(channel * 2, channel * 4, 3, padding=1),
        #             # nn.ReLU(inplace=True),
        #             # # # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        #             # #
        #             # # nn.Conv2d(channel * 4, channel * 8, 3, padding=1),
        #             # # nn.ReLU(inplace=True),
        #             # # nn.ConvTranspose2d(channel * 8, channel * 4, 4, stride=2, padding=1),
        #             # # nn.ReLU(inplace=True),
        #             # # # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        #             #
        #             # nn.ConvTranspose2d(channel * 4, channel * 2, 4, stride=2, padding=1),
        #             # nn.ReLU(inplace=True),
        #             # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #             # 64 -> 32
        #             nn.Conv2d(channel//2, channel//4, 3, padding=1),
        #             nn.ReLU(inplace=True),
        #             # ---------------------------------------------------------------------------------------------
        #             # 32 -> 16
        #             nn.ConvTranspose2d(channel//4, channel//8, 4, stride=2, padding=1),
        #             nn.ReLU(inplace=True),
        #             # 16 -> 3
        #             nn.ConvTranspose2d(
        #                 channel // 8, out_channel, 4, stride=2, padding=1
        #             ),
        #         ]
        #     )
        #
        # elif stride == 2:
        #     # 128 -> 64
        #     blocks.append(nn.Conv2d(channel, channel//2, 3, padding=1))
        #     blocks.append(nn.ReLU(inplace=True))
        #     # 64 -> 32
        #     blocks.append(nn.Conv2d(channel//2, channel//4, 3, padding=1))
        #     blocks.append(nn.ReLU(inplace=True))
        #     # 32 -> 16
        #     blocks.append(
        #         nn.ConvTranspose2d(channel//4, out_channel, 4, stride=2, padding=1)
        #     )

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )


        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        # print("decoder input = ", input.size())
        dec_output = self.blocks(input)
        # print("decoder output = ", dec_output.size())
        return dec_output


class VQVAE(nn.Module):
    def __init__(
        self,
        # in_channel=3,
        in_channel=1,
        channel=128,
        # quantize_t_channel=16,
        quantize_t_channel=128,
        n_res_block=2,
        n_res_channel=32,
        # embed_dim=16,
        embed_dim=128,
        # n_embed=512,
        n_embed=32,
        decay=0.99,
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)
        self.quantize_conv_t = nn.Conv2d(quantize_t_channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + embed_dim, embed_dim, 1)
        self.quantize_b = Quantize(embed_dim, n_embed)
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1
        )
        self.dec = Decoder(
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )

    def forward(self, input):
        quant_t, quant_b, diff,code_b , code_t = self.encode(input)
        dec = self.decode(quant_t, quant_b)
        # print(self.enc_b)

        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        # print(f"1.\n\tquant_t = {quant_t.size()}, quant_b = {quant_b.size()}")
        upsample_t = self.upsample_t(quant_t)
        # print(f"2.\n\tupsample_t = {upsample_t.size()}, quant_b = {quant_b.size()}")
        quant = torch.cat([upsample_t, quant_b], 1)
        # print(f"3.\n\tquant = {quant.size()}")
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec