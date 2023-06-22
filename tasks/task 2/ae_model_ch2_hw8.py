import torch
from torch import nn
from torch.nn import functional as F

# import distributed as dist_fn


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
            nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # ------------------------------------------------------------------------
            nn.Conv2d(channel, channel * 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            nn.Conv2d(channel * 2, channel * 4, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            nn.Conv2d(channel * 4, channel * 8, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel * 8, channel * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
            nn.Conv2d(channel * 4, channel * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # channel  -->  channel * 2
            nn.Conv2d(channel * 2, channel, 3, padding=1),
            # ------------------------------------------------------------------------
            # nn.Conv2d(channel, channel, 3, padding=1),
        ]


        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        blocks.append(nn.Conv2d(channel, channel // 2, 3, padding=1))
        blocks.append(nn.ReLU(inplace=True))
        blocks.append(nn.Conv2d(channel // 2, channel // 4, 3, padding=1))
        blocks.append(nn.ReLU(inplace=True))
        blocks.append(nn.Conv2d(channel // 4, channel // 8, 3, padding=1))
        blocks.append(nn.ReLU(inplace=True))
        blocks.append(nn.Conv2d(channel // 8, channel // 16, 3, padding=1))
        blocks.append(nn.ReLU(inplace=True))
        blocks.append(nn.Conv2d(channel // 16, channel // 32, 3, padding=1))
        blocks.append(nn.ReLU(inplace=True))
        blocks.append(nn.Conv2d(channel // 32, channel // 64, 3, padding=1))
        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        # print("encoder_input = ", input.size())
        output = self.blocks(input)
        # print("encoder_output = ", output.size())
        return output


class Decoder(nn.Module):
    def __init__(
            self, in_channel, out_channel, channel, n_res_block, n_res_channel
    ):
        super().__init__()

        # blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]
        blocks = []

        blocks.append(nn.Conv2d(channel // 64, channel // 32, 3, padding=1))
        blocks.append(nn.ReLU(inplace=True))
        blocks.append(nn.Conv2d(channel // 32, channel // 16, 3, padding=1))
        blocks.append(nn.ReLU(inplace=True))
        blocks.append(nn.Conv2d(channel // 16, channel // 8, 3, padding=1))
        blocks.append(nn.ReLU(inplace=True))
        blocks.append(nn.Conv2d(channel // 8, channel // 4, 3, padding=1))
        blocks.append(nn.ReLU(inplace=True))
        blocks.append(nn.Conv2d(channel // 4, channel // 2, 3, padding=1))
        blocks.append(nn.ReLU(inplace=True))
        blocks.append(nn.Conv2d(channel // 2, channel, 3, padding=1))
        blocks.append(nn.ReLU(inplace=True))

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        blocks.extend(
            [
                # ------------------------------------------------------------------------------------------
                nn.ConvTranspose2d(channel, channel * 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                nn.Conv2d(channel * 2, channel * 4, 3, padding=1),
                nn.ReLU(inplace=True),
                # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

                nn.Conv2d(channel * 4, channel * 8, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel * 8, channel * 4, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

                nn.ConvTranspose2d(channel * 4, channel * 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                nn.Conv2d(channel * 2, channel, 3, padding=1),
                nn.ReLU(inplace=True),
                # ---------------------------------------------------------------------------------------------
                nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    channel // 2, out_channel, 4, stride=2, padding=1
                ),
            ]
        )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        # print("decoder_input = ", input.size())
        output = self.blocks(input)
        # print("decoder_output = ", output.size())
        return output


class AE(nn.Module):
    def __init__(
            self,
            in_channel=3,
            channel=128,
            n_res_block=2,
            n_res_channel=32,
            embed_dim=64,
    ):
        super().__init__()

        self.enc = Encoder(in_channel, channel, n_res_block, n_res_channel)

        self.dec = Decoder(embed_dim, in_channel, channel, n_res_block, n_res_channel)

    def forward(self, input):
        enc = self.enc(input)
        # print("\nenc_b size = ",enc_b.size())
        rec_input = self.dec(enc)

        return rec_input, 1

