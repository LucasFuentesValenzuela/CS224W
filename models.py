import torch
from torch import nn
from typing import Tuple, Optional


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ksize: int = 5,
        kstride: int = 2,
    ):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=kstride,
            padding=(ksize-1)//2,
            padding_mode='replicate',
        )
        self.relu1 = nn.LeakyReLU(0.2)
        # self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor: #-> Tuple[torch.Tensor, torch.Tensor]:
        n, c, h, w = x.shape
        assert c == self.in_channels
        out = self.conv1(x)
        out = self.relu1(out)
        return out
        # out_small = self.pool(out)
        # return out_small, out


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_p: float,
        up_ksize: int = 5,
        up_kstride: int = 2,
        ksize: int = 3,
        kstride: int = 1,
        use_relu: bool = True,
    ):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_relu = use_relu

        self.upres = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=up_ksize,
            stride=up_kstride,
            padding=(up_ksize-1)//2,
            output_padding=1,
            )
        self.conv1 = nn.Conv2d(
            out_channels * 2,
            out_channels,
            kernel_size=ksize,
            stride=kstride,
            padding=(ksize-1)//2,
            padding_mode='replicate'
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=ksize,
            stride=kstride,
            padding=(ksize-1)//2,
            padding_mode='replicate'
        )
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x_small: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        n_small, c_small, h_small, w_small = x_small.shape
        assert c_small == self.in_channels
        x_big = self.upres(x_small)
        x_big = self.relu1(x_big)
        assert x_big.shape == x.shape
        x_cat = torch.cat([x_big, x], dim=1)
        out = self.conv1(x_cat)
        out = self.relu2(out)
        out = self.conv2(out)

        if self.use_relu:
            out = self.relu3(out)
        out = self.dropout(out)
        return out


class UNet(nn.Module):
    def __init__(self, num_features: Optional[int] = None, drop_p: float = 0.5):
        super(UNet, self).__init__()
        layer_channels = [1, 16, 32, 64, 128, 256 , 512]

        self.encoder1 = Encoder(1, 16)
        self.encoder2 = Encoder(16, 32)
        self.encoder3 = Encoder(32, 64)
        self.encoder4 = Encoder(64, 128)
        self.encoder5 = Encoder(128, 256)
        self.encoder6 = Encoder(256, 512)

        self.decoder6 = Decoder(512, 256, drop_p)
        self.decoder5 = Decoder(256, 128, drop_p)
        self.decoder4 = Decoder(128, 64, drop_p)
        self.decoder3 = Decoder(64, 32, 0)
        self.decoder2 = Decoder(32, 16, 0)
        self.decoder1 = Decoder(16, 1, 0, use_relu=False) # A little questionable, maybe should be conv layer

        self.sigmoid = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dtype == torch.float32
        assert len(x.shape) == 4
        e_out1 = self.encoder1(x)
        e_out2 = self.encoder2(e_out1)
        e_out3 = self.encoder3(e_out2)
        e_out4 = self.encoder4(e_out3)
        e_out5 = self.encoder5(e_out4)
        e_out6 = self.encoder6(e_out5)

        d_out5 = self.decoder6(e_out6, e_out5)
        d_out4 = self.decoder5(d_out5, e_out4)
        d_out3 = self.decoder4(d_out4, e_out3)
        d_out2 = self.decoder3(d_out3, e_out2)
        d_out1 = self.decoder2(d_out2, e_out1)
        out    = self.decoder1(d_out1, x) # Not sure on this part lol

        out = self.sigmoid(out)
        return out, x - out


def get_model(model_name: str) -> type:
    models = [UNet]
    for m in models:
        if m.__name__ == model_name:
            return m
    assert False, f'Could not find model {model_name}!'
