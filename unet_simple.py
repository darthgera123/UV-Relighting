import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_params < 1000:
        return f"{total_params} parameters"
    elif total_params < 1_000_000:
        return f"{total_params / 1_000:.1f}K parameters"  # Kilos
    else:
        return f"{total_params / 1_000_000:.1f}M parameters"  # Millions

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(scale_factor),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, scale_factor=2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=scale_factor, stride=scale_factor)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a

        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False, activation=False, ):
        super(UNet, self).__init__()
        self.n_channels = in_channels
        # self.sh_degree = args.sh_degree
        # self.learn_offset = args.learn_offset
        self.out_channels = out_channels
        self.bilinear = bilinear

        factor = 2 if bilinear else 1
    
        self.inc = (DoubleConv(in_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, self.out_channels))
        # else:
        #     self.inc = (DoubleConv(in_channels, 128))
        #     self.down1 = (Down(128, 256, scale_factor=4))
        #     self.down2 = (Down(256, 512, scale_factor=4))
        #     self.down3 = (Down(512, 1024, scale_factor=4))
        #     self.down4 = (Down(1024, 2048 // factor, scale_factor=4))
        #     self.up1 = (Up(2048, 1024 // factor, bilinear, scale_factor=4))
        #     self.up2 = (Up(1024, 512 // factor, bilinear, scale_factor=4))
        #     self.up3 = (Up(512, 256 // factor, bilinear, scale_factor=4))
        #     self.up4 = (Up(256, 256, bilinear, scale_factor=4))
        #     self.outc = (OutConv(256, self.out_channels))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

if __name__ == "__main__":

    # [1,3,1024,1280]  #[1,10,20,3]
    relit = UNet(in_channels=6,out_channels=59)
    source = torch.rand([4,6,256,256])
    # target_light = torch.rand([4,3,16,32])
    output = relit(source)
    print(output.shape)
    print(count_parameters(relit))