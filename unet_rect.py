import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            DoubleConv(3, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(256, 512),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            DoubleConv(512, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            DoubleConv(256, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            DoubleConv(128, 64),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2)
        )

    def forward(self, x):
        # Encoder pass
        x1 = self.encoder[0](x)
        x2 = self.encoder[1](x1)
        x3 = self.encoder[2](x2)
        x4 = self.encoder[3](x3)
        x5 = self.encoder[4](x4)
        print(x5.shape)
        # Decoder pass
        x = self.decoder[0](x5)
        x = self.decoder[1](torch.cat([x, x4], dim=1))
        x = self.decoder[2](x)
        x = self.decoder[3](torch.cat([x, x3], dim=1))
        x = self.decoder[4](x)
        x = self.decoder[5](torch.cat([x, x2], dim=1))
        x = self.decoder[6](x)
        
        return x

if __name__ == "__main__":
# Test the network
    model = UNet()
    input_tensor = torch.randn(1, 3, 256, 512)  # Batch size 1, 3 channels, 256x512 size
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)
