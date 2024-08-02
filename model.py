import torch
import torch.nn as nn
import torch.nn.functional as F

class UpsampleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_scale=2):
        super(UpsampleConvBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=upsample_scale, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

class UVRelit(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(UVRelit, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        # Encoder
        self.enc_conv0 = nn.Conv2d(in_channels=16*32*3+self.in_ch, out_channels=64, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm2d(64)
        self.enc_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.enc_conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.enc_conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.enc_conv4 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)  # New encoder layer
        self.bn4 = nn.BatchNorm2d(1024)
        
        
        # Max pool
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder with Upsample and Convolution Block
        self.dec_upconv4 = UpsampleConvBlock(1024, 512)  # New decoder upsample layer
        self.dec_conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dec_bn4 = nn.BatchNorm2d(512)
        self.dec_upconv3 = UpsampleConvBlock(512+512, 256)  # Corrected for concatenated skip connection
        self.dec_conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dec_bn3 = nn.BatchNorm2d(256)
        self.dec_upconv2 = UpsampleConvBlock(256 + 256, 128)
        self.dec_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(128)
        self.dec_upconv1 = UpsampleConvBlock(128 + 128, 64)
        self.dec_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(64)
        
        # Final output
        self.final_conv = nn.Conv2d(64, self.out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        e0 = F.relu(self.bn0(self.enc_conv0(x)))
        p0 = self.pool(e0)
        e1 = F.relu(self.bn1(self.enc_conv1(p0)))
        p1 = self.pool(e1)
        e2 = F.relu(self.bn2(self.enc_conv2(p1)))
        p2 = self.pool(e2)
        e3 = F.relu(self.bn3(self.enc_conv3(p2)))
        p3 = self.pool(e3)
        e4 = F.relu(self.bn4(self.enc_conv4(p3)))  # New encoder output
        p4 = self.pool(e4)
        
        # Decoder with skip connections
        # d3 = self.dec_upconv3(e3)
        d4 = self.dec_upconv4(e4)  # Process new encoder layer
        d4 = F.relu(self.dec_bn4(self.dec_conv4(d4)))
        d3 = self.dec_upconv3(torch.cat([d4, e3], dim=1))
        d3 = F.relu(self.dec_bn3(self.dec_conv3(d3)))
        d2 = self.dec_upconv2(torch.cat([d3, e2], dim=1))
        d2 = F.relu(self.dec_bn2(self.dec_conv2(d2)))
        d1 = self.dec_upconv1(torch.cat([d2, e1], dim=1))
        d1 = F.relu(self.dec_bn1(self.dec_conv1(d1)))
        
        # Final output
        out = self.final_conv(d1)
        
        return out

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_params < 1000:
        return f"{total_params} parameters"
    elif total_params < 1_000_000:
        return f"{total_params / 1_000:.1f}K parameters"  # Kilos
    else:
        return f"{total_params / 1_000_000:.1f}M parameters"  # Millions




if __name__ == "__main__":
    relit = UVRelit(in_ch=3,out_ch=59)
    tensor_a = torch.rand(1, 3, 16, 32)  # Tensor to be flattened and repeated
    tensor_b = torch.rand(1, 3, 256, 256)  # Target tensor

    # Step 1: Flatten tensor_a
    tensor_a_flattened = tensor_a.flatten(start_dim=1)  # Resulting shape: [1, 1536]

    # Step 2 & 3: Calculate repeat factors and reshape
    # Calculate how many repeats are needed to match the spatial size of tensor B
    repeat_factor_width = tensor_b.size(3)  # 256 (width of tensor B)
    repeat_factor_height = tensor_b.size(2)  # 256 (height of tensor B)
    
    # Repeat the flattened tensor to match the spatial area of tensor B and then reshape
    tensor_a_expanded = tensor_a_flattened.repeat(1, repeat_factor_height * repeat_factor_width).view(1,-1,256,256)
    
    
    # Step 4: Concatenate with tensor B in the channel dimension
    result_tensor = torch.cat((tensor_b, tensor_a_expanded), dim=1)
    
    output = relit(result_tensor)
    # print(output.shape)
    print(output.shape)
    print(count_parameters(relit))


