import torch
import numpy as np


class ConvNormAct(torch.nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, act='prelu'):
        super().__init__()

        self.conv = torch.nn.Conv2d(
            input_channel, output_channel,
            kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2
        )
        self.norm = torch.nn.GroupNorm(min(32, output_channel), output_channel)
        if act == 'prelu':
            self.act = torch.nn.PReLU(output_channel)
        elif act == 'softplus':
            self.act = torch.nn.Softplus()
        elif act == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))



class PortraitRelightingNet(torch.nn.Module):
    def __init__(self, light_size):
        super().__init__()
        self.light_size = light_size

        self.channel_nums = [3, 32, 64, 64, 128, 128, 256, 256, 512, 512, 512]
        
        self.image_encoders = torch.nn.ModuleList([
            ConvNormAct(
                self.channel_nums[i],
                self.channel_nums[i+1] - (self.channel_nums[i] if i == 0 else 0),
                7 if i == 0 else 3,
                2 if i in [1, 3, 5, 7] else 1
            )
            for i in range(len(self.channel_nums) - 1)
        ])
        self.light_decoders = torch.nn.ModuleList([
            ConvNormAct(512, 512, 3, 1),
            ConvNormAct(512, 4 * np.prod(light_size), 3, 1, act='softplus')
        ])
        self.light_encoders = torch.nn.ModuleList([
            ConvNormAct(3 * np.prod(light_size), 512, 1, 1), ConvNormAct(512, 256, 1, 1)
        ])
        self.image_decoders = torch.nn.ModuleList([
            ConvNormAct(
                self.channel_nums[i] + (self.channel_nums[i] if i < len(self.channel_nums) - 1 else 256),
                self.channel_nums[i-1],
                3,
                1,
                'prelu' if i != 1 else 'sigmoid'
            )
            for i in range(len(self.channel_nums) - 1, 0, -1)
        ])
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        


    def forward(self, source_image, target_light):
        x = source_image
        b,c,h,w = source_image.shape
        encoders_features = []
        for i, block in enumerate(self.image_encoders):
            x = block(x) if i != 0 else torch.cat([x, block(x)], axis=1)
            encoders_features.append(x)

        for block in self.light_decoders:
            x = block(x)
        # print(x.shape)
        
        light = x[:, :np.prod(self.light_size) * 3, :, :].reshape(1, np.prod(self.light_size), 3, -1)
        confidence = x[:, np.prod(self.light_size) * 3:, :, :].reshape(1, np.prod(self.light_size), 1, -1)
        
        source_light = torch.sum(light * confidence, axis=-1) / torch.sum(confidence, axis=-1)
        # print(source_light.shape)
        if target_light.ndim <= 2:
            x = torch.roll(
                source_light.view(*self.light_size, 3),
                -target_light.squeeze().int().item(),
                dims=1
            ).view(b, -1, 1, 1)
        else:
            x = target_light.reshape(b, -1, 1, 1)
        
        for block in self.light_encoders:
            x = block(x)
        # print(x.shape)
        x = x.expand(-1, -1, *encoders_features[-1].shape[2:])
        for i, block in enumerate(self.image_decoders):
            x = torch.cat([x, encoders_features[-i-1]], axis=1)
            if i in [2, 4, 6, 8]:
                x = self.upsample(x)
            x = block(x)


        return x
        
if __name__ == "__main__":

    # [1,3,1024,1280]  #[1,10,20,3]
    relit = PortraitRelightingNet(light_size=(16,32))
    source = torch.rand([4,3,512,512])
    target_light = torch.rand([4,3,16,32])
    output = relit(source,target_light)
    print(output.shape)