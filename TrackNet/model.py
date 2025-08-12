import torch
import torch.nn as nn
import torch.nn.functional as F


class TrackNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.conv1 = self.conv_block(9, 64)
        self.conv2 = self.conv_block(64, 64)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.conv3 = self.conv_block(64, 128)
        self.conv4 = self.conv_block(128, 128)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        
        self.conv5 = self.conv_block(128, 256)
        self.conv6 = self.conv_block(256,256)
        self.conv7 = self.conv_block(256, 256)
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.conv8 = self.conv_block(256, 512)
        self.conv9 = self.conv_block(512, 512)
        self.conv10 = self.conv_block(512, 512)

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv11 = self.conv_block(512, 512)
        self.conv12 = self.conv_block(512, 512)
        self.conv13 = self.conv_block(512, 512)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv14 = self.conv_block(512, 256)
        self.conv15 = self.conv_block(256, 256)
        self.conv16 = self.conv_block(256, 256)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv17 = self.conv_block(256, 128)
        self.conv18 = self.conv_block(128, 128)

        # Final layer with softmax
        self.final_conv = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=1)


    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)

        # Decoder
        x = self.up1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)

        x = self.up2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)

        x = self.up3(x)
        x = self.conv17(x)
        x = self.conv18(x)

        x = self.final_conv(x)
        x = self.softmax(x)

        return x
                
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
