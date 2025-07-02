import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 16)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = self.conv_block(16, 32)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = self.conv_block(32, 64)
        self.pool3 = nn.MaxPool3d(2)

        # Bridge
        self.bridge = self.conv_block(64, 128)

        # Decoder
        self.up5 = self.up_conv(128, 64)
        self.dec5 = self.conv_block(128, 64)
        self.up6 = self.up_conv(64, 32)
        self.dec6 = self.conv_block(64, 32)
        self.up7 = self.up_conv(32, 16)
        self.dec7 = self.conv_block(32, 16)

        # Output layer
        self.out = nn.Conv3d(16, out_channels, kernel_size=1, activation=nn.Sigmoid())

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def up_conv(self, in_channels, out_channels):
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))

        # Bridge
        bridge = self.bridge(self.pool3(enc3))

        # Decoder
        up5 = self.up5(bridge)
        dec5 = self.dec5(torch.cat([up5, enc3], dim=1))
        up6 = self.up6(dec5)
        dec6 = self.dec6(torch.cat([up6, enc2], dim=1))
        up7 = self.up7(dec6)
        dec7 = self.dec7(torch.cat([up7, enc1], dim=1))

        return self.out(dec7)
