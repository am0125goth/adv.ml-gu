import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class PixelPredictModel_SegNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        #encoder blocks
        self.enc1 = ConvBlock(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)

        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)

        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)

        self.enc4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2, return_indices=True)

        #bottleneck
        self.bottleneck = ConvBlock(512, 512)

        #decoder blocks
        self.unpooling4 = nn.MaxUnpool2d(2, 2)
        self.dec4 = ConvBlock(512, 256)

        self.unpooling3 = nn.MaxUnpool2d(2, 2)
        self.dec3 = ConvBlock(256, 128)

        self.unpooling2 = nn.MaxUnpool2d(2, 2)
        self.dec2 = ConvBlock(128, 64)

        self.unpooling1 = nn.MaxUnpool2d(2, 2)
        self.dec1 = ConvBlock(64, 64)

        #output
        self.out_conv = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        #encoder
        e1 = self.enc1(x)
        e1_pooled, idx1 = self.pool1(e1)

        e2 = self.enc2(e1_pooled)
        e2_pooled, idx2 = self.pool2(e2)

        e3 = self.enc3(e2_pooled)
        e3_pooled, idx3 = self.pool3(e3)

        e4 = self.enc4(e3_pooled)
        e4_pooled, idx4 = self.pool4(e4)

        #bottleneck
        b = self.bottleneck(e4_pooled)

        #decoder
        d4 = self.unpooling4(b, idx4, output_size=e4.size())
        d4 = self.dec4(d4)

        d3 = self.unpooling3(d4, idx3, output_size=e3.size())
        d3 = self.dec3(d3)

        d2 = self.unpooling2(d3, idx2, output_size=e2.size())
        d2 = self.dec2(d2)

        d1 = self.unpooling1(d2, idx1, output_size=e1.size())
        d1 = self.dec1(d1)

        #output
        out = self.out_conv(d1)
        out = torch.sigmoid(out)
        return out

