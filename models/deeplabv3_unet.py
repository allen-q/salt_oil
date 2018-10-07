# camera-ready

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from .resnet_unet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8
from .aspp_unet import ASPP, ASPP_Bottleneck

class OutConv(nn.Module):
    def __init__(self, in_ch, logits=False):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1, padding=0))
        self.sig = nn.Sigmoid()
        self.logits = logits


    def forward(self, x):
        x_conv = self.conv(x)
        if not self.logits:
            x_out = self.sig(x_conv)
        else:
            x_out = x_conv

        crop_start = (x.shape[-1]-101)//2
        crop_end = crop_start + 101
        x_out = x_out[:,:,crop_start:crop_end,crop_start:crop_end].squeeze()

        return x_out


class Seq_Ex_Block(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, r):
        super(Seq_Ex_Block, self).__init__()
        self.se = nn.Sequential(
            GlobalAvgPool(),
            nn.Linear(in_ch, in_ch//r),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch//r, in_ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        se_weight = self.se(x).unsqueeze(-1).unsqueeze(-1)
        #print(f'x:{x.sum()}, x_se:{x.mul(se_weight).sum()}')
        return x.mul(se_weight)


class SqueezeTensor(nn.Module):
    def __init__(self):
        super(SqueezeTensor, self).__init__()
    def forward(self, x):
        return x.squeeze()

class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()
    def forward(self, x):
        return x.view(*(x.shape[:-2]),-1).mean(-1)


class Decoder(nn.Module):
    def __init__(self, in_ch, ch, out_ch, r=16, upsample=True):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ch, out_ch, kernel_size=3, padding=1)
        self.se = Seq_Ex_Block(out_ch, r)
        self.upsample = upsample

    def forward(self, x, x2=None, ):
        if self.upsample:
            x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        if x2 is not None:
            x = torch.cat([x, x2], 1)

        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = self.se(x)
        #print(x.shape)

        return x

class DeepLabV3(nn.Module):
    def __init__(self, resnet_layer, pretrained=True):
        super(DeepLabV3, self).__init__()
        assert resnet_layer in [18,34,50,101,152]
        resnet = {18:ResNet18_OS8, 34:ResNet34_OS8, 50:ResNet50_OS16,
                  101:ResNet101_OS16, 152:ResNet152_OS16}[resnet_layer]
        aspp_net = {18:ASPP, 34:ASPP, 50:ASPP_Bottleneck,
                    101:ASPP_Bottleneck, 152:ASPP_Bottleneck}[resnet_layer]
        self.num_classes = 1

        #self.resnet = ResNet34_OS8() # NOTE! specify the type of ResNet here
        self.resnet = resnet(pretrained=pretrained)
        self.aspp = aspp_net(num_classes=self.num_classes)

        self.center = nn.Sequential(
                nn.Conv2d(512,512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512,256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
                )

        self.decoder5 = Decoder(256+512, 512, 64)
        self.decoder4 = Decoder(64+256, 256, 64, upsample=False)
        self.decoder3 = Decoder(64+128, 128, 64, upsample=False)
        self.decoder2 = Decoder(64+64, 64, 64)
        self.decoder1 = Decoder(64, 32, 64)

        self.se_f = Seq_Ex_Block(576, 16)
        self.outc = OutConv(576, logits=True)

    def forward(self, x):
        # (x has shape (batch_size, 1, h, w))
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        x = torch.cat([
                (x-mean[2])/std[2],
                (x-mean[1])/std[1],
                (x-mean[0])/std[0],
                ], 1)

        e2,e3,e4,e5 = self.resnet(x)
#        e2.shape : [2, 64, 32, 32]
#        e3.shape:([2, 128, 16, 16]
#        e4.shape: [2, 256, 16, 16]
#        e5.shape: [2, 512, 16, 16]
        dlab = self.aspp(e5)   #[2, 256, 16, 16]

        c = self.center(e5)         #[2, 256, 8, 8]
        d5 = self.decoder5(c, e5)   #[2, 64, 16, 16]
        d4 = self.decoder4(d5, e4)  #[2, 64, 16, 16]
        d3 = self.decoder3(d4, e3)  #[2, 64, 16, 16]
        d2 = self.decoder2(d3, e2)  #[2, 64, 32, 32]
        d1 = self.decoder1(d2)      #[2, 64, 64, 64]


        output = torch.cat((
                F.upsample(dlab, scale_factor=8, mode="bilinear", align_corners=False),
                F.upsample(d1, scale_factor=2, mode='bilinear', align_corners=False),
                F.upsample(d2, scale_factor=4, mode='bilinear', align_corners=False),
                F.upsample(d3, scale_factor=8, mode='bilinear', align_corners=False),
                F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=False),
                F.upsample(d5, scale_factor=8, mode='bilinear', align_corners=False),
                ), 1)               #320, 128, 128
        output = self.se_f(output)

        output = F.dropout2d(output, p=0.5)
        output = self.outc(output)          #1, 101,101


        return output

