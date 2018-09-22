# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class Seq_Ex_Block_C(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, r):
        super(Seq_Ex_Block_C, self).__init__()
        # channel squeeze
        self.cs = nn.Sequential(
            GlobalAvgPool(),
            nn.Linear(in_ch, in_ch//r),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch//r, in_ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        cs_weight = self.cs(x).unsqueeze(-1).unsqueeze(-1)
        #print(f'x:{x.sum()}, x_se:{x.mul(se_weight).sum()}')
        return x.mul(cs_weight)


class Seq_Ex_Block_S(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch):
        super(Seq_Ex_Block_S, self).__init__()
        # spacial squeeze
        self.ss = nn.Sequential(
            nn.Conv2d(in_ch, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        ss_weight = self.ss(x)
        #print(f'x:{x.sum()}, x_se:{x.mul(se_weight).sum()}')
        return x.mul(ss_weight)


class Seq_Ex_Block_CS(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, r):
        super(Seq_Ex_Block_CS, self).__init__()
        # spacial squeeze
        self.ss = Seq_Ex_Block_S(in_ch)
        self.cs = Seq_Ex_Block_C(in_ch, r)


    def forward(self, x):
        out = self.ss(x) + self.cs(x)
        #print(f'x:{x.sum()}, x_se:{x.mul(se_weight).sum()}')
        return out


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


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        #print(f'in: {x.shape}')
        x = self.mpconv(x)
        #print(f'out: {x.shape}')
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, trans_in_ch, bilinear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights

        if bilinear:
            print('Using bilinear for upsampling')
            self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            print('Using transpose conv for upsampling')
            self.upscale = nn.ConvTranspose2d(trans_in_ch, trans_in_ch, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        #print(f'in: {x1.shape}, {x2.shape}')
        x1 = self.upscale(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        #print(f'out: {x.shape}')
        return x


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




def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



def resnet18unet(in_ch=1, bilinear=True, pretrained=False, **kwargs):
    """Constructs a ResNet-18 Unet model.
    Args:

    """
    model = UResNet(BasicBlock, [2, 2, 2, 2], in_ch=in_ch, bilinear=bilinear, **kwargs)
    if pretrained:
        print('Using pre-trained model')
        trained_model = torchvision.models.resnet18(pretrained=True)
        params_trained = trained_model.named_parameters()
        params_new = model.named_parameters()
        dict_params_new = dict(params_new)
        for name1, param1 in params_trained:
            if name1 in dict_params_new:
                dict_params_new[name1].data.copy_(param1.data)

    return model


def resnet34unet(in_ch=1, bilinear=True, pretrained=False, **kwargs):
    """Constructs a ResNet-18 Unet model.
    Args:

    """
    model = UResNet(BasicBlock, [3, 4, 6, 3], in_ch=in_ch, bilinear=bilinear, **kwargs)
    if pretrained:
        print('Using pre-trained model')
        trained_model = torchvision.models.resnet34(pretrained=True)
        params_trained = trained_model.named_parameters()
        params_new = model.named_parameters()
        dict_params_new = dict(params_new)
        for name1, param1 in params_trained:
            if name1 in dict_params_new:
                dict_params_new[name1].data.copy_(param1.data)

    return model


def resnet50unet(in_ch=1, bilinear=True, pretrained=False, **kwargs):
    """Constructs a ResNet-18 Unet model.
    Args:

    """
    model = UResNet(Bottleneck, [3, 4, 6, 3], in_ch=in_ch, bilinear=bilinear, **kwargs)
    if pretrained:
        print('Using pre-trained model')
        trained_model = torchvision.models.resnet50(pretrained=True)
        params_trained = trained_model.named_parameters()
        params_new = model.named_parameters()
        dict_params_new = dict(params_new)
        for name1, param1 in params_trained:
            if name1 in dict_params_new:
                dict_params_new[name1].data.copy_(param1.data)

    return model

def resnet101unet(in_ch=1, bilinear=True, pretrained=False, **kwargs):
    """Constructs a ResNet-18 Unet model.
    Args:

    """
    model = UResNet(Bottleneck, [3, 4, 23, 3], in_ch=in_ch, bilinear=bilinear, **kwargs)
    if pretrained:
        print('Using pre-trained model')
        trained_model = torchvision.models.resnet101(pretrained=True)
        params_trained = trained_model.named_parameters()
        params_new = model.named_parameters()
        dict_params_new = dict(params_new)
        for name1, param1 in params_trained:
            if name1 in dict_params_new:
                dict_params_new[name1].data.copy_(param1.data)

    return model

def resnet152unet(in_ch=1, bilinear=True, pretrained=False, **kwargs):
    """Constructs a ResNet-18 Unet model.
    Args:

    """
    model = UResNet(Bottleneck, [3, 8, 36, 3], in_ch=in_ch, bilinear=bilinear, **kwargs)
    if pretrained:
        print('Using pre-trained model')
        trained_model = torchvision.models.resnet152(pretrained=True)
        params_trained = trained_model.named_parameters()
        params_new = model.named_parameters()
        dict_params_new = dict(params_new)

        for name1, param1 in params_trained:
            if name1 in dict_params_new:
                dict_params_new[name1].data.copy_(param1.data)

    return model
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


class UResNet(nn.Module):
    def __init__(self,pretrained=False):
        print(f'ResNet{"" if pretrained else "not"} using pretrained weights.')
        super(UResNet, self).__init__()
        self.resnet = resnet34(pretrained=pretrained)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            #self.resnet.maxpool
            )

        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4

        self.center = nn.Sequential(
                nn.Conv2d(512,512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512,256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                Seq_Ex_Block_CS(256, 16)
                )

        self.decoder5 = Decoder(256, 512, 32, 32, 16, 16)    #torch.Size([2, 32, 128, 128])
        self.decoder4 = Decoder(32, 256, 32, 1, 8, 16)       #torch.Size([2, 32, 128, 128])
        self.decoder3 = Decoder(32, 128, 32, 1, 4, 16)       #torch.Size([2, 32, 128, 128])
        self.decoder2 = Decoder(32, 64, 32, 1, 2, 16)        #torch.Size([2, 32, 128, 128])
        self.decoder1 = Decoder(32, 64, 32, 1, 2, 16)        #torch.Size([2, 32, 128, 128])

        self.secs_f = Seq_Ex_Block_CS(32, 16)

        self.outc = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1, padding=0))

        self.outc = OutConv(32, logits=True)
    def forward(self, x):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        x = torch.cat([
                (x-mean[2])/std[2],
                (x-mean[1])/std[1],
                (x-mean[0])/std[0],
                ], 1)

        x = self.conv1(x)           #64, 64, 64
        e2 = self.encoder2(x)       #64, 64, 64
        e3 = self.encoder3(e2)      #128, 32, 32
        e4 = self.encoder4(e3)      #256, 16, 16
        e5 = self.encoder5(e4)      #512, 8, 8

        f = self.center(e5)         #256, 4, 4
        d5 = self.decoder5(f, e5)   #torch.Size([2, 32, 128, 128])
        d4 = self.decoder4(d5, e4)  #torch.Size([2, 32, 128, 128])
        d3 = self.decoder3(d4, e3)  #torch.Size([2, 32, 128, 128])
        d2 = self.decoder2(d3, e2)  #torch.Size([2, 32, 128, 128])
        d1 = self.decoder1(d2, x)   #torch.Size([2, 32, 128, 128])


#        f = torch.cat((
#                d1,
#                F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=False),
#                F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=False),
#                F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=False),
#                F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=False),
#                ), 1)               #320, 128, 128
        f = self.secs_f(d1)
        f = F.dropout2d(f, p=0.5)

        out = self.outc(f)          #1, 101,101

        return out


class Decoder(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch, scale_factor1, scale_factor2, r=16):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_ch1, out_ch, 1)
        self.conv2 = nn.Sequential(
                conv3x3(in_ch2, out_ch, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                conv3x3(out_ch, out_ch),
                nn.BatchNorm2d(out_ch))
        if scale_factor1 > 1:
            self.upsampler1 = nn.Upsample(scale_factor=scale_factor1, mode='bilinear', align_corners=False)
        else:
            self.upsampler1 = None
        self.upsampler2 = nn.Upsample(scale_factor=scale_factor2, mode='bilinear', align_corners=False)
        self.relu = nn.ReLU(inplace=True)
        self.secs = Seq_Ex_Block_CS(out_ch, r)

    def forward(self, x, x2=None):
        x1 = self.conv1(x)
        if self.upsampler1 is not None:
            x1 = self.upsampler1(x1)
        x2 = self.conv2(x2)
        x2 = self.upsampler2(x2)
        x2 = self.secs(x2)

        out = x1 + x2
        out = self.relu(out)

        return out
