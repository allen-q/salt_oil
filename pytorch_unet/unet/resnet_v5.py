# add Non Local Neural Network to resnet v1

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

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, mode='embedded_gaussian',
                 sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]
        assert mode in ['embedded_gaussian', 'gaussian', 'dot_product', 'concatenation']

        # print('Dimension: %d, mode: %s' % (dimension, mode))

        self.mode = mode
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 4
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = None
        self.phi = None
        self.concat_project = None

        if mode in ['embedded_gaussian', 'dot_product', 'concatenation']:
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                 kernel_size=1, stride=1, padding=0)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

            if mode == 'embedded_gaussian':
                self.operation_function = self._embedded_gaussian
            elif mode == 'dot_product':
                self.operation_function = self._dot_product
            elif mode == 'concatenation':
                self.operation_function = self._concatenation
                self.concat_project = nn.Sequential(
                    nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
                    nn.ReLU()
                )
        elif mode == 'gaussian':
            self.operation_function = self._gaussian

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
            if self.phi is None:
                self.phi = max_pool(kernel_size=2)
            else:
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        output = self.operation_function(x)
        return output

    def _embedded_gaussian(self, x):
        batch_size = x.size(0)

        # g=>(b, c, t, h, w)->(b, 0.5c, t, h, w)->(b, thw, 0.5c)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta=>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, thw, 0.5c)
        # phi  =>(b, c, t, h, w)[->(b, 0.5c, t, h, w)]->(b, 0.5c, thw)
        # f=>(b, thw, 0.5c)dot(b, 0.5c, twh) = (b, thw, thw)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        # (b, thw, thw)dot(b, thw, 0.5c) = (b, thw, 0.5c)->(b, 0.5c, t, h, w)->(b, c, t, h, w)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _gaussian(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        if self.sub_sample:
            phi_x = self.phi(x).view(batch_size, self.in_channels, -1)
        else:
            phi_x = x.view(batch_size, self.in_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _dot_product(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

    def _concatenation(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # (b, c, N, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
        # (b, c, 1, N)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)

        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        f = self.concat_project(concat_feature)
        b, _, h, w = f.size()
        f = f.view(b, h, w)

        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)

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
                #NONLocalBlock2D(64, mode='dot_product', sub_sample=True, bn_layer=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1, padding=0))
                #NONLocalBlock2D(1, mode='dot_product', sub_sample=True, bn_layer=True))
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


class Decoder(nn.Module):
    def __init__(self, in_ch, ch, out_ch, r=16):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ch, out_ch, kernel_size=3, padding=1)
        self.se = Seq_Ex_Block(out_ch, r)

    def forward(self, x, x2=None):
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        if x2 is not None:
            x = torch.cat([x, x2], 1)

        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = self.se(x)
        #print(x.shape)

        return x



class UResNet(nn.Module):
    def __init__(self,pretrained=False, pretrained_model=None):
        print(f'ResNet{"" if pretrained else "not"} using pretrained weights.')
        super(UResNet, self).__init__()
        if pretrained_model is None:
            self.resnet = resnet34(pretrained=pretrained)
        else:
            self.resnet = pretrained_model

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            #self.resnet.maxpool
            )
        #self.conv1_nl = NONLocalBlock2D(64, mode='dot_product', sub_sample=True, bn_layer=True)
        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        #self.encoder3_nl = NONLocalBlock2D(128, mode='dot_product', sub_sample=True, bn_layer=True)
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4
        self.encoder5_nl = NONLocalBlock2D(512, mode='dot_product', sub_sample=True, bn_layer=True)

        self.center = nn.Sequential(
                nn.Conv2d(512,512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512,256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
                )

        self.decoder5 = Decoder(256+512, 512, 64)
        self.decoder4 = Decoder(64+256, 256, 64)
        self.decoder3 = Decoder(64+128, 128, 64)
        self.decoder3_nl = NONLocalBlock2D(64, mode='dot_product', sub_sample=True, bn_layer=True)
        self.decoder2 = Decoder(64+64, 64, 64)
        self.decoder1 = Decoder(64, 32, 64)


        self.se_f = Seq_Ex_Block(320, 16)
        #self.nlnn_f = NONLocalBlock2D(320, mode='embedded_gaussian', sub_sample=True, bn_layer=True)

        self.outc = OutConv(320, logits=True)
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
        #e3 = self.encoder3_nl(e3)   #128, 32, 32
        e4 = self.encoder4(e3)      #256, 16, 16
        e5 = self.encoder5(e4)      #512, 8, 8
        e5 = self.encoder5_nl(e5)

        f = self.center(e5)         #256, 4, 4
        d5 = self.decoder5(f, e5)   #64, 8, 8
        d4 = self.decoder4(d5, e4)  #64, 16, 16
        d3 = self.decoder3(d4, e3)  #64, 32, 32
        d3 = self.decoder3_nl(d3)  #64, 32, 32
        d2 = self.decoder2(d3, e2)  #64, 64, 64
        d1 = self.decoder1(d2)      #64, 128, 128


        f = torch.cat((
                d1,
                F.upsample(d2, scale_factor=2, mode='bilinear', align_corners=False),
                F.upsample(d3, scale_factor=4, mode='bilinear', align_corners=False),
                F.upsample(d4, scale_factor=8, mode='bilinear', align_corners=False),
                F.upsample(d5, scale_factor=16, mode='bilinear', align_corners=False),
                ), 1)               #320, 128, 128
        f = self.se_f(f)
        f = F.dropout2d(f, p=0.5)
        out = self.outc(f)          #1, 101,101


        return out




