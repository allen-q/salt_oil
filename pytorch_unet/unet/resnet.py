# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.model_zoo as model_zoo


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


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, logits=False):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.conv_mask = nn.Conv2d(in_ch, out_ch, 128)
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

class UResNet(nn.Module):
    def __init__(self, block, layers, in_ch=1, num_classes=1, bilinear=True):
        print('Local ResNet')
        self.inplanes = 64
        super(UResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #self.avgpool = nn.AvgPool2d(7, stride=1)
        #self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.outc = outconv(64, num_classes, logits=True)

        self.up1 = up(1024*block.expansion, 256*block.expansion, 512*block.expansion, bilinear=bilinear)
        self.up2 = up(512*block.expansion, 128*block.expansion, 256*block.expansion, bilinear=bilinear)
        self.up3 = up(256*block.expansion, 64, 128, bilinear=bilinear)
        self.up4 = up(128, 64, 64, bilinear=bilinear)
        
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
        x1 = self.relu(x)
        #x1 = self.maxpool(self.relu(x))
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        #from boxx import g
        #g()
        
        x_up1 = self.up1(x5, x4)
        x_up2 = self.up2(x_up1, x3)
        x_up3 = self.up3(x_up2, x2)
        x_up4 = self.up4(x_up3, x1)
        x_out = self.outc(x_up4)
        


        return x_out
    
    
