# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.utils.model_zoo as model_zoo


class conv_2d_depth_sep(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1):
        super(conv_2d_depth_sep, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   padding=padding, groups=in_channels, dilation=dilation)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


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
    model = UResNet(BasicBlock, [2, 2, 2, 2], up_ch=[768, 384, 192, 128],
                    in_ch=in_ch, bilinear=bilinear, **kwargs)
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
    model = UResNet(BasicBlock, [3, 4, 6, 3], up_ch=[512, 384, 192, 128],
                    in_ch=in_ch, bilinear=bilinear, **kwargs)
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
    model = UResNet(Bottleneck, [3, 4, 6, 3], up_ch=[3072, 1536, 768, 320],
                    in_ch=in_ch, bilinear=bilinear, **kwargs)
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
    model = UResNet(Bottleneck, [3, 4, 23, 3], up_ch=[3072, 1536, 768, 320], 
                    in_ch=in_ch, bilinear=bilinear, **kwargs)
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
    model = UResNet(Bottleneck, [3, 8, 36, 3], up_ch=[3072, 1536, 768, 320],
                    in_ch=in_ch, bilinear=bilinear, **kwargs)
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
    def __init__(self, block, layers, up_ch, in_ch=1, num_classes=1, bilinear=True):
        print('Local ResNet')
        self.inplanes = 64
        super(UResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_atrous_layer(block, 512, layers[3], stride=1, dilation=2)
        
        
        self.pool1 = nn.Conv2d(512, 256, 1)
        self.pool2 = Polling(in_ch=512, out_ch=256, upscale_size=8+2*2, dilation=2)
        self.pool3 = Polling(in_ch=512, out_ch=256, upscale_size=8+2*4, dilation=4)
        self.pool4 = Polling(in_ch=512, out_ch=256, upscale_size=8+2*6, dilation=6)        
        self.conv_pool = nn.Conv2d(1024, 256, 1)        
        self.conv_dec_low_level = nn.Conv2d(64, 256, 1)
        
        self.merg_conv = nn.Sequential(
                conv3x3(512, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                conv3x3(256, 1, 1),
                #nn.BatchNorm2d(1),
                #nn.ReLU(inplace=True)
            )
                                   
        
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
    
    
    def _make_atrous_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv_2d_depth_sep(self.inplanes, planes * block.expansion,
                                  kernel_size=3, padding=dilation, stride=stride,
                                  dilation=dilation),
                nn.BatchNorm2d(planes * block.expansion),
            )           
                

        layers = []
        #print(downsample)
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):        
        print(f'x in: {x.shape}')
        #x in: 128*128
        x = self.conv1(x) #64*64
        x = self.bn1(x)
        print(f'x conv1 out: {x.shape}') 
        #x1 = self.relu(x)
        x1 = self.maxpool(self.relu(x)) #32*32
        print(f'x maxpool out: {x1.shape}')
        x2 = self.layer1(x1) #32*32
        print(f'x2: {x2.shape}')
        x3 = self.layer2(x2) #16*16
        print(f'x3: {x3.shape}')
        x4 = self.layer3(x3) #8*8
        print(f'x4: {x4.shape}')
        x5 = self.layer4(x4) #8*8
        print(f'x5: {x5.shape}')        

        x_pool1 = self.pool1(x5)
        x_pool2 = self.pool2(x5)
        x_pool3 = self.pool3(x5)
        x_pool4 = self.pool4(x5)
        
        enc_out = self.conv_pool(torch.cat((x_pool1, x_pool2, x_pool3, x_pool4), dim=1))
        enc_out_up1 = F.upsample(enc_out, scale_factor=4, mode='bilinear', align_corners=True)
        dec_low_level = self.conv_dec_low_level(x2)
        enc_out_merge = self.merg_conv(torch.cat((enc_out_up1, dec_low_level), dim=1))
        enc_out_final = F.upsample(enc_out_merge, scale_factor=4, mode='bilinear', align_corners=True)
        crop_start = (enc_out_final.shape[-1]-101)//2
        crop_end = crop_start + 101
        x_out = enc_out_final[:,:,crop_start:crop_end,crop_start:crop_end].squeeze()
        #from boxx import g
        #g()
        return x_out


class Polling(nn.Module):
    def __init__(self, in_ch, out_ch, upscale_size, dilation):
        super(Polling, self).__init__()          

        self.conv1 = nn.Conv2d(in_ch, out_ch, 1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        #self.relu = nn.ReLU(inplace=True)
        self.upscale = nn.Upsample(size=upscale_size, mode='bilinear', align_corners=True)																					
        self.conv2 = conv_2d_depth_sep(out_ch, out_ch, kernel_size=3, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.upscale(out)
        #from boxx import g
        #g()
        out = self.conv2(out)
        #out = self.relu(out)
        out = self.bn2(out)

        return out
    
    
    
def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model
