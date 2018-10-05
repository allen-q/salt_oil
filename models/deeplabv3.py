# camera-ready

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from .resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8
from .aspp import ASPP, ASPP_Bottleneck

class DeepLabV3(nn.Module):
    def __init__(self, model_id, project_dir, resnet_layer):
        super(DeepLabV3, self).__init__()
        assert resnet_layer in [18,34,50]
        resnet = {18:ResNet18_OS8, 34:ResNet34_OS8, 50:ResNet50_OS16}[resnet_layer]
        aspp_net = {18:ASPP, 34:ASPP, 50:ASPP_Bottleneck}[resnet_layer]
        self.num_classes = 1

        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        #self.resnet = ResNet34_OS8() # NOTE! specify the type of ResNet here
        self.resnet = resnet()
        #self.aspp = ASPP(num_classes=self.num_classes) # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead
        self.aspp = aspp_net(num_classes=self.num_classes)
    def forward(self, x):
        # (x has shape (batch_size, 1, h, w))
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        x = torch.cat([
                (x-mean[2])/std[2],
                (x-mean[1])/std[1],
                (x-mean[0])/std[0],
                ], 1)


        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))

        output = self.aspp(feature_map) # (shape: (batch_size, num_classes, h/16, w/16))

        output = F.upsample(output, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))
        crop_start = (output.shape[-1]-101)//2
        crop_end = crop_start + 101
        output = output[:,:,crop_start:crop_end,crop_start:crop_end].squeeze()

        return output

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)
