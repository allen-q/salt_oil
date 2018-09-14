# full assembly of the sub-parts to form the complete net

from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, logits=False, apply_se=False, r=16):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128, apply_se=apply_se, r=r)
        self.down2 = down(128, 256, apply_se=apply_se, r=r)
        self.down3 = down(256, 512, apply_se=apply_se, r=r)
        self.down4 = down(512, 512, apply_se=apply_se, r=r)
        self.up1 = up(1024, 256, 512, bilinear=bilinear, apply_se=apply_se, r=r)
        self.up2 = up(512, 128, 256, bilinear=bilinear, apply_se=apply_se, r=r)
        self.up3 = up(256, 64, 128, bilinear=bilinear, apply_se=apply_se, r=r)
        self.up4 = up(128, 64, 64, bilinear=bilinear, apply_se=apply_se, r=r)
        self.outc = outconv(64, n_classes, logits=logits)
        print(f'Use {"Bilinear" if bilinear else "Transpose Conv"} for up-sampling')
        print(f'{"Use" if apply_se else "Does not use"} Sequeeze and Excitation Block')
        print(f'Output is {"logits" if logits else "probability"}')
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        
        return x
    
