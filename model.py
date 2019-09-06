import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes,1,kernel_size=3,stride=1,padding=1,bias=False)


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1, inplace=True)
    )
  
def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    return input[:, :, :target.size(2), :target.size(3)]
    
    
class DispNet(nn.Module):
    expansion = 1

    def __init__(self, batchNorm=True):
        super(DispNet,self).__init__()

        self.batchNorm = batchNorm
        self.conv1   = conv(self.batchNorm,   6,   64, kernel_size=7, stride=2)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256,  256)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm,1024, 1024)

        self.upconv5 = deconv(1024, 512)
        self.upconv4 = deconv(512, 256)
        self.upconv3 = deconv(256,  128)
        self.upconv2 = deconv(128,  64)
        self.upconv1 = deconv(64,  32)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(512)
        self.predict_flow4 = predict_flow(256)
        self.predict_flow3 = predict_flow(128)
        self.predict_flow2 = predict_flow(64)
        self.predict_flow1 = predict_flow(32)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        
        self.iconv5 = nn.Conv2d(1025, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.iconv4 = nn.Conv2d(769, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.iconv3 = nn.Conv2d(385, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.iconv2 = nn.Conv2d(193, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.iconv1 = nn.Conv2d(97, 32, kernel_size=3, stride=1, padding=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3_b = self.conv3_1(self.conv3(out_conv2))
        out_conv4_b = self.conv4_1(self.conv4(out_conv3_b))
        out_conv5_b = self.conv5_1(self.conv5(out_conv4_b))
        out_conv6_b = self.conv6_1(self.conv6(out_conv5_b))

        pr6       = self.predict_flow6(out_conv6_b)
        pr6_up    = self.upsampled_flow6_to_5(pr6)
        
        
        upconv5 = self.upconv5(out_conv6_b)
        iconv5 = self.iconv5(torch.cat([upconv5, pr6_up, out_conv5_b], dim=1))
        pr5       = self.predict_flow5(iconv5)
        pr5_up  = self.upsampled_flow5_to_4(pr5)
        
        upconv4 = self.upconv4(iconv5)
        iconv4 = self.iconv4(torch.cat([upconv4, pr5_up, out_conv4_b], dim=1))
        pr4       = self.predict_flow4(iconv4)
        pr4_up  = self.upsampled_flow4_to_3(pr4)
        
        upconv3 = self.upconv3(iconv4)
        iconv3 = self.iconv3(torch.cat([upconv3, pr4_up, out_conv3_b], dim=1))
        pr3       = self.predict_flow3(iconv3)
        pr3_up  = self.upsampled_flow3_to_2(pr3)
        
        upconv2 = self.upconv2(iconv3)
        iconv2 = self.iconv2(torch.cat([upconv2, pr3_up, out_conv2], dim=1))
        pr2       = self.predict_flow2(iconv2)
        pr2_up  = self.upsampled_flow2_to_1(pr2)
        
        upconv1 = self.upconv1(iconv2)
        iconv1 = self.iconv1(torch.cat([upconv1, pr2_up, out_conv1], dim=1))
        pr1       = self.predict_flow1(iconv1)
        
       

        if self.training:
            return pr1, pr2, pr3, pr4, pr5, pr6
        else:
            return pr1
