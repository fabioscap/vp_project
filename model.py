from numpy import outer
import torch

import torch.nn as nn

from torchvision.models import resnet

# returns a convolution block with batch norm
def convolution_block(in_channels, out_channels, ksize, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels = in_channels,
                  out_channels = out_channels,
                  kernel_size = ksize,
                  stride = stride,
                  padding = padding,
        ),
        nn.BatchNorm2d(out_channels)
    )

# returns a transposed convolution block with batch norm
def upconvolution_block(in_channels, out_channels, ksize, stride, padding,out_padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels = in_channels,
                           out_channels = out_channels,
                           kernel_size = ksize,
                           stride = stride,
                           padding = padding,
                           output_padding= out_padding
        ),
        nn.BatchNorm2d(out_channels)
    )

class Net1(nn.Module):
    def __init__(self,pretrained_encoder = True):
        super().__init__()
        self.rgb_conv1 = nn.Conv2d(in_channels=3,
                                   out_channels=48,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.rgb_bn = nn.BatchNorm2d(48)


        self.d_conv1 = nn.Conv2d(in_channels=1,
                                 out_channels=16,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.d_bn = nn.BatchNorm2d(16)

        pretrained_model = resnet.resnet18(pretrained=pretrained_encoder)
        self.conv2 = pretrained_model._modules['layer1']
        self.conv3 = pretrained_model._modules['layer2']
        self.conv4 = pretrained_model._modules['layer3']
        self.conv5 = pretrained_model._modules['layer4']
        
        del pretrained_model  # save memory by removing the resnet


        self.conv6 = convolution_block(in_channels = 512,
                                       out_channels = 512,
                                       ksize = 3,
                                       stride = 2,
                                       padding = 1
        )

        self.convt5 = upconvolution_block(in_channels = 512,
                                          out_channels = 256,
                                          ksize = 3,
                                          stride = 2,
                                          padding = 1,
                                          out_padding = 1         
        )

        self.convt4 = upconvolution_block(in_channels = 768,
                                          out_channels = 128,
                                          ksize = 3,
                                          stride = 2,
                                          padding = 1,
                                          out_padding = 1
        )

        self.convt3 = upconvolution_block(in_channels = 384,
                                          out_channels = 64,
                                          ksize = 3,
                                          stride = 2,
                                          padding = 1,
                                          out_padding = 1
        )

        self.convt2 = upconvolution_block(in_channels = 192,
                                          out_channels = 64,
                                          ksize = 3,
                                          stride = 2,
                                          padding=1,
                                          out_padding=1
        )

        self.convt1 = upconvolution_block(in_channels = 128,
                                          out_channels = 64,
                                          ksize = 3,
                                          stride = 1,
                                          padding = 1,
                                          out_padding = 0
        )

        self.convtf = convolution_block(in_channels = 128,
                                        out_channels = 1,
                                        ksize = 1,
                                        stride = 1,
                                        padding = 0
        )


        self.relu = nn.ReLU()

    def forward(self,rgb,depth):
        a = self.relu(self.rgb_bn(self.rgb_conv1(rgb)))
        b = self.relu(self.d_bn(self.d_conv1(depth)))
        
        x = torch.cat((a,b),1)

        # encoder
        conv2 = self.conv2(x)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.relu(self.conv6(conv5))

        # decoder
        convt5 = self.relu(self.convt5(conv6))
        y = torch.cat((convt5, conv5), 1)

        convt4 = self.relu(self.convt4(y))
        y = torch.cat((convt4, conv4), 1)

        convt3 = self.relu(self.convt3(y))
        y = torch.cat((convt3, conv3), 1)

        convt2 = self.relu(self.convt2(y))
        y = torch.cat((convt2, conv2), 1)

        convt1 = self.relu(self.convt1(y))
        y = torch.cat((convt1, x), 1)

        y = self.convtf(y)
        
        return y

# I use resnet as feature extractor
class encoder(nn.Module):
    def __init__(self):
        super().__init__()
        pretrained = resnet.resnet18(pretrained=True)

        self.conv2 = pretrained._modules['layer1']
        self.conv3 = pretrained._modules['layer2']
        self.conv4 = pretrained._modules['layer3']
        self.conv5 = pretrained._modules['layer4']
    

    def forward(self,x):
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x