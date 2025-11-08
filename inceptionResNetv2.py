# Structure for inception-ResNet-v2
# not V = same padding, often padding = 1
# V = valid padding, padding = 0

import torch
import numpy as np
import cv2

# Inception V4 structure with ResNet v2
class InceptionV4(torch.nn.Module):
    def __init__(self, output_size = 200):
        super(InceptionV4, self).__init__()
        self.stem = Stem()
        # Inception A
        self.inception_A = InceptionA()
        self.reduction_A = ReductionA()

        # Inception B
        self.inception_B = InceptionB()
        self.reduction_B = ReductionB()

        # Inception C
        self.inception_C_in = InceptionC(in_channels = 2144)
        self.inception_C = InceptionC()

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1)) # output size should be (1, 1)
        self.dropout = torch.nn.Dropout(p = 0.2)
        self.last_linear = torch.nn.Linear(2048, output_size)

    def forward(self, x):
        x = self.stem(x)
        for _ in range(5):
            x = self.inception_A(x)
        x = self.reduction_A(x)

        for _ in range(10):
            x = self.inception_B(x)
        x = self.reduction_B(x)

        for _ in range(5):
            if _ == 0:
                x = self.inception_C_in(x)
            else:
                x = self.inception_C(x)

        x = self.avgpool(x)
        x = torch.reshape(x, [-1, 2048]) # flatten
        x = self.dropout(x)
        x = self.last_linear(x)
        return x

class BasicConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0):
        super(BasicConv2d, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Stem(torch.nn.Module):
    def __init__(self):
        super(Stem, self).__init__()


        self.input = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size = 3, stride = 2, padding = 0, bias = False), # Valid Padding
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),

            torch.nn.Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 0, bias = False), # Valid Padding
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),

            torch.nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1, bias = False),  # Same Padding
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )

        # branch 1
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 96, kernel_size = 3, stride = 2, padding = 0, bias = False), # Valid
            torch.nn.BatchNorm2d(96),
            torch.nn.ReLU()
        )

        self.maxpool1 = torch.nn.MaxPool2d(3, stride = 2, padding = 0) # Valid : channel size 64

        # concat 1
        # [B, C, H, W], so dim = 1 is channel dim.

        self.branch2_R = torch.nn.Sequential(
            torch.nn.Conv2d(64 + 96, 64, kernel_size = 1, stride = 1, padding = 0, bias = False), # Same
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),

            torch.nn.Conv2d(64, 64, kernel_size = (7, 1), stride = 1, padding = (3, 0), bias = False), # Same
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),

            torch.nn.Conv2d(64, 64, kernel_size = (1, 7), stride = 1, padding = (0, 3), bias = False), # Same
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),

            torch.nn.Conv2d(64, 96, kernel_size = 3, stride = 1, padding = 0, bias = False), # Valid
            torch.nn.BatchNorm2d(96),
            torch.nn.ReLU()
        )

        self.branch2_L = torch.nn.Sequential(
            torch.nn.Conv2d(64 + 96, 64, kernel_size = 1, stride = 1, padding = 0, bias = False), # Same
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),

            torch.nn.Conv2d(64, 96, kernel_size = 3, stride = 1, padding = 0, bias = False), # Valid
            torch.nn.BatchNorm2d(96),
            torch.nn.ReLU()
        )

        # concat 2

        self.maxpool2 = torch.nn.MaxPool2d(3, stride = 2, padding = 0) # Valid : channel size 192

        self.conv_last = torch.nn.Sequential(
            torch.nn.Conv2d(96 + 96, 192, kernel_size = 3, stride = 2, padding = 0, bias = False), # Valid
            torch.nn.BatchNorm2d(192),
            torch.nn.ReLU()
        )

        # concat 3

        # output size : 384 channels

    def forward(self, x):
        x = self.input(x)
        x1 = self.conv1(x)
        x2 = self.maxpool1(x)
        # concat 1
        x = torch.cat((x1, x2), dim = 1)

        x1 = self.branch2_R(x)
        x2 = self.branch2_L(x)
        # concat 2
        x = torch.cat((x1, x2), dim = 1)

        x1 = self.maxpool2(x)
        x2 = self.conv_last(x)
        # concat 3
        x = torch.cat((x1, x2), dim = 1)

        return x

# Inception RenNet - V2 Impelementation
# Inception ResNet A block
class InceptionA(torch.nn.Module):
    def __init__(self, in_channels = 384):
        super(InceptionA, self).__init__()

        self.branch1 = torch.nn.Sequential(
            BasicConv2d(in_channels, 32, kernel_size = 1, stride = 1, padding = 0), # same
            BasicConv2d(32, 48, kernel_size = 3, stride = 1, padding = 1), # same
            BasicConv2d(48, 64, kernel_size = 3, stride = 1, padding = 1) # same
        )

        self.branch2 = torch.nn.Sequential(
            BasicConv2d(in_channels, 32, kernel_size = 1, stride = 1, padding = 0), # same
            BasicConv2d(32, 32, kernel_size = 3, stride = 1, padding = 1) # same
        )

        self.branch3 = BasicConv2d(in_channels, 32, kernel_size = 1, stride = 1, padding = 0) # same

        # concat

        self.last_conv = torch.nn.Conv2d(64 + 32 + 32, 384, kernel_size = 1, stride = 1, padding = 0, bias = True) # same
        
        # # projection
        # self.projection = None if in_channels == 384 else torch.nn.Conv2d(in_channels, 384, kernel_size = 1, stride = 1, bias = False)

        self.ReLU = torch.nn.ReLU()

    def forward(self, x):
        residual = x # if self.projection is None else self.projection(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        # concat
        x = torch.cat((x1, x2, x3), dim = 1) # channel size = 64 + 32 + 32 = 128
        x = self.last_conv(x)
        x = x + residual
        x = self.ReLU(x)
        return x
    
class ReductionA(torch.nn.Module):
    def __init__(self, in_channels = 384):
        super(ReductionA, self).__init__()
        
        self.branch1 = torch.nn.Sequential(
            BasicConv2d(in_channels, 256, kernel_size = 1, stride = 1, padding = 0), # same
            BasicConv2d(256, 384, kernel_size = 3, stride = 1, padding = 1), # same 
            BasicConv2d(384, 384, kernel_size = 3, stride = 2, padding = 0) # valid
        )

        self.branch2 = BasicConv2d(in_channels, 384, kernel_size = 3, stride = 2, padding = 0) # valid

        self.Maxpool = torch.nn.MaxPool2d(3, stride = 2, padding = 0) # valid

        # concat

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.Maxpool(x)
        # concat
        x = torch.cat((x1, x2, x3), dim = 1) # channel size = 384 + 384 + in_channels = 1152
        return x

class InceptionB(torch.nn.Module):
    def __init__(self, in_channels = 1152):
        super(InceptionB, self).__init__()
        
        self.branch1 = torch.nn.Sequential(
            BasicConv2d(in_channels, 128, kernel_size = 1, stride = 1, padding = 0), # same
            BasicConv2d(128, 160, kernel_size = (1, 7), stride = 1, padding = (0, 3)), # same
            BasicConv2d(160, 192, kernel_size = (7, 1), stride = 1, padding = (3, 0)) # same
        )

        self.branch2 = BasicConv2d(in_channels, 192, kernel_size = 1, stride = 1, padding = 0) # same

        # concat

        self.last_conv = torch.nn.Conv2d(192 + 192, 1152, kernel_size = 1, stride = 1, padding = 0, bias = True) # same

        self.ReLU = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        # concat
        x = torch.cat((x1, x2), dim = 1) # channel size = 192 + 192 = 384
        x = self.last_conv(x)
        x = x + residual
        x = self.ReLU(x)
        return x
    
class ReductionB(torch.nn.Module):
    def __init__(self, in_channels = 1152):
        super(ReductionB, self).__init__()
        self.branch1 = torch.nn.Sequential(
            BasicConv2d(in_channels, 256, kernel_size = 1, stride = 1, padding = 0), # same
            BasicConv2d(256, 288, kernel_size = 3, stride = 1, padding = 1), # same
            BasicConv2d(288, 320, kernel_size = 3, stride = 2, padding = 0) # valid
        )

        self.branch2 = torch.nn.Sequential(
            BasicConv2d(in_channels, 256, kernel_size = 1, stride = 1, padding = 0), # same
            BasicConv2d(256, 288, kernel_size = 3, stride = 2, padding = 0) # valid
        )

        self.branch3 = torch.nn.Sequential(
            BasicConv2d(in_channels, 256, kernel_size = 1, stride = 1, padding = 0), # same
            BasicConv2d(256, 384, kernel_size = 3, stride = 2, padding = 0) # valid
        )

        self.Maxpool = torch.nn.MaxPool2d(3, stride = 2, padding = 0) # valid

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.Maxpool(x)
        # concat
        x = torch.cat((x1, x2, x3, x4), dim = 1) # channel size = 320 + 288 + 384 + in_channels = 2144
        return x

class InceptionC(torch.nn.Module):
    def __init__(self, in_channels = 2048):
        super(InceptionC, self).__init__()
        
        self.in_channels = in_channels
        self.branch1 = torch.nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size = 1, stride = 1, padding = 0), # same
            BasicConv2d(192, 224, kernel_size = (1, 3), stride = 1, padding = (0, 1)), # same
            BasicConv2d(224, 256, kernel_size = (3, 1), stride = 1, padding = (1, 0)) # same
        )
        
        self.branch2 = torch.nn.Conv2d(in_channels, 192, kernel_size = 1, stride = 1, padding = 0, bias = False) # same

        # concat

        self.last_conv = torch.nn.Conv2d(256 + 192, 2048, kernel_size = 1, stride = 1, padding = 0, bias = True) # same

        # projection
        self.projection = None
        if self.in_channels != 2048:
            self.projection = torch.nn.Conv2d(in_channels, 2048, kernel_size = 1, stride = 1, bias = False)

        self.ReLU = torch.nn.ReLU()

    def forward(self, x):
        residual = x if not self.projection else self.projection(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        # concat
        x = torch.cat((x1, x2), dim = 1) # channel size = 256 + 192 = 448
        x = self.last_conv(x)
        x = x + residual
        x = self.ReLU(x)
        return x


# Not used
class BaseConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0):
        super(BaseConv2d, self).__init__()

        self.ReLU = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm2d(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, bias = False)
        # bn - relu?
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias = False)

        # projection
        if in_channels != out_channels or stride != 1:
            self.projection = torch.nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False)
        else:
            self.projection = None

    def forward(self, x):
        x = self.bn(x)
        x = self.ReLU(x)
        residual = x if self.projection is None else self.projection(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.ReLU(x)
        x = self.conv2(x)
        x = x + residual
        return x
