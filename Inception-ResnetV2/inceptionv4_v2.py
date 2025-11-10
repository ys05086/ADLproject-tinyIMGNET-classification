# Structure for inception-ResNet-v2 second version
# not V = same padding, often padding = 1
# V = valid padding, padding = 0

import torch
import numpy as np
import cv2

# inception-ResNet-v2
class InceptionResNetV2(torch.nn.Module):
    def __init__(self, output_size = 200):
        super(InceptionResNetV2, self).__init__()

        self.stem = Stem()
        # Inception A blocks
        self.inceptionA = torch.nn.ModuleList(
            [InceptionA(in_channels = 384, scale = 0.15) for _ in range(5)]
        )
        self.reductionA = ReductionA(in_channels = 384)

        # Inception B blocks
        self.inceptionB = torch.nn.ModuleList(
            [InceptionB(in_channels = 1152, scale = 0.15) for _ in range(10)]
        )
        self.reductionB = ReductionB(in_channels = 1152)

        # Inception C blocks
        self.inceptionC = torch.nn.ModuleList(
            [InceptionC(in_channels = 2144, scale = 0.15) for _ in range(5)]
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = torch.nn.Dropout(p = 0.2)
        self.output = torch.nn.Linear(2144, output_size)

    def forward(self, x):
        x = self.stem(x)
        for layer in self.inceptionA:
            x = layer(x)
        x = self.reductionA(x)
        for layer in self.inceptionB:
            x = layer(x)
        x = self.reductionB(x)
        for layer in self.inceptionC:
            x = layer(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x


# Basic Conv Block
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
    
# Stem Block
class Stem(torch.nn.Module):
    def __init__(self):
        super(Stem, self).__init__()

        self.input = torch.nn.Sequential(
            BasicConv2d(3, 32, kernel_size = 3, stride = 2, padding = 0), # Valid padding
            BasicConv2d(32, 32, kernel_size = 3, stride = 1, padding = 0), # Valid padding
            BasicConv2d(32, 64, kernel_size = 3, stride = 1, padding = 1)  # Same padding
        )

        self.conv1 = BasicConv2d(64, 96, kernel_size = 3, stride = 2, padding = 0) # Valid padding

        self.maxpool = torch.nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0) # Valid padding

        self.branch2_R = torch.nn.Sequential(
            BasicConv2d(64 + 96, 64, kernel_size = 1, stride = 1, padding = 0), # Same padding
            BasicConv2d(64, 64, kernel_size = (7, 1), stride = 1, padding = (3, 0)), # Same padding
            BasicConv2d(64, 64, kernel_size = (1, 7), stride = 1, padding = (0, 3)), # Same padding
            BasicConv2d(64, 96, kernel_size = 3, stride = 1, padding = 0) # Valid padding
        )

        self.branch2_L = torch.nn.Sequential(
            BasicConv2d(64 + 96, 64, kernel_size = 1, stride = 1, padding = 0), # Same padding
            BasicConv2d(64, 96, kernel_size = 3, stride = 1, padding = 0) # Valid padding
        )

        # concat

        self.maxpool2 = torch.nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0) # Valid padding

        self.conv_last = BasicConv2d(96 + 96, 192, kernel_size = 3, stride = 2, padding = 0) # Valid padding

        # concat
        
        self.bn = torch.nn.BatchNorm2d(384)
        self.relu = torch.nn.ReLU()
        # output size: 384 channels

    def forward(self, x):
        x = self.input(x)

        x1 = self.conv1(x)
        x2 = self.maxpool(x)

        # concat
        x = torch.cat((x1, x2), dim = 1) 

        x1 = self.branch2_R(x)
        x2 = self.branch2_L(x)
        # concat
        x = torch.cat((x1, x2), dim = 1)

        x1 = self.maxpool2(x)
        x2 = self.conv_last(x)

        # concat
        x = torch.cat((x1, x2), dim = 1) # 

        x = self.bn(x)
        x = self.relu(x)

        return x
    

# Inception ResNet A block
class InceptionA(torch.nn.Module):
    def __init__(self, in_channels = 384, scale = 0.1):
        super(InceptionA, self).__init__()

        self.scale = scale
        self.branch1 = torch.nn.Sequential(
            BasicConv2d(in_channels, 32, kernel_size = 1, stride = 1, padding = 0), # Same
            BasicConv2d(32, 48, kernel_size = 3, stride = 1, padding = 1), # Same 
            BasicConv2d(48, 64, kernel_size = 3, stride = 1, padding = 1)  # Same
        )

        self.branch2 = torch.nn.Sequential(
            BasicConv2d(in_channels, 32, kernel_size = 1, stride = 1, padding = 0), # Same
            BasicConv2d(32, 32, kernel_size = 3, stride = 1, padding = 1)  # Same
        )

        self.branch3 = BasicConv2d(in_channels, 32, kernel_size = 1, stride = 1, padding = 0) # Same

        # concat

        self.conv = torch.nn.Conv2d(128, in_channels, kernel_size = 1, stride = 1, padding = 0) # Same
        self.bn = torch.nn.BatchNorm2d(in_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        # concat
        x = torch.cat((x1, x2, x3), dim = 1)

        x = self.conv(x)

        x = x * self.scale + residual
        x = self.bn(x)
        x = self.relu(x)

        return x
    
# Reduction A block
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
        self.bn = torch.nn.BatchNorm2d(1152)
        self.ReLU = torch.nn.ReLU()

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.Maxpool(x)
        # concat
        x = torch.cat((x1, x2, x3), dim = 1) # channel size = 384 + 384 + in_channels = 1152
        x = self.bn(x)
        x = self.ReLU(x)
        return x

# Inception ResNet B block
class InceptionB(torch.nn.Module):
    def __init__(self, in_channels = 1152, scale = 0.1):
        super(InceptionB, self).__init__()
        
        self.scale = scale
        self.branch1 = torch.nn.Sequential(
            BasicConv2d(in_channels, 128, kernel_size = 1, stride = 1, padding = 0), # same
            BasicConv2d(128, 160, kernel_size = (1, 7), stride = 1, padding = (0, 3)), # same
            BasicConv2d(160, 192, kernel_size = (7, 1), stride = 1, padding = (3, 0)) # same
        )

        self.branch2 = BasicConv2d(in_channels, 192, kernel_size = 1, stride = 1, padding = 0) # same

        # concat

        self.last_conv = torch.nn.Conv2d(192 + 192, 1152, kernel_size = 1, stride = 1, padding = 0) # same
        self.bn = torch.nn.BatchNorm2d(1152)
        self.ReLU = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        # concat
        x = torch.cat((x1, x2), dim = 1) # channel size = 192 + 192 = 384
        x = self.last_conv(x)
        x = x * self.scale + residual
        x = self.bn(x)
        x = self.ReLU(x)
        return x

# Reduction B block
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
        # concat
        self.bn = torch.nn.BatchNorm2d(2144)
        self.ReLU = torch.nn.ReLU()

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.Maxpool(x)
        # concat
        x = torch.cat((x1, x2, x3, x4), dim = 1) # channel size = 320 + 288 + 384 + in_channels = 2144
        x = self.bn(x)
        x = self.ReLU(x)
        return x

# Inception ResNet C block
class InceptionC(torch.nn.Module):
    def __init__(self, in_channels = 2144, scale = 0.1):
        super(InceptionC, self).__init__()
        
        self.in_channels = in_channels
        self.scale = scale
        self.branch1 = torch.nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size = 1, stride = 1, padding = 0), # same
            BasicConv2d(192, 224, kernel_size = (1, 3), stride = 1, padding = (0, 1)), # same
            BasicConv2d(224, 256, kernel_size = (3, 1), stride = 1, padding = (1, 0)) # same
        )
        
        self.branch2 = BasicConv2d(in_channels, 192, kernel_size = 1, stride = 1, padding = 0) # same

        # concat

        self.last_conv = torch.nn.Conv2d(256 + 192, 2144, kernel_size = 1, stride = 1, padding = 0) # same

        # projection
        self.projection = None
        if self.in_channels != 2144:
            self.projection = torch.nn.Conv2d(in_channels, 2144, kernel_size = 1, stride = 1, bias = False)
        self.bn = torch.nn.BatchNorm2d(2144)
        self.ReLU = torch.nn.ReLU()

    def forward(self, x):
        residual = x if not self.projection else self.projection(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        # concat
        x = torch.cat((x1, x2), dim = 1) # channel size = 256 + 192 = 448
        x = self.last_conv(x)
        x = x * self.scale + residual
        x = self.bn(x)
        x = self.ReLU(x)
        return x