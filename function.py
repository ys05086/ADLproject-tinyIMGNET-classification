# New Function File for Project 2
# For tiny ImageNet Dataset
# data size 128 * 128 * 3

import os
import torch
import numpy as np
import cv2
from tqdm import tqdm

# Read GT labels
def read_gt(gt_txt, num_img):
    cls = np.zeros(num_img)
    f = open(gt_txt, 'r')
    lines = f.readlines()

    for it in range(len(lines)):
        cls[it] = int((lines[it])[:-1]) - 1 # 1 - 200 txt -> to = 0 - 199
    
    f.close()
    return cls

# Read image and calculate mean and std
def read_image_mean_std(z_file, z_file_list, mean, std, index):
    img_temp = z_file.read(z_file_list[index])
    img_temp = cv2.imdecode(np.frombuffer(img_temp, np.uint8), 1)
    img_temp = img_temp.astype(np.float32)
    img_temp = (img_temp / 255.0 - mean) / std
    return img_temp

# Mean and Std Calculation
# think about deviding to 255 first
def mean_std_calculation(z_file, z_file_list):
    # calculate mean first and std next
    sum_ = np.zeros(3) # sum for each channel
    sum_sq = np.zeros(3) # sum of squares for each channel
    for it in tqdm(range(len(z_file_list)), ncols = 120, desc = 'Mean, Std Calculation'):
        img_temp = z_file.read(z_file_list[it])
        img_temp = cv2.imdecode(np.frombuffer(img_temp, np.uint8), 1)
        img_temp = img_temp / 255.0 # normalize to [0, 1]
        img_temp = img_temp.astype(np.float32)

        sum_ += np.mean(img_temp, axis = (0, 1)) # sum for each channel
        sum_sq += np.mean(img_temp ** 2, axis = (0, 1)) # sum of squares for each channel
        
    # std = sqrt(E(x**2) - (E(x))**2)
    # for formula derivation, check my notes
    mean = sum_ / len(z_file_list)
    std = np.sqrt((sum_sq / len(z_file_list)) - (mean ** 2))

    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    return mean, std

# Image Augmentation
def image_augmentation(image, cutout = False, flip = True, flip_prob = 0.5):
    if cutout:
        H = image.shape[0]
        W = image.shape[1]

        holes = 1 # number of holes to cut out from image
                  # in this case, we use only one hole  
        length = 32 # length of the holes
        mask = np.ones((H, W, 1), np.float32)
        for n in range(holes):
            y = np.random.randint(H)
            x = np.random.randint(W)

            y1 = np.clip(y - length // 2, 0, H)
            y2 = np.clip(y + length // 2, 0, H)
            x1 = np.clip(x - length // 2, 0, W)
            x2 = np.clip(x + length // 2, 0, W)

            mask[y1: y2, x1: x2] = 0.0

        image = image * mask

    if flip:
        prob = np.random.rand()
        image = np.flip(image, 1) if prob > flip_prob else image
    return image

def mini_batch_training_zip(z_file, z_file_list, train_cls, batch_size, mean, std, augmentation = True):
    batch_img = np.zeros((batch_size, 128, 128, 3))
    batch_cls = np.zeros(batch_size)

    # train_img = [20XXXXX, 128, 128, 3]
    rand_num = np.random.randint(0, len(z_file_list), size = batch_size)

    for it in range(batch_size):
        temp = rand_num[it]
        img_temp = z_file.read(z_file_list[temp])
        img_temp = cv2.imdecode(np.frombuffer(img_temp, np.uint8), 1)
        img_temp = img_temp.astype(np.float32)
        img_temp = (img_temp / 255.0 - mean) / std

        batch_img[it, :, :, :] = img_temp if not augmentation else image_augmentation(img_temp)
        batch_cls[it] = train_cls[temp]

    return batch_img, batch_cls

# ResNet 101
# Structure: 3 - 4 - 23 - 3
class ResNet101(torch.nn.Module):
    def __init__(self, outputsize = 200):
        super(ResNet101, self).__init__()
        self.in_channels = 64
        self.mid_channels = 64
        self.ReLU = torch.nn.ReLU()
        self.input_conv = torch.nn.Conv2d(3, self.in_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)

        # Stages
        self.stage1 = StageBlock(self.in_channels, self.mid_channels, num_blocks = 3, stride = 1)
        self.stage2 = StageBlock(self.in_channels * 4, self.mid_channels * 2, num_blocks = 4, stride = 2)
        self.stage3 = StageBlock(self.in_channels * 8, self.mid_channels * 4, num_blocks = 23, stride = 2)
        self.stage4 = StageBlock(self.in_channels * 16, self.mid_channels * 8, num_blocks = 3, stride = 2) # shape = [N, 2048, 8, 8]
        self.bn_final = torch.nn.BatchNorm2d(self.mid_channels * 8 * 4)

        # Average Pooling and Fully Connected Layer
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(2048, 1000)
        self.output = torch.nn.Linear(1000, outputsize)
        
    def forward(self, x):
        x = self.input_conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.bn_final(x)
        x = self.ReLU(x)
        x = self.avgpool(x)
        x = torch.reshape(x, [-1, 2048])
        x = self.fc(x)
        x = self.output(x)

        return x
    
# RoR-3 Net( 110 )

# ResNet 152
# Structure: 3 - 8 - 36 - 3
class ResNet152(torch.nn.Module):
    def __init__(self, outputsize = 200):
        super(ResNet152, self).__init__()
        self.in_channels = 64
        self.mid_channels = 64
        self.ReLU = torch.nn.ReLU()
        self.input_conv = torch.nn.Conv2d(3, self.in_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)

        # Stages
        self.stage1 = StageBlock(self.in_channels, self.mid_channels, num_blocks = 3, stride = 1)
        self.stage2 = StageBlock(self.in_channels * 4, self.mid_channels * 2, num_blocks = 8, stride = 2)
        self.stage3 = StageBlock(self.in_channels * 8, self.mid_channels * 4, num_blocks = 36, stride = 2)
        self.stage4 = StageBlock(self.in_channels * 16, self.mid_channels * 8, num_blocks = 3, stride = 2) # shape = [N, 2048, 8, 8]
        self.bn_final = torch.nn.BatchNorm2d(self.mid_channels * 8 * 4)

        # Average Pooling and Fully Connected Layer
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(2048, 1000)
        self.output = torch.nn.Linear(1000, outputsize)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.bn_final(x)
        x = self.ReLU(x)
        x = self.avgpool(x)
        x = torch.reshape(x, [-1, 2048])
        x = self.fc(x)
        x = self.output(x)

        return x
    
# Residual stage block
class StageBlock(torch.nn.Module):
    def __init__(self, in_channels, mid_channels, num_blocks, stride = 1):
        super(StageBlock, self).__init__()
        
        self.blocks = torch.nn.ModuleList()
        for i in range(num_blocks):
            if i == 0:
                self.blocks.append(BottleNeck(in_channels, mid_channels, stride = stride))
            else:
                self.blocks.append(BottleNeck(mid_channels * 4, mid_channels, stride = 1))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

# Residual Block: Basic Block(pre-activation)
class Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Block, self).__init__()

        self.ReLU = torch.nn.ReLU()
        self.stride = stride
        self.stride_true = False if stride == 1 else True
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = self.stride, padding = 1, bias = False)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)

        self.bn1 = torch.nn.BatchNorm2d(in_channels)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

        # Projection if needed
        self.projection = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride = 2, bias = False)
        
    def forward(self, x):
        residual = self.projection(x) if self.stride_true else x
        
        x = self.bn1(x)
        x = self.ReLU(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.ReLU(x)
        x = self.conv2(x)

        x = x + residual
        return x
    
# Residual BottleNeck Block
class BottleNeck(torch.nn.Module):
    def __init__(self, in_channels, mid_channels, stride = 1):
        super(BottleNeck, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.stride = stride
        self.stride_true = False if stride == 1 else True

        self.ReLU = torch.nn.ReLU()

        # First
        self.bn_in = torch.nn.BatchNorm2d(in_channels)
        self.neck1 = torch.nn.Conv2d(in_channels, mid_channels, kernel_size = 1, stride = 1, padding = 0, bias = False)

        # Second
        self.bn_mid = torch.nn.BatchNorm2d(mid_channels)
        self.conv = torch.nn.Conv2d(mid_channels, mid_channels, kernel_size = 3, stride = self.stride, padding = 1, bias = False)

        # Third
        self.bn_out = torch.nn.BatchNorm2d(mid_channels)
        self.neck2 = torch.nn.Conv2d(mid_channels, mid_channels * 4, kernel_size = 1, stride = 1, padding = 0, bias = False)

        # Projection if needed
        self.projection = torch.nn.Conv2d(in_channels, mid_channels * 4, kernel_size = 1, stride = self.stride, bias = False)

    def forward(self, x):
        residual = self.projection(x) if (self.stride_true == True) or (self.in_channels != self.mid_channels * 4) else x

        x = self.bn_in(x)
        x = self.ReLU(x)
        x = self.neck1(x)

        x = self.bn_mid(x)
        x = self.ReLU(x)
        x = self.conv(x)

        x = self.bn_out(x)
        x = self.ReLU(x)
        x = self.neck2(x)

        x = x + residual
        return x
        
    
# RoR block
class RoR_Block(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(RoR_Block, self).__init__()
        self.stride = stride
        self.stride_true = False if stride == 1 else True # for projection

        self.block1 = Block(in_channels, out_channels, stride) # first block with stride selection
        self.block2 = Block(out_channels, out_channels, 1)
        self.block3 = Block(out_channels, out_channels, 1)
        self.block4 = Block(out_channels, out_channels, 1)
        self.block5 = Block(out_channels, out_channels, 1)
        self.block6 = Block(out_channels, out_channels, 1)
        self.block7 = Block(out_channels, out_channels, 1)
        self.block8 = Block(out_channels, out_channels, 1)
        self.block9 = Block(out_channels, out_channels, 1)
        self.block10 = Block(out_channels, out_channels, 1)
        self.block11 = Block(out_channels, out_channels, 1)
        self.block12 = Block(out_channels, out_channels, 1)
        self.block13 = Block(out_channels, out_channels, 1)
        self.block14 = Block(out_channels, out_channels, 1)
        self.block15 = Block(out_channels, out_channels, 1)
        self.block16 = Block(out_channels, out_channels, 1)
        self.block17 = Block(out_channels, out_channels, 1)
        self.block18 = Block(out_channels, out_channels, 1)

        # Level 2 Projection layer
        self.projection = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride = 1, bias = False)
        self.projection_stride = torch.nn.Conv2d(out_channels, out_channels, kernel_size=1, stride = 2, bias = False)

    def forward(self, x):
        residual = self.projection_stride(x) if self.stride_true else self.projection(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)

        x = x + residual
        return x

# # CBAM Module
# class CBAM(torch.nn.Module):
#     def __init__(self, channels, reduction = 16, kernel_size = 7):
#         super(CBAM, self).__init__()
#         self.channel_attention = ChannelAttention(channels, reduction)
#         self.spatial_attention = SpatialAttention(kernel_size)

#     def forward(self, x):
#         x = self.channel_attention(x)
#         x = self.spatial_attention(x)
#         return x
    

    
