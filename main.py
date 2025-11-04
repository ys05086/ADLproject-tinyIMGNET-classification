import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
import function as ftn
import zipfile, time

# ---------- Device configuration ---- #
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', DEVICE)

# ---------- User parameters --------- #
learning_rate = 0.1
model_save_path = ''
batch_size = 128

restore_iter = 1000
num_training = 2000

restore_lr = 0.001
brestore = False

# ---------- Load data --------------- #
path = 'E:/AdvancedDL/Project2/'
z_train = zipfile.ZipFile(path + 'train.zip', 'r')
z_train_list = z_train.namelist() # name list of all images in zip file
train_cls = ftn.read_gt(path + 'train_gt.txt', len(z_train_list))
print('%d Train labels loaded.' % len(train_cls))

mean, std = ftn.mean_std_calculation(z_train, z_train_list)
print('Mean:', mean)
print('Std:', std)

z_test = zipfile.ZipFile(path + 'test.zip', 'r')
z_test_list = z_test.namelist() # name list of all images in zip file
test_cls = ftn.read_gt(path + 'test_gt.txt', len(z_test_list))
print('%d Test labels loaded.' % len(test_cls))


# train_images, train_cls, mean, std = ftn.load_image(train_path, 207005, preprocessing = True)
# test_images, test_cls, mean_test, std_test = ftn.load_image(test_path, 10000, mean, std, preprocessing = True)

# ----------- Model ------------------ #
model = ftn.ResNet152(outputsize = 200).to(DEVICE)

# ----------- Optimizer -------------- #
optimizer = torch.optim.SGD(model.parameters(), weight_decay = 1e-4, momentum = 0.9, lr = learning_rate)

# ----------- Training --------------- #
for it in tqdm(range(restore_iter if brestore else 0, num_training + 1)):
    batch_img, batch_cls = ftn.mini_batch_training_zip(z_train, z_train_list, train_cls, batch_size, mean, std)
