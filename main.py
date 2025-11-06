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
batch_size = 64

restore_iter = 1000
num_training = 20000

restore_lr = 0.001
brestore = False

# pre-calculated mean and std
mean = [0.40006977, 0.44971865, 0.4779939]
std = [0.2785395, 0.26381946, 0.2719872]

model_save_interval = 100

## path
path = ''
model_save_path = ''


# ---------- Load data --------------- #
z_train = zipfile.ZipFile(path + 'train.zip', 'r')
z_train_list = z_train.namelist() # name list of all images in zip file
train_cls = ftn.read_gt(path + 'train_gt.txt', len(z_train_list))
print('%d Train labels loaded.' % len(train_cls))

# ## Mean and Std Calculation
# mean, std = ftn.mean_std_calculation(z_train, z_train_list)
# print('Mean:', mean)
# print('Std:', std)

z_test = zipfile.ZipFile(path + 'test.zip', 'r')
z_test_list = z_test.namelist() # name list of all images in zip file
test_cls = ftn.read_gt(path + 'test_gt.txt', len(z_test_list))
print('%d Test labels loaded.' % len(test_cls))

# ----------- Model ------------------ #
model = ftn.ResNet50(outputsize = 200).to(DEVICE)

if brestore:
    print('Model Restore form %d iteration.' % restore_iter)
    model.load_state_dict(torch.load(model_save_path + 'model_%d.pt' % restore_iter, map_location = DEVICE))

# ----------- Optimizer -------------- #
optimizer = torch.optim.SGD(model.parameters(), weight_decay = 1e-4, momentum = 0.9, lr = learning_rate if not brestore else restore_lr)
loss = torch.nn.CrossEntropyLoss()

# ----------- Training --------------- #
model_ckpt = []

for it in tqdm(range(restore_iter if brestore else 0, num_training), ncols = 120, desc = 'Training Progress'):
    batch_img, batch_cls = ftn.mini_batch_training_zip(z_train, z_train_list, train_cls, batch_size, mean, std)
    batch_img = np.transpose(batch_img, (0, 3, 1, 2)) # [B, H, W, C] -> [B, C, H, W]

    model.train()
    optimizer.zero_grad()
    pred = model(torch.from_numpy(batch_img.astype(np.float32)).to(DEVICE))
    cls_tensor = torch.tensor(batch_cls, dtype = torch.long).to(DEVICE)

    train_loss = loss(pred, cls_tensor)
    train_loss.backward()
    optimizer.step()

    if it % model_save_interval == 0 and it > (restore_iter if brestore else 0):
        tqdm.write("\niteration: %d " % it)
        tqdm.write("train loss : %f " % train_loss.item())
        tqdm.write('Evaluating the Model...')
        model.eval()

        count = 0
        for itest in tqdm(range(len(z_test_list)), ncols = 120):
            test_img = ftn.read_image_mean_std(z_test, z_test_list, mean, std, itest)
            test_img = np.reshape(test_img, [1, 128, 128, 3]) # adding batch dimension
            test_img = np.transpose(test_img, (0, 3, 1, 2)) # [B, H, W, C] -> [B, C, H, W]

            with torch.no_grad():
                pred = model(torch.from_numpy(test_img.astype(np.float32)).to(DEVICE))

            pred = pred.cpu().numpy()
            pred = np.reshape(pred, 200)
            pred = np.argmax(pred)

            gt = test_cls[itest]

            if int(pred) == int(gt):
                count += 1
        accuracy = count / len(z_test_list) * 100.0

        tqdm.write('Accuracy   : %f' % accuracy)
        tqdm.write('Current LR : %f' % optimizer.param_groups[0]['lr'])


        tqdm.write('SAVING MODEL...')
        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)
        torch.save(model.state_dict(), model_save_path + 'model_%d.pt' % it)
        tqdm.write('MODEL SAVED.\n')

        model_ckpt.append((it, accuracy, model_save_path + 'model_%d.pt' % it))

    # Learning rate decay
    if it == num_training // 10:
        optimizer.param_groups[0]['lr'] = learning_rate * 0.1
    if it == num_training * 3 // 10:
        optimizer.param_groups[0]['lr'] = learning_rate * 0.01
    if it == num_training * 6 // 10:
        optimizer.param_groups[0]['lr'] = learning_rate * 0.001

print('Training Finished.')
print('Best Model')
model_ckpt = sorted(model_ckpt, key = lambda x: x[1], reverse = True)
print(model_ckpt[0])
