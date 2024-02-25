import argparse
import os
import random
import numpy as np
from sqlalchemy import true
import copy
import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data.sampler import SubsetRandomSampler

from pretrained_models_pytorch import pretrainedmodels

from utils import *
from utils.patch_utils import *
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--cuda', action='store_true', help='enables cuda') # --cuda为使用gpu
parser.add_argument('--card', type=int, default=1, help='enables card') # --card  使用第i块卡
parser.add_argument('--target', type=int, default=12, help='The target class: 859 == toaster') # 攻击目标
parser.add_argument('--conf_target', type=float, default=0.6, help='Stop attack on image when target classifier reaches this value for target class') # 当达到指定置信度停止

parser.add_argument('--max_count', type=int, default=500, help='max number of iterations to find adversarial example') # 寻找对抗样本最大迭代次数
parser.add_argument('--patch_type', type=str, default='square', help='patch type: circle or square') # 对抗补丁的形状
parser.add_argument('--patch_size', type=float, default=0.05, help='patch size. E.g. 0.05 ~= 5% of image ') # 对抗补丁的大小

parser.add_argument('--train_size', type=int, default=5000, help='Number of training images') # 训练集大小
parser.add_argument('--test_size', type=int, default=5000, help='Number of test images') # 测试集大小

parser.add_argument('--image_size', type=int, default=224, help='the height / width of the input image to network')  # 网络的输入大小

parser.add_argument('--plot_all', type=int, default=0, help='1 == plot all successful adversarial images') # 画出所有成功的对抗补丁

parser.add_argument('--netClassifier', default='resnet50', help="The target classifier") # 要攻击的目标网络

parser.add_argument('--outf', default='./logs', help='folder to output images and model checkpoints') # 输出图像和模型检查点的文件夹
parser.add_argument('--manualSeed', type=int, default=None, help='manual seed') # 手工种子

opt = parser.parse_args() #得到参数
opt.cuda = True
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.card)
print(opt)

test_acc = []
# 构建log目录
patch_save_path = './patch/' + opt.netClassifier
try:
    os.makedirs(patch_save_path)
except OSError:
    pass

try:
    os.makedirs(opt.outf)
except OSError:
    pass
with open("log.txt",'a+') as f:
    f.writelines("args:{} Time:{}\n".format(opt, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))

# 设置随机数种子
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
# 加速网络训练
# cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
# 取出opt中的参数
target = opt.target
conf_target = opt.conf_target
max_count = opt.max_count
patch_type = opt.patch_type
patch_size = opt.patch_size
image_size = opt.image_size
train_size = opt.train_size
test_size = opt.test_size
plot_all = opt.plot_all 

# 条件为false时 执行断言
assert train_size + test_size <= 50000, "Traing set size + Test set size > Total dataset size"
# 根据参数创建模型
print("=> creating model ")
netClassifier = pretrainedmodels.__dict__[opt.netClassifier](num_classes=1000, pretrained='imagenet')
if opt.cuda:
    netClassifier.cuda()


print('==> Preparing data..')
# 标准化 imagenet的均值和方差
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean,
                                 std=std)
idx = np.arange(50000)
np.random.shuffle(idx)
# 取出train_size和test_size大小的idx 并且数据不重复
training_idx = idx[:train_size]
test_idx = idx[train_size:train_size+test_size]
# 加载训练数据 对数据进行缩放 裁剪等处理
train_loader = torch.utils.data.DataLoader(
    dset.ImageFolder('imagenetdata/val', transforms.Compose([
        transforms.Scale(round(max(netClassifier.input_size)*1.050)),
        transforms.CenterCrop(max(netClassifier.input_size)),
        transforms.ToTensor(),
        ToSpaceBGR(netClassifier.input_space=='BGR'),
        ToRange255(max(netClassifier.input_range)==255),
        normalize,
    ])),
    batch_size=1, shuffle=False, sampler=SubsetRandomSampler(training_idx),
    num_workers=opt.workers, pin_memory=True)
 
test_loader = torch.utils.data.DataLoader(
    dset.ImageFolder('imagenetdata/val', transforms.Compose([
        transforms.Resize([image_size,image_size]),
        transforms.ToTensor(),
        ToSpaceBGR(netClassifier.input_space=='BGR'),
        ToRange255(max(netClassifier.input_range)==255),
        normalize,
    ])),
    batch_size=1, shuffle=False, sampler=SubsetRandomSampler(test_idx),
    num_workers=opt.workers, pin_memory=True)

min_in, max_in = netClassifier.input_range[0], netClassifier.input_range[1]
min_in, max_in = np.array([min_in, min_in, min_in]), np.array([max_in, max_in, max_in])
min_out, max_out = np.min((min_in-mean)/std), np.max((max_in-mean)/std)


def train(epoch, patch, patch_shape):
    netClassifier.eval()
    success = 0
    total = 0
    recover_time = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        if opt.cuda:
            data = data.cuda()
            labels = labels.cuda()
        prediction = netClassifier(data)
 
        # 只攻击分类正确样本，如果分类标签不正确则继续        
        if prediction.data.max(1)[1][0] != labels.data[0]:
            continue
        
        total += 1
        
        # transform path
        data_shape = data.data.cpu().numpy().shape
        if patch_type == 'circle':
            patch, mask, patch_shape = circle_transform(patch, data_shape, patch_shape, image_size)
        elif patch_type == 'square':
            patch, mask  = square_transform(patch, data_shape, patch_shape, image_size)
        patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask) # 转换成torch的tensor
        
        if opt.cuda:
            patch, mask = patch.cuda(), mask.cuda()
        adv_x, mask, patch = attack(data, patch, mask)
        
        adv_label = netClassifier(adv_x).data.max(1)[1][0]
        ori_label = labels.data[0]
        # 如果攻击成功 success+1
        if adv_label == target:
            success += 1
            # 如果画图为1
            if plot_all == 1: 
                # plot source image 第几批 原始标签 或者 第几批 对抗标签
                # vutils.save_image(data.data, "./%s/%d_%d_original.png" %(opt.outf, batch_idx, ori_label), normalize=True)
                
                # plot adversarial image
                vutils.save_image(adv_x.data, "./%s/%d_%d_%d_adversarial.png" %(opt.outf, epoch, batch_idx, adv_label), normalize=True)
 
        masked_patch = torch.mul(mask, patch)
        patch = masked_patch.data.cpu().numpy()
        new_patch = np.zeros(patch_shape)
        for i in range(new_patch.shape[0]): 
            for j in range(new_patch.shape[1]): 
                new_patch[i][j] = submatrix(patch[i][j])
 
        patch = new_patch

        # log to file  
        progress_bar(batch_idx, len(train_loader), "Train Patch Success: {:.5f}".format(success/total))
        
    return patch


def attack(x, patch, mask):
    start_patch = copy.deepcopy(patch)
    netClassifier.eval()

    x_out = F.softmax(netClassifier(x))
    target_prob = x_out.data[0][target]
    # 让patch作为可更新的参数
    patch.requires_grad = True
    optimizer = torch.optim.Adam([patch])

    count = 0 
    # 给x贴上patch
    adv_x = torch.mul((1-mask),x) + torch.mul(mask,patch)
    adv_x = torch.clamp(adv_x, min_out, max_out) # 将值限制到合理范围内
    # 当 攻击目标的分类概率 还没达到指定置信度的时候
    while conf_target > target_prob:
        count += 1
        adv_x = torch.mul((1-mask),x) + torch.mul(mask,patch)
        adv_x = torch.clamp(adv_x, min_out, max_out) # 将值限制到合理范围内
        optimizer.zero_grad()
        adv_out = F.log_softmax(netClassifier(adv_x))
        
        Loss = -adv_out[0][target]
        Loss.backward()

        optimizer.step()

        out = F.softmax(adv_out)
        target_prob = out.detach()[0][target]

        if count >= opt.max_count:
            patch = start_patch
            break
           
    return adv_x, mask, patch 


if __name__ == '__main__':
    if patch_type == 'circle':
        patch, patch_shape = init_patch_circle(image_size, patch_size) 
    elif patch_type == 'square':
        patch, patch_shape = init_patch_square(image_size, patch_size) 
    else:
        sys.exit("Please choose a square or circle patch")
    
    for epoch in range(1, opt.epochs + 1):
        patch = train(epoch, patch, patch_shape)
        np.save("{}/Patch_{}_{}_{}.npy".format(patch_save_path, opt.netClassifier, epoch, patch_size),patch)
        vutils.save_image(torch.tensor(patch), "{}/Patch_{}_{}_{}.png".format(patch_save_path, opt.netClassifier, epoch, patch_size), normalize=True)