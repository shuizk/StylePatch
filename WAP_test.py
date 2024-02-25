import argparse
import os
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from pretrained_models_pytorch import pretrainedmodels
from torch.utils.data.sampler import SequentialSampler
from utils import *
from utils.patch_utils import *
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--patch_file', type=str,default='./patch/AP/Patch_resnet50.npy', help='patch file to test')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--cuda', action='store_true', help='enables cuda') # --cuda为使用gpu
parser.add_argument('--card', type=str, default=0, help='enables card') # --card 1 使用第1块卡
parser.add_argument('--target', type=int, default=859, help='The target class: 859 == toaster') # 攻击目标
parser.add_argument('--patch_type', type=str, default='square', help='patch type: circle or square') # 对抗补丁的形状
parser.add_argument('--patch_size', type=float, default=0.05, help='patch size. E.g. 0.05 ~= 5% of image ') # 对抗补丁的大小
parser.add_argument('--test_size', type=int, default=200, help='Number of test images') # 测试集大小
parser.add_argument('--image_size', type=int, default=224, help='the height / width of the input image to network')  # 网络的输入大小
parser.add_argument('--plot_all', type=int, default=0, help='1 == plot all successful adversarial images') # 画出所有成功的对抗补丁
parser.add_argument('--netClassifier', default='resnet101', help="The target classifier") # 要攻击的目标网络
parser.add_argument('--outf', default='./logs', help='folder to output images and model checkpoints') # 输出图像和模型检查点的文件夹
parser.add_argument('--manualSeed', type=int, default=None, help='manual seed') # 手工种子

opt = parser.parse_args() #得到参数
opt.cuda = True
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.card)
print(opt)

test_acc = []
# 构建log目录
try:
    os.makedirs(opt.outf)
except OSError:
    pass

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
patch_file = opt.patch_file
target = opt.target
patch_type = opt.patch_type
patch_size = opt.patch_size
image_size = opt.image_size
test_size = opt.test_size
plot_all = opt.plot_all

# 条件为false时 执行断言
assert test_size <= 50000, "Traing set size + Test set size > Total dataset size"
# 根据参数创建模型
print("=> creating model ")
netClassifier = pretrainedmodels.__dict__[opt.netClassifier](num_classes=1000, pretrained='imagenet')
if opt.cuda:
    netClassifier.cuda()

print('==> Preparing data..')
# 标准化
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean, std=std)
idx = np.arange(50000)
# np.random.shuffle(idx)
# 取出test_size大小的idx
test_idx = idx[:test_size]
# 加载训练数据 对数据进行缩放 裁剪等处理
 
test_loader = torch.utils.data.DataLoader(
    dset.ImageFolder('imagenetdata/val', transforms.Compose([
        transforms.Resize([image_size,image_size]),
        transforms.ToTensor(),
        ToSpaceBGR(netClassifier.input_space=='BGR'),
        ToRange255(max(netClassifier.input_range)==255),
        normalize,
    ])),
    batch_size=1, shuffle=False, sampler=SequentialSampler(test_idx),
    num_workers=opt.workers, pin_memory=True)

min_in, max_in = netClassifier.input_range[0], netClassifier.input_range[1]
min_in, max_in = np.array([min_in, min_in, min_in]), np.array([max_in, max_in, max_in])
min_out, max_out = np.min((min_in-mean)/std), np.max((max_in-mean)/std)

def weighted_cam_test(patch, patch_shape):
    netClassifier.eval()
    success = 0
    top5 = 0
    untarget = 0
    total = 0
    for batch_idx, (data, labels) in enumerate(test_loader):
        if opt.cuda:
            data = data.cuda()
            labels = labels.cuda()

        prediction = netClassifier(data)

        # only computer adversarial examples on examples that are originally classified correctly        
        # if prediction.data.max(1)[1][0] != labels.data[0]:
        #     continue
      
        total += 1
        
        # transform path
        data_shape = data.data.cpu().numpy().shape
        # if patch_type == 'circle':
        #     patch, mask, patch_shape = circle_transform(patch, data_shape, patch_shape, image_size)
        # elif patch_type == 'square':
        #     patch, mask = square_transform(patch, data_shape, patch_shape, image_size)
        patch, mask = weighted_cam_transform(data, patch, data_shape, plot_all, image_size, mean, std, batch_idx, labels)

        patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask)
        if opt.cuda:
            patch, mask = patch.cuda(), mask.cuda()
 
        adv_x = torch.mul((1-mask),data) + torch.mul(mask,patch)
        adv_x = torch.clamp(adv_x, min_out, max_out)
        
        result = netClassifier(adv_x).data
        adv_label = result.max(1)[1][0]
        ori_label = labels.data[0]

        _ ,topall = torch.sort(result, descending=True)
        if target in topall[0][0:5]:
            top5 += 1
        
        if ori_label != adv_label:
            untarget += 1
        
        if adv_label == target:
            success += 1

        
        # if adv_label == target:
        #     success += 1
        #     if plot_all == 1:
        #         vutils.save_image(adv_x.data, "./%s/%d_%d_weighted_adversarial_success.png" %(opt.outf, batch_idx, adv_label), normalize=True)
        # else:
        #     if plot_all == 1:
        #         vutils.save_image(adv_x.data, "./%s/%d_%d_weighted_adversarial_fail.png" %(opt.outf, batch_idx, adv_label), normalize=True)    
        masked_patch = torch.mul(mask, patch)
        patch = masked_patch.data.cpu().numpy()
        new_patch = np.zeros(patch_shape)
        for i in range(new_patch.shape[0]): 
            for j in range(new_patch.shape[1]): 
                new_patch[i][j] = submatrix(patch[i][j])
 
        patch = new_patch

        # log to file  
        progress_bar(batch_idx, len(test_loader), "Weighted Test Success: {:.5f}".format(success/total))

    print("Weighted Test top1 target Success: {:.5f}".format(success/total))
    print("Weighted Test top5 target Success: {:.5f}".format(top5/total))
    print("Weighted Test untarget Success: {:.5f}".format(untarget/total))
   


def test(patch, patch_shape):
    """
        orgin patch attack test
        args:
            patch: patch to be tested
            patch_shape: size of patch
    """
    netClassifier.eval()
    success = 0
    top5 = 0
    untarget = 0
    total = 0
    for batch_idx, (data, labels) in enumerate(test_loader):
        if opt.cuda:
            data = data.cuda()
            labels = labels.cuda()

        prediction = netClassifier(data)

        # only computer adversarial examples on examples that are originally classified correctly        
        if prediction.data.max(1)[1][0] != labels.data[0]:
            continue
      
        total += 1
        
        # transform path
        data_shape = data.data.cpu().numpy().shape
        if patch_type == 'circle':
            patch, mask, patch_shape = circle_transform(patch, data_shape, patch_shape, image_size)
        elif patch_type == 'square':
            patch, mask = square_transform(patch, data_shape, patch_shape, image_size)
        patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask)
        if opt.cuda:
            patch, mask = patch.cuda(), mask.cuda()
 
        adv_x = torch.mul((1-mask),data) + torch.mul(mask,patch)
        adv_x = torch.clamp(adv_x, min_out, max_out)
        
        result = netClassifier(adv_x).data
        adv_label = result.max(1)[1][0]
        ori_label = labels.data[0]

        _ ,topall = torch.sort(result, descending=True)
        if target in topall[0][0:5]:
            top5 += 1
        
        if ori_label != adv_label:
            untarget += 1
        
        if adv_label == target:
            success += 1
    
        # if adv_label == target:
        #     success += 1
        #     if plot_all == 1:
        #         vutils.save_image(adv_x.data, "./%s/%d_%d_adversarial_success.png" %(opt.outf, batch_idx, adv_label), normalize=True)
        # else:
        #     if plot_all == 1:
        #         vutils.save_image(adv_x.data, "./%s/%d_%d_adversarial_fail.png" %(opt.outf, batch_idx, adv_label), normalize=True)
        masked_patch = torch.mul(mask, patch)
        patch = masked_patch.data.cpu().numpy()
        new_patch = np.zeros(patch_shape)
        for i in range(new_patch.shape[0]): 
            for j in range(new_patch.shape[1]): 
                new_patch[i][j] = submatrix(patch[i][j])
 
        patch = new_patch

        # log to file  
        progress_bar(batch_idx, len(test_loader), "Test Success: {:.5f}".format(success/total))
    print("Test top1 target Success: {:.5f}".format(success/total))
    print("Test top5 target Success: {:.5f}".format(top5/total))
    print("Test untarget Success: {:.5f}".format(untarget/total))


if __name__ == '__main__':
    patch = np.load(patch_file)
    lavan_patch = np.load("patch/LaVAN/Patch_resnet50.npy")
    print("==================================AP=====================================")
    test(patch, patch.shape)
    print("=================================LaVAN===================================")
    test(lavan_patch, lavan_patch.shape)
    print("==================================WAP====================================")
    weighted_cam_test(patch, patch.shape)

