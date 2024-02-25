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
parser.add_argument('--epochs', type=int, default=4, help='number of epochs to train for')
parser.add_argument('--cuda', action='store_true', help='enables cuda') # --cuda为使用gpu
parser.add_argument('--card', type=int, default=1, help='enables card') # --card  使用第i块卡
parser.add_argument('--max_count', type=int, default=50000, help='max number of iterations to find adversarial example') # 寻找对抗样本最大迭代次数
parser.add_argument('--patch_type', type=str, default='weighted', help='patch type: circle or square') # 对抗补丁的形状
parser.add_argument('--patch_size', type=float, default=0.05, help='patch size. E.g. 0.05 ~= 5% of image ') # 对抗补丁的大小

parser.add_argument('--train_size', type=int, default=1000, help='Number of training images') # 训练集大小
parser.add_argument('--test_size', type=int, default=1000, help='Number of test images') # 测试集大小

parser.add_argument('--image_size', type=int, default=224, help='the height / width of the input image to network')  # 网络的输入大小

parser.add_argument('--plot_all', type=int, default=1, help='1 == plot all successful adversarial images') # 画出所有成功的对抗补丁

parser.add_argument('--netClassifier', default='vgg19', help="The target classifier") # 要攻击的目标网络

parser.add_argument('--outf', default='./result/patch_images', help='folder to output images and model checkpoints') # 输出图像和模型检查点的文件夹
parser.add_argument('--manualSeed', type=int, default=None, help='manual seed') # 手工种子

img_nums = 12814
opt = parser.parse_args() #得到参数
opt.cuda = True
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.card)
print(opt)

test_acc = []

try:
    os.makedirs(opt.outf)
except OSError:
    pass

try:
    os.makedirs("./result/origin_images")
except OSError:
    pass

try:
    os.makedirs("./result/cam_images")
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

max_count = opt.max_count
patch_type = opt.patch_type
patch_size = opt.patch_size
image_size = opt.image_size
train_size = opt.train_size
test_size = opt.test_size
plot_all = opt.plot_all 

# 条件为false时 执行断言
assert train_size + test_size <= img_nums, "Traing set size + Test set size > Total dataset size"
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

idx = np.arange(img_nums)
np.random.shuffle(idx)
# 取出train_size和test_size大小的idx 并且数据不重复
training_idx = idx[:train_size]
test_idx = idx[train_size:train_size+test_size]
# 加载训练数据 对数据进行缩放 裁剪等处理
train_loader = torch.utils.data.DataLoader(
    dset.ImageFolder('realImage', transforms.Compose([
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
    dset.ImageFolder('realImage', transforms.Compose([
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


# --------------------------------风格损失与内容损失部分---------------------------------------------
default_content_layers = ['conv_4']
default_style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
# 风格损失权重 和 内容损失权重




def train(epoch):
    netClassifier.eval()
    success = 0
    total = 0

    for batch_idx, (data, labels) in enumerate(train_loader):
        if opt.cuda:
            data = data.cuda()
            labels = labels.cuda()
        prediction = netClassifier(data)
        vutils.save_image(data.data, "./result/origin_images/%d_%d_.png" %(batch_idx, labels.item()), normalize=True)
        # 只攻击分类正确样本，如果分类标签不正确则继续        
        if prediction.data.max(1)[1][0] != labels.data[0]:
            continue
        
        total += 1
        
        # transform path
        # if patch_type == 'circle':
        #     patch, patch_shape = init_patch_circle(image_size, patch_size) 
        # elif patch_type == 'square':
        patch, patch_shape = init_patch_square(image_size, patch_size) 
        # else:
        #     sys.exit("Please choose a square or circle patch")
        data_shape = data.data.cpu().numpy().shape

        if patch_type == 'circle':
            patch, mask, patch_shape = circle_transform(patch, data_shape, patch_shape, image_size)
        elif patch_type == 'square':
            patch, mask  = square_transform(patch, data_shape, patch_shape, image_size)
        # 加权热力图定位
        elif patch_type == 'weighted':
            patch, mask = weighted_cam_transform(data, patch, data_shape, plot_all, image_size, mean, std, batch_idx, labels.item())
        
        patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask) # 转换成torch的tensor
        if opt.cuda:
            patch, mask = patch.cuda(), mask.cuda()
        style_img = get_style_img(data, mask)
        adv_x, mask, patch = attack(data, labels, style_img, patch, mask)
        
        adv_label = netClassifier(adv_x).data.max(1)[1][0]
        ori_label = labels.data[0]
        # 如果攻击成功 success+1
        if adv_label != ori_label:
            success += 1
            # 如果画图为1
            if plot_all == 1: 
                # plot source image 第几批 原始标签 或者 第几批 对抗标签
                # vutils.save_image(data.data, "./%s/%d_%d_original.png" %(opt.outf, batch_idx, ori_label), normalize=True)
                
                # plot adversarial image
                vutils.save_image(adv_x.data, "./%s/%d_%d_%d_adversarial_success.png" %(opt.outf, epoch, batch_idx, adv_label), normalize=True)
        else:
            if plot_all == 1: 
                # plot source image 第几批 原始标签 或者 第几批 对抗标签
                # vutils.save_image(data.data, "./%s/%d_%d_original.png" %(opt.outf, batch_idx, ori_label), normalize=True)
                
                # plot adversarial image
                vutils.save_image(adv_x.data, "./%s/%d_%d_%d_adversarial_fail.png" %(opt.outf, epoch, batch_idx, adv_label), normalize=True)
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


def attack(x, labels, style_img, patch, mask):
    # 将局部图像插值扩大到 整体大小
    ori_label = labels.data[0]
    adv_label = ori_label
    style_img = F.interpolate(style_img, size=(image_size,image_size), mode='bicubic')
    content_img = copy.deepcopy(x)
    model, content_losses, style_losses = get_model_and_losses(netClassifier, content_img, style_img, default_content_layers, default_style_layers)

    netClassifier.eval()
    
    # 让patch作为可更新的参数
    patch = torch.mul(mask,content_img)
    patch.requires_grad = True
    optimizer = torch.optim.Adam([patch])

    count = 0 
    # 给x贴上patch
    adv_x = torch.mul((1-mask),x) + torch.mul(mask,patch)
    adv_x = torch.clamp(adv_x, min_out, max_out) # 将值限制到合理范围内
    # 当 攻击目标的分类概率 还没达到指定置信度的时候
    while adv_label.item() == ori_label.item() or Loss.item() > 5:
        count += 1
        adv_x = torch.mul((1-mask),x) + torch.mul(mask,patch)
        adv_x = torch.clamp(adv_x, min_out, max_out) # 将值限制到合理范围内
        optimizer.zero_grad()
        adv_out = F.log_softmax(netClassifier(adv_x))
        adv_Loss = adv_out[0][ori_label.item()]
        model(patch)
        content_loss = 0
        style_loss = 0
        for l in content_losses:
            content_loss += l.loss
        for l in style_losses:
            style_loss += l.loss
        style_weight = 20
        content_weight = 40
        adv_weight = 20
        Loss = adv_weight * adv_Loss + content_weight * content_loss + style_weight * style_loss
        Loss.backward()
        optimizer.step()

        adv_label = netClassifier(adv_x).data.max(1)[1][0]

        if count >= opt.max_count:
            break
           
    return adv_x, mask, patch 


if __name__ == '__main__':

    for epoch in range(opt.epochs, opt.epochs + 1):
        train(epoch)
        