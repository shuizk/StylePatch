import argparse
import os
import random
import numpy as np
import copy
import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data.sampler import SubsetRandomSampler,SequentialSampler

from pretrained_models_pytorch import pretrainedmodels

from utils import *
from utils.patch_utils import *
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--cuda', action='store_true', help='enables cuda') # --cuda为使用gpu
parser.add_argument('--card', type=int, default=0, help='enables card') # --card  使用第i块卡
parser.add_argument('--target', type=int, default=859, help='The target class: 859 == toaster') # 攻击目标
parser.add_argument('--conf_target', type=float, default=0.6, help='Stop attack on image when target classifier reaches this value for target class') # 当达到指定置信度停止
parser.add_argument('--max_count', type=int, default=1000, help='max number of iterations to find adversarial example') # 寻找对抗样本最大迭代次数
parser.add_argument('--patch_type', type=str, default='weighted', help='patch type: circle or square') # 对抗补丁的形状
parser.add_argument('--patch_size', type=float, default=0.05, help='patch size. E.g. 0.05 ~= 5% of image ') # 对抗补丁的大小
parser.add_argument('--train_size', type=int, default=5, help='Number of training images') # 训练集大小
parser.add_argument('--image_size', type=int, default=224, help='the height / width of the input image to network')  # 网络的输入大小
parser.add_argument('--plot_all', type=int, default=1, help='1 == plot all successful adversarial images') # 画出所有成功的对抗补丁
parser.add_argument('--netClassifier', default='vgg19', help="The target classifier") # 要攻击的目标网络
parser.add_argument('--manualSeed', type=int, default=None, help='manual seed') # 手工种子

# img_nums = 12814
img_nums = 5
opt = parser.parse_args() #得到参数
opt.cuda = True
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.card)
print(opt)


# 设置随机数种子
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

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
plot_all = opt.plot_all 

# 条件为false时 执行断言
assert train_size  <= img_nums, "Traing set size + Test set size > Total dataset size"
# 根据参数创建模型
print("=> creating model ")
netClassifier = pretrainedmodels.__dict__[opt.netClassifier](num_classes=1000, pretrained='imagenet')
if opt.cuda:
    netClassifier.cuda()

print('==> Preparing data..')
# 标准化 imagenet的均值和方差
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean, std=std)

idx = np.arange(img_nums)
# np.random.shuffle(idx)
# 取出train_size和test_size大小的idx 并且数据不重复
training_idx = idx[:train_size]
# 加载训练数据 对数据进行缩放 裁剪等处理
train_loader = torch.utils.data.DataLoader(
    dset.ImageFolder('./StylePatch/input/', transforms.Compose([
        transforms.Resize([image_size,image_size]),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=1, shuffle=False, sampler=MySampler(training_idx),
    num_workers=opt.workers, pin_memory=True)

min_in, max_in = netClassifier.input_range[0], netClassifier.input_range[1]
min_in, max_in = np.array([min_in, min_in, min_in]), np.array([max_in, max_in, max_in])
min_out, max_out = np.min((min_in-mean)/std), np.max((max_in-mean)/std)

# --------------------------------风格损失与内容损失部分---------------------------------------------

default_content_layers = ['conv_4']
default_style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
# 风格损失权重 和 内容损失权重

def StylePatchtrain():
    method_name = "Style"
    storage_dir = f"./StylePatch/result/{method_name}/{opt.netClassifier}"
    try:
        os.makedirs(f"{storage_dir}/patch_images/000")
    except OSError:
        pass
    netClassifier.eval()
    success = 0
    total = 0
    top5 = 0
    untarget = 0

    # for batch_idx, (data, labels) in enumerate(train_loader):
    for batch_idx, (_, (data,labels)) in zip(training_idx, enumerate(train_loader)):
        if opt.cuda:
            data = data.cuda()
            labels = labels.cuda()
        
        total += 1
        patch, patch_shape = init_patch_square(image_size, patch_size) 
        data_shape = data.data.cpu().numpy().shape
        patch, mask = weighted_cam_transform_demo(data, patch, data_shape, plot_all, image_size, mean, std, batch_idx, labels.item())
        patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask) # 转换成torch的tensor
        if opt.cuda:
            patch, mask = patch.cuda(), mask.cuda()
        style_img = get_style_img(data, mask)
        adv_x, mask, patch = StylePatchattack(data, labels, style_img, patch, mask)
        
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
        # 如果攻击成功 success+1
        if adv_label == target:
            # 如果画图为1
            if plot_all == 1: 
                vutils.save_image(adv_x.data,  f"{storage_dir}/patch_images/000/{batch_idx}_success_{adv_label.item()}.png" , normalize=True)
        else:
            if plot_all == 1: 
                vutils.save_image(adv_x.data,  f"{storage_dir}/patch_images/000/{batch_idx}_fail_{adv_label.item()}.png" , normalize=True)
        masked_patch = torch.mul(mask, patch)
        patch = masked_patch.data.cpu().numpy()
        new_patch = np.zeros(patch_shape)
        for i in range(new_patch.shape[0]): 
            for j in range(new_patch.shape[1]): 
                new_patch[i][j] = submatrix(patch[i][j])
        patch = new_patch
        # log to file  
        progress_bar(batch_idx, len(train_loader), "StylePatch Success: {:.5f}".format(success/total))
    print("Test top1 target Success: {:.5f}".format(success/total))
    print("Test top5 target Success: {:.5f}".format(top5/total))
    print("Test untarget Success: {:.5f}".format(untarget/total))

def StylePatchattack(x, labels, style_img, patch, mask):
    # 将局部图像插值扩大到 整体大小
    style_img = F.interpolate(style_img, size=(image_size,image_size), mode='bicubic')
    content_img = copy.deepcopy(x)
    model, content_losses, style_losses = get_model_and_losses(netClassifier, content_img, style_img, default_content_layers, default_style_layers, opt.netClassifier)

    netClassifier.eval()

    x_out = F.softmax(netClassifier(x))
    target_prob = x_out.data[0][target]
    # 让patch作为可更新的参数
    patch = torch.mul(mask,content_img)
    patch.requires_grad = True
    optimizer = torch.optim.SGD([patch], lr=1)
    # optimizer = torch.optim.Adam([patch])
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
        adv_Loss = -adv_out[0][target]
        model(patch)
        content_loss = 0
        style_loss = 0
        for l in content_losses:
            content_loss += l.loss
        for l in style_losses:
            style_loss += l.loss
        style_weight = 3
        content_weight = 3
        adv_weight = 6
        Loss = adv_weight * adv_Loss + content_weight * content_loss + style_weight * style_loss
        Loss.backward()
        optimizer.step()
        out = F.softmax(adv_out)
        target_prob = out.detach()[0][target]

        if count >= opt.max_count:
            break
           
    return adv_x, mask, patch 
# --------------------------------风格损失与内容损失部分---------------------------------------------

# -----------------------------------------AP------------------------------------------------------

def APtrain():
    method_name = "AP"
    storage_dir = f"./StylePatch/result/{method_name}/{opt.netClassifier}"
    try:
        os.makedirs(f"{storage_dir}/patch_images/000")
    except OSError:
        pass
    netClassifier.eval()
    success = 0
    total = 0
    top5 = 0
    untarget = 0

    # for batch_idx, (data, labels) in enumerate(train_loader):
    for batch_idx, (_, (data,labels)) in zip(training_idx, enumerate(train_loader)):
        if opt.cuda:
            data = data.cuda()
            labels = labels.cuda()
        total += 1
        patch, patch_shape = init_patch_square(image_size, patch_size) 
        data_shape = data.data.cpu().numpy().shape
        patch, mask = weighted_cam_transform_demo(data, patch, data_shape, plot_all, image_size, mean, std, batch_idx, labels.item())
        patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask) # 转换成torch的tensor
        if opt.cuda:
            patch, mask = patch.cuda(), mask.cuda()
        adv_x, mask, patch = APattack(data, patch, mask)

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
        # 如果攻击成功 success+1
        if adv_label == target:
            # 如果画图为1
            if plot_all == 1: 
                vutils.save_image(adv_x.data,  f"{storage_dir}/patch_images/000/{batch_idx}_success_{adv_label.item()}.png" , normalize=True)
        else:
            if plot_all == 1: 
                vutils.save_image(adv_x.data,  f"{storage_dir}/patch_images/000/{batch_idx}_fail_{adv_label.item()}.png" , normalize=True)
        masked_patch = torch.mul(mask, patch)
        patch = masked_patch.data.cpu().numpy()
        new_patch = np.zeros(patch_shape)
        for i in range(new_patch.shape[0]): 
            for j in range(new_patch.shape[1]): 
                new_patch[i][j] = submatrix(patch[i][j])
        patch = new_patch
        # log to file  
        progress_bar(batch_idx, len(train_loader), "AP Success: {:.5f}".format(success/total))
    print("Test top1 target Success: {:.5f}".format(success/total))
    print("Test top5 target Success: {:.5f}".format(top5/total))
    print("Test untarget Success: {:.5f}".format(untarget/total))

def APattack(x, patch, mask):
    netClassifier.eval()
    x_out = F.softmax(netClassifier(x))
    target_prob = x_out.data[0][target]
    # 给x贴上patch
    adv_x = torch.mul((1-mask),x) + torch.mul(mask,patch)
    count = 0 
    # 当 攻击目标的分类概率 还没达到指定置信度的时候
    while conf_target > target_prob:
        count += 1
        adv_x.requires_grad=True
        adv_out = F.log_softmax(netClassifier(adv_x))
        Loss = -adv_out[0][target]
        Loss.backward()
        adv_grad = adv_x.grad.clone()
        adv_x.grad.data.zero_()
        # 只更新patch部分
        patch -= 3*adv_grad 
        patch = torch.clamp(patch, min_out, max_out)
        adv_x = torch.mul((1-mask),x) + torch.mul(mask,patch) # 利用矩阵的点乘 将patch和x 合成到一起
        adv_x = torch.clamp(adv_x, min_out, max_out) # 将值限制到合理范围内
        out = F.softmax(netClassifier(adv_x))
        target_prob = out.data[0][target]
        if count >= opt.max_count:
            break
    return adv_x, mask, patch
# -----------------------------------------AP------------------------------------------------------

# ----------------------------------------LaVAN----------------------------------------------------
def LaVANtrain():
    method_name = "LaVAN"
    storage_dir = f"./StylePatch/result/{method_name}/{opt.netClassifier}"
    try:
        os.makedirs(f"{storage_dir}/patch_images/000")
    except OSError:
        pass
    netClassifier.eval()
    success = 0
    total = 0
    top5 = 0
    untarget = 0
    # for batch_idx, (data, labels) in enumerate(train_loader):
    for batch_idx, (_, (data,labels)) in zip(training_idx, enumerate(train_loader)):
        if opt.cuda:
            data = data.cuda()
            labels = labels.cuda()
        total += 1
        patch, patch_shape = init_patch_square(image_size, patch_size) 
        data_shape = data.data.cpu().numpy().shape
        patch, mask = weighted_cam_transform_demo(data, patch, data_shape, plot_all, image_size, mean, std, batch_idx, labels.item())
        patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask) # 转换成torch的tensor
        if opt.cuda:
            patch, mask = patch.cuda(), mask.cuda()
        adv_x, mask, patch = LaVANattack(data, labels, patch, mask)

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
        # 如果攻击成功 success+1
        if adv_label == target:
            # 如果画图为1
            if plot_all == 1: 
                vutils.save_image(adv_x.data,  f"{storage_dir}/patch_images/000/{batch_idx}_success_{adv_label.item()}.png" , normalize=True)
        else:
            if plot_all == 1: 
                vutils.save_image(adv_x.data,  f"{storage_dir}/patch_images/000/{batch_idx}_fail_{adv_label.item()}.png" , normalize=True)
        masked_patch = torch.mul(mask, patch)
        patch = masked_patch.data.cpu().numpy()
        new_patch = np.zeros(patch_shape)
        for i in range(new_patch.shape[0]): 
            for j in range(new_patch.shape[1]): 
                new_patch[i][j] = submatrix(patch[i][j])
        patch = new_patch
        # log to file  
        progress_bar(batch_idx, len(train_loader), "LaVAN Success: {:.5f}".format(success/total))
    print("Test top1 target Success: {:.5f}".format(success/total))
    print("Test top5 target Success: {:.5f}".format(top5/total))
    print("Test untarget Success: {:.5f}".format(untarget/total))

def LaVANattack(x, labels, patch, mask):
    targets = target * torch.ones_like(labels)
    netClassifier.eval()
    x_out = F.softmax(netClassifier(x))
    target_prob = x_out.data[0][target]
    # 给x贴上patch
    adv_x = torch.mul((1-mask),x) + torch.mul(mask,patch)
    count = 0 
    # 当 攻击目标的分类概率 还没达到指定置信度的时候
    while conf_target > target_prob:
        # optimizer.zero_grad()
        count += 1
        adv_x.requires_grad=True
        adv_out = netClassifier(adv_x)
        Loss = 2*F.cross_entropy(adv_out, targets) - F.cross_entropy(adv_out, labels)
        Loss.backward()

        adv_grad = adv_x.grad.clone()
        
        adv_x.grad.data.zero_()
        # 只更新patch部分
        patch -= 3*adv_grad 
        patch = torch.clamp(patch, min_out, max_out)
        # optimizer.step()
        adv_x = torch.mul((1-mask),x) + torch.mul(mask,patch) # 利用矩阵的点乘 将patch和x 合成到一起
        adv_x = torch.clamp(adv_x, min_out, max_out) # 将值限制到合理范围内
 
        out = F.softmax(netClassifier(adv_x))
        target_prob = out.data[0][target]

        if count >= opt.max_count:
            break

    return adv_x, mask, patch 

# ----------------------------------------LaVAN----------------------------------------------------

# ----------------------------------------PS-GAN---------------------------------------------------
def PSGANtrain():
    method_name = "PSGAN"
    storage_dir = f"./StylePatch/result/{method_name}/{opt.netClassifier}"
    try:
        os.makedirs(f"{storage_dir}/patch_images/000")
    except OSError:
        pass
    netClassifier.eval()
    success = 0
    total = 0
    top5 = 0
    untarget = 0

    # for batch_idx, (data, labels) in enumerate(train_loader):
    for batch_idx, (_, (data,labels)) in zip(training_idx, enumerate(train_loader)):
        if opt.cuda:
            data = data.cuda()
            labels = labels.cuda()
        total += 1

        patch, patch_shape = init_patch_square(image_size, patch_size, patch_file="full_numpy_bitmap_cruise_ship.npy") 
        data_shape = data.data.cpu().numpy().shape
        patch, mask = weighted_cam_transform_demo(data, patch, data_shape, plot_all, image_size, mean, std, batch_idx, labels.item())
        patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask) # 转换成torch的tensor
        if opt.cuda:
            patch, mask = patch.cuda(), mask.cuda()
        origin_patch = copy.deepcopy(patch)
        style_img = copy.deepcopy(patch)
        adv_x, mask, patch = PSGANattack(data, labels, origin_patch, style_img, patch, mask)
        
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
        # 如果攻击成功 success+1
        if adv_label == target:
            # 如果画图为1
            if plot_all == 1: 
                vutils.save_image(adv_x.data,  f"{storage_dir}/patch_images/000/{batch_idx}_success_{adv_label.item()}.png" , normalize=True)
                
        else:
            if plot_all == 1: 
                vutils.save_image(adv_x.data,  f"{storage_dir}/patch_images/000/{batch_idx}_fail_{adv_label.item()}.png" , normalize=True)
        masked_patch = torch.mul(mask, patch)
        patch = masked_patch.data.cpu().numpy()
        new_patch = np.zeros(patch_shape)
        for i in range(new_patch.shape[0]): 
            for j in range(new_patch.shape[1]): 
                new_patch[i][j] = submatrix(patch[i][j])
 
        patch = new_patch
        # log to file  
        progress_bar(batch_idx, len(train_loader), "PS-GAN Success: {:.5f}".format(success/total))
    print("Test top1 target Success: {:.5f}".format(success/total))
    print("Test top5 target Success: {:.5f}".format(top5/total))
    print("Test untarget Success: {:.5f}".format(untarget/total))

def PSGANattack(x, labels, content_img, style_img, patch, mask):
    model, PS_losses, _ = get_model_and_PSlosses(netClassifier, content_img, style_img, default_content_layers, default_style_layers, opt.netClassifier)
    netClassifier.eval()
    x_out = F.softmax(netClassifier(x))
    target_prob = x_out.data[0][target]
    # 让patch作为可更新的参数
    patch = torch.mul(mask,content_img)
    patch.requires_grad = True
    optimizer = torch.optim.SGD([patch], lr=1)
    # optimizer = torch.optim.Adam([patch])
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
        adv_Loss = -adv_out[0][target]
        model(patch)
        PS_loss = 0
        for l in PS_losses:
            PS_loss += l.loss
        PS_weight = 20
        adv_weight = 1
        Loss = adv_weight * adv_Loss + PS_weight * PS_loss
        Loss.backward()
        optimizer.step()
        out = F.softmax(adv_out)
        target_prob = out.detach()[0][target]

        if count >= opt.max_count:
            break

    return adv_x, mask, patch 
# ----------------------------------------PS-GAN---------------------------------------------------

if __name__ == '__main__':
    StylePatchtrain()
    APtrain()
    LaVANtrain()
    PSGANtrain()
        