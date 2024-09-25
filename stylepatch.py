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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"#根据显卡余量选择
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--cuda', action='store_true', help='enables cuda') # --cuda为使用gpu
#parser.add_argument('--card', type=int, default=0, help='enables card') # --card  使用第i块卡
parser.add_argument('--target', type=int, default=859, help='The target class: 859 == toaster') # 攻击目标
parser.add_argument('--conf_target', type=float, default=0.6, help='Stop attack on image when target classifier reaches this value for target class') # 当达到指定置信度停止
parser.add_argument('--max_count', type=int, default=1000, help='max number of iterations to find adversarial example') # 寻找对抗样本最大迭代次数
parser.add_argument('--patch_type', type=str, default='weighted', help='patch type: circle or square') # 对抗补丁的形状
parser.add_argument('--patch_size', type=float, default=0.7, help='patch size. E.g. 0.05 ~= 5% of image ') # 对抗补丁的大小
parser.add_argument('--train_size', type=int, default=120, help='Number of training images') # 训练集大小，可以任意改动
parser.add_argument('--image_size', type=int, default=224, help='the height / width of the input image to network')  # 网络的输入大小
parser.add_argument('--plot_all', type=int, default=1, help='1 == plot all successful adversarial images') # 画出所有成功的对抗补丁
parser.add_argument('--netClassifier', default='vgg19', help="The target classifier") # 要攻击的目标网络


# img_nums = 12814
img_nums = 10000
opt = parser.parse_args() #得到参数
opt.cuda = True
#os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.card)
print(opt)


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


# 根据参数创建模型
print("=> creating model ")
netClassifier = pretrainedmodels.__dict__[opt.netClassifier](num_classes=1000, pretrained='imagenet')
if opt.cuda:
    netClassifier.cuda()

print('==> Preparing data..')
# 标准化 imagenet的均值和方差
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
#normalize = transforms.Normalize(mean=mean, std=std)

idx = np.arange(img_nums)
# np.random.shuffle(idx)
# 取出train_size和test_size大小的idx 并且数据不重复
training_idx = idx[:train_size]
# 加载训练数据 对数据进行缩放 裁剪等处理
train_loader = torch.utils.data.DataLoader(
    dset.ImageFolder('./StylePatch/input/', transforms.Compose([
        transforms.Resize([image_size,image_size]),
        transforms.ToTensor(),])),
    batch_size=1, shuffle=False, sampler=MySampler(training_idx),
    num_workers=opt.workers, pin_memory=True)

min_in, max_in = netClassifier.input_range[0], netClassifier.input_range[1]
min_in, max_in = np.array([min_in, min_in, min_in]), np.array([max_in, max_in, max_in])
min_out, max_out = np.min((min_in-mean)/std), np.max((max_in-mean)/std)

# --------------------------------风格损失与内容损失部分---------------------------------------------

default_content_layers = ['conv_4']
default_style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
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
        _ ,topall = torch.sort(result, descending=True)
        
        
        path_imgs="./StylePatch/input/000/"
        list1=os.listdir(path_imgs)
        list1.sort()
        a=list1[batch_idx]
        vutils.save_image(adv_x.data,  f"{storage_dir}/patch_images/000/{a}" )
        masked_patch = torch.mul(mask, patch)

        
        patch = masked_patch.data.cpu().numpy()
        #np.set_printoptions(threshold=np.inf)
        new_patch = np.zeros(patch_shape)
        for i in range(new_patch.shape[0]): 
        	for j in range(new_patch.shape[1]):
        		new_patch[i][j] = submatrix(patch[i][j])
        patch = new_patch

def StylePatchattack(x, labels, style_img, patch, mask):
    # 将局部图像插值扩大到整体大小
    style_img = F.interpolate(style_img, size=(image_size, image_size), mode='bicubic')
    content_img = copy.deepcopy(x)
    model, content_losses, style_losses = get_model_and_losses(netClassifier, content_img, style_img, default_content_layers, default_style_layers, opt.netClassifier)

    netClassifier.eval()

    x_out = F.softmax(netClassifier(x))
    target_prob = x_out.data[0][target]
    
    # patch作为可更新的参数
    patch.requires_grad = True
    optimizer = torch.optim.SGD([patch], lr=1)

    count = 0
    while conf_target > target_prob:
        count += 1
        
        # 在给x贴上patch时，确保补丁部分的扰动不超过64
        patch_diff = patch - content_img  # 计算patch和原图之间的差异
        patch_diff = torch.clamp(patch_diff, -64 / 255.0, 64 / 255.0)  # 限制补丁像素精度变化在 [-64, 64] 之间，可以根据限制调整
        
        # 更新补丁，使其符合限制
        patch = content_img + patch_diff
        
        # 给x贴上更新后的patch
        adv_x = torch.mul((1-mask), x) + torch.mul(mask, patch)
        adv_x = torch.clamp(adv_x, min_out, max_out)  # 将整体图像的像素值限制在合法范围

        optimizer.zero_grad()
        adv_out = F.log_softmax(netClassifier(adv_x))
        adv_Loss = -adv_out[0][target]
        model(patch)
        
        # 计算内容和风格损失
        content_loss = sum([l.loss for l in content_losses])
        style_loss = sum([l.loss for l in style_losses])
        
        style_weight = 3
        content_weight = 3
        adv_weight = 6
        Loss = adv_weight * adv_Loss + content_weight * content_loss + style_weight * style_loss
        
        # 反向传播，使用retain_graph=True确保图不会被释放
        Loss.backward(retain_graph=True)
        optimizer.step()

        out = F.softmax(adv_out)
        target_prob = out.detach()[0][target]

        if count >= opt.max_count:
            break

    return adv_x, mask, patch




if __name__ == '__main__':
    StylePatchtrain()
    
path_imgs="./StylePatch/input/000/"
path_npy="./WAP/cam_npy/"
list1=os.listdir(path_imgs)
list1.sort()
list2=os.listdir(path_npy)
list2.sort()
for i in range(120):#数值为训练集的大小
	a=list1[i]
	c=list2[i]
	b=path_imgs+a
	d=path_npy+c
	os.remove(b)
	os.remove(d)
	
