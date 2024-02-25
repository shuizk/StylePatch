import argparse
import torch
import torchvision.datasets as dset
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils
from pretrained_models_pytorch import pretrainedmodels
import os

parser = argparse.ArgumentParser()
parser.add_argument('--netClassifier', default='vgg19', help="The source classifier") # 攻击图像来源
parser.add_argument('--targetNet', default='vgg16', help="The target classifier") # 要攻击的目标网络
parser.add_argument('--method', default='LaVAN', help="The target classifier") # 攻击方法
parser.add_argument('--card', type=int, default=0, help='enables card') # --card  使用第i块卡

opt = parser.parse_args() #得到参数
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.card)
net = pretrainedmodels.__dict__[opt.targetNet](num_classes=1000, pretrained='imagenet')

patch_path = f"./result/{opt.method}/{opt.netClassifier}"

class ImageNetClass:
    def __init__(self, label, nums, success_nums, success_rate):
        self.label = label
        self.nums = nums
        self.success_nums = success_nums
        self.success_rate = success_rate

target = 859
target_img_num = 0 # 原来就是目标类的
cuda = True
label_nums = [0] * 1000
success_nums = [0] * 1000


# net = models.resnet18(pretrained=True)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean, std=std)

origin_loader = torch.utils.data.DataLoader(
    dset.ImageFolder('./result/origin_images', transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=1, shuffle=False, 
    num_workers=2)
 
patch_loader = torch.utils.data.DataLoader(
    dset.ImageFolder(patch_path, transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=1, shuffle=False, 
    num_workers=2)

net = net.cuda()
net.eval()
origin_labels = []
patch_labels = []
top5_labels = []


for batch_idx, (data, labels) in enumerate(origin_loader):
        if cuda:
            data = data.cuda()
            labels = labels.cuda()
        prediction = net(data)
        # vutils.save_image(data.data, "./temp/origin_images/%d_.png" %(batch_idx), normalize=True)
        origin_label = prediction.data.max(1)[1][0].item()
        origin_labels.append(origin_label)
        label_nums[origin_label] += 1
        if origin_label == target:
            target_img_num += 1

for batch_idx, (data, labels) in enumerate(patch_loader):
        if cuda:
            data = data.cuda()
            labels = labels.cuda()
        prediction = net(data)
        # vutils.save_image(data.data, "./temp/patch_images/%d_.png" %(batch_idx), normalize=True)
        top5_labels.append(torch.sort(prediction, descending=True)[1][0][0:5])
        patch_labels.append(prediction.data.max(1)[1][0].item())

untarget_num = 0
target_num = 0
top5_num = 0

total = len(patch_labels)

for i in range(total):
    if origin_labels[i] != patch_labels[i]:
        untarget_num += 1
        success_nums[origin_labels[i]] += 1
    if patch_labels[i] == target:
        target_num += 1
    if target in top5_labels[i]:
        top5_num += 1
        

    # print(origin_labels[i],patch_labels[i])
total -= target_img_num
target_num -= target_img_num
top5_num -= target_img_num

imagenet_classes = []

for i in range(1000):
    if label_nums[i] != 0:
        imagenet_class = ImageNetClass(i, label_nums[i], success_nums[i], round(success_nums[i]/label_nums[i], 4))
        imagenet_classes.append(imagenet_class)
        # print(f'{i} num is {label_nums[i]}\t success num is {success_nums[i]}\t untarget success rate: {round(success_nums[i]/label_nums[i], 4)}')
imagenet_classes.sort(key = lambda x: x.success_rate)

with open(f'{opt.method}_{opt.netClassifier}_attack_{opt.targetNet}.txt', 'w') as f:
    f.writelines(f'标签 数量 成功数量 成功率\n')
    for i in imagenet_classes:
        print(f'{i.label} num is {i.nums}\t success num is {i.success_nums}\t untarget success rate: {i.success_rate}')
        f.write(f'{i.label} {i.nums} {i.success_nums} {i.success_rate}\n')

print(f'{opt.method} {opt.netClassifier} attack {opt.targetNet} total is {total}')
print(f'Target Black Test {target_num} rate:{round(target_num/total, 4)}')
print(f'Top5 target Black Test {top5_num} rate:{round(top5_num/total, 4)}')
print(f'Untarget Black Test {untarget_num} rate:{round(untarget_num/total, 4)}')