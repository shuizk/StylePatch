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
from utils import *
from utils.patch_utils import *
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--rpatch_file', type=str,default='RAP', help='patch file to test')
parser.add_argument('--bpatch_file', type=str,default='AP', help='patch file to test')
parser.add_argument('--lpatch_file', type=str,default='LaVAN', help='patch file to test')
parser.add_argument('--patch_net', type=str,default='vgg19', help='patch file to test')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--cuda', action='store_true', help='enables cuda') # --cuda为使用gpu
parser.add_argument('--card', type=str, default=0, help='enables card') # --card 1 使用第1块卡
parser.add_argument('--target', type=int, default=859, help='The target class: 859 == toaster') # 攻击目标
parser.add_argument('--patch_type', type=str, default='square', help='patch type: circle or square') # 对抗补丁的形状
parser.add_argument('--test_size', type=int, default=50, help='Number of test images') # 测试集大小
parser.add_argument('--image_size', type=int, default=224, help='the height / width of the input image to network')  # 网络的输入大小
parser.add_argument('--plot_all', type=int, default=1, help='1 == plot all successful adversarial images') # 画出所有成功的对抗补丁
parser.add_argument('--netClassifier', default='vgg19', help="The target classifier") # 要攻击的目标网络
parser.add_argument('--outf', default='./result/patch_images', help='folder to output images and model checkpoints') # 输出图像和模型检查点的文件夹
parser.add_argument('--manualSeed', type=int, default=None, help='manual seed') # 手工种子

opt = parser.parse_args() #得到参数
opt.cuda = True
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.card)
print(opt)

test_acc = []
# 构建log目录
# try:
#     os.makedirs(opt.outf)
# except OSError:
#     pass

# 设置随机数种子
if opt.manualSeed is None:
    opt.manualSeed = 4785
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
rpatch_file = opt.rpatch_file
bpatch_file = opt.bpatch_file
lpatch_file = opt.lpatch_file
target = opt.target
patch_type = opt.patch_type
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
idx = np.arange(10000)
np.random.shuffle(idx)
# 取出test_size大小的idx
test_idx = idx[:test_size]
# 加载训练数据 对数据进行缩放 裁剪等处理
 
test_loader = torch.utils.data.DataLoader(
    dset.ImageFolder('realImage', transforms.Compose([
        transforms.Resize([image_size,image_size]),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=1, shuffle=False, sampler=MySampler(test_idx),
    num_workers=opt.workers, pin_memory=True)

min_in, max_in = netClassifier.input_range[0], netClassifier.input_range[1]
min_in, max_in = np.array([min_in, min_in, min_in]), np.array([max_in, max_in, max_in])
min_out, max_out = np.min((min_in-mean)/std), np.max((max_in-mean)/std)

all_mask = []

def weighted_cam_test(patch, patch_shape, transform_type, trans_parameter):
    netClassifier.eval()
    success = 0
    untarget = 0
    total = 0
    top5 = 0
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
        patch, mask = weighted_cam_transform(data, patch, data_shape, plot_all, image_size, mean, std, batch_idx, labels.item())

        patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask)
        if opt.cuda:
            patch, mask = patch.cuda(), mask.cuda()

        adv_x = torch.mul((1-mask),data) + torch.mul(mask,patch)
        adv_x = torch.clamp(adv_x, min_out, max_out)

        if transform_type == "rotation":
            adv_x = rotation(adv_x, trans_parameter)
        if transform_type == "homography":
            adv_x = Homography(adv_x, trans_parameter)
        if transform_type == "barrel_distortion":
            adv_x = barrel_distortion(adv_x, trans_parameter)
        if transform_type == "occipital_distortion":
            adv_x = barrel_distortion(adv_x, trans_parameter)
        result = netClassifier(adv_x).data
        adv_label = result.max(1)[1][0]
        ori_label = labels.data[0]

        _ , topall = torch.sort(result, descending=True)
        if opt.target in topall[0][0:5]:
            top5 += 1

        if ori_label != adv_label:
            untarget += 1

        # target Top-1
        if adv_label == target:
            success += 1
            if plot_all == 1:
                vutils.save_image(adv_x.data, "./%s/%d_%d_adversarial_success.png" %(opt.outf, batch_idx, adv_label), normalize=True)
        else:
            if plot_all == 1:
                vutils.save_image(adv_x.data, "./%s/%d_%d_adversarial_fail.png" %(opt.outf, batch_idx, adv_label), normalize=True)    
        masked_patch = torch.mul(mask, patch)
        patch = masked_patch.data.cpu().numpy()
        new_patch = np.zeros(patch_shape)
        for i in range(new_patch.shape[0]): 
            for j in range(new_patch.shape[1]): 
                new_patch[i][j] = submatrix(patch[i][j])
 
        patch = new_patch

        # log to file  
        progress_bar(batch_idx, len(test_loader), "Test Success: {:.5f}".format(success/total))
    print("Weighted Test top1 target Success: {:.5f}".format(success/total))
    print("Weighted Test untarget Success: {:.5f}".format(untarget/total))
    print("Weighted Test top5 target Success: {:.5f}".format(top5/total))
    test_acc.append(success/total)
    temp = [success/total, untarget/total, top5/total]
    return temp


def test(patch, patch_type, patch_shape, transform_type, trans_parameter):
    """
        orgin patch attack test
        args:
            patch: patch to be tested
            patch_shape: size of patch
    """
    try:
        os.makedirs(f"./RAP/{patch_type}/{transform_type}")
    except OSError:
        pass
    netClassifier.eval()
    untarget = 0
    success = 0
    total = 0
    top5 = 0
    for batch_idx, (data, labels) in enumerate(test_loader):
        if opt.cuda:
            data = data.cuda()
            labels = labels.cuda()

        # prediction = netClassifier(data)

        # only computer adversarial examples on examples that are originally classified correctly        
        # if prediction.data.max(1)[1][0] != labels.data[0]:
        #     continue
      
        total += 1
        
        # transform path
        data_shape = data.data.cpu().numpy().shape

        if len(all_mask) < test_size:
            patch, mask = square_transform(patch, data_shape, patch_shape, image_size)
            all_mask.append(mask)
        else:
            mask = all_mask[batch_idx]
            x = np.zeros(data_shape)
            for i in range(x.shape[0]):
                random_x = np.nonzero(mask[0][0])[0][0]
                random_y = np.nonzero(mask[0][0])[1][0]    
                # apply patch to dummy image  
                x[i][0][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][0]
                x[i][1][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][1]
                x[i][2][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][2]
            patch = x

        patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask)
        if opt.cuda:
            patch, mask = patch.cuda(), mask.cuda()

        adv_x = torch.mul((1-mask),data) + torch.mul(mask,patch)
        adv_x = torch.clamp(adv_x, min_out, max_out)

        if transform_type == "rotation":
            adv_x = rotation(adv_x, trans_parameter)
        if transform_type == "homography":
            adv_x = Homography(adv_x, trans_parameter)
        if transform_type == "barrel_distortion":
            adv_x = barrel_distortion(adv_x, trans_parameter)
        if transform_type == "occipital_distortion":
            adv_x = barrel_distortion(adv_x, trans_parameter)

        result = netClassifier(adv_x).data
        adv_label = result.max(1)[1][0]
        ori_label = labels.data[0]

        
        _ , topall = torch.sort(result, descending=True)
        if opt.target in topall[0][0:5]:
            top5 += 1

        if ori_label != adv_label:
            untarget += 1

        if adv_label == target:
            success += 1
            if plot_all == 1:
                vutils.save_image(adv_x.data, f"./RAP/{patch_type}/{transform_type}/{batch_idx}_adversarial_success.png", normalize=True)
        else:
            if plot_all == 1:
                vutils.save_image(adv_x.data, f"./RAP/{patch_type}/{transform_type}/{batch_idx}_adversarial_fail.png", normalize=True)
        masked_patch = torch.mul(mask, patch)
        patch = masked_patch.data.cpu().numpy()
        new_patch = np.zeros(patch_shape)
        for i in range(new_patch.shape[0]): 
            for j in range(new_patch.shape[1]): 
                new_patch[i][j] = submatrix(patch[i][j])
 
        patch = new_patch

        # log to file  
        progress_bar(batch_idx, len(test_loader), "Test Success: {:.5f}".format(success/total))
    print("{} Test top1 target Success: {:.5f}".format(patch_type,success/total))
    print("{} Test untarget Success: {:.5f}".format(patch_type,untarget/total))
    print("{} Test top5 target Success: {:.5f}".format(patch_type,top5/total))
    test_acc.append(success/total)
    temp = [success/total, untarget/total, top5/total]
    return temp


if __name__ == '__main__':

    basic_patch = np.load(f"./patch/{bpatch_file}/Patch_{opt.patch_net}.npy")
    LaVAN_patch = np.load(f"./patch/{lpatch_file}/Patch_{opt.patch_net}.npy")
    robust_patch = np.load(f"./patch/{rpatch_file}/Patch_{opt.patch_net}.npy")

    basic_occipital_acc = []
    robust_occipital_acc = []
    LaVAN_occipital_acc = []
    print("====================枕形畸变20===============================")
    rota = 20
    temp1 = test(basic_patch,"AP",basic_patch.shape, transform_type="occipital_distortion", trans_parameter=-rota*1e-15)
    basic_occipital_acc.append(temp1)
    temp2 = test(robust_patch, "RAP", robust_patch.shape, transform_type="occipital_distortion", trans_parameter=-rota*1e-15)
    robust_occipital_acc.append(temp2)
    temp3 = test(LaVAN_patch, "LaVAN", LaVAN_patch.shape, transform_type="occipital_distortion", trans_parameter=-rota*1e-15)
    LaVAN_occipital_acc.append(temp3)

    
    print("====================桶形畸变20===============================")
    basic_barrel_acc = []
    robust_barrel_acc = []
    LaVAN_barrel_acc = []
    rota = 20
    temp1 = test(basic_patch,"AP", basic_patch.shape, transform_type="barrel_distortion", trans_parameter=rota*1e-15)
    basic_barrel_acc.append(temp1)
    temp2 = test(robust_patch, "RAP", robust_patch.shape, transform_type="barrel_distortion", trans_parameter=rota*1e-15)
    robust_barrel_acc.append(temp2)
    temp3 = test(LaVAN_patch, "LaVAN", LaVAN_patch.shape, transform_type="barrel_distortion", trans_parameter=rota*1e-15)
    LaVAN_barrel_acc.append(temp3)    

    
    print("=====================单应性变换13==============================")
    basic_homo_acc = []
    robust_homo_acc = []
    LaVAN_homo_acc = []
    rota = 13
    temp1 = test(basic_patch,"AP", basic_patch.shape, transform_type="homography", trans_parameter=rota)
    basic_homo_acc.append(temp1)
    temp2 = test(robust_patch, "RAP", robust_patch.shape, transform_type="homography", trans_parameter=rota)
    robust_homo_acc.append(temp2)
    temp3 = test(LaVAN_patch, "LaVAN", LaVAN_patch.shape, transform_type="homography", trans_parameter=rota)
    LaVAN_homo_acc.append(temp3)

    
    print("=====================旋转30度==============================")
    basic_rot_acc = []
    robust_rot_acc = []
    LaVAN_rot_acc = []
    rota = 30
    temp1 = test(basic_patch,"AP", basic_patch.shape, transform_type="rotation", trans_parameter=rota)
    basic_rot_acc.append(temp1)
    temp2 = test(robust_patch, "RAP", robust_patch.shape, transform_type="rotation", trans_parameter=rota)
    robust_rot_acc.append(temp2)
    temp3 = test(LaVAN_patch, "LaVAN", LaVAN_patch.shape, transform_type="rotation", trans_parameter=rota)
    LaVAN_rot_acc.append(temp3)
    
    # weighted_cam_test(robust_patch, "RAP", robust_patch.shape)

    
    import matplotlib.pyplot as plt
    # 数据
    categories = ['rotation 30', 'homography 13', 'barrel 20', 'occipital 20']  # 类别
    names = ['Top-1','Untarget','Top-5']
    for i in range(len(names)):
        AP = [basic_rot_acc[0][i], basic_homo_acc[0][i], basic_barrel_acc[0][i], basic_occipital_acc[0][i]]  # 第一组数据
        LaVAN = [LaVAN_rot_acc[0][i], LaVAN_homo_acc[0][i], LaVAN_barrel_acc[0][i], LaVAN_occipital_acc[0][i]]  # 第二组数据
        RAP = [robust_rot_acc[0][i], robust_homo_acc[0][i], robust_barrel_acc[0][i], robust_occipital_acc[0][i]]  # 第三组数据
        # 设置柱状图的宽度
        bar_width = 0.25
        # 计算柱状图的位置
        bar_positions1 = np.arange(len(categories))
        bar_positions2 = bar_positions1 + bar_width
        bar_positions3 = bar_positions2 + bar_width
        # 创建画布和子图
        fig, ax = plt.subplots()
        # 绘制柱状图
        ax.bar(bar_positions1, AP, width=bar_width, label='AP')
        ax.bar(bar_positions2, LaVAN, width=bar_width, label='LaVAN')
        ax.bar(bar_positions3, RAP, width=bar_width, label='RAP')
        # 设置刻度和标签
        ax.set_xticks(bar_positions2)
        ax.set_xticklabels(categories)
        ax.set_title(names[i])
        # 添加图例
        ax.legend(loc = 'upper left')
        # 显示图表
        plt.savefig(f"./RAP/{names[i]}.png")



