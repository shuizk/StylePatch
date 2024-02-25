import os
import numpy as np
import torchvision.models as models
from cam.scorecam import *
import torch
from PIL import Image
from utils import *
import argparse
import random
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from pretrained_models_pytorch import pretrainedmodels
from utils.patch_utils import *
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--cuda', action='store_true', help='enables cuda') # --cuda为使用gpu
parser.add_argument('--card', type=str, default=0, help='enables card') # --card 1 使用第1块卡
parser.add_argument('--test_size', type=int, default=5, help='Number of test images') # 测试集大小
parser.add_argument('--image_size', type=int, default=224, help='the height / width of the input image to network')  # 网络的输入大小
parser.add_argument('--plot_all', type=int, default=1, help='1 == plot all successful adversarial images') # 画出所有成功的对抗补丁
parser.add_argument('--netClassifier', default='resnet50', help="The target classifier") # 要攻击的目标网络
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


if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
# 取出opt中的参数

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
idx = np.arange(5)
# np.random.shuffle(idx)
# 取出test_size大小的idx
test_idx = idx[:test_size]
# 加载训练数据 对数据进行缩放 裁剪等处理

test_loader = torch.utils.data.DataLoader(
    dset.ImageFolder('./WAP/cam_input', transforms.Compose([
        transforms.Resize([image_size,image_size]),
        transforms.ToTensor(),
        ToSpaceBGR(netClassifier.input_space=='BGR'),
        ToRange255(max(netClassifier.input_range)==255),
        normalize,
    ])),
    batch_size=1, shuffle=False, sampler=MySampler(test_idx),
    num_workers=opt.workers, pin_memory=True)

min_in, max_in = netClassifier.input_range[0], netClassifier.input_range[1]
min_in, max_in = np.array([min_in, min_in, min_in]), np.array([max_in, max_in, max_in])
min_out, max_out = np.min((min_in-mean)/std), np.max((max_in-mean)/std)


def weighted_cam(confidences, scorecam_maps, input_size=299):
    weighted_cam_map = np.zeros((input_size, input_size))
    for confidence, scorecam_map in zip(confidences, scorecam_maps):
        if scorecam_map.shape[2] != input_size:
            if scorecam_map.shape[2] == 299:
                img = scipy.ndimage.zoom(scorecam_map.numpy()[0][0],0.7491)
            else:
                img = scipy.ndimage.zoom(scorecam_map.numpy()[0][0],1.3348)
        else:
            img = scorecam_map.numpy()[0][0]
        weighted_cam_map += confidence * img
    weighted_cam_map = weighted_cam_map / sum(confidences)
    return weighted_cam_map

def chooseNet(input_, model_name, layer_name, input_size, label):
    if model_name == "alexnet":
        model = models.alexnet(pretrained=True).eval()
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=True).eval()
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True).eval()
    elif model_name == "inception_v3":
        model = models.inception_v3(pretrained=True).eval()
        input_size = (299, 299)
    # layer_name 是分类前最后一个卷积层激活后的层
    with torch.no_grad():
        model_dict = dict(type=model_name, arch=model, layer_name=layer_name, input_size=input_size)
        scorecam = ScoreCAM(model_dict)
        scorecam_map, confidence = scorecam(input_, class_idx=label)
        # print("{} is done!".format(model_name))
        return confidence, scorecam_map[0]

def toImage(input, mean, std):   
    """
        denormalize and detoTensor
    """
    x = input.clone()
    x = x.squeeze()
    for i in range(3):
        x[i] = x[i] * std[i] + mean[i]
    x = x.detach().cpu().numpy() * 255
    x = x.astype('uint8')
    x = np.transpose(x,(1,2,0))
    x = Image.fromarray(x)
    return x


def weighted_cam_transform(input, image_size, mean, std, batch_idx, label, plot_all=1):
    # nets = [("alexnet", "features_10"), ("vgg16", "features_29"), ("resnet18", "layer4"), ("inception_v3", "Mixed_7c")
    nets = [("alexnet", "features_10"), ("vgg16", "features_29"), ("resnet18", "layer4")]
    confidences = []
    scorecam_maps = []
    input_image = toImage(input, mean, std)
    for net in nets:
        # 针对不同网络变换输入
        if net[0] == "inception_v3":
            input_ = apply_transforms(input_image, 299)
        else:
            input_ = apply_transforms(input_image)
        if torch.cuda.is_available():
            input_ = input_.cuda()
        # 生成热力图
        confidence, scorecam_map = chooseNet(input_=input_, model_name=net[0], layer_name=net[1], input_size=image_size, label=label)
        confidences.append(confidence)
        scorecam_maps.append(scorecam_map.type(torch.FloatTensor).cpu()) # shape (1,1,size,size)
    input_ = apply_transforms(input_image, image_size)
    # 生成加权热力图
    weighted_cam_map = weighted_cam(confidences=confidences,scorecam_maps=scorecam_maps,input_size=image_size) # numpy格式

    if plot_all == 1:
        try:
            os.makedirs(f"./WAP/cam_images")
        except OSError:
            pass
        my_visualize(nets, input_.cpu(), scorecam_maps, confidences, weighted_cam_map, save_path="./WAP/cam_images/" + '{}_{}_weighted_cam.png'.format(batch_idx, label.item()))
    
    try:
        os.makedirs(f"./WAP/cam_npy")
    except OSError:
        pass
    scorecam_maps.append(weighted_cam_map)
    np.save(f"./cam_npy/{batch_idx}.npy", scorecam_maps)
    # 根据加权热力图寻找mask
    contirbution = 0
    window = np.zeros((50,50))
    mask_x = 0
    mask_y = 0
    # 滑动窗口寻找贡献度最大的部分
    for i in range(weighted_cam_map.shape[0]-window.shape[0]):
        for j in range(weighted_cam_map.shape[1]-window.shape[1]):
            window = weighted_cam_map[i:i+window.shape[0],j:j+window.shape[1]]
            current_contribution = np.sum(window)
            if current_contribution > contirbution:
                contirbution = current_contribution
                mask_x = i
                mask_y = j
    # 根据坐标制作mask
    # 将patch放入和图片同大小的矩阵中
    # rot = np.random.choice(4)
    x = np.zeros((1,3,224,224))
    for i in range(3):
        # patch[0][i] = np.rot90(patch[0][i], rot)
        x[0][i][mask_x: mask_x+window.shape[0], mask_y: mask_y+window.shape[1]] = np.ones_like(window)
    mask = np.copy(x)
    mask[mask != 0] = 1.0
    np.save(f"./WAP/cam_npy/{batch_idx}_mask.npy", mask)
    return x, mask

def test():
    netClassifier.eval()
    for batch_idx, (data, _) in enumerate(test_loader):
        print(f'Image {batch_idx} is processing...')
        if opt.cuda:
            data = data.cuda()

        prediction = netClassifier(data)

        # transform path
        data_shape = data.data.cpu().numpy().shape
        # if patch_type == 'circle':
        #     patch, mask, patch_shape = circle_transform(patch, data_shape, patch_shape, image_size)
        # elif patch_type == 'square':
        #     patch, mask = square_transform(patch, data_shape, patch_shape, image_size)
        weighted_cam_transform(data, image_size, mean, std, test_idx[batch_idx], prediction.data.max(1)[1])


if __name__ == '__main__':
    test()