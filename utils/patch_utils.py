import os
import sys
import time
import math
import numpy as np
import torchvision.models as models
from cam.scorecam import *
import torch
from torch.utils.data.sampler import Sampler
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
from PIL import Image
from scipy.ndimage.interpolation import rotate
from utils import *
import cv2

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 35.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    if msg:
        L.append(' ' + msg)
    L.append(' | Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def submatrix(arr):
    x, y = np.nonzero(arr)
    # Using the smallest and largest x and y indices of nonzero elements, 
    # we can find the desired rectangular bounds.  
    # And don't forget to add 1 to the top bound to avoid the fencepost problem.
    return arr[x.min():x.max()+1, y.min():y.max()+1]


class ToSpaceBGR(object):
    def __init__(self, is_bgr):
        self.is_bgr = is_bgr
    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor


class ToRange255(object):
    def __init__(self, is_255):
        self.is_255 = is_255
    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor

class MySampler(Sampler):
    def __init__(self, indices):
        self.indices = indices
    
    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def init_patch_circle(image_size, patch_size):
    image_size = image_size**2
    noise_size = int(image_size*patch_size)
    radius = int(math.sqrt(noise_size/math.pi)) # 圆的面积除以π 开根号 得到半径
    patch = np.zeros((1, 3, radius*2, radius*2))  # 初始化patch 1*3*直径*直径
    for i in range(3):
        a = np.zeros((radius*2, radius*2))    
        cx, cy = radius, radius # The center of circle 
        y, x = np.ogrid[-radius: radius, -radius: radius]
        index = x**2 + y**2 <= radius**2
        a[cy-radius:cy+radius, cx-radius:cx+radius][index] = np.random.rand()
        idx = np.flatnonzero((a == 0).all((1)))
        a = np.delete(a, idx, axis=0)
        patch[0][i] = np.delete(a, idx, axis=1)
    return patch, patch.shape


def circle_transform(patch, data_shape, patch_shape, image_size):
    # get dummy image 
    x = np.zeros(data_shape)
   
    # get shape
    m_size = patch_shape[-1]
    
    for i in range(x.shape[0]):

        # random rotation 从0-360中随机抽取一个数 旋转patch的RGB三个通道
        rot = np.random.choice(360)
        for j in range(patch[i].shape[0]):
            patch[i][j] = rotate(patch[i][j], angle=rot, reshape=False)
        
        # random location
        random_x = np.random.choice(image_size)
        if random_x + m_size > x.shape[-1]:
            while random_x + m_size > x.shape[-1]:
                random_x = np.random.choice(image_size)
        random_y = np.random.choice(image_size)
        if random_y + m_size > x.shape[-1]:
            while random_y + m_size > x.shape[-1]:
                random_y = np.random.choice(image_size)
       
        # apply patch to dummy image  把x随机一块儿位置替换成patch x是和输入图片相同的0矩阵
        x[i][0][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][0]
        x[i][1][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][1]
        x[i][2][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][2]
    
    mask = np.copy(x)
    mask[mask != 0] = 1.0
    
    return x, mask, patch.shape


def init_patch_square(image_size, patch_size, patch_file=None):
    # get mask
    if patch_file is None:
        image_size = image_size**2
        noise_size = image_size*patch_size
        noise_dim = int(noise_size**(0.5))
        patch = np.random.rand(1,3,noise_dim,noise_dim)
    else:
        patch = cv2.imread("ship.png")[:,:,::-1]
        patch = patch.astype(int) / 255
        patch[patch == 0] = 1e-5
        patch = np.array([patch.transpose(2,0,1)])

    return patch, patch.shape


def square_transform(patch, data_shape, patch_shape, image_size):
    # get dummy image
    x = np.zeros(data_shape)
    
    # get shape
    m_size = patch_shape[-1]
    
    for i in range(x.shape[0]):

        # random rotation 按照90度旋转
        # rot = np.random.choice(4)
        # for j in range(patch[i].shape[0]):
        #     patch[i][j] = np.rot90(patch[i][j], rot)
        
        # random location
        random_x = np.random.choice(image_size)
        if random_x + m_size > x.shape[-1]:
            while random_x + m_size > x.shape[-1]:
                random_x = np.random.choice(image_size)
        random_y = np.random.choice(image_size)
        if random_y + m_size > x.shape[-1]:
            while random_y + m_size > x.shape[-1]:
                random_y = np.random.choice(image_size)
       
        # apply patch to dummy image  
        x[i][0][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][0]
        x[i][1][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][1]
        x[i][2][random_x:random_x+patch_shape[-1], random_y:random_y+patch_shape[-1]] = patch[i][2]
    
    mask = np.copy(x)
    mask[mask != 0] = 1.0
    
    return x, mask

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

def weighted_cam_transform(input, patch, data_shape, plot_all, image_size, mean, std, batch_idx, label):
    # nets = [("alexnet", "features_10"), ("vgg16", "features_29"), ("resnet18", "layer4"), ("inception_v3", "Mixed_7c")]
    if not os.path.exists(f"./cam_npy/{batch_idx}_mask.npy"):

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
                os.makedirs(f"./cam_images_1")
            except OSError:
                pass
            my_visualize(nets, input_.cpu(), scorecam_maps, confidences, weighted_cam_map, save_path="./cam_images_1/" + '{}_{}_weighted_cam.png'.format(batch_idx, label))
        
        try:
            os.makedirs(f"./cam_npy")
        except OSError:
            pass
        scorecam_maps.append(weighted_cam_map)
        np.save(f"./cam_npy/{batch_idx}.npy", scorecam_maps)
        # 根据加权热力图寻找mask
        contirbution = 0
        window = np.zeros(patch[0][0].shape)
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
        x = np.zeros(data_shape)
        for i in range(patch[0].shape[0]):
            # patch[0][i] = np.rot90(patch[0][i], rot)
            x[0][i][mask_x: mask_x+window.shape[0], mask_y: mask_y+window.shape[1]] = patch[0][i]
        mask = np.copy(x)
        mask[mask != 0] = 1.0
        np.save(f"./cam_npy/{batch_idx}_mask.npy", mask)
        return x, mask 
    else:
        mask = np.load(f"./cam_npy/{batch_idx}_mask.npy", allow_pickle=True)
        mask_x = np.nonzero(mask[0][0])[0][0]
        mask_y = np.nonzero(mask[0][0])[1][0]
        x = np.zeros(data_shape)
        for i in range(patch[0].shape[0]):
            # patch[0][i] = np.rot90(patch[0][i], rot)
            x[0][i][mask_x: mask_x+patch[0][0].shape[0], mask_y: mask_y+patch[0][0].shape[1]] = patch[0][i]
        return x, mask


def weighted_cam_transform_demo(input, patch, data_shape, plot_all, image_size, mean, std, batch_idx, label):
    # nets = [("alexnet", "features_10"), ("vgg16", "features_29"), ("resnet18", "layer4"), ("inception_v3", "Mixed_7c")]
    if not os.path.exists(f"./WAP/cam_npy/{batch_idx}_mask.npy"):

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
            my_visualize(nets, input_.cpu(), scorecam_maps, confidences, weighted_cam_map, save_path="./WAP/cam_images/" + '{}_{}_weighted_cam.png'.format(batch_idx, label))
        
        try:
            os.makedirs(f"./WAP/cam_npy")
        except OSError:
            pass
        scorecam_maps.append(weighted_cam_map)
        np.save(f"./WAP/cam_npy/{batch_idx}.npy", scorecam_maps)
        # 根据加权热力图寻找mask
        contirbution = 0
        window = np.zeros(patch[0][0].shape)
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
        x = np.zeros(data_shape)
        for i in range(patch[0].shape[0]):
            # patch[0][i] = np.rot90(patch[0][i], rot)
            x[0][i][mask_x: mask_x+window.shape[0], mask_y: mask_y+window.shape[1]] = patch[0][i]
        mask = np.copy(x)
        mask[mask != 0] = 1.0
        np.save(f"./WAP/cam_npy/{batch_idx}_mask.npy", mask)
        return x, mask 
    else:
        mask = np.load(f"./WAP/cam_npy/{batch_idx}_mask.npy", allow_pickle=True)
        mask_x = np.nonzero(mask[0][0])[0][0]
        mask_y = np.nonzero(mask[0][0])[1][0]
        x = np.zeros(data_shape)
        for i in range(patch[0].shape[0]):
            # patch[0][i] = np.rot90(patch[0][i], rot)
            x[0][i][mask_x: mask_x+patch[0][0].shape[0], mask_y: mask_y+patch[0][0].shape[1]] = patch[0][i]
        return x, mask


    

def get_style_img(img, patch_mask):
    style_img = torch.mul(img, patch_mask)
    loc = torch.nonzero(style_img)
    style_img = style_img[:,:,loc[0][2]:loc[-1][2],loc[0][3]:loc[-1][3]]
    return style_img

class ContentLoss(torch.nn.Module):
    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.target = target.detach()

    def forward(self, input):
        # 均方误差损失函数 sum((a-b)^2)/n 
        # self.loss = F.mse_loss(input, self.target)
        self.loss = 1 - torch.cosine_similarity(input.view(input.shape[1],-1), self.target.view(input.shape[1],-1)).sum() / input.shape[1]
        return input

# 计算x的格里姆矩阵
def gram(x: torch.Tensor):
    # x is a [n, c, h, w] array
    n, c, h, w = x.shape

    features = x.reshape(n * c, h * w)
    features = torch.mm(features, features.T) / n / c / h / w
    return features

# 定义风格损失
class StyleLoss(torch.nn.Module):
    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.target = gram(target.detach()).detach()

    def forward(self, input):
        G = gram(input)
        # 输入和目标的 格里姆矩阵 的 均方损失函数
        # self.loss = F.mse_loss(G, self.target)
        self.loss = 1 - torch.cosine_similarity(G.view(input.shape[1],-1), self.target.view(input.shape[1],-1)).sum() / input.shape[1]
        return input

def get_model_and_losses(netClassifier, content_img, style_img, content_layers, style_layers, net_type):
    num_loss = 0
    expected_num_loss = len(content_layers) + len(style_layers)
    content_losses = []
    style_losses = []

    model = torch.nn.Sequential()
    if "resnet" in net_type:
        model.add_module("conv1", netClassifier.conv1)
        model.add_module("bn1", netClassifier.bn1)
        model.add_module("relu", netClassifier.relu)
        model.add_module("maxpool", netClassifier.maxpool)

        model.add_module("layer1", netClassifier.layer1)
        target_feature = model(style_img)
        style_loss = StyleLoss(target_feature)
        model.add_module(f'style_loss_1', style_loss)
        style_losses.append(style_loss)

        model.add_module("layer2", netClassifier.layer2)
        
        target = model(content_img)
        content_loss = ContentLoss(target)
        model.add_module(f'content_loss_1', content_loss)
        content_losses.append(content_loss)

        target_feature = model(style_img)
        style_loss = StyleLoss(target_feature)
        model.add_module(f'style_loss_2', style_loss)
        style_losses.append(style_loss)


    elif "vgg" in net_type:
        cnn = netClassifier.features.eval()
        i = 0
        for layer in cnn.children():
            if isinstance(layer, torch.nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, torch.nn.ReLU):
                name = f'relu_{i}'
                layer = torch.nn.ReLU(inplace=False)
            elif isinstance(layer, torch.nn.MaxPool2d):
                name = f'pool_{i}'
            elif isinstance(layer, torch.nn.BatchNorm2d):
                name = f'bn_{i}'
            else:
                raise RuntimeError(
                    f'Unrecognized layer: {layer.__class__.__name__}')

            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img)
                content_loss = ContentLoss(target)
                model.add_module(f'content_loss_{i}', content_loss)
                content_losses.append(content_loss)
                num_loss += 1

            if name in style_layers:
                target_feature = model(style_img)
                style_loss = StyleLoss(target_feature)
                model.add_module(f'style_loss_{i}', style_loss)
                style_losses.append(style_loss)
                num_loss += 1

            if num_loss >= expected_num_loss:
                break

    return model, content_losses, style_losses

class MySampler(Sampler):
    r"""Samples elements for my own sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(self.data_source)

    def __len__(self):
        return len(self.data_source)

class PSLoss(torch.nn.Module):
    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.target = target.detach()

    def forward(self, input):
        # 均方误差损失函数 sum((a-b)^2)/n 
        # self.loss = F.mse_loss(input, self.target)
        self.loss = F.l1_loss(input, self.target)
        return input


def get_model_and_PSlosses(netClassifier, content_img, style_img, content_layers, style_layers, net_type):
    num_loss = 0
    expected_num_loss = len(content_layers) + len(style_layers)
    content_losses = []
    style_losses = []

    model = torch.nn.Sequential()
    if "resnet" in net_type:
        model.add_module("conv1", netClassifier.conv1)
        model.add_module("bn1", netClassifier.bn1)
        model.add_module("relu", netClassifier.relu)
        model.add_module("maxpool", netClassifier.maxpool)

        model.add_module("layer1", netClassifier.layer1)
        target_feature = model(style_img)
        style_loss = StyleLoss(target_feature)
        model.add_module(f'style_loss_1', style_loss)
        style_losses.append(style_loss)

        model.add_module("layer2", netClassifier.layer2)
        
        target = model(content_img)
        content_loss = PSLoss(target)
        model.add_module(f'content_loss_1', content_loss)
        content_losses.append(content_loss)

        target_feature = model(style_img)
        style_loss = StyleLoss(target_feature)
        model.add_module(f'style_loss_2', style_loss)
        style_losses.append(style_loss)


    elif "vgg" in net_type:
        cnn = netClassifier.features.eval()
        i = 0
        for layer in cnn.children():
            if isinstance(layer, torch.nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, torch.nn.ReLU):
                name = f'relu_{i}'
                layer = torch.nn.ReLU(inplace=False)
            elif isinstance(layer, torch.nn.MaxPool2d):
                name = f'pool_{i}'
            elif isinstance(layer, torch.nn.BatchNorm2d):
                name = f'bn_{i}'
            else:
                raise RuntimeError(
                    f'Unrecognized layer: {layer.__class__.__name__}')

            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img)
                content_loss = PSLoss(target)
                model.add_module(f'content_loss_{i}', content_loss)
                content_losses.append(content_loss)
                num_loss += 1

            if name in style_layers:
                target_feature = model(style_img)
                style_loss = StyleLoss(target_feature)
                model.add_module(f'style_loss_{i}', style_loss)
                style_losses.append(style_loss)
                num_loss += 1

            if num_loss >= expected_num_loss:
                break

    return model, content_losses, style_losses

def rotation(img, angle):
   
    angle = angle*math.pi/180
    theta = torch.tensor([
        [math.cos(angle),math.sin(-angle),0],
        [math.sin(angle),math.cos(angle) ,0]
    ]).cuda()
    grid = F.affine_grid(theta.unsqueeze(0), img.size())
    x = F.grid_sample(img, grid)
    return x

def Homography(img, angle):
   
    angle = angle*math.pi/180
    theta = torch.tensor([
        [math.cos(angle),math.sin(angle)*1.1,0],
        [math.sin(angle)*1.5,math.cos(angle) ,0]
    ]).cuda()

    grid = F.affine_grid(theta.unsqueeze(0), img.size())
    x = F.grid_sample(img, grid)
    return x

def barrel_distortion(img, k=5e-15):
    # 获取图像的高度和宽度
    h, w = img.shape[-2:]
    # 计算中心点坐标
    cx, cy = w // 2, h // 2
    # 创建mapx和mapy的网格坐标矩阵
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    # 计算极坐标
    r = np.sqrt((map_x - cx) ** 2 + (map_y - cy) ** 2)
    theta = np.arctan2(map_y - cy, map_x - cx)
    # 计算畸变
    k1 = k
    k2 = k * 2
    k3 = k * 3
    # 计算畸变半径
    r_distort = r * (1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6)
    # 计算mapx和mapy的新坐标
    map_x_distort = r_distort * np.cos(theta) + cx
    map_y_distort = r_distort * np.sin(theta) + cy
    # 应用畸变
    distorted_img = cv2.remap(img.squeeze().cpu().detach().numpy().transpose(1,2,0), map_x_distort.astype(np.float32), map_y_distort.astype(np.float32), cv2.INTER_LINEAR)
    return torch.from_numpy(distorted_img.transpose(2,0,1)).unsqueeze(0).cuda()