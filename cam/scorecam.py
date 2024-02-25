import torch
import torch.nn.functional as F
from cam.basecam import *
import numpy as np

class ScoreCAM(BaseCAM):

    """
        ScoreCAM, inherit from BaseCAM

    """

    def __init__(self, model_dict):
        super().__init__(model_dict)

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()
        
        # predication on raw input
        logit = self.model_arch(input).cuda()
        logit = F.softmax(logit)

        n = 1 #logit的前n个最大值
        n_max_index = np.argsort(logit.cpu().detach().numpy())[0, -n:][::-1]
        score_saliency_maps = []
        # 计算n个显著图
        for index in range(n):
            # 循环n次算出置信度前n的类的CAM

            if class_idx is None:

                predicted_class = torch.tensor([n_max_index[index]]).cuda()
                #predicted_class = logit.max(1)[-1]
                score = logit[:, predicted_class].squeeze()
            else:
                predicted_class = torch.LongTensor([class_idx])
                score = logit[:, class_idx].squeeze()
                confidence = score.item()

            if torch.cuda.is_available():
              predicted_class= predicted_class.cuda()
              score = score.cuda()
              logit = logit.cuda()

            self.model_arch.zero_grad()
            #最后一遍释放所有参数
            if index < n-1:
                retain_graph = True
            else:
                retain_graph = False
            # score.backward(retain_graph=retain_graph)
            activations = self.activations['value']
            b, k, u, v = activations.size()

            score_saliency_map = torch.zeros((1, 1, h, w))

            if torch.cuda.is_available():
              activations = activations.cuda()
              score_saliency_map = score_saliency_map.cuda()

            with torch.no_grad():
              for i in range(k):

                  # upsampling
                  saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
                  # 根据给定 size 或 scale_factor，上采样或下采样输入数据input.
                  saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

                  if saliency_map.max() == saliency_map.min():
                    continue

                  # normalize to 0-1
                  norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

                  # how much increase if keeping the highlighted region
                  # predication on masked input 原图和掩码相乘  掩码即norm_saliency_map 相应的softmax输出为权重
                  output = self.model_arch(input * norm_saliency_map)
                  output = F.softmax(output)
                  score = output[0][predicted_class]

                  score_saliency_map +=  score * saliency_map

            score_saliency_map = F.relu(score_saliency_map)
            score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

            if score_saliency_map_min == score_saliency_map_max:
                return None

            score_saliency_map = (score_saliency_map - score_saliency_map_min).div(score_saliency_map_max - score_saliency_map_min).data
            score_saliency_maps.append(score_saliency_map)

        return score_saliency_maps, confidence

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)