from __future__ import print_function
from cv2 import rotate
import torch.nn.functional as F
import torch
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
from PIL import Image
import cv2


def read_image(img_path, mode="RGB"):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError(f"{img_path} does not exist")
    while not got_img:
        try:
            img = Image.open(img_path).convert("RGB")
            if mode == "BGR":
                r, g, b = img.split()
                img = Image.merge("RGB", (b, g, r))
            got_img = True
        except IOError:
            print(f"IOError incurred when reading '{img_path}'. Will redo.")
            pass
    return img


def collate_fn(batch):
    imgs, labels = zip(*batch)
    labels = [int(k) for k in labels]
    labels = torch.tensor(labels, dtype=torch.int64)
    return torch.stack(imgs, dim=0), labels


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.eval()


def proxies_reducer(num_classes, num_centers, logit, gamma=64):
    if num_centers == 1:
        return logit
    elif num_centers is None:
        return logit
    else:
        logit = logit.view(-1, num_centers, num_classes)
        # prob = F.softmax(gamma * logit, dim=1)
        # sim_to_classes = torch.sum(prob * logit, dim=1)
        sim_to_classes = torch.logsumexp(gamma * logit, dim=1) / gamma
        # sim_to_classes = torch.max(logit, dim=2)[0]
        return sim_to_classes


def cm_plot(num_classes, label, matrix, fig_name=""):

    print(matrix)
    plt.imshow(matrix, cmap=plt.cm.Blues)

    # 设置x轴坐标label
    plt.xticks(range(num_classes), label)
    # 设置y轴坐标label
    plt.yticks(range(num_classes), label)
    # 显示colorbar
    # plt.colorbar()
    plt.ylabel("True Labels")
    plt.xlabel("Predicted Labels")
    plt.title("Confusion matrix")

    # 在图中标注数量/概率信息
    thresh = matrix.max() / 2
    for x in range(num_classes):
        for y in range(num_classes):
            # 注意这里的matrix[y, x]不是matrix[x, y]
            info = int(matrix[y, x])
            plt.text(
                x,
                y,
                info,
                verticalalignment="center",
                horizontalalignment="center",
                color="white" if info > thresh else "black",
            )
    plt.tight_layout()
    plt.savefig(fig_name + "confusion_matrix.png", dpi=300)
    plt.show()

def percentage_cm_plot(num_classes, label, matrix, fig_name=""):
    print(matrix)
    matrix = matrix.astype('float')
    for i in range(num_classes):
        matrix[i] = matrix[i]/matrix[i].sum()
    plt.imshow(matrix, cmap=plt.cm.Blues)
    # 设置x轴坐标label
    plt.xticks(range(num_classes), label, rotation=45, fontsize=15)
    # 设置y轴坐标label
    plt.yticks(range(num_classes), label, rotation=-45, fontsize=15)
    # 显示colorbar
    # plt.colorbar()
    plt.ylabel('True Labels', fontsize=15)
    plt.xlabel('Predicted Labels', fontsize=15)
    plt.title(fig_name.split("_")[-3]+" "+fig_name.split("_")[-2], fontsize=15)
    # plt.title('Confusion matrix')

    # 在图中标注数量/概率信息
    thresh = matrix.max() / 2
    for x in range(num_classes):
        for y in range(num_classes):
            # 注意这里的matrix[y, x]不是matrix[x, y]
            info = matrix[y, x]
            # flt = info/matrix[y].sum()
            percentage = format(info, '.1%')
            plt.text(x, y, percentage,
                     verticalalignment='center',
                     horizontalalignment='center',fontsize=15
                     ,color="white" if info > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(fig_name + "confusion_matrix.png", dpi=300)
    plt.show()


def tsne_feature_visualization(features, n_components):
    features_tsne = TSNE(n_components=n_components).fit_transform(features)
    return features_tsne


def tsne_plot(name, features, labels, classes):
    ship_labels = labels
    label_com = classes
    marker = ["v", "o", "s", "P", "*"]
    figsize = 10, 8

    plt.figure(figsize=figsize)
    for index in range(len(label_com)):
        data_ship = features[ship_labels == index]
        data_ship_x = data_ship[:, 0]
        data_ship_y = data_ship[:, 1]
        # plt.scatter(data_ship_x, data_ship_y, marker=marker[int(index / 10)])
        plt.scatter(data_ship_x, data_ship_y, marker=marker[index], s=144)
    # plt.legend(
    #     labels=label_com,
    #     loc="best",
    #     ncol=1,
    #     borderpad=0,
    #     frameon=False,
    #     markerscale=1,
    #     labelspacing=0,
    #     fontsize=20,
    # )
    # plt.axes().get_xaxis().set_visible(False) # 隐藏x坐标轴
    plt.axis("off")
    plt.savefig(name + "_tsne.png", dpi=300)
    plt.show()


def visualize_cam(mask, img):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]

    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze().cpu()), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])

    result = heatmap + img.cpu()
    result = result.div(result.max()).squeeze()

    return heatmap, result


def find_resnet_layer(arch, target_layer_name):
    """Find resnet layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'conv1'
            target_layer_name = 'layer1'
            target_layer_name = 'layer1_basicblock0'
            target_layer_name = 'layer1_basicblock0_relu'
            target_layer_name = 'layer1_bottleneck0'
            target_layer_name = 'layer1_bottleneck0_conv1'
            target_layer_name = 'layer1_bottleneck0_downsample'
            target_layer_name = 'layer1_bottleneck0_downsample_0'
            target_layer_name = 'avgpool'
            target_layer_name = 'fc'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    if "layer" in target_layer_name:
        hierarchy = target_layer_name.split("_")
        layer_num = int(hierarchy[0].lstrip("layer"))
        if layer_num == 1:
            target_layer = arch.layer1
        elif layer_num == 2:
            target_layer = arch.layer2
        elif layer_num == 3:
            target_layer = arch.layer3
        elif layer_num == 4:
            target_layer = arch.layer4
        else:
            raise ValueError("unknown layer : {}".format(target_layer_name))

        if len(hierarchy) >= 2:
            bottleneck_num = int(
                hierarchy[1].lower().lstrip("bottleneck").lstrip("basicblock")
            )
            target_layer = target_layer[bottleneck_num]

        if len(hierarchy) >= 3:
            target_layer = target_layer._modules[hierarchy[2]]

        if len(hierarchy) == 4:
            target_layer = target_layer._modules[hierarchy[3]]

    else:
        target_layer = arch._modules[target_layer_name]

    return target_layer


def find_densenet_layer(arch, target_layer_name):
    """Find densenet layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_transition1'
            target_layer_name = 'features_transition1_norm'
            target_layer_name = 'features_denseblock2_denselayer12'
            target_layer_name = 'features_denseblock2_denselayer12_norm1'
            target_layer_name = 'features_denseblock2_denselayer12_norm1'
            target_layer_name = 'classifier'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """

    hierarchy = target_layer_name.split("_")
    target_layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        target_layer = target_layer._modules[hierarchy[1]]

    if len(hierarchy) >= 3:
        target_layer = target_layer._modules[hierarchy[2]]

    if len(hierarchy) == 4:
        target_layer = target_layer._modules[hierarchy[3]]

    return target_layer


def find_vgg_layer(arch, target_layer_name):
    """Find vgg layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_42'
            target_layer_name = 'classifier'
            target_layer_name = 'classifier_0'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    hierarchy = target_layer_name.split("_")

    if len(hierarchy) >= 1:
        target_layer = arch.features

    if len(hierarchy) == 2:
        target_layer = target_layer[int(hierarchy[1])]

    return target_layer


def find_alexnet_layer(arch, target_layer_name):
    """Find alexnet layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_0'
            target_layer_name = 'classifier'
            target_layer_name = 'classifier_0'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    hierarchy = target_layer_name.split("_")

    if len(hierarchy) >= 1:
        target_layer = arch.features

    if len(hierarchy) == 2:
        target_layer = target_layer[int(hierarchy[1])]

    return target_layer


def find_squeezenet_layer(arch, target_layer_name):
    """Find squeezenet layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features_12'
            target_layer_name = 'features_12_expand3x3'
            target_layer_name = 'features_12_expand3x3_activation'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    hierarchy = target_layer_name.split("_")
    target_layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        target_layer = target_layer._modules[hierarchy[1]]

    if len(hierarchy) == 3:
        target_layer = target_layer._modules[hierarchy[2]]

    elif len(hierarchy) == 4:
        target_layer = target_layer._modules[hierarchy[2] + "_" + hierarchy[3]]

    return target_layer


def denormalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError("tensor should be 4D")

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean)


def normalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError("tensor should be 4D")

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.sub(mean).div(std)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return self.do(tensor)

    def do(self, tensor):
        return normalize(tensor, self.mean, self.std)

    def undo(self, tensor):
        return denormalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


def distance_matrix(inputs):
    nB = inputs.size(0)
    dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(nB, nB)
    dist = dist + dist.t()
    # use squared
    dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2).clamp_(min=1e-12)
    return dist

# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import copy
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - dataset (BaseDataSet).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, dataset, batch_size, num_instances, max_iters):
        self.label_index_dict = dataset.label_index_dict
        self.batch_size = batch_size
        self.K = num_instances
        self.num_labels_per_batch = self.batch_size // self.K
        self.max_iters = max_iters
        self.labels = list(self.label_index_dict.keys())

    def __len__(self):
        return self.max_iters

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"|Sampler| iters {self.max_iters}| K {self.K}| M {self.batch_size}|"

    def _prepare_batch(self):
        batch_idxs_dict = defaultdict(list)

        for label in self.labels:
            idxs = copy.deepcopy(self.label_index_dict[label])
            if len(idxs) < self.K:
                idxs.extend(np.random.choice(idxs, size=self.K - len(idxs), replace=True))
            random.shuffle(idxs)

            batch_idxs_dict[label] = [idxs[i * self.K: (i + 1) * self.K] for i in range(len(idxs) // self.K)]

        avai_labels = copy.deepcopy(self.labels)
        return batch_idxs_dict, avai_labels

    def __iter__(self):
        batch_idxs_dict, avai_labels = self._prepare_batch()
        for _ in range(self.max_iters):
            batch = []
            if len(avai_labels) < self.num_labels_per_batch:
                batch_idxs_dict, avai_labels = self._prepare_batch()

            selected_labels = random.sample(avai_labels, self.num_labels_per_batch)
            for label in selected_labels:
                batch_idxs = batch_idxs_dict[label].pop(0)
                batch.extend(batch_idxs)
                if len(batch_idxs_dict[label]) == 0:
                    avai_labels.remove(label)
            yield batch
