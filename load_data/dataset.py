from __future__ import absolute_import, print_function
import torch.utils.data as data
from util import read_image
import re
from collections import defaultdict
import os
from collections import defaultdict
import torchvision.transforms as T


def build_transforms(mode="RGB", is_train=True):
    normMean = [0.4948052, 0.48568845, 0.44682974]  # 第一行
    normStd = [0.24580306, 0.24236229, 0.2603115]  # 第二行
    if mode == "BGR":
        normalize_transform = T.Normalize(
            mean=[104.0 / 255, 117.0 / 255, 128.0 / 255], std=3 * [1.0 / 255]
        )
    else:
        normalize_transform = T.Normalize(mean=normMean, std=normStd)
    if is_train:
        transform = T.Compose(
            [
                T.Resize([256, 256]),
                T.RandomResizedCrop(size=224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    else:
        transform = T.Compose(
            [T.Resize(size=256), T.CenterCrop(224), T.ToTensor(), normalize_transform]
        )
    return transform


class BaseDataSet(data.Dataset):
    """
    Basic Dataset read image path from img_source
    img_source: list of img_path and label
    """

    def __init__(self, img_source, transforms=None, mode="RGB"):
        self.mode = mode
        self.transforms = transforms
        self.root = os.path.dirname(img_source)
        assert os.path.exists(img_source), f"{img_source} NOT found."
        self.img_source = img_source
        self.label_list = list()
        self.path_list = list()
        self._load_data()
        self.label_index_dict = self._build_label_index_dict()
        self.classes = self._get_classes()

    def __len__(self):
        return len(self.label_list)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"| Dataset Info |datasize: {self.__len__()}|num_labels: {len(set(self.label_list))}|"

    def _load_data(self):
        with open(self.img_source, "r") as f:
            for line in f:
                _path, _label = re.split(r",| ", line.strip())
                self.path_list.append(_path)
                self.label_list.append(_label)

    def _build_label_index_dict(self):
        index_dict = defaultdict(list)
        for i, label in enumerate(self.label_list):
            index_dict[label].append(i)
        return index_dict

    def __getitem__(self, index):
        path = self.path_list[index]
        img_path = os.path.join(self.root, path)
        label = self.label_list[index]

        img = read_image(img_path, mode=self.mode)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label

    def _get_classes(self):
        label = [int(c) for c in list(set(self.label_list))]
        classes = sorted(label)
        return classes
