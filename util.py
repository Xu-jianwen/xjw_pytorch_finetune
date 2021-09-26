from __future__ import print_function
import torch
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
from PIL import Image


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


def tsne_feature_visualization(name, features, n_components):
    features_tsne = TSNE(n_components=n_components).fit_transform(features)
    return features_tsne


def D2_images_sar_plot(name, features, labels):
    ship_labels = labels
    label_com = ["cargo", "container", "tanker"]
    colors = ["r", "g", "b"]
    marker = ["o", "v", "s"]
    figsize = 10, 8

    figure, ax = plt.subplots(figsize=figsize)
    for index in range(len(label_com)):
        data_ship = features[ship_labels == index]
        data_ship_x = data_ship[:, 0]
        data_ship_y = data_ship[:, 1]
        plt.scatter(data_ship_x, data_ship_y, c=colors[index], marker=marker[index])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels=label_com, loc="best")
    plt.savefig(name + "_tsne.png", dpi=300)
    plt.show()
