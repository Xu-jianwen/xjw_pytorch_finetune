from .AlexNet import alexnet
from .vgg import vgg16, vgg16_bn, vgg11
from .ResNet import resnet18, resnet34, resnet50, resnet101
from .ResNet_fusion import resnet50_ff
from .mobilenet_v3 import mobilenet_v3_large, mobilenet_v3_small
from.BNInception import bninception
from.Xception import xception


__factory = {
    "AlexNet": alexnet,
    "VGG16": vgg16,
    "VGG11": vgg11,
    "VGG16_BN": vgg16_bn,
    "bninception": bninception,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "mobilenetv3_large": mobilenet_v3_large,
    "mobilenetv3_small": mobilenet_v3_small,
    "xception": xception,
    "resnet50ff": resnet50_ff,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a model instance.

    Parameters
    ----------
    name : str
        the name of loss model
    """
    if name not in __factory:
        raise KeyError("Unknown network:", name)
    return __factory[name](*args, **kwargs)
