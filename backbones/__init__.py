from .AlexNet import alexnet
from .vgg import vgg16, vgg16_bn
from .ResNet import resnet18, resnet34, resnet50, resnet101


__factory = {
    "AlexNet": alexnet,
    "VGG16": vgg16,
    "VGG16_BN": vgg16_bn,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
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
