from .AlexNet import alexnet
from .vgg import vgg_net
from .ResNet import resnet

__factory = {
    'AlexNet': alexnet,
    'VGGNet': vgg_net, 
    'resnet18': resnet,
    'resnet34': resnet,
    'resnet50': resnet,
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