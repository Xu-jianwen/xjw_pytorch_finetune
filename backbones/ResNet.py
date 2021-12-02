import torch.nn as nn
import math
from PIL import Image
from torch.utils import model_zoo
# from backbones.spp_layer import spatial_pyramid_pool


__all__ = ["ResNet", "resnet18", "resnet34", "resnet50", "resnet101"]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, dim=256, num_classes=3, embedding=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.output_num = [3, 2, 1]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.embeddings = nn.Sequential(
            nn.Linear(512 * block.expansion, dim),
            nn.LeakyReLU(inplace=True),
        )
        self.embedding = embedding
        if self.embedding:
            self.classifier = nn.Linear(dim, num_classes, bias=False)
        else:
            self.classifier = nn.Linear(512 * block.expansion, num_classes, bias=False)

        self.baselayer = [
            self.conv1,
            self.bn1,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        ]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # ft = spatial_pyramid_pool(
        #     previous_conv=x,
        #     num_sample=x.size(0),
        #     previous_conv_size=[int(x.size(2)), int(x.size(3))],
        #     out_pool_size=self.output_num,
        # )
        x = self.avgpool(x)
        ft = x.view(x.size(0), -1)
        if self.embedding:
            ft = self.embeddings(ft)
        ft = nn.functional.normalize(ft, p=2, dim=1)
        output = self.classifier(ft)

        return ft, output


def resnet18(pretrained, embedding=False, dim=256, num_classes=3):
    model = ResNet(
        BasicBlock,
        [2, 2, 2, 2],
        embedding=embedding,
        dim=dim,
        num_classes=num_classes,
    )
    url = model_urls["resnet18"]
    if pretrained is True:
        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(url)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def resnet34(pretrained, embedding=False, dim=256, num_classes=3):
    model = ResNet(
        BasicBlock,
        [3, 4, 6, 3],
        embedding=embedding,
        dim=dim,
        num_classes=num_classes,
    )
    url = model_urls["resnet34"]
    if pretrained is True:
        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(url)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def resnet50(pretrained, embedding=False, dim=256, num_classes=3):
    model = ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        embedding=embedding,
        dim=dim,
        num_classes=num_classes,
    )
    url = model_urls["resnet50"]
    if pretrained is True:
        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(url)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def resnet101(pretrained, embedding=False, dim=256, num_classes=3):
    model = ResNet(
        Bottleneck,
        [3, 4, 23, 3],
        embedding=embedding,
        dim=dim,
        num_classes=num_classes,
    )
    url = model_urls["resnet101"]
    if pretrained is True:
        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(url)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model
