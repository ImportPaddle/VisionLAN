import os, sys
import paddle.nn as nn
import math
DIR = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(DIR, '../'))
from api import normal_, fill_, zero_


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2D(in_planes, out_planes, kernel_size=1, stride=stride, bias_attr=False)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias_attr=False)


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2D(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2D(planes)
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


class ResNet(nn.Layer):

    def __init__(self, block, layers, strides, compress_layer=True):
        self.inplanes = 32
        super(ResNet, self).__init__()
        self.conv1_new = nn.Conv2D(3, 32, kernel_size=3, stride=strides[0], padding=1,
                                   bias_attr=False)
        self.bn1 = nn.BatchNorm2D(32)
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer(block, 32, layers[0], stride=strides[1])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=strides[2])
        self.layer3 = self._make_layer(block, 128, layers[2], stride=strides[3])
        self.layer4 = self._make_layer(block, 256, layers[3], stride=strides[4])
        self.layer5 = self._make_layer(block, 512, layers[4], stride=strides[5])

        self.compress_layer = compress_layer
        if compress_layer:
            # for handwritten
            self.layer6 = nn.Sequential(
                nn.Conv2D(512, 256, kernel_size=(3, 1), padding=(0, 0), stride=(1, 1)),
                nn.BatchNorm2D(256),
                nn.ReLU())

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2D):
                fill_(m.weight, 1)
                zero_(m.bias)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, multiscale=False):
        out_features = []
        x = self.conv1_new(x)
        x = self.bn1(x)
        x = self.relu(x)
        tmp_shape = x.shape[2:]
        x = self.layer1(x)
        if x.shape[2:] != tmp_shape:
            tmp_shape = x.shape[2:]
            out_features.append(x)
        x = self.layer2(x)
        if x.shape[2:] != tmp_shape:
            tmp_shape = x.shape[2:]
            out_features.append(x)
        x = self.layer3(x)
        if x.shape[2:] != tmp_shape:
            tmp_shape = x.shape[2:]
            out_features.append(x)
        x = self.layer4(x)
        if x.shape[2:] != tmp_shape:
            tmp_shape = x.shape[2:]
            out_features.append(x)
        x = self.layer5(x)
        if not self.compress_layer:
            out_features.append(x)
        else:
            if x.shape[2:] != tmp_shape:
                tmp_shape = x.shape[2:]
                out_features.append(x)
            x = self.layer6(x)
            out_features.append(x)
        return out_features


def resnet45(strides, compress_layer):
    model = ResNet(BasicBlock, [3, 4, 6, 6, 3], strides, compress_layer)
    return model
