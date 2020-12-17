from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from imblearn.metrics import sensitivity_specificity_support
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torchvision.models as models
import scipy.io as sio
import torch.nn as nn
import math 
import torch.nn.functional as F



def resnet18(dataset, train_csv):
  class_length = 0
  resnet18 = models.resnet18()
  if(dataset=="chexpert"):
      resnet18.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      resnet18.fc = torch.nn.Linear(512, len(train_csv['class'].unique()))
      class_length = len(train_csv['class'].unique())
  if(dataset == "flowers"):
      resnet18.fc = torch.nn.Linear(512, len(train_csv['labels'].unique()))
      class_length = len(train_csv['labels'].unique())
  return class_length,resnet18
      

""" constructed on Github user Kaihau Tang in the follwoing link: https://github.com/KaihuaTang/ResNet50-Pytorch-Face-Recognition/blob/master/ResNet.py """

""" a resnet50 with smaller layers that can be used for images with one channel and size 98x98 (avgpool2d converted from 7 to 4)"""

def tiny_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet([3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(model.modelPath))
    return model


class ResNet(nn.Module):
    """
    block: A sub module
    """
    def __init__(self, layers, num_classes=40, model_path="model.pkl"):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.modelPath = model_path
        self.conv1 = nn.Conv2d(1, 16, kernel_size = 7, stride = 2, padding = 3,
                               bias = False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.stack1 = self.make_stack(16, layers[0])
        self.stack2 = self.make_stack(32, layers[1], stride=2)
        self.stack3 = self.make_stack(64, layers[2], stride=2)
        self.stack4 = self.make_stack(128, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride = 1)
        self.fc = nn.Linear(128 * Bottleneck.expansion, num_classes)
        # initialize parameters
        self.init_param()

    def init_param(self):
        # The following is initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.shape[0] * m.weight.shape[1]
                m.weight.data.normal_(0, math.sqrt(2./n))
                m.bias.data.zero_()

    def make_stack(self, planes, blocks, stride = 1):
        downsample = None
        layers = []
            
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
                )

        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        x = self.stack4(x)
        #print(x.size())
        x = self.avgpool(x)
        #print(x.size())
        x = x.view(x.size(0), -1)
        #print(x.size())
        x = self.fc(x)

        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
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
