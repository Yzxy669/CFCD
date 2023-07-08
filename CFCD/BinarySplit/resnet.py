import torch
import torch.nn as nn
from torchvision import models
from CAM import ChannelAttention


class base_resnet(nn.Module):
    def __init__(self, net_type='resnet18'):
        super(base_resnet, self).__init__()
        if net_type == 'resnet18':
            self.model = models.resnet18()
            print(self.model)
        if net_type == 'resnet34':
            self.model = models.resnet34()
            print(self.model)

        self.conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.cam = ChannelAttention(512)
        self.feat_fc = nn.Linear(512, 64)  # 特征维度
        self.fc = nn.Linear(512, 2)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.cam(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        features = self.feat_fc(x)
        x = self.fc(x)
        x = self.softmax(x)
        return [x, features]


if __name__ == "__main__":
    net = base_resnet('resnet18')
    print(net)
    img = torch.rand(2, 9, 32, 32)
    result = net(img)
    print(result)
