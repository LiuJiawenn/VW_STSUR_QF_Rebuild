import torch.nn as nn
import torch.nn.functional as F


# 与SPQ区别： conv1 输入是2通道的
class TPQSubNet(nn.Module):
    def __init__(self):
        super(TPQSubNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv9 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv10 = nn.Conv2d(512, 512, 3, padding=1)

    def forward(self, x):
        # 输入一个三通道patch 32*32*3 输出patch feature 1*512
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv7(h))
        h = F.relu(self.conv8(h))
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv9(h))
        h = F.relu(self.conv10(h))
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        return h
