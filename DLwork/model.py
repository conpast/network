import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
        def __init__(self):
                super(Classifier, self).__init__()

                # 第一层卷积
                self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                # 第二层卷积
                self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
                # 第三层卷积
                self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
                self.flatten = nn.Flatten()
                # 全连接层
                self.fc1 = nn.Linear(128 * 16 * 16, 1024)  # 根据卷积输出调整输入特征数
                self.fc2 = nn.Linear(1024, 10)  # 输出层有两个神经元，对应猫和狗两个类别

                # Dropout层用于防止过拟合（可选）
                self.dropout = nn.Dropout(p=0.5)

        def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))  # 64x64
                x = self.pool(F.relu(self.conv2(x)))  # 32x32
                x = self.pool(F.relu(self.conv3(x)))  # 16x16
                # x = self.pool(F.relu(self.conv4(x)))  # 8x8 （如果加了第四层卷积和池化）

                # 如果没有加第四层卷积和池化，则注释掉上一行，并确保下一行的32x32是正确的
                x = self.flatten(x)
                x = F.relu(self.fc1(x))
                x = self.dropout(x)  # 应用Dropout（可选）
                x = self.fc2(x)
                return x

        # 实例化模型


