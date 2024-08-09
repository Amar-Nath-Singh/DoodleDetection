import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 101):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(5, 5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14x5
            nn.Conv2d(5, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7x8
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU()  # 7x7x16
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 16, 100),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(100, num_classes)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out

class resnet(nn.Module):
    def __init__(self, in_channels=1, num_classes = 101):
        super(resnet, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                3,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.ReLU(),
        )
        self.model = torchvision.models.resnet18(pretrained = True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # x = self.initial(x)
        x = self.model(x)
        return x

class effnet(nn.Module):
    def __init__(self, in_channels=1, num_classes = 101):
        super(effnet, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                3,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.ReLU(),
        )
        self.model = torchvision.models.efficientnet_b0()
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.initial(x).detach()
        x = self.model(x)
        return x
    


class CNNWithSE(nn.Module):
    def __init__(self, in_channels=1, num_classes=10 ,ratio = 16):
        super(CNNWithSE, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels,16,3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.se_block1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 32 // ratio, 1),
            nn.SiLU(),
            nn.Conv2d(32 // ratio, 32, 1),
            nn.Sigmoid()
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.se_block2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 128 // ratio, 1),
            nn.SiLU(),
            nn.Conv2d(128 // ratio, 128, 1),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 14 * 14, 256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = x * self.se_block1(x)
        x = self.conv_block2(x)
        x = x * self.se_block2(x)
        x = self.classifier(x)

        return x

class SEBlock(nn.Module):
    def __init__(self, channels, ratio):
        super(SEBlock, self).__init__()
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // ratio, 1),
            nn.SiLU(),
            nn.Conv2d(channels // ratio, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        se = self.se_block(x)
        return x * se


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_prob = 0.2):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=101, dropout_prob = 0.2):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 64, layers[0])
        self.layer2 = self._make_layer(block, 64, 128, layers[1], stride=2, se_ratio=16)
        self.layer3 = self._make_layer(block, 128, 256, layers[2], stride=2, se_ratio=32)
        self.layer4 = self._make_layer(block, 256, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1, se_ratio = None):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        
        if se_ratio:
            layers.append(SEBlock(out_channels, se_ratio))

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

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

def resnet_model(in_channels,num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


    
if __name__ == '__main__':
    res = resnet(3, 101)

    print(resnet_model(3, 101))
