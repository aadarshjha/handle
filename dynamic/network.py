import math
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet101
from timesformer.models.vit import TimeSformer


# IPN implementation, use 3D CNN w/ ResNeXt block
class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
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


# ResNeXt implementation from IPN
class ResNeXt(nn.Module):
    def __init__(self, layers, sample_size, sample_duration,
                 shortcut_type='B', cardinality=32, num_classes=400):
        block = ResNeXtBottleneck
        self.inplanes = 64
        
        super(ResNeXt, self).__init__()
        
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2),
                               padding=(3, 3, 3), bias=False)
        
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, 
                                    padding=1)
        
        self.layer1 = self.make_layer(block, 128, layers[0], 
                                       shortcut_type, cardinality)
        
        self.layer2 = self.make_layer(block, 256, layers[1], 
                                       shortcut_type, cardinality, 
                                       stride=2)
        
        self.layer3 = self.make_layer(block, 512, layers[2], 
                                       shortcut_type, cardinality, 
                                       stride=2)
        
        self.layer4 = self.make_layer(block, 1024, layers[3], 
                                       shortcut_type, cardinality, 
                                       stride=2)
        
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        
        self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size),
                                    stride=1)
        
        self.fc = nn.Linear(cardinality * 32 * block.expansion, 
                            num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, 
                                                  mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_layer(self, block, planes, blocks, shortcut_type, cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, cardinality, 
                            stride, downsample))
        
        self.inplanes = planes * block.expansion
        
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        in_s = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# CNN-LSTM implementation w/ ResNet-101
# No need to rework model here
class ResNet_LSTM(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet_LSTM, self).__init__()
        self.cnn = resnet101(pretrained=True)
        
        # Replace last layer of ResNet w/ modified linear to pass to LSTM
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 512)
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=6, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        b_size, C, sample_duration, H, W = x.size()
        x = x.reshape((b_size, sample_duration, C, H, W))
        x = x.view(b_size * sample_duration, C, H, W)

        with torch.no_grad():
            x = self.cnn(x)
        
        x = x.view(b_size, sample_duration, -1)

        hidden_st = None            
        out, hidden_st = self.lstm(x, hidden_st)
        out = self.fc(out[:, -1, :])

        return out


def fix_model_layers(model, config):
    # Need to modify layers appropriately for specific pretrained models
    # No need to modify Jester pretrained model

    if config.pretrain_dataset == 'ipn':
        model.conv1 = nn.Conv3d(3, model.conv1.out_channels, kernel_size=(3,7,7),
                                stride=(1,2,2), padding=(1,3,3), bias=False)
    elif config.pretrain_dataset == 'egogesture':
        model.conv1 = nn.Conv3d(3, model.conv1.out_channels, kernel_size=(3,7,7),
                                stride=(1,2,2), padding=(1,3,3), bias=False)

    return model


# Load pretrained resnext model, change architecture basesd on model
def load_resnext101(config, device):
    pretrain_pth = config.pretrain_path
    
    print(f'Loading model {pretrain_pth}')

    model = ResNeXt([3, 4, 23, 3], sample_size=config.sample_size, sample_duration=config.sample_duration,
                    num_classes=13)
    pretrain_dict = torch.load(pretrain_pth)
    pretrain_dict['state_dict'] = {key.replace('module.', ''): value for key, value in pretrain_dict['state_dict'].items()}

    del pretrain_dict['state_dict']['fc.weight']
    del pretrain_dict['state_dict']['fc.bias']

    model = fix_model_layers(model, config)
    model.load_state_dict(pretrain_dict['state_dict'], strict=False)

    # Replace last layer for MiniIPN (3 classes)
    model.fc = nn.Linear(model.fc.in_features, 3)
    
    model.fc = model.fc.to(device)
    model = model.to(device)
    print(model)

    return model, model.fc.parameters()


# Load CNN-LSTM
def load_cnn_lstm(config, device):
    # Load CNN model
    model = ResNet_LSTM().to(device)
    return model, model.parameters()


# Load TimeSformer
def load_timesformer(config, device):
    model = TimeSformer(img_size=config.sample_size, num_classes=3, num_frames=config.sample_duration,
                        attention_type='divided_space_time', 
                        pretrained_model='D:\Projects\ML\models\stformer.pyth')
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    model.model.head.weight.requires_grad = True
    model.model.head.bias.requires_grad = True
    
    return model, model.model.head.parameters()