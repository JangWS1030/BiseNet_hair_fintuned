from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

RESNET18_URL = "https://download.pytorch.org/models/resnet18-5c106cde.pth"


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(x)
        residual = F.relu(self.bn1(residual), inplace=True)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x if self.downsample is None else self.downsample(x)
        out = self.relu(shortcut + residual)
        return out


def create_layer_basic(in_chan: int, out_chan: int, blocks: int, stride: int = 1) -> nn.Sequential:
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    for _ in range(blocks - 1):
        layers.append(BasicBlock(out_chan, out_chan, stride=1))
    return nn.Sequential(*layers)


class Resnet18(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, blocks=2, stride=1)
        self.layer2 = create_layer_basic(64, 128, blocks=2, stride=2)
        self.layer3 = create_layer_basic(128, 256, blocks=2, stride=2)
        self.layer4 = create_layer_basic(256, 512, blocks=2, stride=2)
        if pretrained:
            self.init_weight()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = F.relu(self.bn1(x), inplace=True)
        x = self.maxpool(x)
        x = self.layer1(x)
        feat8 = self.layer2(x)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat8, feat16, feat32

    def init_weight(self) -> None:
        state_dict = load_state_dict_from_url(RESNET18_URL, progress=True)
        own_state = self.state_dict()
        for key, value in state_dict.items():
            if "fc" in key:
                continue
            if key in own_state:
                own_state[key] = value
        self.load_state_dict(own_state)

    def get_params(self) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
        wd_params: list[torch.nn.Parameter] = []
        nowd_params: list[torch.nn.Parameter] = []
        for _, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if module.bias is not None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params.extend(list(module.parameters()))
        return wd_params, nowd_params

