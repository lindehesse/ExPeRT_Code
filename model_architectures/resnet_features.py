import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from typing import Optional, Tuple


model_urls = {"resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth"}

model_dir = "./pretrained_models"


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    # class attribute
    expansion = 1
    num_layers = 2

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        last_act: str = "relu",
    ) -> None:
        super(BasicBlock, self).__init__()
        # only conv with possibly not 1 stride
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        if last_act == "sigmoid":
            self.sigmoid = nn.Sigmoid()
        elif last_act == "relu":
            pass
        else:
            raise ValueError(f"last_act must be either 'relu' or 'sigmoid', not {last_act}")

        # if stride is not 1 then self.downsample cannot be None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # the residual connection
        out += identity

        if hasattr(self, "sigmoid"):
            out = self.sigmoid(out)
        else:
            out = self.relu(out)

        return out

    def block_conv_info(self) -> Tuple[list, list, list]:
        block_kernel_sizes = [3, 3]
        block_strides = [self.stride, 1]
        block_paddings = [1, 1]

        return block_kernel_sizes, block_strides, block_paddings


class ResNet_features(nn.Module):
    """
    the convolutional layers of ResNet
    the average pooling and final fully convolutional layer is removed
    """

    def __init__(
        self,
        block: type[BasicBlock],
        layers: list,
        zero_init_residual: bool = False,
        in_channels: int = 3,
        last_act: str = "relu",
    ) -> None:
        super(ResNet_features, self).__init__()

        self.inplanes = 64

        # the first convolutional layer before the structured sequence of blocks
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # comes from the first conv and the following max pool
        self.kernel_sizes = [7, 3]
        self.strides = [2, 2]
        self.paddings = [3, 1]

        # the following layers, each layer is a sequence of blocks
        self.block = block
        self.layers = layers
        self.layer1 = self._make_layer(block=block, planes=64, num_blocks=self.layers[0])
        self.layer2 = self._make_layer(block=block, planes=128, num_blocks=self.layers[1], stride=2)
        self.layer3 = self._make_layer(block=block, planes=256, num_blocks=self.layers[2], stride=2)
        self.layer4 = self._make_layer(
            block=block,
            planes=512,
            num_blocks=self.layers[3],
            stride=2,
            last_act=last_act,
        )

        # initialize the parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: type[BasicBlock],
        planes: int,
        num_blocks: int,
        stride: int = 1,
        last_act: str = "relu",
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # only the first block has downsample that is possibly not None
        layers.append(block(self.inplanes, planes, stride, downsample, last_act=last_act))

        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        # keep track of every block's conv size, stride size, and padding size
        for each_block in layers:
            (
                block_kernel_sizes,
                block_strides,
                block_paddings,
            ) = each_block.block_conv_info()
            self.kernel_sizes.extend(block_kernel_sizes)
            self.strides.extend(block_strides)
            self.paddings.extend(block_paddings)

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def conv_info(self) -> Tuple[list, list, list]:
        return self.kernel_sizes, self.strides, self.paddings

    def num_layers(self) -> int:
        """
        the number of conv layers in the network, not counting the number
        of bypass layers
        """

        return (
            self.block.num_layers * self.layers[0]
            + self.block.num_layers * self.layers[1]
            + self.block.num_layers * self.layers[2]
            + self.block.num_layers * self.layers[3]
            + 1
        )

    def __repr__(self) -> str:
        return f"resnet{self.num_layers() + 1}_features"


def resnet18_features(pretrained: str = "imagenet", **kwargs: bool) -> ResNet_features:
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    """

    model = ResNet_features(BasicBlock, [2, 2, 2, 2], **kwargs)  # type: ignore
    if pretrained == "imagenet":
        my_dict = model_zoo.load_url(model_urls["resnet18"], model_dir=model_dir)
        my_dict.pop("fc.weight")
        my_dict.pop("fc.bias")
        my_dict.pop("conv1.weight")  # remove the first layer because we have 1 channel input instead of 3
        model.load_state_dict(my_dict, strict=False)
    elif pretrained is not None:
        raise NotImplementedError('Only "imagenet" is supported for as pretrained model')

    return model
