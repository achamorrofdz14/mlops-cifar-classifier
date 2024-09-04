# ruff: noqa
"""File containing the model architecture for CIFAR-10 problem."""

import torch
import torch.nn.functional as F
from torch import nn


class CIFARCNN(nn.Module):
    """CNN model."""

    # Constructor
    def __init__(
        self,
        out_1: int = 32,
        out_2: int = 64,
        out_3: int = 128,
        p: int = 0,
    ) -> None:
        """Initialize the CIFARCNN model.

        :param out_1: number of output channels for the first convolutional layer.
        :param out_2: number of output channels for the second convolutional layer.
        :param out_3: number of output channels for the third convolutional layer.
        :param p: dropout probability.
        """
        super().__init__()
        self.cnn1 = nn.Conv2d(
            in_channels=3,
            out_channels=out_1,
            kernel_size=5,
            padding=2,
        )
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv1_bn = nn.BatchNorm2d(out_1)
        self.drop_conv = nn.Dropout(p=0.2)

        self.cnn2 = nn.Conv2d(
            in_channels=out_1,
            out_channels=out_2,
            kernel_size=5,
            padding=2,
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv2_bn = nn.BatchNorm2d(out_2)

        self.cnn3 = nn.Conv2d(
            in_channels=out_2,
            out_channels=out_3,
            kernel_size=5,
            padding=2,
        )
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv3_bn = nn.BatchNorm2d(out_3)

        # Hidden layer 1
        self.fc1 = nn.Linear(out_3 * 4 * 4, 1000)
        self.drop = nn.Dropout(p=p)
        self.fc1_bn = nn.BatchNorm1d(1000)

        # Hidden layer 2
        self.fc2 = nn.Linear(1000, 1000)
        self.fc2_bn = nn.BatchNorm1d(1000)

        # Hidden layer 3
        self.fc3 = nn.Linear(1000, 1000)
        self.fc3_bn = nn.BatchNorm1d(1000)

        # Hidden layer 4
        self.fc4 = nn.Linear(1000, 1000)
        self.fc4_bn = nn.BatchNorm1d(1000)

        # Final layer
        self.fc5 = nn.Linear(1000, 10)
        self.fc5_bn = nn.BatchNorm1d(10)

    # Predictiona
    def forward(self, x) -> torch.Tensor:
        """Forward pass of the model.

        :param x: input tensor

        :return: output tensor
        """
        x = self.cnn1(x)
        x = self.conv1_bn(x)
        x = self.maxpool1(x)
        x = self.drop_conv(x)

        x = self.cnn2(x)
        x = self.conv2_bn(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = self.drop_conv(x)

        x = self.cnn3(x)
        x = self.conv3_bn(x)
        x = torch.relu(x)
        x = self.maxpool3(x)
        x = self.drop_conv(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc1_bn(x)

        x = F.relu(self.drop(x))
        x = self.fc2(x)
        x = self.fc2_bn(x)

        x = F.relu(self.drop(x))
        x = self.fc3(x)
        x = self.fc3_bn(x)

        x = F.relu(self.drop(x))
        x = self.fc4(x)
        x = self.fc4_bn(x)

        x = F.relu(self.drop(x))
        x = self.fc5(x)
        x = self.fc5_bn(x)

        return x
