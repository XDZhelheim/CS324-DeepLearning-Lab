from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn


class ConvBNReLU(nn.Module):
    """
    This module references to a class from
    https://github.com/ShiqiYu/SimpleCNNbyCPP
    
    which combines convolution, batch normalization and ReLU layers.
    
    author: Prof. Shiqi Yu & "fengyuentau"
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0):
        super(ConvBNReLU, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.out_channels,
                              kernel_size=self.kernel_size,
                              stride=self.stride,
                              padding=self.padding,
                              bias=True)
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out
    
class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.reshape(self.shape)


class CNN(nn.Module):
    def __init__(self, n_channels, n_classes):
        """
        Initializes CNN object. 
        
        Args:
        n_channels: number of input channels
        n_classes: number of classes of the classification problem
        """
        super(CNN, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        # self.layers=nn.Sequential(
        #     ConvBNReLU(self.n_channels, 64, kernel_size=3, stride=1, padding=1),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
        #     ConvBNReLU(64, 128, kernel_size=3, stride=1, padding=1),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
        #     ConvBNReLU(128, 256, kernel_size=3, stride=1, padding=1),
        #     ConvBNReLU(256, 256, kernel_size=3, stride=1, padding=1),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
        #     ConvBNReLU(256, 512, kernel_size=3, stride=1, padding=1),
        #     ConvBNReLU(512, 512, kernel_size=3, stride=1, padding=1),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
        #     ConvBNReLU(512, 512, kernel_size=3, stride=1, padding=1),
        #     ConvBNReLU(512, 512, kernel_size=3, stride=1, padding=1),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
        #     Reshape(-1, 512),
            
        #     nn.Linear(512, self.n_classes),
        #     nn.Softmax(dim=1)
        # )
        
        self.layers = nn.Sequential(
            ConvBNReLU(3, 32, 3, 1),      # 32 -> 30
            nn.MaxPool2d(2, 2),            # 30 -> 15
            ConvBNReLU(32, 32, 3, 2, 1),    # 15 -> 8
            
            Reshape(-1, 32*8*8),
            
            nn.Linear(32*8*8, self.n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        Performs forward pass of the input.
        
        Args:
        x: input to the network
        Returns:
        out: outputs of the network
        """
        out = self.layers(x)
        # print(x.shape)
        # for layer in self.layers:
        #     print(layer.__class__.__name__)
        #     x=layer.forward(x)
        #     print(x.shape)
        # out=x
        return out
