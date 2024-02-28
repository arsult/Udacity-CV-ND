## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 1)
        self.conv1_pool = nn.MaxPool2d((2, 2), 2)
        self.conv1_bn = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 2, stride = 1)
        self.conv2_pool = nn.MaxPool2d((2, 2), 2)
        self.conv2_bn = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2, stride = 1)
        self.conv3_pool = nn.MaxPool2d((2, 2), 2)
        self.conv3_bn = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 27 * 27, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 136)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.conv1_pool(F.relu(self.conv1_bn(self.conv1(x)))) 
        x = self.conv2_pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.conv3_pool(F.relu(self.conv3_bn(self.conv3(x))))
 
        x = x.view(-1, 64 * 27 * 27)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
