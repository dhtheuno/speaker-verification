import math
import torch
import torch.nn as nn
from model.ECAPA_TDNN.SEBlock import SEBlock

class SE_Res2Block(nn.Module):
    def __init__(
        self,
        input_channel,
        current_channel,
        kernel_size = None,
        dialation = None,
        scale = 8
    ):
        super(SE_Res2Block, self).__init__()
        width = int(math.floor(current_channel/scale))
        self.conv1 = nn.Conv1d(input_channel, width*scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width*scale)
        self.nums = scale-1
        convs = []
        bns = []
        
        num_pad = math.floor(kernel_size/2)*dialation
        for _ in range(self.nums):
            convs.append(
                nn.Conv1d(
                    width,
                    width, 
                    kernel_size = kernel_size,
                    dilation = dialation,
                    padding = num_pad
                )
            )
            
            bns.append(
                nn.BatchNorm1d(
                    width
                )
            )
        
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width*scale, current_channel, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(current_channel)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEBlock(current_channel)
    
    def forward(self, input):
        residual = input
        
        output = self.conv1(input)
        output = self.relu(output)
        output = self.bn1(output)

        splits = torch.split(output, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                split = splits[i]
            else:
                split = split+splits[i]
            split = self.convs[i](split)
            split = self.relu(split)
            split = self.bns[i](split)
            
            if i == 0:
                output = split
            else:
                output = torch.cat((output, split),1)
        output = torch.cat((output, splits[self.nums]), 1)
        
        output = self.conv3(output)
        output = self.relu(output)
        output = self.bn3(output)

        output = self.se(output)
        output += residual
        return output

        