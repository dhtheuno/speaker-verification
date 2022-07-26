import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(
        self,
        input_channel,
        scaled_channel=128
    ):
        super(SEBlock, self).__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(input_channel, scaled_channel, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(scaled_channel,input_channel, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(
        self,
        input
    ):
        output = self.block(input)
        return output*input