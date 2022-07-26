import torch
import torchaudio
import torch.nn as nn

class FbankAug(nn.Module):
    def __init__(
        self,
        freq_mask_width = (0,8),
        time_mask_width = (0,10)
    ):
        super().__init__()
        self.freq_mask_width = freq_mask_width
        self.time_mask_wdith = time_mask_width
    def mask_along_axis(self, input, axis):
        original_size = input.shape
        batch_size, features, times = input.shape
        
        if axis == 1:
            axis_type = features
            width_range = self.freq_mask_width
        else:
            axis_type = times
            width_range = self.time_mask_wdith
    
        mask_len = torch.randint(width_range[0], width_range[1], (batch_size,1), device=input.device).unsqueeze(2)
        mask_pos = torch.randint(0, max(1, axis_type-mask_len.max()), (batch_size,1), device=input.device).unsqueeze(2)
        arange= torch.arange(axis_type, device=input.device)
        arange= torch.arange(axis_type, device=input.device).view(1, 1, -1)
        mask = (mask_pos <= arange) * (arange <(mask_pos+mask_len))
        mask = mask.any(dim=1)

        if axis == 1:
            mask = mask.unsqueeze(2)
        else:
            mask = mask.unsqueeze(1)

        input = input.masked_fill(mask, 0.0)
        return input.view(*original_size)
    
    def forward(self, input):
        input = self.mask_along_axis(input, axis=2)
        input = self.mask_along_axis(input, axis=1)
        return input