import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import math

from model.ECAPA_TDNN.FbankAug import FbankAug
from model.ECAPA_TDNN.PreEmphasis import PreEmphasis
from model.ECAPA_TDNN.SE_Res2Block import SE_Res2Block



class ScaleDotProductAttention(nn.Module):
    def __init__(self,input_dim):
        super(ScaleDotProductAttention, self).__init__()
        self.w_q = nn.Linear(input_dim, input_dim)
        self.w_v = nn.Linear(input_dim, input_dim)
        self.w_k = nn.Linear(input_dim, input_dim)

        self.softmax = nn.Softmax(dim=2)
    def forward(self, x):
        q = self.w_q(x)
        v = self.w_v(x)
        k = self.w_k(x)

        score = torch.matmul(q, k.permute(0,2,1))/math.sqrt(k.size(-1))
        score = self.softmax(score)
        output = torch.matmul(score,v)
        
        return output, score

class ECAPA_TDNN(nn.Module):
    def __init__(
        self,
        channel
    ):
        super(ECAPA_TDNN, self).__init__()
        self.feature_extraction = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(
                sample_rate = 16000,
                n_fft= 512,
                win_length = 400,
                hop_length = 160,
                f_min = 20,
                f_max = 7600,
                window_fn = torch.hamming_window,
                n_mels=80
            )
        )
        self.specaug = FbankAug()

        self.conv1 = nn.Conv1d(32, channel, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(channel)
        
        self.layer1 = SE_Res2Block(channel, channel, kernel_size=3, dialation=2, scale=8)
        self.layer2 = SE_Res2Block(channel, channel, kernel_size=3, dialation=3, scale=8)
        self.layer3 = SE_Res2Block(channel, channel, kernel_size=3, dialation=4, scale=8)
        
        self.layer4 = nn.Conv1d(3*channel, 1536, kernel_size=1, dilation=1)
    
        self.attentive_state_pooling = nn.Sequential(
            nn.Conv1d(1536, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256,1536, kernel_size=1),
            nn.Softmax(dim=2)
        )
        self.ScaleDotProductAttention = ScaleDotProductAttention(1536)
        self.bn5 = nn.BatchNorm1d(6144)
        self.fc6 = nn.Linear(6144, 192)
        self.bn6 = nn.BatchNorm1d(192)

    def forward(
        self,
        input,
        aug
    ):
        with torch.no_grad():
            input = self.feature_extraction(input)+1e-6
            input = input.log()
            input = input - torch.mean(input, dim=-1, keepdim=True)
            if aug == True:
                input = self.specaug(input)
        input = self.conv1(input)
        input = self.relu(input)
        input = self.bn1(input)

        input1 = self.layer1(input)
        input2 = self.layer2(input + input1)
        input3 = self.layer3(input + input1 + input2)

        input = self.layer4(torch.cat((input1,input2,input3), dim=1))
        input = self.relu(input)

        t = input.size()[-1]
        '''
        global_input = torch.cat((input,torch.mean(input, dim=2,keepdim=True).repeat(1,1,t), \
            torch.sqrt(torch.var(input,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        '''
        w = self.attentive_state_pooling(input)
        attention_input = input.permute(0,2,1)
        
        attention_output, attention_score = self.ScaleDotProductAttention(attention_input)
        attention_output = attention_output.permute(0,2,1)
    
        mu = torch.sum(input*w, dim=2)
        sg = torch.sqrt((torch.sum((input**2)*w,dim=2)-mu**2).clamp(min=1e-4))
 
        mu_attension = torch.mean(attention_output, dim=2)
        sg_attention = torch.std(attention_output, dim=2).clamp(min=1e-4)
        
        input = torch.cat((mu,sg,mu_attension, sg_attention), 1)
        
        input = self.bn5(input)
        input = self.fc6(input)
        input = self.bn6(input)

        return input