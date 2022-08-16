import torch 
import torch.nn as nn

from model.ECAPA_TDNN.ECAPATDNNmdoel import ECAPATDNNmdoel

class classification(nn.Moduel):
    def __init__(self, config):
        super(classification, self).__init__()
        model = ECAPATDNNmodel(config)
        #print("Loading Model", config['eval_model_path'])    
        
        model.load_parameters(config['eval_model_path'])
        self.speaker_encoder = model.speaker_encoder
        '''
        self.lstm  = nn.LSTM(
            80,
            256,
            2, 
            batch_first=True, 
            dropout = 0.1, 
            bidirectional=True
        )
        
        self.averge_pooling = nn.
        '''
        self.linear = nn.Linear(192,2)
        
        self.softmax = nn.Softmax(dim=1)

    def forawrd(self, x):
        x = self.speaker_encoder.forward(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x