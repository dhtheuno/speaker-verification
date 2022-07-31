import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ECAPA_TDNN.ECAPA_TDNN import ECAPA_TDNN
from model.ECAPA_TDNN.loss import AAMsoftmax

class ECAPATDNNmodel(nn.Module):
    def __init__(self, hparams):
        super(ECAPATDNNmodel, self).__init__()
        channel = hparams['C']
        n_class = hparams['n_class']
        m = hparams['m']
        s = hparams['s']
        #self.aug = hparams['specaug']
        self.device = hparams['device']
        
        self.speaker_encoder = ECAPA_TDNN(channel = channel)
        self.speaker_loss = AAMsoftmax(n_class = n_class, m = m, s=s)
    
    def train_step(self, features, labels, aug):
        speaker_embedding = self.speaker_encoder.forward(features, aug = aug)
        nloss, prec = self.speaker_loss.forward(speaker_embedding,labels)
        return speaker_embedding, nloss, prec

    def eval_step(self, features_1, features_2):
        with torch.no_grad():
            embedding_1 = self.speaker_encoder.forward(features_1, aug=False)
            embedding_1 = F.normalize(embedding_1, p=2, dim=1)
            
            embedding_2 = self.speaker_encoder(features_2, aug=False)
            embedding_2 = F.normalize(embedding_2, p=2, dim=1)

        return embedding_1, embedding_2
    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model."%origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)