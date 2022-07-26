import time
import torch
import tqdm
from tools import *

def train(epoch, model, train_dataloader, hparams, logger, optim, scheduler):
    model.train() 
    
    device = hparams['device']
    fbankaug = hparams['fbankaug']

    
    index, top1, loss = 0, 0, 0
    lr = optim.param_groups[0]['lr']
    logger.info("start Traing Epoch %d"%(epoch+1))
    for num, (audio, label) in enumerate(train_dataloader):
        optim.zero_grad()
        label = torch.LongTensor(label)
        
        if device == 'cuda':
            label = label.cuda()
            audio = audio.cuda()
        speaker_embedding, nloss, prec = model.train_step(audio, label, aug=fbankaug)
        nloss.backward()
        optim.step()
        index += len(label)
        top1 += prec
        loss += nloss.detach().cpu().numpy()
        logger.info(" [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * ((num+1) / train_dataloader.__len__())) + \
			" Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num+1), top1/index*len(label)))
    scheduler.step(epoch)

    return loss/(index+1), lr , top1/(index+1)*len(label)

def validation(model, test_dataloader, hparams, logger):
    model.eval()
    logger.info("Starting Valdiation")
    scores, labels = [], []
    for index, (audio_1, audio_2, ans) in enumerate(test_dataloader):
        embedding_1, embedding_2 = model.eval_step(audio_1, audio_2)
        score = torch.mean(torch.matmul(embedding_1, embedding_2))
        score = score.detach().cpu().numpy()
        scores.append(score)
        labels.append(ans)
    EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
    fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
    minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1 )
    return EER, minDCF