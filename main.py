import os
import time
from tqdm import tqdm
import argparse
import torch
import shutil

from hyperpyyaml import load_hyperpyyaml
from torch.utils.data import DataLoader

from train import train, validation
from dataloader import load_trainset, load_testset
from model.ECAPA_TDNN.ECAPATDNNmdoel import ECAPATDNNmodel

from tools import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/model.yaml')
    opt = parser.parse_args()


    with open(opt.config) as fin:
        config = load_hyperpyyaml(fin, overrides=None)

    output_folder = config['output_folder']
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    
    train_log = config['train_log']
    logger = init_logger(train_log)

    shutil.copyfile(opt.config, os.path.join(output_folder, 'config.yaml'))
    logger.info('Save config info.')

    
    logger.info('data augmentation is')
    logger.info(config['aug'])
    
    
    batch_size = config['dataloader_options']['batch_size']
    num_workers = config['dataloader_options']['num_workers']
    
    logger.info('loading train_dataset')
    train_dataset = load_trainset(config['dataset_options'])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers= num_workers
    )

    logger.info('loading test_dataset')
    test_dataset = load_testset(config['dataset_options'])
    '''
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle = False,
        num_workers = num_workers
    )
    '''
    lr = config['lr']
    lr_decay = config['lr_decay']
    weight_decay = float(config['weight_decay'])
    
    num_epoch = config['num_epoch']
    eval_interval = config['eval_interval']
    
    logger.info("Loading model, optimizer and scheudler")
    model = ECAPATDNNmodel(config['model_options'])
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=eval_interval, gamma=lr_decay)
    
    EERs = []
    if config['device'] == 'cuda':
        model = model.cuda()
    for epoch in tqdm(range(num_epoch)):
        loss, lr, acc = train(epoch, model, train_dataloader, config, logger, optimizer, scheduler)
        if (epoch+1)%eval_interval == 0:
            output_path = os.path.join(config['save_folder'],f"ECAPA_TDNN_{epoch}.pt")
            model.save_parameters(output_path)
            logger.info("save model in %s"%(output_path))
            
            logger.info("start validation")
            EER, minDCF = validation(model, test_dataset, config, logger)
            EERs.append(EER)
            logger.info("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%"%(epoch, lr, loss, acc, EERs[-1], min(EERs)))        
if __name__ == "__main__":
    main()