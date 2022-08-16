import os
import time
import torch
import random
import numpy as np
from queue import Empty
from pathlib import Path
from tqdm.auto import tqdm
import argparse
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.multiprocessing import Manager
from hyperpyyaml import load_hyperpyyaml
from dataloader import load_testset, loadWAV
from torch.utils.data import DataLoader
from train import test_validation
#from model.ECAPA_TDNN.ECAPATDNNmdoel import ECAPATDNNmodel
from model.MFA
from tools import *
import concurrent.futures

import librosa
import soundfile
from tqdm import tqdm
from multiprocessing import Pool, Process

#Pyotrch Multiprocessing Code based from 
#Perform Inference in multiprocessing manner
#https://m.ibric.org/miniboard/read.php?id=149713&Board=isori&BackLink=L21haW4vP3RhYj03

'''
def main():

    print("Start Evaluation")
    test_file = open('eval.txt', 'w')
    
    file_path = dict()
    for root, dir, files in os.walk("/root/project/speaker-verification/data/val/DB"):
        for file in files:
            file_path[file] = root
    with open("/root/project/speaker-verification/data/test_set/testset/trials") as fin:
        lines = fin.readlines()
        for line in tqdm(lines, total=len(lines)):
            audio_1, audio_2, label = line.split()
            audio_1 = audio_1 + '.wav'
            audio_1 = audio_1[5:]
            
            audio_2 = audio_2 + '.wav'
            audio_2 = audio_2[5:]
            
            audio_1_path = file_path.get(audio_1)
            if audio_1_path == None:
                continue 
            else: 
                audio_1_path = os.path.join(audio_1_path, audio_1)
            
            audio_2_path = file_path.get(audio_2)
            if audio_2_path == None:
                continue 
            else: 
                audio_2_path = os.path.join(audio_2_path, audio_2)
        
            
            #print(audio_1_path)
            #print(audio_2_path)
            if label == 'target':
                ans = str(1)
            else:
                ans = str(0)
            test_file.write(audio_1_path)
            test_file.write('\t')
            test_file.write(audio_2_path)
            test_file.write('\t')
            test_file.write(ans)
            test_file.write('\n')
            
            #print(line)
            #audio_1, audio_2, label = line.split()
            #print(audio_1)
            #print(audio_2)
            #print(label)
            #break

    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/model.yaml')
    opt = parser.parse_args()


    with open(opt.config) as fin:
        config = load_hyperpyyaml(fin, overrides=None)
    model = ECAPATDNNmodel(config)
    print("Loading Model", config['eval_model_path'])    
    model.load_parameters(config['eval_model_path'])
    model = model.cuda()
    
    print("Loading Evaluation Set")
    test_dataset = load_testset(config)
    EER,minDCF = test_validation(model, test_dataset, config)
    print(EER,minDCF)
'''
def read_files_into_q(test_file, queue, event, psend_pipe):
    lines = open(test_file).readlines()
    audio_1_list = list()
    audio_2_list = list()
    ans_list = list()

    for line in lines:
        audio_1, audio_2, ans = line.split()
        audio_1_list.append(audio_1)
        audio_2_list.append(audio_2)
        ans_list.append(ans)
    #audio_1_list = audio_1_list[:30000]
    #audio_2_list = audio_2_list[:30000]
    #ans_list = ans_list[:30000]

    #List provided my AIhub is in order; randomly shuffle them
    #check = list(zip(audio_1_list, audio_2_list, ans_list))
    #random.shuffle(check)
    #audio_1_list, audio_2_list, ans_list = zip(*check)
    
    #audio_1_list = list(audio_1_list)
    
    #audio_2_list = list(audio_2_list)
    #ans_list = list(ans_list)

    #audio_1_list = audio_1_list[:100000]
    #audio_2_list = audio_2_list[:100000]
    #ans_list = ans_list[:100000]
    #print(ans_list)
    
    print(f"processing {len(audio_1_list)} audio pairs...")
    
    while len(audio_1_list) >0:
        if queue.full():
            time.sleep(0.01)
            continue
        else:
            audio_1 = audio_1_list.pop()
            #print(audio_1)
            audio_2 = audio_2_list.pop()
            #print(audio_2)
            ans = ans_list.pop()
            data_1 = loadWAV(
                audio_1,
                max_frames=0,
                evalmode=True
            )
            data_2 = loadWAV(
                audio_2,
                max_frames=0,
                evalmode=True
            )
            queue.put((
                torch.FloatTensor(data_1),
                torch.FloatTensor(data_2),
                ans
            ))
            psend_pipe.send((len(audio_1_list)))
    event.set()
    queue.join()

def extract_embedding(queue, similarity, event, model, device, output):
    model.eval().to(device)
   
    while not (event.is_set() and queue.empty()):
        try: 
            audio_1, audio_2, ans = queue.get(block=True, timeout=0.1)
        except Empty:
            continue 
        with torch.no_grad():
            audio_1 = audio_1.cuda()
            audio_2 = audio_2.cuda()
            embedding_1, embedding_2 = model.eval_step(audio_1, audio_2)
            #score = similarity(embedding_1, embedding_2)
            score = torch.mean(torch.matmul(embedding_1, embedding_2.T))
            score = score.detach().cpu().numpy()
        output.append((score, int(ans)))
        queue.task_done()
    

def print_qsize(event, precv_pipe, queue):
    try:
        pbar = tqdm(bar_format="{desc}")
        while not (event.is_set() and queue.empty()):
            if not precv_pipe.poll(): continue
            remaining = precv_pipe.recv()
            pbar.desc = f"rem : {remaining:4}, " + \
                f"qsize : {queue.qsize():2}"
            pbar.update()
            time.sleep(0.01)
        pbar.close()
    except NotImplementedError as err:
        print("JoinableQueue.qsize has not been implemented;"+
            "remaining can't be shown")

def caller(device, eval_path, config, detector_count, qsize):
    start = time.time()
    queue = mp.JoinableQueue(qsize)
    event = mp.Event()
    precv_pipe, psend_pipe = mp.Pipe(duplex=False)
    closables = [queue, precv_pipe, psend_pipe]
    lock = mp.Lock()
    manager = Manager()
    similarity = torch.nn.CosineSimilarity()
    output = manager.list()
    
    reader_process = mp.Process(
        target = read_files_into_q,
        args = (eval_path, queue,  event, psend_pipe)
    )
    extract_processes = [mp.Process(target=extract_embedding,\
        args=(queue, similarity, event, get_model(config),\
            device, output))\
            for i in range(detector_count)]

    reader_process.start()
    [dp.start() for dp in extract_processes]

    print_qsize(event, precv_pipe, queue)

    # Waiting for processes to complete
    [dp.join() for dp in extract_processes]
    reader_process.join()
    
    
    scores = [i[0] for i in output]
    labels = [i[1] for i in output]
    EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
    fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
    minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1 )
    print("EER: ", EER)
    print("minDCF: ", minDCF)
    
    
    
    # Closing everything
    [c.close() for c in closables]
    print(f"time taken : {time.time() - start} s.")

def get_model(config):
    
    model = ECAPATDNNmodel(config)
    print("Loading Model", config['eval_model_path'])    
    model.load_parameters(config['eval_model_path'])
    return model
def resample(audio):
    #audio_1, audio_2, _ = line.split()
    audio, sr = librosa.load(audio)
    if sr != 16000:
        #print("wow")
        audio = librosa.resample(audio, orig_sr=sr, target_sr= 16000)
        sf.write(audio_1, audio, 16000)
    #audio, sr = librosa.load(audio_2)
    #if sr != 16000:
        #print("wow")
        #audio = librosa.resample(audio, orig_sr=sr, target_sr= 16000)
        #sf.write(audio_2, audio, 16000)
    return 0    
def check_and_resample(path):
    try:
        audio, sr = soundfile.read(path)
    except Exception as e:
        print(path)
        print(e)
        #print("check")
        return
    else:
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr= 16000)
            try:
                librosa.output.write_wav(path, audio, sr=16000)
            except Exception as e:
                print(e)
            return path
        else:
            return path
if __name__ == "__main__":
    
    for i in range( )
        mp.set_start_method("spawn", force=True)
        #main()
        parser = argparse.ArgumentParser()
        parser.add_argument('-config', type=str, default='config/model.yaml')
        opt = parser.parse_args()
        with open(opt.config) as fin:
            config = load_hyperpyyaml(fin, overrides=None)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        eval_path = config['eval_file']
        caller(device,eval_path, config, detector_count=17, qsize=17)
    
    
    
    
    '''  
    fin = open("/root/project/speaker-verification/data/train.txt")
    lines = fin.readlines()
    files = list()
    for line in lines:
        audio_1, _ = line.split()
        files.append(audio_1.strip())
        #files.append(audio_2.strip())
    files = list(set(files))
    #print(files)

     
    with concurrent.futures.ProcessPoolExecutor(20) as executor:
        clean_wav_files = list(executor.map(check_and_resample, files))
    
    '''
