import os
import argparse
from tqdm import tqdm

from hyperpyyaml import load_hyperpyyaml
from dataloader import load_testset
from torch.utils.data import DataLoader
from model.ECAPA_TDNN.ECAPATDNNmdoel import ECAPATDNNmodel
def main():
    '''
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
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/model.yaml')
    opt = parser.parse_args()


    with open(opt.config) as fin:
        config = load_hyperpyyaml(fin, overrides=None)
    test_dataset = load_testset(config)
    model = ECAPATDNNmodel(config)
    #print(model)
    print("Loading Model", config['eval_model_path'])    
    model.load_parameters(config['eval_model_path'])
    print("Loading Evaluation Set")
    test_dataset = load_trainset(config)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size = 32
        shuffle=False,
        num_workers = 10
    )
    
if __name__ == "__main__":
    main()