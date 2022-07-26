import os
import glob  
import argparse
import librosa
import collections
from tqdm import tqdm
import soundfile
import random
import concurrent.futures

from hyperpyyaml import load_hyperpyyaml

def check_and_resample(path):
    try:
        audio, sr = soundfile.read(path)
    except:
        print("check")
        return
    else:
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr= 16000)
            librosa.output.write_wav(path, audio, sr=16000)
            return path
        else:
            return path
def get_wav_files(path):
    wav_files = glob.glob(os.path.join(path, '*/*/*/*.wav'))
    #clean_wav_files = list()
    '''
    for i in tqdm(wav_files, total=len(wav_files)):
        try:
            audio, sr = soundfile.read(i)
        except:
            print("check")
            continue
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr= 16000)
            librosa.output.write_wav(i, audio, sr=16000)
        clean_wav_files.append(i)
    '''
    with concurrent.futures.ProcessPoolExecutor(10) as executor:
        clean_wav_files = list(executor.map(check_and_resample, wav_files))
    wav_files = list(filter(None,clean_wav_files))
    speaker_ids = [os.path.basename(wav_file).split("-")[1][0:4] for wav_file in wav_files]
    
    dictkeys = list(set(speaker_ids))
    dictkeys.sort()
    dictkeys = {key:num for num, key in enumerate(dictkeys)}
    
    return wav_files, speaker_ids, dictkeys
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str)
    opt = parser.parse_args()


    with open(opt.config) as fin:
        config = load_hyperpyyaml(fin, overrides=None)
    
    wav_files, speakers_ids, dictkeys = get_wav_files(config['train_data_path'])
    data_path= config['data_folder']
    if not os.path.isdir(data_path):
        os.mkdir(data_path)
    train_file = config['dataset_options']['train_file']
    with open(train_file, 'w') as f:
        for wav_file, speakers_id in tqdm(zip(wav_files, speakers_ids), total=len(wav_files)):
            '''
            audio, sr = librosa.load(wav_file)
            if sr != 16000:
                resample_audio = librosa.resample(y, orig_sr=sr, target_sr= 16000)
                librosa.output.write_wav(wav_file, resample_audio, 16000)
            '''
            f.write(wav_file)
            f.write("\t")
            f.write(str(dictkeys[speakers_id]))
            f.write("\n") 
    print("total number of ids: ", len(dictkeys))

    wav_files, speakers_ids, dictkeys = get_wav_files(config['val_data_path'])
    wav_dict = collections.defaultdict(list)
    for wav_file, speakers_id in zip(wav_files, speakers_ids):
        wav_dict[speakers_id].append(wav_file)
    ids = [key for key, _ in wav_dict.items()]
    test_pair_files = list()
    ans = list()
    print("total number of validation audios: ", len(wav_files))
    test_set_num = len(wav_files)//2
    print("Creating %d number of validation test set"%test_set_num)
    for i in range(test_set_num):
        random_float = random.uniform(0,1)
        if random_float >0.5:
            test_id_1, test_id_2 = random.sample(ids,2)
            assert test_id_1 != test_id_2
            test_id_1_wav = random.choice(wav_dict[test_id_1])
            test_id_2_wav = random.choice(wav_dict[test_id_2])
            test_pair_files.append((test_id_1_wav, test_id_2_wav))
            ans.append(0)
        else:
            test_id_1 = random.sample(ids,1)[0]
            test_pair_files.append(tuple(random.choices(wav_dict[test_id_1],k=2)))
            ans.append(1)
    assert len(ans) == len(test_pair_files)
    print("writing them to text file ")
    test_file = config['dataset_options']['test_file']
    with open(test_file, 'w') as f:
        for (audio_1_path, audio_2_path), label in tqdm(zip(test_pair_files, ans), total=len(ans)):

            f.write(audio_1_path)
            f.write("\t")
            f.write(audio_2_path)
            f.write("\t")
            f.write(str(label))
            f.write("\n") 
    print("DONE!!")
if __name__ == "__main__":
        main()
        #test = librosa.load("data/train/DB/continuous/2021-12-16/2988/D0123-2988M1021-0__000_0-05918066.wav")
