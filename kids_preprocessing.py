import os
import librosa
import numpy as np
from multiprocessing import Pool, Process, Queue
from tqdm import tqdm
import soundfile as sf
import matplotlib.pyplot as plt


def check_duration(audio_path, q):
    audio,sr = sound.load(audio_path)
    #print("sampling rate: ",sr)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr= 16000)
        sf.write(audio_path, audio, 16000)
    #assert sr == 16000
    duration = len(audio)/16000
    q.put(duration)
    #return duration

def check(duration_list):
    durations = np.array(duration_list)
    mean_duration = np.mean(durations)
    low_duration = np.min(durations)
    max_duration = np.max(durations)
    return mean_duration, low_duration, max_duration

def get_audios(base_dir):
    audios = list()
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('wav'):
                audios.append(f"{root}/{file}")
    
    audio_num = len(audios)
    return audio_num, audios


if __name__ == "__main__":
    base_dir = "/root/project/speaker-verification/data/kids/IPTV_kids"
    audio_num, audios = get_audios(base_dir)
    print(audio_num)
    print(audios[0])
    durations = list()
    q = Queue()
    #audios = audios[0:100]
    for audio in tqdm(audios, total=len(audios)):
        proc = Process(target=check_duration, args=(audio,q))
        proc.start()
    for i in range(len(audios)):
        durations.append(q.get())
    print(len(durations))
    #print(durations[0:100])
    means, lows, maxs = check(durations)
    print(means)
    print(lows)
    print(maxs)
    fig = plt.figure(figsize=(10,7))
    plt.boxplot(durations)
    plt.savefig("data.png")
    plt.show()
        #duration = check_duration(audio)
        #durations.append(duration)
    #print(len(durations))

    '''
    with Pool(5) as p:
        duration_list = p.map(check_duration, audios)
        print(len(duration_list))
    '''