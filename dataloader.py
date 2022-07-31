#From clovaai/voxcelb_trainer/DatasetLoader
#Modified so it can be used with hyperpyyaml

import os
import torch
import glob
import numpy 
import random
import soundfile
import librosa
from scipy import signal
import numpy as np

from torch.utils.data import Dataset

'''
def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)
'''

def loadWAV(filename, max_frames, evalmode=True, num_eval=10):
    max_audio = max_frames * 160 + 240
    audio, sr = soundfile.read(filename)
    '''
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr= 16000)
    '''
    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1
        audio = numpy.pad(audio, (0, shortage), 'wrap')
        audiosize = audio.shape[0]
    
    if evalmode:
        startframe = numpy.linspace(0, audiosize-max_audio, num=num_eval)
    else:
        startframe = numpy.array([numpy.int64(random.random()*(audiosize-max_audio))])

    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])

    feat = numpy.stack(feats, axis=0).astype(numpy.float)
    return feat

class AugmentWAV(object):
    def __init__(self, musan_path, rir_path, max_frames):
        self.max_frames = max_frames
        self.max_audio = max_frames * 160 + 240
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr   = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise   = {'noise':[1,1], 'speech':[3,7],  'music':[1,1] }
        self.noiselist  = {}
        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*.wav'));

        for file in augment_files:
            if not file.split('/')[-3] in self.noiselist:
                self.noiselist[file.split('/')[-3]] = []
            self.noiselist[file.split('/')[-3]].append(file)

        self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))
        #print(self.rir_files)

    def additive_noise(self, noisecat, audio):

        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 

        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))

        noises = []

        for noise in noiselist:

            noiseaudio  = loadWAV(noise, self.max_frames, evalmode=False)
            noise_snr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio[0] ** 2)+1e-4) 
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)

        return numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True) + audio
    
    def reverberate(self, audio):

        rir_file    = random.choice(self.rir_files)
        
        rir, fs     = soundfile.read(rir_file)
        rir         = numpy.expand_dims(rir.astype(numpy.float),0)
        rir         = rir / numpy.sqrt(numpy.sum(rir**2))

        return signal.convolve(audio, rir, mode='full')[:,:self.max_audio]

class load_trainset(Dataset):
    def __init__(self, hparams):

        self.augment = hparams['aug']
        self.train_file = hparams['train_file']
        self.max_frames = hparams['max_frames']
        self.musan_path = hparams['musan_path']
        self.rir_path = hparams['rir_path']
        self.num_eval = hparams['num_eval']
        self.augment_wav = AugmentWAV(
            musan_path = self.musan_path,
            rir_path = self.rir_path,
            max_frames = self.max_frames
        )

        self.audio_list = list() 
        self.label_list = list()
        
        lines = open(self.train_file).read().splitlines()
        for index, line in enumerate(lines):
            wav_path, label = line.strip().split()
            self.audio_list.append(wav_path)
            self.label_list.append(label)
        
        #self.audio_list = self.audio_list[0:2400]
        #self.label_list = self.label_list[0:2400]
    def __getitem__(self, index):
        audio = self.audio_list[index]
        label = self.label_list[index]
    
        audio = loadWAV(
            audio,
            self.max_frames,
            evalmode = False,
            num_eval=self.num_eval
        )
        if self.augment:
            augtype = random.randint(0,4)
            if augtype == 1:
                audio   = self.augment_wav.reverberate(audio)
            elif augtype == 2:
                audio   = self.augment_wav.additive_noise('music',audio)
            elif augtype == 3:
                audio   = self.augment_wav.additive_noise('speech',audio)
            elif augtype == 4:
                audio   = self.augment_wav.additive_noise('noise',audio)
        
        return torch.FloatTensor(audio[0]), int(label)
    def __len__(self):
        return len(self.audio_list)

class load_testset(Dataset):
    def __init__(self, hparams):

        self.test_files = hparams['eval_file']
        self.eval_frames = hparams['eval_frames']

        lines = open(self.test_files).readlines()
        
        self.audio_1_list = list()
        self.audio_2_list = list()
        self.ans_list = list()

        for line in lines:
            audio_1, audio_2, ans = line.split()
            
            self.audio_1_list.append(audio_1)
            self.audio_2_list.append(audio_2)
            self.ans_list.append(ans)
        self.audio_1_list = self.audio_1_list
        self.audio_2_list = self.audio_2_list
        self.ans_list = self.ans_list
    def __len__(self):
        return len(self.audio_1_list)
    def __getitem__(self, index):
        audio_1 = self.audio_1_list[index]
        audio_2 = self.audio_2_list[index]
        ans = self.ans_list[index]
        
        data_1 = loadWAV(
            audio_1,
            max_frames=self.eval_frames,
            evalmode=True
        )
        data_2 = loadWAV(
            audio_2,
            max_frames=self.eval_frames,
            evalmode=True
        )
        return torch.FloatTensor(data_1[0]), torch.FloatTensor(data_2[0]), ans


