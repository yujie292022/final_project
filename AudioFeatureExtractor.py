import numpy as np
#from numpy.fft import fft, ifft
#from matplotlib import pyplot as plt
import pandas as pd
import librosa  as lb
import librosa.display as lbd
from glob import glob
import re
from matplotlib import pyplot as plt

class AudioFeatureExtractor():
    def __init__(self,data_dir):
        #self.audio_files = glob(data_dir+'/*.wav')
        self.audio_files = glob(data_dir+'0_01_1.wav')
    
    def getTargetLabel(self, file):
        filename="".join(re.split('./sound_data/mnist/',file))
        target = filename[0]
  
        return target

    def splitSignal(self, audioSignal, nsegments):
        signalLength = len(audioSignal)
        segmentLength = int(np.ceil(signalLength/nsegments))
        audioSegments = []
        for i in range(nsegments):
            first = i*segmentLength
            last = first  + segmentLength - 1
            if last > (signalLength-1):
                last  = signalLength-1
            audioSegments.append(audioSignal[first:last])
    
        return audioSegments
    
    def constructMFCCFeatures(self, nsegments=10, num_mfcc=13):
        column_labels=["Target"]
        for q in range(num_mfcc):
            column_labels.append("MFCC "+str(q))
        data = []

        for file in self.audio_files:
            audio_data, Fs = lb.load(file,sr=48000)
            # yfft, freqs = self.performFFT(audio_data, Fs, freq_cutoff)
            # segments = self.splitSignal(audio_data, nsegments)
            segments = self.splitSignal(audio_data, nsegments)
            target = self.getTargetLabel(file)
            # freq_cutoff = 1000            
            
            for j in range(nsegments):
                #D = lb.stft(data)
                D= lb.feature.mfcc(y=segments[j],sr=Fs, n_mfcc=num_mfcc)
                s_db = np.mean(lb.amplitude_to_db(np.abs(D),ref = np.max),1)
                data_entry = [target] + s_db.tolist()
                data.append(data_entry)
        self.mfcc_data_frame = pd.DataFrame(data, columns = column_labels)
        return self.mfcc_data_frame

directory = "./sound_data/mnist/"       
Ad = AudioFeatureExtractor(directory)
s_db =Ad.constructMFCCFeatures()
mfcc = s_db.iloc[:,1:].to_numpy().T
print(s_db)
print(mfcc)
Fs = 48000
plt.figure()
lbd.specshow(mfcc, x_axis='time', y_axis='mel', sr=Fs)
plt.colorbar(format="%+2.f dB")
plt.show()