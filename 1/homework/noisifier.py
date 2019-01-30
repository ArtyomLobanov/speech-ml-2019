#!/usr/bin/python
import numpy as np 
import os 
import sys
import pandas as pd
import numpy as np
import random
import itertools
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import random
import ntpath
import math
import soundfile as sf

def get_sounds(path, sr):
    res = []
    for file in os.listdir(path):
        if file.endswith("wav"):
            data, _ = librosa.core.load(os.path.join(path, file), sr)
            res.append(data)
    return res
    

def add_music(data, music, alpha):
    p = 0
    n = len(data)
    m = len(music)
    while p < n:
        r = min(n - p, m)
        data[p:p + r] += alpha * music[:r]
        p += m
    np.clip(data, -1, 1, out=data)
        
def add_beep(data, beep, time, alpha):
    r = min(len(beep), len(data) - time)
    data[time:time + r] += alpha * beep[:r]
    np.clip(data[time:time + r], -1, 1, out=data[time:time + r])
    
def add_random_noise_mono(data, musics, beeps, music_alpha, beep_alpha, beep_frequency, rate):
    n = len(data)
    music = random.choice(musics)
    add_music(data, music, music_alpha)
    beep_count = int(math.ceil(n / rate * beep_frequency))
    for i in range(beep_count):
        add_beep(data, random.choice(beeps), random.randint(0, n - 1), beep_alpha)
        
    
def add_random_noise_stereo(data, musics, beeps, music_alpha, beep_alpha, beep_frequency, rate):
    n = len(data)
    music = random.choice(musics)
    add_music(data[:, 0], music, music_alpha)
    add_music(data[:, 1], music, music_alpha)
    beep_count = int(math.ceil(n / rate * beep_frequency))
    for i in range(beep_count):
        beep = random.choice(beeps)
        time = random.randint(0, n - 1)
        add_beep(data[:, 0], beep, time, beep_alpha)
        add_beep(data[:, 1], beep, time, beep_alpha)

def generate_noised_versions(data_path, target_path, noise_path, music_alpha, beep_alpha, beep_frequency):
    os.makedirs(target_path, exist_ok=True)
    
    musics_cache = {}
    beeps_cache = {}
    
    def get_noise_for_rate(rate):
        if rate in musics_cache:
            return musics_cache[rate], beeps_cache[rate]
        musics = get_sounds(os.path.join(noise_path, "music"), rate)
        beeps = get_sounds(os.path.join(noise_path, "beep"), rate)
        musics_cache[rate] = musics
        beeps_cache[rate] = beeps
        return musics, beeps
    
    for file in os.listdir(data_path):
        if file.endswith("wav"):
            data, rate = librosa.core.load(os.path.join(data_path, file))
            musics, beeps = get_noise_for_rate(rate)
            add_random_noise_mono(data, musics, beeps, music_alpha, beep_alpha, beep_frequency, rate)
            librosa.output.write_wav(os.path.join(target_path, file), data, rate)
        if file.endswith("flac"):
            data, rate = sf.read(os.path.join(data_path, file))
            musics, beeps = get_noise_for_rate(rate)
            add_random_noise_stereo(data, musics, beeps, music_alpha, beep_alpha, beep_frequency, rate)
            sf.write(os.path.join(target_path, file), data, rate)

def main(argv):
   data_path = argv[0]
   target_path = argv[1]
   noise_path = argv[2]
   music_alpha = float(argv[3])
   beep_alpha = float(argv[4])
   beep_frequency = float(argv[5])
   generate_noised_versions(data_path, target_path, noise_path, music_alpha, beep_alpha, beep_frequency)

if __name__ == "__main__":
   main(sys.argv[1:])