import os
import tempfile

import librosa
import numpy as np
import pandas as pd
from scipy.io import wavfile


class FeatureExtractor:
    def extract_features(self, wav_path):
        """
        Extracts features for classification ny frames for .wav file

        :param wav_path: string, path to .wav file
        :return: pandas.DataFrame with features of shape (n_chunks, n_features)
        """
        raise NotImplementedError("Should have implemented this")


class PyAAExtractor(FeatureExtractor):
    """Python Audio Analysis features extractor"""

    def __init__(self):
        self.extract_script = "./extract_pyAA_features.py"
        self.py_env_name = "ipykernel_py2"

    def extract_features(self, wav_path):
        with tempfile.NamedTemporaryFile() as tmp_file:
            feature_save_path = tmp_file.name
            cmd = "python \"{}\" --wav_path=\"{}\" " \
                  "--feature_save_path=\"{}\"".format(self.extract_script, wav_path, feature_save_path)
            os.system("source activate {}; {}".format(self.py_env_name, cmd))

            feature_df = pd.read_csv(feature_save_path)
        return feature_df


class LibrosaExtractor(FeatureExtractor):
    def __init__(self, frame_sec=0.5):
        self.frame_sec = frame_sec

    def extract_features(self, wav_path):
        rate, data = wavfile.read(wav_path)
        data = np.float64(data)
        frame_size = int(rate * self.frame_sec)
        features = []
        for i in range(0, len(data), frame_size):
            mfcc = librosa.feature.mfcc(data[i:i + frame_size], rate)
            spec = librosa.feature.melspectrogram(data[i:i + frame_size], rate)
            features.append(
                np.concatenate((np.mean(mfcc.T, axis=0),
                                np.mean(spec.T, axis=0))))
        return pd.DataFrame(np.array(features))
