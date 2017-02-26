import librosa
import numpy as np


def extract_melspec(in_fp, sr, win_size, hop_size, n_mels):
    sig, sr = librosa.core.load(in_fp, sr=sr)
    feat = librosa.feature.melspectrogram(sig, sr=sr,
                                          n_fft=win_size,
                                          hop_length=hop_size,
                                          n_mels=n_mels).T
    feat = np.log(1+10000*feat)
    return feat


if __name__ == '__main__':
    in_fp = '../data/data.magnatagatune/sample_audio/sample_1.mp3'

    sr = 16000
    win_size = 512  # 512, 1024, 2048, 4096, 8192, 16384
    hop_size = 512
    n_mels = 128
    diff_order = 0

    feat = extract_melspec(in_fp, sr, win_size, hop_size, n_mels)
    print('Feature shape: {}'.format(feat.shape))
