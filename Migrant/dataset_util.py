import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from util import pad_sequence, process_TW

class TransDataset(Dataset):
    def __init__(self, X, Y, length):
        self.X = X
        self.Y = Y
        self.length = length

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        data = self.X[idx]
        label = self.Y[idx]

        timestamp = data[:, 0].astype(float)
        packet_length = data[:, 1].astype(int)

        timestamp = pad_sequence(timestamp, self.length)
        packet_length = pad_sequence(packet_length, self.length)

        matrix = process_sign_matrix(timestamp, packet_length,
                                     bin_ms=0.044,
                                     matrix_w=1800,
                                     )

        return torch.from_numpy(matrix), label


class MigrantDataset(Dataset):
    def __init__(self, X, Y, length, win, loaded_ratio):
        self.X = X
        self.Y = Y
        self.length = length
        self.loaded_ratio = loaded_ratio
        self.win = win

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        data = self.X[index]
        label = self.Y[index]

        timestamp = data[:, 0]
        packet_length = data[:, 1]


        loading_time = timestamp.max()
        threshold = loading_time * self.loaded_ratio / 100

        timestamp = np.trim_zeros(timestamp, "b")
        valid_index = np.where(timestamp <= threshold)

        timestamp = timestamp[valid_index]
        packet_length = packet_length[valid_index]

        timestamp = pad_sequence(timestamp, self.length)
        packet_length = pad_sequence(packet_length, self.length)

        return self.process_data(packet_length,timestamp,self.win), label

    def process_data(self, packet_length, timestamp, win):
        TW_raw = process_TW(packet_length, timestamp, win)

        N_raw = TW_raw.shape[1]
        if N_raw == 1800:
            TW_fix = TW_raw
        elif N_raw < 1800:
            TW_fix = np.zeros((4, 1800), dtype=np.float32)
            TW_fix[:, :N_raw] = TW_raw
        else:
            TW_fix = TW_raw[:, :1800]

        return TW_fix.astype(np.float32)



class DirectionDataset(Dataset):
    def __init__(self, X, Y, length, loaded_ratio=100):
        self.X = X
        self.Y = Y
        self.length = length
        self.loaded_ratio = loaded_ratio

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        data = self.X[index]
        label = self.Y[index]

        timestamp = data[:, 0]
        sign = np.sign(data[:, 1])

        loading_time = timestamp.max()
        threshold = loading_time * self.loaded_ratio / 100

        timestamp = np.trim_zeros(timestamp, "b")
        valid_index = np.where(timestamp <= threshold)

        sign = sign[valid_index]

        return self.process_data(sign), label

    def process_data(self, direction):
        direction = pad_sequence(direction, self.length)

        return direction.reshape(1, -1).astype(np.float32)
