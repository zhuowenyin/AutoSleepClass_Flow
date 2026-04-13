import os
import re
import numpy as np
import scipy.io as so
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import sys

def load_stateidx(ppath, name, ann_name=''):
    ddir = os.path.join(ppath, name)
    ppath, name = os.path.split(ddir)

    if ann_name == '':
        ann_name = name

    remidxfile3 = os.path.join(ppath, name, '3_remidx_' + ann_name + '.txt')
    remidxfile_regular = os.path.join(
        ppath, name, 'remidx_' + ann_name + '.txt')

    if os.path.exists(remidxfile3):
        remidxfile = remidxfile3
        print('3_remdix')
    else:
        remidxfile = remidxfile_regular
        print('remdix')

    f = open(remidxfile, 'r')
    lines = f.readlines()
    f.close()

    n = 0
    for l in lines:
        if re.match(r'\d', l):
            n += 1

    M = np.zeros(n, dtype='int')
    K = np.zeros(n, dtype='int')

    i = 0
    for l in lines:
        if re.search(r'^\s+$', l) or re.search(r'\s*#', l):
            continue
        if re.match(r'\d+\s+-?\d+', l):
            a = re.split(r'\s+', l)
            M[i] = int(a[0])
            K[i] = int(a[1])
            i += 1

    return M, K

class CustomDataset(Dataset):
    def __init__(self, data, labels, window_size, num_classes=3):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.uint8)
        self.window_size = window_size
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data) // self.window_size

    def __getitem__(self, index):
        x = self.data[index*self.window_size:index*self.window_size+self.window_size][:]
        
        if (index*self.window_size >= (len(self.data) - self.window_size)):
            y = self.labels[index*self.window_size-1]
        else:
            y = self.labels[index*self.window_size]

        y = y.long()
        y = y - 1  # shift the labels to start at 0
        y_encoded = F.one_hot(y, num_classes=self.num_classes)
        x = x.view(-1, 2)

        return x, y_encoded

class CustomDatasetFC(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y

def create_segments(array, segment_length, overlap):
    segments = []
    stride = segment_length - overlap
    for i in range(0, len(array) - segment_length + 1, stride):
        segments.append(array[i:i+segment_length])
    return segments

def my_bpfilter(x, w0, w1, N=4, bf=True):
    from scipy import signal
    b, a = signal.butter(N, [w0, w1], 'bandpass')
    if bf:
        y = signal.filtfilt(b, a, x)
    else:
        y = signal.lfilter(b, a, x)
    return y

def get_sr(ppath, name):
    fid = open(os.path.join(ppath, name, 'info.txt'), newline=None)
    lines = fid.readlines()
    fid.close()
    values = []
    for l in lines:
        a = re.search("^" + 'SR' + ":" + r"\s+(.*)", l)
        if a:
            values.append(a.group(1))
    return float(values[0])

def minmax_scale(x, axis=None):
    min_val = np.min(x, axis=axis, keepdims=True)
    max_val = np.max(x, axis=axis, keepdims=True)
    result = (x-min_val)/(max_val-min_val)
    return result

def zscore_scale(x, axis=None):
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std = np.where(std == 0, 1, std)
    result = (x-mean)/std
    return result

def safe_load_model(path, device):
    from models import CPU_Unpickler
    import torch
    try:
        # Attempt modern PyTorch loading for recently saved models
        return torch.load(path, map_location=device, weights_only=False)
    except Exception:
        # Fallback to custom legacy unpickler for the older original .pkl files
        return CPU_Unpickler(open(path, 'rb'), device).load()
