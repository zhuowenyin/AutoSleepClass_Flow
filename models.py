import torch
import torch.nn as nn
import pickle
import io

class biLSTM(nn.Module):
    def __init__(self, input_size, window_len, hidden_size=128, num_layers=2, dropout=0.2, device='cpu', num_classes=3):
        super(biLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            dropout=dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_size*2),
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, num_channels = x.size()

        # fallback if device is missing dynamically
        dev = getattr(self, 'device', x.device) 
        h0 = torch.randn(2 * self.num_layers, batch_size,
                         self.hidden_size).double().to(dev)
        c0 = torch.randn(2 * self.num_layers, batch_size,
                         self.hidden_size).double().to(dev)

        # Transpose the tensor to match the shape (batch, seq_len, num_channels)
        x = x.permute(0, 1, 2)

        self.lstm.flatten_parameters()
        outl, _ = self.lstm(x, (h0, c0))
        out = outl[:, -1, :]

        out = self.fc(out)
        return out


class FCNet(nn.Module):
    def __init__(self, segment_length=51, dropout=0.2, num_classes=3):
        super(FCNet, self).__init__()
        self.segment_length = segment_length
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(segment_length * num_classes, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(32),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        seg_len = getattr(self, 'segment_length', 51)
        out = self.fc(x) + x[:, int((seg_len-1)/2+0.1)]
        return out


class CPU_Unpickler(pickle.Unpickler):
    def __init__(self, file, device):
        super().__init__(file)
        self.device = device

    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=self.device, weights_only=False)
        else:
            return super().find_class(module, name)
