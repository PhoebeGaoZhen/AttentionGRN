import numpy as np
import torch
import torch.nn as nn
import csv
import math
from sklearn.metrics import roc_auc_score,average_precision_score
from torch.utils.data import (DataLoader)
import warnings
warnings.filterwarnings('ignore')
torch.set_default_tensor_type(torch.DoubleTensor)

def load_index(file):
    with open(file, 'r') as f:
        csv_r = list(csv.reader(f, delimiter='\n'))
    return np.array(csv_r).flatten().astype(int)

def numpy2loader(X, y, batch_size):
    X_set = torch.from_numpy(X)
    X_loader = DataLoader(X_set, batch_size=batch_size)
    y_set = torch.from_numpy(y)
    y_loader = DataLoader(y_set, batch_size=batch_size)

    return X_loader, y_loader

def loaderToList(data_loader):
    length = len(data_loader)
    data = []
    for i in data_loader:
        data.append(i)
    return data

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # a = self.pe[:x.size(0), :]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class STGRNS(nn.Module):
    def __init__(self, input_dim, nhead=2, d_model=80, num_classes=2, dropout=0.1):
        super().__init__()
        self.prenet = nn.Linear(input_dim, d_model)
        self.positionalEncoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=256, nhead=2, dropout=dropout
        )

        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, window_size):
        out = window_size.permute(1, 0, 2)
        out = self.positionalEncoding(out)
        out = self.encoder_layer(out)
        out = out.transpose(0, 1)
        stats = out.mean(dim=1)
        out = self.pred_layer(stats)
        return out

def load_data_TF2(data_path):
    xxdata_list = []
    yydata = []
    # count_set = [0]
    # count_setx = 0

    xdata = np.load(data_path + 'Nxdata_tf.npy')
    ydata = np.load(data_path + 'ydata_tf.npy')
    for k in range(int(len(ydata) / 2)):
        xxdata_list.append(xdata[2 * k, :,
                           :])
        xxdata_list.append(xdata[2 * k + 1, :,
                           :])
        yydata.append(1)
        yydata.append(0)

    yydata_array = np.array(yydata)
    yydata_x = yydata_array.astype('int')

    return ((np.array(xxdata_list), yydata_x))


def Evaluation(y_true, y_pred,flag=False):


    AUC = roc_auc_score(y_true=y_true, y_score=y_pred)


    AUPR = average_precision_score(y_true=y_true,y_score=y_pred)
    AUPR_norm = AUPR/np.mean(y_true)


    return AUC, AUPR, AUPR_norm