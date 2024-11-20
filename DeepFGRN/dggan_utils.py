import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="0,1"
from keras import backend as K
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# import re
# import numpy as np
import pandas as pd
# from keras import models,layers,regularizers
# from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer,MinMaxScaler
# from sklearn.metrics import confusion_matrix,roc_auc_score,matthews_corrcoef,roc_curve,auc
# from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score,precision_recall_curve
# from keras import backend as K
# from math import log2
# import seaborn as sns
# import matplotlib.pyplot as plt


def minmaxstandard(rawdata):
    # new_data1 = np.zeros((rawdata.shape[0],rawdata.shape[1]))
    # for i in range(rawdata.shape[0]):
    #     for j in range(rawdata.shape[1]):
    #         new_data1[i][j] = log2(rawdata[i][j])
    Standard_data = MinMaxScaler().fit_transform(rawdata)
    return Standard_data

def read_graph_DREAM4(path):

    nodes = set()
    nodes_s = set()
    egs = []
    graph = [{}, {}]

    egs_nodes = pd.read_csv(path, sep='\t', header=None)

    for i in range(egs_nodes.shape[0]):

        source_node = egs_nodes[0][i]
        target_node = egs_nodes[1][i]
        source_node = [i for i in source_node if i.isnumeric()]
        source_node = ''.join(source_node)
        target_node = [i for i in target_node if i.isnumeric()]
        target_node = ''.join(target_node)

        source_node = int(source_node)-1
        target_node = int(target_node)-1

        nodes.add(source_node)
        nodes.add(target_node)
        nodes_s.add(source_node)
        egs.append([source_node, target_node])

        if source_node not in graph[0]:
            graph[0][source_node] = []
        if target_node not in graph[1]:
            graph[1][target_node] = []

        graph[0][source_node].append(target_node)
        graph[1][target_node].append(source_node)

    n_node = len(nodes)
    return graph, n_node, list(nodes), list(nodes_s), egs

def read_graph_Ecoli(pathnetwork,pathnode):

    nodes = set()
    nodes_s = set()
    egs = []
    graph = [{}, {}]

    egs_nodes = pd.read_csv(pathnetwork, sep='\t', header=None)

    for i in range(egs_nodes.shape[0]):

        source_node = egs_nodes[0][i]
        target_node = egs_nodes[1][i]

        nodes_s.add(source_node)
        egs.append([source_node, target_node])

        if source_node not in graph[0]:
            graph[0][source_node] = []
        if target_node not in graph[1]:
            graph[1][target_node] = []

        graph[0][source_node].append(target_node)
        graph[1][target_node].append(source_node)

    genename = pd.read_csv(pathnode, sep='\t', engine='python')
    nodes = genename.ids
    n_node = len(nodes)
    return graph, n_node, list(nodes), list(nodes_s), egs


def spectral_norm(w, r=5):
    w_shape = K.int_shape(w)
    in_dim = np.prod(w_shape[:-1]).astype(int)
    out_dim = w_shape[-1]
    w = K.reshape(w, (in_dim, out_dim))
    u = K.ones((1, in_dim))
    for i in range(r):
        v = K.l2_normalize(K.dot(u, w))
        u = K.l2_normalize(K.dot(v, K.transpose(w)))
    return K.sum(K.dot(K.dot(u, w), K.transpose(v)))

def spectral_normalization(w):
    return w / spectral_norm(w)