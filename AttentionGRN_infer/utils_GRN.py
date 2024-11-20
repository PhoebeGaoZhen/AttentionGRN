import torch
import dgl
import copy
from torch_geometric.data import Data
import  pandas as pd
import numpy as np
from dgl.data.utils import save_graphs, load_graphs
from torch_geometric.utils import to_networkx
import networkx as nx
from torch_geometric.nn import SAGEConv
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
import dgl.function as fn

import utils_gz
from torch.utils.data import (DataLoader)


def get_centrality(train_g, N_nx, katz_alpha):
    idc = nx.in_degree_centrality(train_g)
    odc = nx.out_degree_centrality(train_g)
    cc = nx.closeness_centrality(train_g)
    bc = nx.betweenness_centrality(train_g)
    # ec = nx.eigenvector_centrality(train_g)
    katz_ec = nx.katz_centrality(train_g, alpha=katz_alpha, max_iter=100000)

    idc = np.array([item for item in idc.values()])
    odc = np.array([item for item in odc.values()])
    cc = np.array([item for item in cc.values()])
    bc = np.array([item for item in bc.values()])
    # ec = np.array([item for item in ec.values()])
    katz_ec = np.array([item for item in katz_ec.values()])

    IA = np.zeros((N_nx, 5))
    # IA = np.zeros((N_nx, 4))

    IA[:, 0] = idc
    IA[:, 1] = odc
    IA[:, 2] = cc
    IA[:, 3] = bc
    IA[:, 4] = katz_ec

    return IA


def transform_savebinI(train_og_O, data_all, train_og_O_T, katz_alpha, k_hop):

    train_g = to_networkx(train_og_O, to_undirected=False)
    train_g_T = to_networkx(train_og_O_T, to_undirected=False)
    data_all = to_networkx(data_all, to_undirected=False)

    N_nx = train_g.number_of_nodes()


    IA = get_centrality(train_g, N_nx, katz_alpha)

    Kedge = utils_gz.buildPE_Kindices(data_all, k_hop) #

    Kindices = torch.LongTensor(Kedge)
    I = utils_gz.getI_directed(train_g, train_g_T, IA, k_hop)

    I = torch.from_numpy(I)

    g = utils_gz.g_dgl_I(train_og_O, I, Kindices, N_nx)

    return g



def genepair_to_dgl_I(labels, gene_pair_tf_trainList, gene_pair_target_trainList, train_g_pos):
    k_edge_idx1 = []
    k_edge_idx2 = []

    all_tf = []
    all_target = []
    # count = 0
    for i in range(len(labels)):
        tf = gene_pair_tf_trainList[i]
        target = gene_pair_target_trainList[i]
        # count += 1
        all_tf.append(tf)
        all_target.append(target)
        label = labels[i]
        if label == 1:
            k_edge_idx1.append(tf)
            k_edge_idx2.append(target)

    g = dgl.graph((k_edge_idx1, k_edge_idx2), num_nodes=train_g_pos.num_nodes())
    g.ndata['x'] = train_g_pos.ndata['x']
    g.ndata['I'] = train_g_pos.ndata['I']

    return all_tf, all_target, g, k_edge_idx1



def numpy2loader(X, y, genepair_tf, genepair_target, batch_size):
    X_set = torch.from_numpy(X)
    X_loader = DataLoader(X_set, batch_size=batch_size)
    y_set = torch.from_numpy(y)
    y_loader = DataLoader(y_set, batch_size=batch_size)
    genepair_tf_set = torch.from_numpy(genepair_tf)
    genepair_tf_loader = DataLoader(genepair_tf_set, batch_size=batch_size)
    genepair_target_set = torch.from_numpy(genepair_target)
    genepair_target_loader = DataLoader(genepair_target_set, batch_size=batch_size)

    return X_loader, y_loader, genepair_tf_loader, genepair_target_loader

def loaderToList(data_loader):
    length = len(data_loader)
    data = []
    for i in data_loader:
        data.append(i)
    return data
