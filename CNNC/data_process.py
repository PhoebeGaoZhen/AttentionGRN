import csv
import pandas as pd
import time, random
import scipy.sparse as sp
from sklearn.preprocessing import MaxAbsScaler
from scipy.sparse import csr_matrix
from utils import load_data5
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizer_v2.gradient_descent import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')




def normalize_features(features):
    scaler = MaxAbsScaler()
    normalized_features = scaler.fit_transform(features)
    return normalized_features

def load_npz_dataset(file_name):
    """Load a graph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    graph : dict
        Dictionary that contains:
            * 'A' : The adjacency matrix in sparse matrix format
            * 'X' : The attribute matrix in sparse matrix format
            * 'z' : The ground truth class labels
            * Further dictionaries mapping node, class and attribute IDs

    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        edge_index = loader['adj_indices'].copy()
        A = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                           loader['adj_indptr']), shape=loader['adj_shape'])

        X = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                           loader['attr_indptr']), shape=loader['attr_shape'])
        # scaler = MaxAbsScaler()
        # X = scaler.fit_transform(X)

        graph = {
            'A': A,
            'X': X,
        }
        return graph

def load_data(pathway="./data/Benchmark Dataset/", dataset='Specific Dataset hHEP TF1000+'):
    os.makedirs(pathway, exist_ok=True)
    dataset_path = os.path.join(pathway, '{}.npz'.format(dataset))
    g = load_npz_dataset(dataset_path)
    adj, feature = g['A'], g['X']
    return adj, feature, dataset

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges_general_link_prediction(adj, test_percent=20., val_percent=20.):

    adj = adj - sp.dia_matrix((adj.diagonal()[None, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    edges_positive, _, _ = sparse_to_tuple(adj)

    num_test = int(np.floor(edges_positive.shape[0] / (100. / test_percent)))
    num_val = int(np.floor(edges_positive.shape[0] / (100. / val_percent)))

    edges_positive_idx = np.arange(edges_positive.shape[0])
    np.random.shuffle(edges_positive_idx)
    val_edge_idx = edges_positive_idx[:num_val]
    val_edges = edges_positive[val_edge_idx]
    test_edge_idx = edges_positive_idx[num_val:(num_val + num_test)]
    test_edges = edges_positive[test_edge_idx]
    train_edges = np.delete(edges_positive, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    positive_idx, _, _ = sparse_to_tuple(adj)
    positive_idx = positive_idx[:, 0] * adj.shape[0] + positive_idx[:, 1]

    test_edges_false = np.empty((0, 2), dtype='int64')
    idx_test_edges_false = np.empty((0,), dtype='int64')
    while len(test_edges_false) < len(test_edges):
        idx = np.random.choice(adj.shape[0] ** 2, 2 * (num_test - len(test_edges_false)), replace=True)
        idx = idx[~np.in1d(idx, positive_idx, assume_unique=True)]
        idx = idx[~np.in1d(idx, idx_test_edges_false, assume_unique=True)]
        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx, colidx)).transpose()
        coords = np.unique(coords, axis=0)
        np.random.shuffle(coords)
        coords = coords[coords[:, 0] != coords[:, 1]]
        coords = coords[:min(num_test, len(idx))]
        test_edges_false = np.append(test_edges_false, coords, axis=0)
        idx_test_edges_false = np.append(idx_test_edges_false, idx[:min(num_test, len(idx))])

    val_edges_false = np.empty((0, 2), dtype='int64')
    idx_val_edges_false = np.empty((0,), dtype='int64')
    while len(val_edges_false) < len(val_edges):
        idx = np.random.choice(adj.shape[0] ** 2, 2 * (num_val - len(val_edges_false)), replace=True)
        idx = idx[~np.in1d(idx, positive_idx, assume_unique=True)]
        idx = idx[~np.in1d(idx, idx_test_edges_false, assume_unique=True)]
        idx = idx[~np.in1d(idx, idx_val_edges_false, assume_unique=True)]
        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx, colidx)).transpose()
        coords = np.unique(coords, axis=0)
        np.random.shuffle(coords)
        coords = coords[coords[:, 0] != coords[:, 1]]
        coords = coords[:min(num_val, len(idx))]
        val_edges_false = np.append(val_edges_false, coords, axis=0)
        idx_val_edges_false = np.append(idx_val_edges_false, idx[:min(num_val, len(idx))])

    train_edges_false = np.empty((0, 2), dtype='int64')
    idx_train_edges_false = np.empty((0,), dtype='int64')
    while len(train_edges_false) < len(train_edges):
        idx = np.random.choice(adj.shape[0] ** 2, 2 * (len(train_edges) - len(train_edges_false)), replace=True)
        idx = idx[~np.in1d(idx, positive_idx, assume_unique=True)]
        idx = idx[~np.in1d(idx, idx_test_edges_false, assume_unique=True)]
        idx = idx[~np.in1d(idx, idx_val_edges_false, assume_unique=True)]
        idx = idx[~np.in1d(idx, idx_train_edges_false, assume_unique=True)]
        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx, colidx)).transpose()
        coords = np.unique(coords, axis=0)
        np.random.shuffle(coords)
        coords = coords[coords[:, 0] != coords[:, 1]]
        coords = coords[:min(len(train_edges), len(idx))]
        train_edges_false = np.append(train_edges_false, coords, axis=0)
        idx_train_edges_false = np.append(idx_train_edges_false, idx[:min(len(train_edges), len(idx))])

    train_edges_linear = train_edges[:, 0] * adj.shape[0] + train_edges[:, 1]
    test_edges_linear = test_edges[:, 0] * adj.shape[0] + test_edges[:, 1]
    assert not np.any(np.in1d(idx_test_edges_false, positive_idx))
    assert not np.any(np.in1d(idx_val_edges_false, positive_idx))
    assert not np.any(np.in1d(val_edges[:, 0] * adj.shape[0] + val_edges[:, 1], train_edges_linear))
    assert not np.any(np.in1d(test_edges_linear, train_edges_linear))
    assert not np.any(np.in1d(val_edges[:, 0] * adj.shape[0] + val_edges[:, 1], test_edges_linear))

    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false


def label10(edges, edges_false):
    mixed_edges = []
    mixed_labels = []

    for pos_edge, neg_edge in zip(edges, edges_false):
        mixed_edges.append(pos_edge)
        mixed_edges.append(neg_edge)
        mixed_labels.append(1)
        mixed_labels.append(0)

    mixed_edges_data = np.array(mixed_edges)
    mixed_labels_data = np.array(mixed_labels)
    return mixed_edges_data,mixed_labels_data



def get_gene_expression(file_expression):
    node_map = []
    s = pd.read_csv(file_expression, delimiter=',', header=None)
    for row in s.values:
        gene_expression = [float(x) for x in row[:]]
        node_map.append(gene_expression)
    return node_map


def process_edges(edges_with_labels, node_map, path, data_name):
    x = []
    y = []
    z = []

    for edge, label in edges_with_labels:
        x_gene_name, y_gene_name = edge
        y.append(label)
        z.append(str(x_gene_name) + '\t' + str(y_gene_name))
        # x_tf = [np.log10(value + 10 ** -2) for value in node_map[x_gene_name]]
        # x_gene = [np.log10(value + 10 ** -2) for value in node_map[y_gene_name]]
        # # x_tf = node_map[x_gene_name]
        # n = len(x_tf)
        # # x_gene =node_map[y_gene_name]
        # H_T, xedges, yedges = np.histogram2d(x_tf, x_gene, bins=32)
        epsilon = 1e-2

        x_tf = [np.log10(value + epsilon) if value > 0 else np.nan for value in node_map[x_gene_name]]
        x_gene = [np.log10(value + epsilon) if value > 0 else np.nan for value in node_map[y_gene_name]]
        n = len(x_tf)

        x_tf = np.nan_to_num(x_tf, nan=0)
        x_gene = np.nan_to_num(x_gene, nan=0)

        H_T, xedges, yedges = np.histogram2d(x_tf, x_gene, bins=32)
        H = H_T.T
        HT = (np.log10(H / n + 10 ** -4) + 4) / 4
        x.append(HT)

    if len(x) > 0:
        xx = np.array(x)[:, :, :, np.newaxis]
    else:
        xx = np.array(x)
    save_dir = os.path.join(path + '/NEPDF_data')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    np.save(save_dir + '/Nxdata_tf7_' + data_name + '.npy', xx)
    np.save(save_dir + '/ydata_tf7_' + data_name + '.npy', np.array(y))
    np.save(save_dir + '/zdata_tf7_' + data_name + '.npy', np.array(z))



def preprocess_dataset(dataset_path):
    file_name = dataset_path + '/Final_expression.csv'
    file_path = dataset_path + '/Final_GRNorTRN_pos_index.csv'

    df = pd.read_csv(file_name, header=0)

    df = df.iloc[:, 1:]

    exp_file = df.T

    geneName = exp_file.index
    loader = load_data5(exp_file)
    exp_file = loader.exp_data()
    geneNum = exp_file.shape[0]

    df = pd.read_csv(file_path, header=None, names=['index', 'TF', 'target'],skiprows=1)

    df = df.drop(columns=df.columns[0])

    rows = df['TF'].values.astype(int)
    cols = df['target'].values.astype(int)
    data = pd.Series([1] * len(df))

    adj_init = csr_matrix((data, (rows, cols)), shape=(geneNum, geneNum))
    data_name = dataset_path

    file_name = data_name + "_feature.csv"
    print(data_name)

    if not os.path.exists(file_name):
        df = pd.DataFrame(exp_file)

        df.to_csv(file_name, index=False)
        print("file save successfully")
    else:
        print("file already exists.")

    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
        mask_test_edges_general_link_prediction(adj_init)

    train_edges_data, train_edge_labels = label10(edges=train_edges, edges_false=train_edges_false)
    val_edges_data, val_edge_labels = label10(edges=val_edges, edges_false=val_edges_false)
    test_edges_data, test_edge_labels = label10(edges=test_edges, edges_false=test_edges_false)

    train_edges_with_labels = list(zip(train_edges_data, train_edge_labels))
    val_edges_with_labels = list(zip(val_edges_data, val_edge_labels))
    test_edges_with_labels = list(zip(test_edges_data, test_edge_labels))

    node_map = get_gene_expression(file_name)

    process_edges(train_edges_with_labels, node_map, data_name, 'train_edges_data')
    process_edges(val_edges_with_labels, node_map, data_name, 'validation_edges')
    process_edges(test_edges_with_labels, node_map, data_name, 'test_edges_data')
    print("Successful!")



def preprocess_dataset_GRN(dataset_path, save_path):
    file_name = dataset_path + '/Final_expression.csv'
    file_path = dataset_path + '/Final_GRNorTRN_pos_index.csv'

    df = pd.read_csv(file_name, header=0)

    df = df.iloc[:, 1:]

    exp_file = df.T

    geneName = exp_file.index
    loader = load_data5(exp_file)
    exp_file = loader.exp_data()
    geneNum = exp_file.shape[0]

    df = pd.read_csv(file_path, header=None, names=['index', 'TF', 'target'],skiprows=1)

    df = df.drop(columns=df.columns[0])

    rows = df['TF'].values.astype(int)
    cols = df['target'].values.astype(int)
    data = pd.Series([1] * len(df))

    adj_init = csr_matrix((data, (rows, cols)), shape=(geneNum, geneNum))
    data_name = dataset_path

    file_name = data_name + "_feature.csv"
    print(data_name)

    if not os.path.exists(file_name):

        df = pd.DataFrame(exp_file)

        df.to_csv(file_name, index=False)
        print("save successfully！")
    else:
        print("file already exists.")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    train_set_file = save_path + 'Train_set.csv'
    test_set_file = save_path + 'Test_set.csv'
    val_set_file = save_path + 'Validation_set.csv'
    metric_file = save_path + 'metric.csv'

    tf_file = dataset_path + '/Final_TF_common_index.csv'
    target_file = dataset_path + '/Final_gene_list.csv'
    label_file = dataset_path + '/Final_GRNorTRN_pos_index.csv'

    GRN_type = 'tf-gene-posneg'  # tf-gene；  gene-gene; tf-gene-posneg
    dataset_type = 'unbalanced'  # balanced；  unbalanced

    train_val_test_set(label_file, target_file, tf_file, train_set_file, val_set_file, test_set_file,
                       GRN_type, dataset_type)

    train_data = pd.read_csv(train_set_file, index_col=0).values  # tf ID -- target ID -- label
    validation_data = pd.read_csv(val_set_file, index_col=0).values  # tf ID -- target ID -- label
    test_data = pd.read_csv(test_set_file, index_col=0).values  # tf ID -- target ID -- label
    train_edges_data, train_edge_labels = train_data[:,0:2], train_data[:,2]
    val_edges_data, val_edge_labels = validation_data[:,0:2], validation_data[:,2]
    test_edges_data, test_edge_labels = test_data[:,0:2], test_data[:,2]

    train_edges_with_labels = list(zip(train_edges_data, train_edge_labels))
    val_edges_with_labels = list(zip(val_edges_data, val_edge_labels))
    test_edges_with_labels = list(zip(test_edges_data, test_edge_labels))

    node_map = get_gene_expression(file_name)
    process_edges(train_edges_with_labels, node_map, data_name, 'train_edges_data')
    process_edges(val_edges_with_labels, node_map, data_name, 'validation_edges')
    process_edges(test_edges_with_labels, node_map, data_name, 'test_edges_data')
    print("Successful!")


def train_val_test_set(label_file,Gene_file,TF_file,train_set_file,val_set_file,test_set_file,GRN_type, dataset_type,p_val=0.5):
    print(GRN_type,dataset_type,p_val)

    gene_set = pd.read_csv(Gene_file, index_col=0)['index'].values  # 779

    tf_set = pd.read_csv(TF_file, index_col=0)['index'].values  # 84

    label = pd.read_csv(label_file, index_col=0)  #（2478,2）

    tf = label['TF'].values

    tf_list = np.unique(tf)   # 14

    pos_dict = {}
    for i in tf_list:
        pos_dict[i] = []
    for i, j in label.values:
        pos_dict[i].append(j)


    neg_dict = {}
    if GRN_type == 'gene-gene':
        gene_set_without_tf = list(set(gene_set)-set(tf_list))

        for i in gene_set:
            neg_dict[i] = []
        for i in gene_set_without_tf:
            neg_dict[i] = list(range(0,len(gene_set)))
        for i in tf_list:
            temp_a = list(range(0, len(gene_set)))
            temp_b = pos_dict[i]
            neg_dict[i] = list(set(temp_a)-set(temp_b))
    elif GRN_type=='tf-gene':

        for i in tf_set:
            neg_dict[i] = []

        for i in tf_set:
            if i in pos_dict.keys():
                pos_item = pos_dict[i]
                pos_item.append(i)

                neg_item = np.setdiff1d(gene_set, pos_item)
                neg_dict[i].extend(neg_item)
                pos_dict[i] = np.setdiff1d(pos_dict[i], i)

            else:
                neg_item = np.setdiff1d(gene_set, i)
                neg_dict[i].extend(neg_item)
    elif GRN_type=='tf-gene-posneg':

        for i in tf_list:
            neg_dict[i] = []

        for i in tf_list:
            if i in pos_dict.keys():
                pos_item = pos_dict[i]
                pos_item.append(i)

                neg_item = np.setdiff1d(gene_set, pos_item)
                neg_dict[i].extend(neg_item)
                pos_dict[i] = np.setdiff1d(pos_dict[i], i)

            else:
                print('negtive sample tf error')

    else:
        print("error")



    train_pos = {}
    val_pos = {}
    test_pos = {}

    for k in pos_dict.keys():
        if len(pos_dict[k]) <= 1:
            p = np.random.uniform(0,1)
            if p <= p_val:
                train_pos[k] = pos_dict[k]
            else:
                test_pos[k] = pos_dict[k]

        elif len(pos_dict[k]) == 2:

            train_pos[k] = [pos_dict[k][0]]
            test_pos[k] = [pos_dict[k][1]]

        else:

            np.random.shuffle(pos_dict[k])
            train_pos[k] = pos_dict[k][:len(pos_dict[k]) * 3 // 5]
            val_pos[k] = pos_dict[k][len(pos_dict[k]) * 3 // 5:len(pos_dict[k]) * 4 // 5]
            test_pos[k] = pos_dict[k][len(pos_dict[k]) * 4 // 5:]


    train_neg = {}
    val_neg = {}
    test_neg = {}
    if dataset_type == 'balanced':

        neg_dict_balanced = {}
        for i in range(label.shape[0]):
            key, value = get_random_kvpair(neg_dict)
            if key not in neg_dict_balanced:
                neg_dict_balanced[key] = [value]
            else:
                neg_dict_balanced[key].append(value)

        for k in neg_dict_balanced.keys():
            if len(neg_dict_balanced[k]) <= 1:
                p = np.random.uniform(0, 1)
                if p <= p_val:
                    train_neg[k] = neg_dict_balanced[k]
                else:
                    test_neg[k] = neg_dict_balanced[k]

            elif len(neg_dict_balanced[k]) == 2:

                train_neg[k] = [neg_dict_balanced[k][0]]
                test_neg[k] = [neg_dict_balanced[k][1]]

            else:

                np.random.shuffle(neg_dict_balanced[k])
                train_neg[k] = neg_dict_balanced[k][:len(neg_dict_balanced[k]) * 3 // 5]
                val_neg[k] = neg_dict_balanced[k][
                              len(neg_dict_balanced[k]) * 3 // 5:len(neg_dict_balanced[k]) * 4 // 5]
                test_neg[k] = neg_dict_balanced[k][len(neg_dict_balanced[k]) * 4 // 5:]

    elif dataset_type == 'unbalanced':

        for k in pos_dict.keys():
            if k in neg_dict:
                np.random.shuffle(neg_dict[k])
                train_neg[k] = neg_dict[k][:len(neg_dict[k]) * 3 // 5]
                val_neg[k] = neg_dict[k][
                              len(neg_dict[k]) * 3 // 5:len(neg_dict[k]) * 4 // 5]
                test_neg[k] = neg_dict[k][len(neg_dict[k]) * 4 // 5:]
            else:
                print('unbalanced: '+ str(k))
    else:
        print('error')

    final_num_label1 = 0
    for i in train_pos:
        final_num_label1 += len(train_pos[i])
    for i in val_pos:
        final_num_label1 += len(val_pos[i])
    for i in test_pos:
        final_num_label1 += len(test_pos[i])
    print('final label1 number')
    print(final_num_label1)
    final_num_label0 = 0
    for i in train_neg:
        final_num_label0 += len(train_neg[i])
    for i in val_neg:
        final_num_label0 += len(val_neg[i])
    for i in test_neg:
        final_num_label0 += len(test_neg[i])

    print('final label0 number')
    print(final_num_label0)


    save_datasets(train_pos, train_neg, train_set_file)
    save_datasets(val_pos, val_neg, val_set_file)
    save_datasets(test_pos, test_neg, test_set_file)

    print('traing set, validation set, and test set are saved successfully.')

def save_datasets(pos,neg,path_file):
    pos_set = transform_genepair(pos)
    # pos_set = []
    # for k in pos.keys():
    #     # print('key:', k)
    #     for j in pos[k]:
    #         # print('value:', j)
    #         pos_set.append([k, j])
    pos_label = [1 for _ in range(len(pos_set))]

    # neg_set = []
    # for k in neg.keys():
    #     for j in neg[k]:
    #         neg_set.append([k, j])
    neg_set = transform_genepair(neg)
    neg_label = [0 for _ in range(len(neg_set))]

    all_set = pos_set + neg_set
    all_set_label = pos_label + neg_label

    all_set_a = np.array(all_set)
    all_sample = pd.DataFrame()
    all_sample['TF'] = all_set_a[:, 0]
    all_sample['Target'] = all_set_a[:, 1]
    all_sample['Label'] = all_set_label
    all_sample.to_csv(path_file)


def get_random_kvpair(dictionary):
    neg_pair = {}
    key_value_list = list(dictionary.items())
    random_kvpair = random.choice(key_value_list)
    key = random_kvpair[0]
    values = random_kvpair[1]
    value = random.choice(values)
    neg_pair[key] = value
    return key, value

def transform_genepair(dict):
    dict_set = []
    for k in dict.keys():
        # print('key:', k)
        for j in dict[k]:
            # print('value:', j)
            dict_set.append([k, j])
    return dict_set


