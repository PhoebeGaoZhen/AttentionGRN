import numpy as np
import os
import csv
import pandas as pd
import time
import scipy.sparse as sp
from sklearn.preprocessing import MaxAbsScaler
from scipy.sparse import csr_matrix
from utils import load_data5
import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizer_v2.gradient_descent import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from numpy import *
import matplotlib
matplotlib.use('Agg')




def normalize_features(features):
    scaler = MaxAbsScaler()
    normalized_features = scaler.fit_transform(features)
    return normalized_features


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

def get_separation_index(file_name):
    import numpy as np
    index_list = []
    with open(file_name, 'r') as s:
        for line in s:
            index_list.append(int(line.split(',')[0]))
    return np.array(index_list)


def process_edges(edges_with_labels, node_map, path, tf_list, data_name):
    gene_pair_label_array = array(edges_with_labels)
    gene_pair_index = tf_list
    for i in range(len(gene_pair_index)):  #### many sperations
        # print(i)
        start_index = gene_pair_index[i]
        end_index = gene_pair_index[i + 1]
        start_index_value = start_index[0]
        end_index_value = end_index[0]
        x = []
        y = []
        z = []
        for idx in range(start_index_value, end_index_value):
            gene_pair = gene_pair_label_array[idx]
            x_gene_name, y_gene_name, label = gene_pair[0], gene_pair[1], gene_pair[2]
            y.append(label)

            z.append(str(x_gene_name) + '\t' + str(y_gene_name))
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
        np.save(save_dir + '/Nxdata_tf7_' + data_name + str(i)+'.npy', xx)
        np.save(save_dir + '/ydata_tf7_' + data_name + str(i)+ '.npy', np.array(y))
        np.save(save_dir + '/zdata_tf7_' + data_name +  str(i)+'.npy', np.array(z))

def process_edges_TRN(edges_with_labels, node_map, path, tf_list, data_name):
    gene_pair_label_array = array(edges_with_labels)
    gene_pair_index = tf_list

    for i in range(len(tf_list)):
        tf_id = tf_list[i][0]

        tf_edges = []
        x = []
        y = []
        z = []
        for j in range(gene_pair_label_array.shape[0]):
            if gene_pair_label_array[j][0] == tf_id:
                tf_edge = gene_pair_label_array[j]
                tf_edges.append(tf_edge)
                x_gene_name, y_gene_name, label = tf_edge[0], tf_edge[1], tf_edge[2]
                y.append(label)
                z.append(str(x_gene_name) + '\t' + str(y_gene_name))

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
        # print(tf_edges)
        if len(x) > 0:
            xx = np.array(x)[:, :, :, np.newaxis]
        else:
            xx = np.array(x)
        save_dir = os.path.join(path + '/NEPDF_data')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        np.save(save_dir + '/Nxdata_tf7_' + data_name + str(i)+'.npy', xx)
        np.save(save_dir + '/ydata_tf7_' + data_name + str(i)+ '.npy', np.array(y))
        np.save(save_dir + '/zdata_tf7_' + data_name +  str(i)+'.npy', np.array(z))



def preprocess_dataset(dataset_path,tf_list):
    file_name = dataset_path + '/Final_expression.csv'
    file_path = dataset_path + '/pos_neg_balanced_id.csv'

    df = pd.read_csv(file_name, header=0)
    df = df.iloc[:, 1:]

    exp_file = df.T

    geneName = exp_file.index
    loader = load_data5(exp_file)
    exp_file = loader.exp_data()
    geneNum = exp_file.shape[0]

    df = pd.read_csv(file_path, header=None, names=['TF', 'target','label'],skiprows=1)
    data_edges_with_labels = df.to_numpy()

    data_name = dataset_path

    file_name = data_name + "_feature.csv"
    # print(data_name)

    if not os.path.exists(file_name):

        df = pd.DataFrame(exp_file)

        df.to_csv(file_name, index=False)
        print("save file successfully！")
    else:
        print("file already exists.")

    node_map = get_gene_expression(file_name)

    # transform edges into graph
    # process_edges(data_edges_with_labels, node_map, data_name, tf_list,'data_process_TRN')
    process_edges_TRN(data_edges_with_labels, node_map, data_name, tf_list,'data_process_TRN')

    print("process successful!")









