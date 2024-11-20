
import numpy as np
import pandas as pd
import csv
from numpy import *
import os
import utils_data as utils

# Ranknums = [500, 1000]
Ranknums = [1000]

data_source = 'DATA1_AttentionGRN\\'

# dataset_names = [ 'hESC','hHEP','mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']
dataset_names = [ 'hHEP']

for dataset_name in dataset_names:
    network_types = []
    if dataset_name == 'hESC':
        network_types = ['hESC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']

    elif dataset_name == 'hHEP' :
        network_types = ['HepG2-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']

    elif dataset_name == 'mDC':
        network_types = ['mDC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']

    elif dataset_name == 'mESC':
        network_types = ['mESC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network',
                         'mESC-lofgof-network']

    elif dataset_name == 'mHSC-E':
        network_types = ['mHSC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']

    elif dataset_name == 'mHSC-GM':
        network_types = ['mHSC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']

    elif dataset_name == 'mHSC-L':
        network_types = ['mHSC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']

    else:
        print("network type error")
    final_rank_nums = []
    for network_type in network_types:

        for Rank_num in Ranknums:
            print('\n\n****************************'+ dataset_name + '————' + network_type + '————' + str(Rank_num) + '****************************')
            # raw data path
            if dataset_name == 'hESC' or dataset_name == 'hHEP':
                gold_network_path = 'rawData\\Networks\\human\\' + network_type + '.csv'
                pathHumanTF = 'rawData\\Networks\\human-tfs.csv'
            else:
                gold_network_path = 'rawData\\Networks\\mouse\\' + network_type + '.csv'
                pathHumanTF = 'rawData\\Networks\\mouse-tfs.csv'

            gene_expression_path = 'rawData\\scRNA-Seq\\' + dataset_name + '\\ExpressionData.csv'
            gene_order_path = 'rawData\\scRNA-Seq\\' + dataset_name + '\\GeneOrdering.csv'

            # save path
            result_path = data_source + dataset_name + '\\' + gold_network_path[23:-4] + '\\Top' + str(Rank_num) + '\\'
            if not os.path.isdir(result_path):
                os.makedirs(result_path)

            utils.preprocess_DATA_AttentionGRN(result_path, gold_network_path, gene_expression_path, dataset_name, gene_order_path, Rank_num, pathHumanTF)

utils.dataset_Info(Ranknums, data_source)








