import os
import pandas as pd
import numpy as np
import utils_data

Rank_num = 500
dataset_names = ['hESC', 'mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L', 'hHEP']

for dataset_name in dataset_names:
    network_types = []
    if dataset_name == 'hESC':
        network_types = ['hESC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']
    elif dataset_name == 'hHEP':
        network_types = ['Non-specific-ChIP-seq-network', 'STRING-network','HepG2-ChIP-seq-network']
    elif dataset_name == 'mDC':
        network_types = ['mDC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']
    elif dataset_name == 'mESC':
        network_types = ['mESC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network',
                         'mESC-lofgof-network']
    elif dataset_name == 'mHSC-E' or dataset_name == 'mHSC-GM' or dataset_name == 'mHSC-L':
        network_types = ['mHSC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']
    else:
        print("network type error")

    print(dataset_name + ': ')
    print(network_types)
    for network_type in network_types:
    # for Rank_num in Ranknums:
        print('\n\n****************************'+ dataset_name + '————' + network_type + '————' + str(Rank_num) + '****************************')
        # 路径
        data_path = 'DATA2\\' + dataset_name + '\\' + network_type + '\\Top' + str(Rank_num) + '\\'
        exp_file = data_path + 'Final_expression.csv'  # 细胞*基因名称
        tf_file = data_path + 'Final_TF_index.csv'
        target_file = data_path + 'Final_gene_list.csv'
        label_file = data_path + 'Final_GRNorTRN_pos_index.csv'
        label_name_file = data_path + 'Final_GRNorTRN_pos.csv'


        gene_pairs_dir = data_path + 'gene_pairs.txt'
        gene_pairs_num_dir = data_path + "gene_pairs_num.txt"

        # 读取靶基因文件中的靶基因index
        gene_set = pd.read_csv(target_file, index_col=0)['index'].values

        # 读取TF文件中的index，这里是全部TF，可以是基因列表与人类全部TF的交集，也可以是这个交集再加上已知关联对中的TF的并集，在数据预处理部分已经得到
        # tf_set = pd.read_csv(tf_file, index_col=0)['index'].values

        # 读取已知关联文件 tf index，靶基因 index
        label = pd.read_csv(label_file, index_col=0)
        # label_name = pd.read_csv(label_name_file, index_col=0)

        # 已知关联文件中出现的所有tf
        tf = label['TF'].values
        # 选出已知关联文件中的tf 列表，不重复
        tf_list = np.unique(tf)

        # 构建数据集的正样本。找到已知关联中所有的正样本，并存储到字典中，TF为键，靶基因为值
        pos_dict = {}
        for i in tf_list:
            pos_dict[i] = []
        for i, j in label.values:
            pos_dict[i].append(j)

        pos_list = []  # 2478个基因对
        for tf in pos_dict.keys():
            all_target = pos_dict[tf]
            for target in all_target:
                pos_list.append([tf, target])
        # print(len(pos_list))

        # 为每个正样本找到一个负样本, 并组合在一起
        neg_dict = {}
        neg_list = []
        pos_neg_balanced_id = []  # tf id-- 基因id--标签

        for pair in pos_list:
            # print(pair)
            signal_gene_set = gene_set.tolist()

            pos_tf = pair[0]
            pos_target = pair[1]
            neg_dict[pos_tf] = []
            neg_target = np.random.choice(gene_set)

            while neg_target == pos_tf or neg_target in pos_dict[pos_tf] or neg_target in neg_dict[pos_tf]:
                neg_target = np.random.choice(gene_set)
                if neg_target in signal_gene_set:
                    signal_gene_set.remove(neg_target)
                if len(signal_gene_set)==0:
                    break
            # print(pos_tf, pos_target, '1')
            # print(pos_tf, neg_target, '0')

            neg_list.append([pos_tf, neg_target])
            neg_dict[pos_tf].append(neg_target)
            # print('pos: ' + str(pos_tf) + '--' + str(pos_target))
            # print('neg: ' + str(pos_tf) + '--' + str(neg_target))
            # print('\n')
            # 正样本
            pos_neg_balanced_id.append([pos_tf, pos_target, 1])
            # hard负样本
            pos_neg_balanced_id.append([pos_tf, neg_target, 0])



        gene_name_set = pd.read_csv(target_file, index_col=0)['gene'].values
        pos_neg_balanced_name = []
        for item in pos_neg_balanced_id:
            # print(item)
            tf_id = item[0]
            target_id = item[1]
            label = item[2]

            tf_name = gene_name_set[tf_id]
            target_name = gene_name_set[target_id]
            pos_neg_balanced_name.append([tf_name, target_name, label])
            # print(tf_name,tf_id)

        pos_neg_balanced_name = pd.DataFrame(pos_neg_balanced_name)
        pos_neg_balanced_name.to_csv(data_path + 'pos_neg_balanced_name.csv', sep=',', header=0,index=0)

        # return tf_list, pos_neg_balanced_id
        # 保存
        tf_list_save = pd.DataFrame(tf_list)
        tf_list_save.to_csv(data_path + "tf_list.csv", sep=',', header=0,index=0)

        pos_neg_balanced_id = pd.DataFrame(pos_neg_balanced_id)
        pos_neg_balanced_id.to_csv(data_path + 'pos_neg_balanced_id.csv', sep=',', header=0,index=0)