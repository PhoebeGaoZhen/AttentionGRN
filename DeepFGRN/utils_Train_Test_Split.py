import pandas as pd
import numpy as np
import os,random
# from collections import Counter
# from sklearn.model_selection import train_test_split
# from Code.utils import Network_Statistic
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--ratio', type=float, default=0.67, help='the ratio of the training set')
# parser.add_argument('--Rank_num', type=int, default= 500, help='network scale')
parser.add_argument('--p_val', type=float, default=0.5, help='the position of the target with degree equaling to one')
# parser.add_argument('--data', type=str, default='hESC', help='data type')
# parser.add_argument('--net', type=str, default='Specific', help='network type')
args = parser.parse_args()

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

def save_datasets(pos,neg,path_file):
    pos_set = transform_genepair(pos)
    pos_label = [1 for _ in range(len(pos_set))]

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

def get_all_set(pos, neg):
    pos_set = transform_genepair(pos)
    pos_label = [1 for _ in range(len(pos_set))]

    neg_set = transform_genepair(neg)
    neg_label = [0 for _ in range(len(neg_set))]

    all_set = pos_set + neg_set
    all_set_label = pos_label + neg_label

    return all_set,all_set_label


def train_val_test_set(label_file,Gene_file,TF_file,train_set_file,val_set_file,test_set_file,GRN_type, dataset_type,p_val=0.5):
    gene_set = pd.read_csv(Gene_file, index_col=0)['index'].values
    tf_set = pd.read_csv(TF_file, index_col=0)['index'].values
    label = pd.read_csv(label_file, index_col=0)
    tf = label['TF'].values
    tf_list = np.unique(tf)


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
    num_label0 = 0
    for i in neg_dict:
        num_label0 += len(neg_dict[i])
    print(num_label0)

    num_label1 = 0
    for i in pos_dict:
        num_label1 += len(pos_dict[i])
    print(num_label1)

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
                print(k)
    else:
        print('error')

    save_datasets(train_pos, train_neg, train_set_file)
    save_datasets(val_pos, val_neg, val_set_file)
    save_datasets(test_pos, test_neg, test_set_file)

    print('traing set, validation set, and test set are saved successfully.')


def train_val_test_set_hard_3CV(label_file,Gene_file,TF_file):
    gene_set = pd.read_csv(Gene_file, index_col=0)['index'].values
    tf_set = pd.read_csv(TF_file, index_col=0)['index'].values
    label = pd.read_csv(label_file, index_col=0)
    tf = label['TF'].values
    tf_list = np.unique(tf)


    pos_dict = {}
    for i in tf_list:
        pos_dict[i] = []
    for i, j in label.values:
        pos_dict[i].append(j)

    pos_list = []
    for tf in pos_dict.keys():
        all_target = pos_dict[tf]
        for target in all_target:
            pos_list.append([tf,target])

    neg_dict = {}
    neg_list = []
    pos_neg_balanced = []
    for pair in pos_list:

        pos_tf = pair[0]
        pos_target = pair[1]
        neg_dict[pos_tf] = []
        neg_target = np.random.choice(gene_set)

        while neg_target == pos_tf or neg_target in pos_dict[pos_tf] or neg_target in neg_dict[pos_tf]:
            neg_target = np.random.choice(gene_set)

        neg_list.append([pos_tf,neg_target])
        neg_dict[pos_tf].append(neg_target)
        pos_neg_balanced.append([pos_tf,pos_target,1])
        pos_neg_balanced.append([pos_tf,neg_target,0])
    pos_neg_balanced = pd.DataFrame(pos_neg_balanced)

    print(len(pos_neg_balanced))

    return tf_list, pos_neg_balanced



def train_val_test_set_2(label_file,Gene_file,TF_file,train_set_file,val_set_file,test_set_file,GRN_type, dataset_type,p_val=0.5):
    print(GRN_type,dataset_type,p_val)
    gene_set = pd.read_csv(Gene_file, index_col=0)['index'].values
    tf_set = pd.read_csv(TF_file, index_col=0)['index'].values
    label = pd.read_csv(label_file, index_col=0)
    tf = label['TF'].values

    tf_list = np.unique(tf)

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
            # print('tf', i)
            if i in pos_dict.keys():

                pos_item = pos_dict[i]
                pos_item.append(i)
                neg_item = np.setdiff1d(gene_set, pos_item)
                neg_dict[i].extend(neg_item)
                pos_dict[i] = np.setdiff1d(pos_dict[i], i)

            else:
                neg_item = np.setdiff1d(gene_set, i)
                neg_dict[i].extend(neg_item)


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

    pos = []
    for p in train_pos:
        for s in range(len(train_pos[p])):
            pos.append(train_pos[p][s])

    for p in val_pos:
        for s in range(len(val_pos[p])):
            pos.append(val_pos[p][s])
    for p in test_pos:
        for s in range(len(test_pos[p])):
            pos.append(test_pos[p][s])
    # print(pos)
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
        pos_dict_keys = []
        for k in pos_dict.keys():
            if k in neg_dict:
                np.random.shuffle(neg_dict[k])
                train_neg[k] = neg_dict[k][:len(neg_dict[k]) * 3 // 5]
                val_neg[k] = neg_dict[k][
                              len(neg_dict[k]) * 3 // 5:len(neg_dict[k]) * 4 // 5]
                test_neg[k] = neg_dict[k][len(neg_dict[k]) * 4 // 5:]
        neg = []
        for p in train_neg:
            for s in range(len(train_neg[p])):
                neg.append(train_neg[p][s])

        for p in val_neg:
            for s in range(len(val_neg[p])):
                neg.append(val_neg[p][s])
        for p in test_neg:
            for s in range(len(test_neg[p])):
                neg.append(test_neg[p][s])
        # print(neg)

        supp_neg_dict = {}
        if len(neg)==0:
            for i in range(len(pos)):
                key, value = get_random_kvpair(neg_dict)
                if key not in supp_neg_dict:
                    supp_neg_dict[key] = [value]
                else:
                    supp_neg_dict[key].append(value)
        for ke in supp_neg_dict:
            train_neg[ke] = supp_neg_dict[ke][:len(supp_neg_dict[ke]) * 3 // 5]
            val_neg[ke] = supp_neg_dict[ke][
                         len(supp_neg_dict[ke]) * 3 // 5:len(supp_neg_dict[ke]) * 4 // 5]
            test_neg[ke] = supp_neg_dict[ke][len(supp_neg_dict[ke]) * 4 // 5:]

        neg = []
        for p in train_neg:
            for s in range(len(train_neg[p])):
                neg.append(train_neg[p][s])

        for p in val_neg:
            for s in range(len(val_neg[p])):
                neg.append(val_neg[p][s])
        for p in test_neg:
            for s in range(len(test_neg[p])):
                neg.append(test_neg[p][s])


    else:
        print('error')

    save_datasets(train_pos, train_neg, train_set_file)
    save_datasets(val_pos, val_neg, val_set_file)
    save_datasets(test_pos, test_neg, test_set_file)

    print('traing set, validation set, and test set are saved successfully.')



def train_val_test_set_hard(label_file,Gene_file,TF_file,train_set_file,val_set_file,test_set_file,GRN_type, dataset_type, density, p_val=0.5):
    print(GRN_type,dataset_type,p_val)
    gene_set = pd.read_csv(Gene_file, index_col=0)['index'].values
    tf_set = pd.read_csv(TF_file, index_col=0)['index'].values
    label = pd.read_csv(label_file, index_col=0)
    tf = label['TF'].values
    tf_list = np.unique(tf)

    pos_dict = {}
    for i in tf_list:
        pos_dict[i] = []
    for i, j in label.values:
        pos_dict[i].append(j)

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
    for k in train_pos.keys():
        train_neg[k] = []
        for i in range(len(train_pos[k])):
            neg = np.random.choice(gene_set)

            while neg == k or neg in pos_dict[k] or neg in train_neg[k]:
                neg = np.random.choice(gene_set)
            train_neg[k].append(neg)

    train_set, train_set_label = get_all_set(train_pos, train_neg)

    val_neg = {}
    for k in val_pos.keys():
        val_neg[k] = []
        for i in range(len(val_pos[k])):
            neg = np.random.choice(gene_set)
            while neg == k or neg in pos_dict[k] or neg in train_neg[k] or neg in val_neg[k]:
                neg = np.random.choice(gene_set)
            val_neg[k].append(neg)

    val_set, val_set_label = get_all_set(train_pos, train_neg)

    count = 0
    for k in test_pos.keys():
        count += len(test_pos[k])
    test_neg_num = int(count // density - count)
    test_neg = {}
    for k in tf_set:
        test_neg[k] = []

    test_pos_set = transform_genepair(test_pos)

    test_neg_set = []
    for i in range(test_neg_num):
        t1 = np.random.choice(tf_set)
        t2 = np.random.choice(gene_set)
        while t1 == t2 or [t1, t2] in train_set or [t1, t2] in test_pos_set or [t1, t2] in val_set or [t1,
                                                                                                       t2] in test_neg_set:
            t2 = np.random.choice(gene_set)

        test_neg_set.append([t1, t2])

    test_pos_label = [1 for _ in range(len(test_pos_set))]
    test_neg_label = [0 for _ in range(len(test_neg_set))]

    test_set = test_pos_set + test_neg_set
    test_set_label = test_pos_label + test_neg_label


    train_set = np.array(train_set)
    train_set = pd.DataFrame(train_set)
    train_set['TF'] = train_set.iloc[:, 0]
    train_set['Target'] = train_set.iloc[:, 1]
    train_set['Label'] = train_set_label
    train_set.to_csv(train_set_file)

    val_set = np.array(val_set)
    val_set = pd.DataFrame(val_set)
    val_set['TF'] = val_set.iloc[:, 0]
    val_set['Target'] = val_set.iloc[:, 1]
    val_set['Label'] = val_set_label
    val_set.to_csv(val_set_file)

    test_set = np.array(test_set)
    test_set = pd.DataFrame(test_set)
    test_set['TF'] = test_set.iloc[:, 0]
    test_set['Target'] = test_set.iloc[:, 1]
    test_set['Label'] = test_set_label
    test_set.to_csv(test_set_file)

    print('traing set, validation set, and test set --  hard sample -- are saved successfully.')


def Network_Statistic(data_type,net_scale,net_type):

    if net_type =='STRING':
        dic = {'hESC500': 0.024, 'hESC1000': 0.021, 'hHEP500': 0.028, 'hHEP1000': 0.024, 'mDC500': 0.038,
               'mDC1000': 0.032, 'mESC500': 0.024, 'mESC1000': 0.021, 'mHSC-E500': 0.029, 'mHSC-E1000': 0.027,
               'mHSC-GM500': 0.040, 'mHSC-GM1000': 0.037, 'mHSC-L500': 0.048, 'mHSC-L1000': 0.045}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale



    elif net_type == 'Non-specific-ChIP-seq-network':

        dic = {'hESC500': 0.016, 'hESC1000': 0.014, 'hHEP500': 0.015, 'hHEP1000': 0.013, 'mDC500': 0.019,
               'mDC1000': 0.016, 'mESC500': 0.015, 'mESC1000': 0.013, 'mHSC-E500': 0.022, 'mHSC-E1000': 0.020,
               'mHSC-GM500': 0.030, 'mHSC-GM1000': 0.029, 'mHSC-L500': 0.048, 'mHSC-L1000': 0.043}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'hESC-ChIP-seq-network':
        dic = {'hESC500': 0.164, 'hESC1000': 0.165,'hHEP500': 0.379, 'hHEP1000': 0.377,'mDC500': 0.085,
               'mDC1000': 0.082,'mESC500': 0.345, 'mESC1000': 0.347,'mHSC-E500': 0.578, 'mHSC-E1000': 0.566,
               'mHSC-GM500': 0.543, 'mHSC-GM1000': 0.565,'mHSC-L500': 0.525, 'mHSC-L1000': 0.507}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'Lofgof':
        dic = {'mESC500': 0.158, 'mESC1000': 0.154}

        query = 'mESC' + str(net_scale)
        scale = dic[query]
        return scale

    else:
        raise ValueError


def get_samples(exp_file, target_file, train_data):
    expression = pd.read_csv(exp_file, index_col=0).T  # 779*758
    gene_list = pd.read_csv(target_file)

    feature = []
    labels = []
    for i in range(train_data.shape[0]):
        item = train_data[i]
        tf = item[0]
        target = item[1]
        label = item[2]

        tf_name = gene_list.iloc[tf, 1]
        target_name = gene_list.iloc[target, 1]

        tf_exp = expression.loc[[tf_name]]
        target_exp = expression.loc[[target_name]]
        temp = np.hstack((tf_exp, target_exp))

        feature.append(temp)
        labels.append(label)

    return list(zip(feature, labels))

def get_samples_DeepFGRN(exp_file, target_file, train_data, GRN_embedding_s, GRN_embedding_t):
    expression = pd.read_csv(exp_file, index_col=0).T  # 779*758
    gene_list = pd.read_csv(target_file)

    feature_tf_exp = []
    feature_target_exp = []
    feature_tf_net_tf_s = []
    feature_tf_net_tf_t = []
    feature_target_net_tf_s = []
    feature_target_net_tf_t = []
    labels = []
    for i in range(train_data.shape[0]):
        item = train_data[i]
        tf = item[0]
        target = item[1]
        label = item[2]

        tf_name = gene_list.iloc[tf, 1]
        target_name = gene_list.iloc[target, 1]

        tf_exp = expression.loc[[tf_name]].values.flatten()
        target_exp = expression.loc[[target_name]].values.flatten()

        tf_s = GRN_embedding_s[tf]
        tf_t = GRN_embedding_t[tf]
        target_s = GRN_embedding_s[target]
        target_t = GRN_embedding_t[target]

        feature_tf_exp.append(tf_exp)
        feature_target_exp.append(target_exp)
        feature_tf_net_tf_s.append(tf_s)
        feature_tf_net_tf_t.append(tf_t)
        feature_target_net_tf_s.append(target_s)
        feature_target_net_tf_t.append(target_t)
        labels.append(label)

    train_data_sample = list(zip(feature_tf_exp, feature_target_exp, feature_tf_net_tf_s, feature_tf_net_tf_t,
                feature_target_net_tf_s, feature_target_net_tf_t, labels))  # len

    return train_data_sample

def get_samples_DeepFGRN_3CV(exp_file, target_file, train_data, GRN_embedding_s, GRN_embedding_t):
    expression = pd.read_csv(exp_file, index_col=0).T  # 779*758
    gene_list = pd.read_csv(target_file)

    feature_tf_exp = []
    feature_target_exp = []
    feature_tf_net_tf_s = []
    feature_tf_net_tf_t = []
    feature_target_net_tf_s = []
    feature_target_net_tf_t = []
    labels = []

    train_data = np.array(train_data)
    for i in range(train_data.shape[0]):
        item = train_data[i]
        tf = item[0]
        target = item[1]
        label = item[2]

        tf_name = gene_list.iloc[tf, 1]
        target_name = gene_list.iloc[target, 1]

        tf_exp = expression.loc[tf_name].values.flatten()
        target_exp = expression.loc[target_name].values.flatten()

        tf_s = GRN_embedding_s[tf]
        tf_t = GRN_embedding_t[tf]
        tf_s = np.squeeze(tf_s)
        tf_t = np.squeeze(tf_t)
        target_s = GRN_embedding_s[target]
        target_t = GRN_embedding_t[target]

        feature_tf_exp.append(tf_exp)
        feature_target_exp.append(target_exp)
        feature_tf_net_tf_s.append(tf_s)
        feature_tf_net_tf_t.append(tf_t)
        feature_target_net_tf_s.append(target_s)
        feature_target_net_tf_t.append(target_t)
        labels.append(label)

    train_data_sample = list(zip(feature_tf_exp, feature_target_exp, feature_tf_net_tf_s, feature_tf_net_tf_t,
                feature_target_net_tf_s, feature_target_net_tf_t, labels))  # len

    return train_data_sample




def get_samples_pair(pos_neg_balanced, subset_TF, exp_file):
    samples_feature = []
    samples_pair = []
    samples_label = []


    all_pos_tf = pos_neg_balanced.iloc[:, 0]

    expression = pd.read_csv(exp_file, index_col=0).T  # 779*758

    for tf in subset_TF:
        tf_expression = expression.iloc[tf,:]
        target_index_list = all_pos_tf[all_pos_tf.isin([tf])].index.tolist()

        for i in target_index_list:
            target_index = pos_neg_balanced.iloc[i, 1]
            label = pos_neg_balanced.iloc[i, 2]

            target_expression = expression.iloc[target_index, :]

            samples_pair.append([tf, target_index, label])
            samples_label.append(label)
    # samples_feature = pd.DataFrame(samples_feature)
    samples_pair = pd.DataFrame(samples_pair)
    samples_label = pd.DataFrame(samples_label)


    return samples_pair,samples_label





def get_samples_DeepFGRN_noC(exp_file, target_file, train_data,):
    expression = pd.read_csv(exp_file, index_col=0).T  # 779*758
    gene_list = pd.read_csv(target_file)

    feature_tf_exp = []
    feature_target_exp = []
    labels = []
    for i in range(train_data.shape[0]):
        item = train_data[i]
        tf = item[0]
        target = item[1]
        label = item[2]

        tf_name = gene_list.iloc[tf, 1]
        target_name = gene_list.iloc[target, 1]

        tf_exp = expression.loc[[tf_name]].values.flatten()
        target_exp = expression.loc[[target_name]].values.flatten()

        feature_tf_exp.append(tf_exp)
        feature_target_exp.append(target_exp)
        labels.append(label)

    train_data_sample = list(zip(feature_tf_exp, feature_target_exp, labels))  # len

    return train_data_sample



def get_single_result(result, iteration, metric):
    location = 100
    if metric == 'AUROC':
        location = 0
    elif metric == 'AUPR':
        location = 1
    elif metric == 'Accuracy':
        location = 2
    elif metric == 'F1':
        location = 3
    elif metric == 'SPE':
        location = 4
    elif metric == 'MCC':
        location = 5
    elif metric == 'Precision':
        location = 6
    elif metric == 'Recall':
        location = 7
    else:
        print('error')

    indices = []
    indices_float = []
    if result.shape[1] > iteration:
        for i in range(iteration):
            temp = result.iloc[location].to_list()[i]
            # print('temp:', temp)
            if i == 0:
                tempa = temp.split('[')[1]
            else:
                tempa = temp
            indices.append(tempa)
            tempb = float(tempa)
            indices_float.append(tempb)
    else:
        for i in range(iteration):
            temp = result.iloc[location].to_list()[i]
            # print('temp:', temp)
            if i == 0:
                tempa = temp.split('[')[1]
            elif i == iteration - 1:
                tempa = temp.split(']')[0].split(' ')[1]
            else:
                tempa = temp
            indices.append(tempa)
            tempb = float(tempa)
            indices_float.append(tempb)

    return indices, indices_float


def get_metric_all_311(path_oriresult, dataset_names,network_types, Rank_num, modelname, iteration, metric):
    indice_all_dataset = []
    metric_head = []

    indice_all_mean = []
    indice_all_std = []

    for i in range(len(dataset_names)):
        dataset_name = dataset_names[i]
        network_type = network_types[i]
        print(dataset_name, network_type)

        path_info = path_oriresult + dataset_name + '\\' + network_type + '\\Top' + str(
            Rank_num) + '\\' + modelname + '_311__all.csv'
        print(path_info)
        result = pd.read_csv(path_info, header=None)

        indice, indices_float = get_single_result(result, iteration, metric=metric)

        indice_mean = np.mean(indices_float)
        indice_std = np.std(indices_float)

        # indice_all_dataset.append(indice)
        indice_all_dataset.extend(indice)

        # indice_all_mean.append(indice_mean)
        indice_all_mean.append([indice_mean, indice_std])
        # indice_all_std.append(indice_std)


        # metric_head.append(metric)

    indice_all_dataset = np.array(indice_all_dataset)
    indice_all_dataset = pd.DataFrame(indice_all_dataset)
    heads = pd.DataFrame([metric])
    # metric_head = pd.DataFrame(metric_head)
    # heads = pd.concat([head, metric_head], axis=1)
    indice_all_dataset = pd.concat([heads, indice_all_dataset], axis=0)
    indice_all_dataset.reset_index(drop=True, inplace=True)

    columnsa = [metric + '_mean', metric + '_std']
    columnsa = pd.DataFrame(columnsa).T
    indice_all_mean = pd.DataFrame(indice_all_mean)
    indice_all_mean = pd.concat([columnsa, indice_all_mean], axis=0)

    # indice_all_std = pd.DataFrame(in)

    return indice_all_dataset, indice_all_mean


def get_metric_all_3cv(path_oriresult, dataset_names, network_types, Rank_num, modelname, iteration, metric):
    indice_all_dataset = []
    indice_all_mean = []

    for i in range(len(dataset_names)):
        dataset_name = dataset_names[i]
        network_type = network_types[i]
        print(dataset_name, network_type)

        path_info = path_oriresult + dataset_name + '\\' + network_type + '\\Top' + str(
            Rank_num) + '\\' + modelname + '_3CV__all.csv'
        print(path_info)
        result = pd.read_csv(path_info, header=None)

        indice, indices_float = get_single_result(result, iteration, metric=metric)

        indice_mean = np.mean(indices_float)
        indice_std = np.std(indices_float)

        indice_all_dataset.extend(indice)

        indice_all_mean.append([indice_mean, indice_std])

    indice_all_dataset = np.array(indice_all_dataset)
    indice_all_dataset = pd.DataFrame(indice_all_dataset)
    heads = pd.DataFrame([metric])
    indice_all_dataset = pd.concat([heads, indice_all_dataset], axis=0)
    indice_all_dataset.reset_index(drop=True, inplace=True)

    columnsa = [metric + '_mean', metric + '_std']
    columnsa = pd.DataFrame(columnsa).T
    indice_all_mean = pd.DataFrame(indice_all_mean)
    indice_all_mean = pd.concat([columnsa, indice_all_mean], axis=0)

    return indice_all_dataset, indice_all_mean


def results_summary(path, modelname, iteration, evaluation, Rank_num):
    avg_results = []
    avg_results = pd.DataFrame(avg_results)
    for network_type_t in ['specific', 'non-specific', 'string', 'lofgof']:

        save_path = path + 'Top' + str(Rank_num) + '\\'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        dataset_names = []
        network_types = []
        if network_type_t == 'specific':
            dataset_names = ['hESC', 'hHEP', 'mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']
            network_types = ['hESC-ChIP-seq-network', 'HepG2-ChIP-seq-network', 'mDC-ChIP-seq-network',
                             'mESC-ChIP-seq-network', 'mHSC-ChIP-seq-network', 'mHSC-ChIP-seq-network',
                             'mHSC-ChIP-seq-network']
            network_types = network_types[:len(dataset_names)]
        elif network_type_t == 'non-specific':
            dataset_names = ['hESC', 'hHEP', 'mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']
            network_types = ['Non-specific-ChIP-seq-network', 'Non-specific-ChIP-seq-network',
                             'Non-specific-ChIP-seq-network', 'Non-specific-ChIP-seq-network',
                             'Non-specific-ChIP-seq-network', 'Non-specific-ChIP-seq-network',
                             'Non-specific-ChIP-seq-network']
            network_types = network_types[:len(dataset_names)]
        elif network_type_t == 'string':
            dataset_names = ['hESC', 'hHEP', 'mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']
            network_types = ['STRING-network', 'STRING-network', 'STRING-network', 'STRING-network', 'STRING-network',
                             'STRING-network', 'STRING-network']
            network_types = network_types[:len(dataset_names)]
        elif network_type_t == 'lofgof':
            dataset_names = ['mESC']
            network_types = ['mESC-lofgof-network']
            # network_types = network_types[:len(dataset_names)]
        else:
            print('error')

        all_results = []
        all_results = pd.DataFrame(all_results)

        all_results_mean = []
        all_results_mean = pd.DataFrame(all_results_mean)

        # indices = ['AUROC', 'AUPR', 'Accuracy','F1',"SPE",'MCC','Precision','Recall']
        indices = ['AUROC', 'AUPR']

        for i in range(len(indices)):
            if evaluation == '3cv':
                indice_all_dataset, indice_all_mean = get_metric_all_3cv(path, dataset_names, network_types, Rank_num,
                                                                     modelname, iteration, indices[i])
            elif evaluation == '311':
                indice_all_dataset, indice_all_mean = get_metric_all_311(path, dataset_names, network_types, Rank_num,
                                                                     modelname, iteration, indices[i])
            # AUPR_all_dataset = get_metric_all(path, dataset_names,network_types, Rank_num, modelname, iteration, metric='AUPR')
            if i == 0:
                all_results = indice_all_dataset
                all_results_mean = indice_all_mean
            else:
                all_results = pd.concat([all_results, indice_all_dataset], axis=1)
                all_results_mean = pd.concat([all_results_mean, indice_all_mean], axis=1)

        # print(all_results, all_results_mean)

        columns = ['metric']
        for i in dataset_names:
            for j in range(iteration):
                col_name = i
                columns.append(col_name)
        columns = pd.DataFrame(columns)
        all_results = pd.concat([columns, all_results], axis=1)
        dataset_names_pd = pd.DataFrame(dataset_names)
        all_results_mean = pd.concat([dataset_names_pd, all_results_mean], axis=1)

        all_results.to_csv(save_path + network_type_t + '_' + modelname + '_results.csv', index=False)

        if network_type_t == 'specific':
            avg_results = all_results_mean
        else:
            avg_results = pd.concat([avg_results, all_results_mean], axis=0)
        # print('hee')

    avg_results.reset_index(drop=True, inplace=True)
    print(avg_results)
    avg_results.to_csv(save_path + 'average_results.csv', index=False)














