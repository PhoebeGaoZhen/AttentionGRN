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

def get_all_set(pos, neg):
    # pos_set = []
    # for k in pos.keys():
    #     # print('key:', k)
    #     for j in pos[k]:
    #         # print('value:', j)
    #         pos_set.append([k, j])
    pos_set = transform_genepair(pos)
    pos_label = [1 for _ in range(len(pos_set))]

    # neg_set = []
    # for k in neg.keys():
    #     for j in neg[k]:
    #         neg_set.append([k, j])
    neg_set = transform_genepair(neg)
    neg_label = [0 for _ in range(len(neg_set))]

    all_set = pos_set + neg_set
    all_set_label = pos_label + neg_label

    return all_set,all_set_label


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


def train_val_test_set_2(label_file,Gene_file,TF_file,train_set_file,val_set_file,test_set_file,GRN_type, dataset_type,p_val=0.5):
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
                pos_item.append(i)  #

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
    #
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
        all_pos_dict_keys = []
        for k in pos_dict.keys():
            all_pos_dict_keys.append(k)
            if k in neg_dict:
                pos_dict_keys.append(k)
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

        supp_neg_dict = {}
        if len(neg) == 0:

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



def train_val_test_set_hard(label_file,Gene_file,TF_file,train_set_file,val_set_file,test_set_file,density, p_val=0.5):

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
    #
    neg_dict = {}
    neg_list = []
    pos_neg_balanced = []
    for pair in pos_list:
        print(pair)

        pos_tf = pair[0]
        pos_target = pair[1]
        neg_dict[pos_tf] = []
        neg_target = np.random.choice(gene_set)

        while neg_target == pos_tf or neg_target in pos_dict[pos_tf] or neg_target in neg_dict[pos_tf]:
        # while neg_target == pos_tf or neg_target == pos_target or neg_target in neg_dict[pos_tf]:
            neg_target = np.random.choice(gene_set)

        neg_list.append([pos_tf,neg_target])
        neg_dict[pos_tf].append(neg_target)

        pos_neg_balanced.append([pos_tf,pos_target,1])

        pos_neg_balanced.append([pos_tf,neg_target,0])
    pos_neg_balanced = pd.DataFrame(pos_neg_balanced)

    print(len(pos_neg_balanced))

    return tf_list, pos_neg_balanced




def dataset_Info(final_nums, path, dataset_method, dataset_name, modelname, network_types):
    '''

    :param final_nums:
    :param path:
    :param dataset_method: 'result_Data_1' 'result_Data_2'
    :param dataset_name:
    :param network_types:
    :return:
    '''
    head = []
    infos = pd.DataFrame()
    indexss = pd.DataFrame()
    for network_type in network_types:
        for Rank_num in final_nums:
            head.append(network_type + '/' + str(Rank_num))

            path_info = dataset_method + '\\' + dataset_name + '\\' + network_type + '\\Top' + str(
                Rank_num) + '\\' + modelname + '_311__avg.csv'

            a = pd.read_csv(path_info)
            infos = pd.concat([infos, a.iloc[:, 1]], axis=1)

            indexss = a.iloc[:, 0]
            # print(a.iloc[:,1])
    head = pd.DataFrame(head)
    # print(head)
    # print(infos)
    infos.columns = head
    infos.index = indexss
    infos.to_csv(path,encoding="utf_8_sig")