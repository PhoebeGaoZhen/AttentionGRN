import numpy as np
import pandas as pd
import os


def get_nodeid(target_file, save_path):
    node = pd.read_csv(target_file)
    name = node.iloc[:,1]
    ids = node.iloc[:,0]
    data = {}
    for i in range(len(name)):
        data[name[i]] = ids[i]

    file = open(save_path, 'w')
    for k, v in data.items():
        # print(k,v)
        file.write(str(k) + ' ' + str(v) + '\n')

    file.close()


def get_gene_list(file_name):
    import re
    h={}
    s = open(file_name,'r') #gene symbol ID list of sc RNA-seq
    for line in s:
        search_result = re.search(r'^([^\s]+)\s+([^\s]+)',line)
        h[search_result.group(1).lower()]=search_result.group(2) # h [gene symbol] = gene ID
    s.close()
    return h


def get_samples_pair(pos_neg_balanced, subset_TF, save_path):
    '''
    :param pos_neg_balanced: DataFrame, Each TF has a positive sample and a hard negative sample
    :param subset_TF: TF index of the training set/test set
    :return: All positive and negative sample gene pairs for training set/test set (including label)
    '''
    pos_neg_balanced_col = pos_neg_balanced
    pos_neg_balanced_col.columns = ['tf_name', 'target_name', 'label']
    xdata = np.load(save_path + 'Nxdata_tf.npy')
    # ydata = np.load(save_path + 'ydata_tf.npy')

    samples_feature = []
    samples_pair = []
    samples_label = []

    all_pos_tf = pos_neg_balanced.iloc[:, 0]

    for tf in subset_TF:

        target_index_list = all_pos_tf[all_pos_tf.isin([tf])].index.tolist()

        for i in target_index_list:
            target_index = pos_neg_balanced.iloc[i, 1]
            label = pos_neg_balanced.iloc[i, 2]

            samples_pair.append([tf, target_index, label])
            samples_label.append(label)

            feature = xdata[i, :]
            samples_feature.append(feature)

    samples_pair = pd.DataFrame(samples_pair)
    samples_feature = np.array(samples_feature)
    samples_label = np.array(samples_label)

    return samples_feature,samples_label, samples_pair


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


