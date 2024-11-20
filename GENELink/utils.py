import pandas as pd
import torch, os
from torch.utils.data import Dataset
import random as rd
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
import torch.nn as nn
from sklearn.metrics import confusion_matrix,roc_auc_score,matthews_corrcoef,roc_curve,auc
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score,precision_recall_curve
import warnings
warnings.filterwarnings("ignore")


class scRNADataset(Dataset):
    def __init__(self,train_set,num_gene,flag=False):
        super(scRNADataset, self).__init__()
        self.train_set = train_set
        self.num_gene = num_gene
        self.flag = flag


    def __getitem__(self, idx):
        train_data = self.train_set[:,:2]
        train_label = self.train_set[:,-1]


        data = train_data[idx].astype(np.int64)
        label = train_label[idx].astype(np.float32)


        return data, label

    def __len__(self):
        return len(self.train_set)

    def Adj_Generate(self,TF_set,direction=False, loop=False):
        print("direction = " + str(direction))
        adj = sp.dok_matrix((self.num_gene, self.num_gene), dtype=np.float32) # (910,910)

        for pos in self.train_set:

            tf = pos[0]
            target = pos[1]

            if direction == False:
                if pos[-1] == 1:
                    adj[tf, target] = 1.0
                    adj[target, tf] = 1.0
            else:
                if pos[-1] == 1:
                    adj[tf, target] = 1.0
                    if target in TF_set:
                        adj[target, tf] = 1.0


        if loop:
            adj = adj + sp.identity(self.num_gene)

        adj = adj.todok()


        return adj



class load_data():
    def __init__(self, data, normalize=True):
        self.data = data
        self.normalize = normalize

    def data_normalize(self,data):
        standard = StandardScaler()
        epr = standard.fit_transform(data.T)

        return epr.T


    def exp_data(self):
        data_feature = self.data.values

        if self.normalize:
            data_feature = self.data_normalize(data_feature)

        data_feature = data_feature.astype(np.float32)

        return data_feature


def adj2saprse_tensor(adj):
    coo = adj.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()

    adj_sp_tensor = torch.sparse_coo_tensor(i, v, coo.shape)
    return adj_sp_tensor


def Evaluation(y_true, y_pred,flag=False):
    if flag:
        # y_p = torch.argmax(y_pred,dim=1)
        y_p = y_pred[:,-1]
        y_p = y_p.cpu().detach().numpy()
        y_p = y_p.flatten()
    else:
        y_p = y_pred.cpu().detach().numpy()
        y_p = y_p.flatten()


    y_t = y_true.cpu().numpy().flatten().astype(int)

    AUC = roc_auc_score(y_true=y_t, y_score=y_p)


    AUPR = average_precision_score(y_true=y_t,y_score=y_p)
    # AUPR_norm = AUPR/np.mean(y_t)


    return AUC, AUPR


def normalize(expression):
    std = StandardScaler()
    epr = std.fit_transform(expression)

    return epr



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



def embed2file(tf_embed,tg_embed,gene_file,tf_path,target_path):
    tf_embed = tf_embed.cpu().detach().numpy()
    tg_embed = tg_embed.cpu().detach().numpy()

    gene_set = pd.read_csv(gene_file, index_col=0)

    tf_embed = pd.DataFrame(tf_embed,index=gene_set['Gene'].values)
    tg_embed = pd.DataFrame(tg_embed, index=gene_set['Gene'].values)

    tf_embed.to_csv(tf_path)
    tg_embed.to_csv(target_path)


def get_samples(pos_neg_balanced, subset_TF, expression):
    samples_feature = []
    samples_pair = []
    sample_label = []

    all_pos_tf = pos_neg_balanced.iloc[:, 0]
    for tf in subset_TF:
        print(tf)
        tf_expression = expression.iloc[tf,:]
        target_index_list = all_pos_tf[all_pos_tf.isin([tf])].index.tolist()
        print('target_index_list', target_index_list)
        print('----')

        for i in target_index_list:
            target_index = pos_neg_balanced.iloc[i, 1]
            label = pos_neg_balanced.iloc[i, 2]

            target_expression = expression.iloc[target_index, :]

            samples_pair.append([tf, target_index, label])
            sample_label.append(label)
    # samples_feature = pd.DataFrame(samples_feature)
    samples_pair = pd.DataFrame(samples_pair)
    sample_label = pd.DataFrame(sample_label)

    # print(samples.shape)
    # print(samples)

    return samples_pair, sample_label



def get_samples_pair(pos_neg_balanced, subset_TF):
    samples_feature = []
    samples_pair = []
    samples_label = []


    all_pos_tf = pos_neg_balanced.iloc[:, 0]
    for tf in subset_TF:
        target_index_list = all_pos_tf[all_pos_tf.isin([tf[0]])].index.tolist()

        for i in target_index_list:
            target_index = pos_neg_balanced.iloc[i, 1]
            label = pos_neg_balanced.iloc[i, 2]
            samples_pair.append([tf[0], target_index, label])
            samples_label.append(label)
    # samples_feature = pd.DataFrame(samples_feature)
    samples_pair = pd.DataFrame(samples_pair)
    samples_label = pd.DataFrame(samples_label)

    return samples_pair,samples_label



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

    avg_results.reset_index(drop=True, inplace=True)
    print(avg_results)
    avg_results.to_csv(save_path + 'average_results.csv', index=False)






















