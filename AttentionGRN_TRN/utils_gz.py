from torch_geometric.data import Data
import pandas as pd
import numpy as np
import torch, copy, os, random
import dgl
# from torch_geometric.utils import to_networkx
import networkx as nx
# from sklearn.preprocessing import normalize
# from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix,roc_auc_score,matthews_corrcoef,roc_curve,auc,average_precision_score
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score,precision_recall_curve

def get_pos_neg_T_2(gene_pair_tf_train,gene_pair_target_train, y_train, exp_file, gene_list_file):

    train_tf = gene_pair_tf_train
    train_target = gene_pair_target_train
    train_label = y_train
    train_edge_1_tf = []
    train_edge_1_target = []
    train_edge_0_tf = []
    train_edge_0_target = []

    # train_set1_node = []
    for i in range(train_label.shape[0]):
        if train_label[i] == 1:
            train_edge_1_tf.append(train_tf[i])
            train_edge_1_target.append(train_target[i])
            # train_set1_node.append(train_tf[i])
            # train_set1_node.append(train_target[i])

        elif train_label[i] == 0:
            train_edge_0_tf.append(train_tf[i])
            train_edge_0_target.append(train_target[i])
        else:
            print('link type error')

    # positive edges for DGL building
    train_edge_1 = []
    train_edge_1.append(train_edge_1_tf)
    train_edge_1.append(train_edge_1_target)
    train_edge_index_pos = torch.tensor(train_edge_1, dtype=torch.long)
    train_edge_1_T = [] #
    train_edge_1_T.append(train_edge_1_target)
    train_edge_1_T.append(train_edge_1_tf)
    train_edge_index_pos_T = torch.tensor(train_edge_1_T, dtype=torch.long)
    # negative edges for DGL building
    train_edge_0 = []
    train_edge_0.append(train_edge_0_tf)
    train_edge_0.append(train_edge_0_target)
    train_edge_index_neg = torch.tensor(train_edge_0, dtype=torch.long)
    train_edge_0_T = []
    train_edge_0_T.append(train_edge_0_target)
    train_edge_0_T.append(train_edge_0_tf)
    train_edge_index_neg_T = torch.tensor(train_edge_0_T, dtype=torch.long)

    train_set_node = []
    for node in train_edge_1_tf:
        if node not in train_set_node:
            train_set_node.append(node)
    for node in train_edge_1_target:
        if node not in train_set_node:
            train_set_node.append(node)
    for node in train_edge_0_tf:
        if node not in train_set_node:
            train_set_node.append(node)
    for node in train_edge_0_target:
        if node not in train_set_node:
            train_set_node.append(node)
    # print(len(train_set_node))

    all_node_feature = pd.read_csv(exp_file, index_col=0).T  # 779*758
    gene_list = pd.read_csv(gene_list_file, index_col=0)
    # print(gene_list)
    gene_name = gene_list.loc[:, 'gene'].values.tolist()
    gene_index = gene_list.loc[:, 'index'].values.tolist()
    train_set_node_name = []
    for node in train_set_node:
        if node in gene_index:
            temp = gene_index.index(node)
            temp_name = gene_name[temp]
            train_set_node_name.append(temp_name)
    train_set_node_feature = all_node_feature.loc[train_set_node_name, :]

    # Subsets do not necessarily cover all nodes, and these nodes need to be considered
    if len(train_set_node) != all_node_feature.shape[0]:
        no_feature_x = []
        for i in gene_name:
            if i not in train_set_node_name:
                no_feature_x.append(i)
        no_feature = pd.DataFrame(0, index=no_feature_x, columns=range(all_node_feature.shape[1])) # feature 0
        X_ = pd.concat((train_set_node_feature, no_feature))
    else:
        X_ = train_set_node_feature  # (779,758)
    X_ = X_.values.tolist()
    X = torch.tensor(X_, dtype=torch.float)

    train_og_pos = Data(x=X, edge_index=train_edge_index_pos)
    train_og_neg = Data(x=X, edge_index=train_edge_index_neg)
    train_og_pos_T = Data(x=X, edge_index=train_edge_index_pos_T)
    train_og_neg_T = Data(x=X, edge_index=train_edge_index_neg_T)

    return train_og_pos, train_og_neg, train_og_pos_T, train_og_neg_T


def get_corr(exp_file, gene_list_file, method, cutoffbeishu):

    if method == 'cosine':
        exp = pd.read_csv(exp_file, index_col=0).T.values
        corr_score = np.zeros((exp.shape[0], exp.shape[0]))
        for i in range(exp.shape[0]):
            node_i = exp[i,:]
            for j in range(exp.shape[0]):
                node_j = exp[j,:]
                temp = cosine_similarity(node_i, node_j)
                corr_score[i][j] = temp
    elif method == 'pearson':
        exp = pd.read_csv(exp_file, index_col=0)
        corr_score_d = exp.corr(method="pearson")
        corr_score = corr_score_d.values
    elif method == 'spearman':
        exp = pd.read_csv(exp_file, index_col=0)
        corr_score_d = exp.corr(method="spearman")
        corr_score = corr_score_d.values
    elif method == 'kendall':
        exp = pd.read_csv(exp_file, index_col=0)
        corr_score_d = exp.corr(method="kendall")
        corr_score = corr_score_d.values
    else:
        print('correlation score types error')

    print('correlation score compute done')

    # Sort the correlation score matrix
    corr_node_list = []
    corr_score_list = []
    for i in range(corr_score.shape[0]):
        struc_scores = corr_score[i, :]
        indices = np.argsort(struc_scores)[::-1]
        struc_score = struc_scores[indices]
        corr_node_list.append(indices)
        corr_score_list.append(struc_score)
    corr_node_list = np.array(corr_node_list)
    corr_score_list = np.array(corr_score_list)

    corr_score_list = corr_score_list[:, 1:]
    corr_node_list = corr_node_list[:, 1:]

    final_corr_scores = np.zeros((corr_score_list.shape[0], corr_score_list.shape[0] - 1))
    final_corr_nodes = np.zeros((corr_score_list.shape[0], corr_score_list.shape[0] - 1))
    nodeA = []
    nodeB = []
    for i in range(corr_score_list.shape[0]):
        score = corr_score_list[i, :-1]
        mean = np.mean(score)
        std = np.std(score)
        cutoff = mean + std*cutoffbeishu
        # cutoff = 0.65
        for j in range(score.shape[0]):
            if i==j:
                continue
            elif abs(score[j]) > cutoff:
                # print(score[j], abs(score[j]))
                final_corr_scores[i][j] = score[j]
                final_corr_nodes[i][j] = corr_node_list[i][j]
                nodeA.append(i)
                nodeB.append(j)
    corr_edge_1 = []
    corr_edge_1.append(nodeA)
    corr_edge_1.append(nodeB)
    corr_edge_1_tensor = torch.tensor(corr_edge_1, dtype=torch.long)  # (2,1478)
    # print(st_edge_1)

    all_node_feature = pd.read_csv(exp_file, index_col=0).T  # 779*758
    gene_list = pd.read_csv(gene_list_file, index_col=0)
    # print(gene_list)
    gene_name = gene_list.loc[:, 'gene'].values.tolist()
    gene_index = gene_list.loc[:, 'index'].values.tolist()
    all_node_name = []
    for node in gene_index:
        temp = gene_index.index(node)
        temp_name = gene_name[temp]
        all_node_name.append(temp_name)
    corr_node_feature = all_node_feature.loc[all_node_name, :]
    # print(corr_node_feature.shape)
    corr_X_ = corr_node_feature.values.tolist()
    corr_X = torch.tensor(corr_X_, dtype=torch.float)

    corr_g = Data(x=corr_X, edge_index=corr_edge_1_tensor)

    return corr_g

def add_original_graph(og_data, st_data, weight=1.0):
    st_data = copy.deepcopy(st_data)
    e_i = torch.cat((og_data.edge_index, st_data.edge_index), dim=1) # tensor: [2,11611]
    st_data.edge_index = e_i

    return st_data

def find123Nei(G, node):
    nodes = list(nx.nodes(G))
    nei1_li = []
    nei2_li = []
    nei3_li = []
    for FNs in list(nx.neighbors(G, node)):  # find 1_th neighbors
        nei1_li.append(FNs)

    for n1 in nei1_li:
        for SNs in list(nx.neighbors(G, n1)):  # find 2_th neighbors
            nei2_li.append(SNs)
    nei2_li = list(set(nei2_li) - set(nei1_li))
    if node in nei2_li:
        nei2_li.remove(node)

    for n2 in nei2_li:
        for TNs in nx.neighbors(G, n2):
            nei3_li.append(TNs)
    nei3_li = list(set(nei3_li) - set(nei2_li) - set(nei1_li))
    if node in nei3_li:
        nei3_li.remove(node)

    return nei1_li, nei2_li, nei3_li


def getSI_D_2(G, IA, khop_neighborlist):
    length = len(khop_neighborlist)
    Tk_I = []
    if length == 0:
        min_d_in = 0
        max_d_in = 0
        mean_d_in = 0
        sigma_d_in = 0
        min_d_out = 0
        max_d_out = 0
        mean_d_out = 0
        sigma_d_out = 0
        centrality1 = 0
        centrality2 = 0
        centrality3 = 0
        centrality4 = 0
        centrality5 = 0
    else:
        list_degree_in = []
        list_degree_out = []
        idc = []
        odc = []
        cc = []
        bc = []
        katz_ec = []

        for dst_node in khop_neighborlist:
            #SI
            in_degree = G.in_degree(dst_node)
            out_degree = G.out_degree(dst_node)
            list_degree_in.append(in_degree)
            list_degree_out.append(out_degree)

            centrality = IA[dst_node].tolist()
            idc.append(centrality[0])
            odc.append(centrality[1])
            cc.append(centrality[2])
            bc.append(centrality[3])
            katz_ec.append(centrality[4])

        min_d_in = min(list_degree_in)
        max_d_in = max(list_degree_in)
        mean_d_in = np.round(sum(list_degree_in) / length, 3)
        sigma_d_in = np.round(np.std(list_degree_in), 3)

        min_d_out = min(list_degree_out)
        max_d_out = max(list_degree_out)
        mean_d_out = np.round(sum(list_degree_out) / length,3)
        sigma_d_out = np.round(np.std(list_degree_out),3)

        centrality1 = np.round(sum(idc) / length,3)
        centrality2 = np.round(sum(odc) / length,3)
        centrality3 = np.round(sum(cc) / length,3)
        centrality4 = np.round(sum(bc) / length,3)
        centrality5 = np.round(sum(katz_ec) / length,3)


    Tk_I.extend([centrality1, centrality2, centrality3, centrality4, centrality5, min_d_in, max_d_in, mean_d_in,sigma_d_in, min_d_out, max_d_out, mean_d_out,sigma_d_out])

    return Tk_I

def getI_directed(nx_g, nx_g_T, IA, k_hop):
    I = []
    for target_node in nx_g.nodes():
        node_SI = []
        node_indegree = nx_g.in_degree(target_node)  # G.add_edge(1, 3)  1————>3
        node_outdegree = nx_g.out_degree(target_node)
        node_SI.append(node_indegree)
        node_SI.append(node_outdegree)
        node_SI.extend(IA[target_node])

        neighbors = find123Nei(nx_g, target_node)
        one_hopt = neighbors[0]  #
        two_hopt = neighbors[1]
        three_hopt = neighbors[2]

        T1t = getSI_D_2(nx_g, IA, one_hopt)
        T2t = getSI_D_2(nx_g, IA, two_hopt)

        neighbors = find123Nei(nx_g_T, target_node)
        one_hops = neighbors[0]  #
        two_hops = neighbors[1]
        three_hops = neighbors[2]
        T1s = getSI_D_2(nx_g_T, IA, one_hops)
        T2s = getSI_D_2(nx_g_T, IA, two_hops)

        if k_hop == 1:
            node_SI.extend(T1s)
            node_SI.extend(T1t)
        elif k_hop == 2:
            node_SI.extend(T1s)
            node_SI.extend(T1t)
            node_SI.extend(T2s)
            node_SI.extend(T2t)
        # print('target_node: ',target_node, node_SI)
        node_SI = np.array(node_SI)

        # I[target_node] = node_SI
        I.append(node_SI)
    I = np.array(I)

    return I


def buildPE_Kindices(networkx_g, k_hop):
    edge_idx1 = []
    edge_idx2 = []

    for e in networkx_g.edges:
        edge_idx1.append(e[0])
        edge_idx2.append(e[1])

    path = dict(nx.all_pairs_bellman_ford_path(networkx_g))

    nodes_ids = list(path.keys())
    all_path = list(map(path.get, nodes_ids))

    src = []
    dst = []
    for s_idx, s_node in enumerate(nodes_ids):
        spd_from_idx = all_path[s_idx]
        for target_node, path in spd_from_idx.items():
            len_of_path = len(path)
            if len_of_path == 1:
                continue
            elif len_of_path == 2 and k_hop == 1:
                src.append(s_node)
                dst.append(target_node)
            elif len_of_path == 2 and k_hop == 2:
                src.append(s_node)
                dst.append(target_node)
            elif len_of_path == 3 and k_hop == 2:
                src.append(s_node)
                dst.append(target_node)
            else:
                continue

    Kindices = get_k_indicaces(src, dst)

    return Kindices


def get_k_indicaces(src,dst):
    map1={}
    ind = []
    # for i in range(len(src) - 1):
    for i in range(len(src)):
        v1 = str(src[i]) +"_" + str(dst[i]) # v1: '61_671'
        # v2 = str(dst[i]) +"_" + str(src[i]) # v2: '671_61'
        if not map1.get(v1):
            map1[v1] = True
            # map1[v2] = True
        # if not map1.get(v2):
        #     map1[v1] = True
        #     map1[v2] = True
    sd =list(map1.keys()) # ['61_671','671_61','61_597','597_61' ...
    source = []
    destination= []
    for i in range(len(sd)):
        x = sd[i].split("_")
        source.append(int(x[0]))
        destination.append(int(x[1]))
    return np.stack((np.array(source), np.array(destination)))



def g_dgl_I(data, I, Kindices, num_nodes):
    edge_idx1 = Kindices[0]
    edge_idx2 = Kindices[1]
    # edge_idx1 = train_edge_tf
    # edge_idx2 = train_edge_target
    k_edge_idx1 = []
    k_edge_idx2 = []
    count = 0
    for i in range(len(edge_idx1)):
        count += 1

        n1 = edge_idx1[i]
        n2 = edge_idx2[i]

        k_edge_idx1.append(n1)
        k_edge_idx2.append(n2)

    I = normalizeTensor(I)

    g = dgl.graph((k_edge_idx1, k_edge_idx2), num_nodes=num_nodes)
    g.ndata['x'] = data.x
    g.ndata['I'] = I

    print("DONE construct dgl graph.")

    return g


def normalizeTensor(x):
    # x_normed = x / x.max(0, keepdim=True)[0]
    temp = x.max(0, keepdim=True)[0]
    temp = temp + 1e-8
    x_normed = x / temp

    # MA = x.max(0, keepdim=True)
    return x_normed


def cosine_similarity(vector1, vector2):
    vec1 = np.array(vector1)
    vec2 = np.array(vector2)

    dot_product = np.dot(vec1, vec2)

    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    similarity = dot_product / (norm_vec1 * norm_vec2)
    similarity = np.round(similarity, 4)

    return similarity


def metric_scores(y_test, y_pred, th=0.5):
    #y_predlabel = [(0 if item<th else 1) for item in y_pred]
    #tn,fp,fn,tp = confusion_matrix(y_test,y_predlabel).flatten()
    #SPE = tn*1./(tn+fp)
    #MCC = matthews_corrcoef(y_test,y_predlabel)
    #Recall = recall_score(y_test, y_predlabel)
    #Precision = precision_score(y_test, y_predlabel)
    #F1 = f1_score(y_test, y_predlabel)
    #Acc = accuracy_score(y_test, y_predlabel)
    AUC = roc_auc_score(y_test, y_pred)
    precision_aupr, recall_aupr, _ = precision_recall_curve(y_test, y_pred)
    AUPR = auc(recall_aupr, precision_aupr)

    return AUC, AUPR

def get_samples_pair(pos_neg_balanced, subset_TF, save_path):
    '''
    :param pos_neg_balanced: DataFrame, Each TF has a positive sample + a hard negative sample, tf- target gene - tag
    :param subset_TF: TF index of the training set/test set
    :return: All positive and negative sample gene pairs for training set/test set (including tags)
    '''
    pos_neg_balanced_col = pos_neg_balanced
    pos_neg_balanced_col.columns = ['tf_name', 'target_name', 'label']
    xdata = np.load(save_path + 'Nxdata_tf.npy')
    # ydata = np.load(save_path + 'ydata_tf.npy')

    samples_feature = []
    samples_pair = []
    samples_label = []
    samples_pair_tf = []
    samples_pair_target = []

    all_pos_tf = pos_neg_balanced.iloc[:, 0]

    for tf in subset_TF:
        target_index_list = all_pos_tf[all_pos_tf.isin([tf])].index.tolist()

        for i in target_index_list:
            target_index = pos_neg_balanced.iloc[i, 1]
            label = pos_neg_balanced.iloc[i, 2]
            samples_pair.append([tf, target_index, label])
            samples_pair_tf.append(tf[0])
            samples_pair_target.append(target_index)

            samples_label.append(label)

            feature = xdata[i, :]
            samples_feature.append(feature)

    samples_feature = np.array(samples_feature)
    samples_label = np.array(samples_label)
    samples_pair_tf = np.array(samples_pair_tf)
    samples_pair_target = np.array(samples_pair_target)

    return samples_feature,samples_label, samples_pair_tf, samples_pair_target



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

        indice_all_mean.append([indice_mean, indice_std])

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


