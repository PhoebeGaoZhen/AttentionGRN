import numpy as np
import pandas as pd
from utils import load_data
import os, random
# from input_data import load_data2
from scipy.sparse import csr_matrix
import scipy.sparse as sp


def adj_to_bias(adj, sizes, nhood = 1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0

    return -1e9 * (1.0-mt)



def load_data1(file_name,file_path):
    # 读取CSV文件，跳过第一行（标题行）
    df = pd.read_csv(file_name, header=0)  # header=0 表示第一行是列名

    # 删除第一列（列名）
    df = df.iloc[:, 1:] # 基因表达数据 cell*gene

    # 转置DataFrame
    data_input = df.T  # 基因表达数据
    geneName = data_input.index
    loader = load_data(data_input)
    feature = loader.exp_data() # 正则化，并转换为 float32
    geneNum = feature.shape[0]
    # 假设CSV文件路径为'your_file.csv'

    # 使用pandas读取CSV文件，指定没有列名，并提供列名
    df = pd.read_csv(file_path, header=None, names=['index', 'TF', 'target'],skiprows=1)

    # 由于第一行是列标题，我们需要跳过它，只从第二行开始读取数据
    # 这可以通过使用 `skiprows=1` 参数来实现，但需要在读取文件后删除第一行
    df = df.drop(columns=df.columns[0])


    # 提取行索引和列索引
    rows = df['TF'].values.astype(int)  # 将索引列转换为整数
    cols = df['target'].values.astype(int)  # 使用'TF'列作为列索引
    data = pd.Series([1] * len(df))  # 邻接矩阵中的权重都是1

    # 创建CSR矩阵
    adj_init = csr_matrix((data, (rows, cols)), shape=(geneNum, geneNum))

    # adj_init, exp_file, data_name = load_data2()
    # # 拼接文件名
    # file_name = data_name + "_feature.csv"
    #
    # # 检查文件是否存在
    # if not os.path.exists(file_name):
    #     # 将 CSR 矩阵转换为稠密矩阵
    #     dense_matrix = exp_file.toarray()
    #
    #     # 创建 DataFrame 对象
    #     df = pd.DataFrame(dense_matrix)
    #
    #     # 保存 DataFrame 为 CSV 文件
    #     df.to_csv(file_name, index=False)
    #     print("文件保存成功！")
    # else:
    #     print("文件已存在，无需保存。")

    # # 保存 DataFrame 为 CSV 文件
    # data_input = pd.read_csv(file_name)  # .../

    train_edges, train_edges_false,  val_edges, val_edges_false, test_edges, test_edges_false = \
        mask_test_edges_general_link_prediction(adj_init)


    # 将边和标签放在一起 【正负样本合并到一个列表，每个元素是tf target label】
    validation_data = [(edge[0], edge[1], 1) for edge in val_edges] + [(edge[0], edge[1], 0) for edge in val_edges_false]

    validation_data = np.array(validation_data)
    test_data = [(edge[0], edge[1], 1) for edge in test_edges] + [(edge[0], edge[1], 0) for edge in test_edges_false]
    test_data = np.array(test_data)
    train_data = [(edge[0], edge[1], 1) for edge in train_edges] + [(edge[0], edge[1], 0) for edge in
                                                                       train_edges_false]

    train_data = np.array(train_data)
    # # 拼接文件名
    # adj_name = data_name + "_adj.csv"
    #
    # # 检查文件是否存在
    # if not os.path.exists(adj_name):
    #     # 将 CSR 矩阵转换为稠密矩阵
    #     dense_matrix = adj_init.toarray()
    #
    #     # 创建 DataFrame 对象
    #     df = pd.DataFrame(dense_matrix)
    #
    #     # 保存 DataFrame 为 CSV 文件
    #     df.to_csv(adj_name, index=False)
    #     print("文件保存成功！")
    # else:
    #     print("文件已存在，无需保存。")
    #
    #     # 保存 DataFrame 为 CSV 文件
    # train_file = pd.read_csv(adj_name)  # .../
    #
    #
    #
    # 假设你的CSV文件名为'gene_expression_matrix.csv'
    # 假设CSV文件路径
    #
    #
    # train_file = 'Specific Dataset hESC TF1000+_adj.csv'  # .../Demo/
    # test_file = 'Data/train_test_val/Non-Specific/hESC500/Test_set.csv'
    # val_file = 'Data/train_test_val/Non-Specific/hESC500/Validation_set.csv'
    # train_data = pd.read_csv(train_file, index_col=0).values
    # validation_data = pd.read_csv(val_file, index_col=0).values
    # test_data = pd.read_csv(test_file, index_col=0).values

    train_data = train_data[np.lexsort(-train_data.T)]    # 按照 train_data 中最后一列的降序对 train_data 进行重新排序，返回重新排序后的数组。
    train_index = np.sum(train_data[:,2])

    validation_data = validation_data[np.lexsort(-validation_data.T)]
    validation_index = np.sum(validation_data[:, 2])

    test_data = test_data[np.lexsort(-test_data.T)]
    test_index = np.sum(test_data[:, 2])


    logits_train = sp.csr_matrix((train_data[0:train_index,2], (train_data[0:train_index,0] , train_data[0:train_index,1])),shape=(geneNum, geneNum)).toarray()
    neg_logits_train = sp.csr_matrix((np.ones(train_data[train_index:, 2].shape), (train_data[train_index:, 0], train_data[train_index:, 1])),
                                 shape=(geneNum, geneNum)).toarray()
    interaction = logits_train
    interaction = interaction + np.eye(interaction.shape[0])
    interaction = sp.csr_matrix(interaction)
    logits_train = logits_train.reshape([-1, 1])
    neg_logits_train = neg_logits_train.reshape([-1, 1])

    logits_test = sp.csr_matrix((test_data[0:test_index, 2], (test_data[0:test_index, 0], test_data[0:test_index, 1] )),
                                 shape=(geneNum, geneNum)).toarray()
    neg_logits_test = sp.csr_matrix((np.ones(test_data[test_index:, 2].shape), (test_data[test_index:, 0], test_data[test_index:, 1])),
                                shape=(geneNum, geneNum)).toarray()
    logits_test = logits_test.reshape([-1, 1])
    neg_logits_test = neg_logits_test.reshape([-1, 1])
    logits_validation = sp.csr_matrix((validation_data[0:validation_index, 2], (validation_data[0:validation_index, 0], validation_data[0:validation_index, 1])),
                               shape=(geneNum, geneNum)).toarray()
    neg_logits_validation = sp.csr_matrix(
        (np.ones(validation_data[validation_index:, 2].shape), (validation_data[validation_index:, 0], validation_data[validation_index:, 1])),
        shape=(geneNum, geneNum)).toarray()
    logits_validation = logits_validation.reshape([-1, 1])
    neg_logits_validation = neg_logits_validation.reshape([-1, 1])

    train_mask = np.array(logits_train[:, 0], dtype=np.bool).reshape([-1, 1])
    test_mask = np.array(logits_test[:, 0], dtype=np.bool).reshape([-1, 1])
    validation_mask = np.array(logits_validation[:, 0], dtype=np.bool).reshape([-1, 1])

    return geneName, feature, logits_train, logits_test, logits_validation, train_mask, test_mask, validation_mask, interaction, neg_logits_train, neg_logits_test, neg_logits_validation, train_data, validation_data, test_data


def load_data_gao(file_name, file_path, pos_neg_balanced_id):
    # 1. 读取基因表达数据，获得基因名称、节点特征、基因数量等信息
    # 读取CSV文件，跳过第一行（标题行）
    df = pd.read_csv(file_name, header=0)  # header=0 表示第一行是列名
    # 删除第一列（列名）
    df = df.iloc[:, 1:] # 基因表达数据 cell*gene

    # 转置DataFrame
    data_input = df.T  # 基因表达数据
    geneName = data_input.index
    loader = load_data(data_input)
    feature = loader.exp_data() # 正则化，并转换为 float32
    geneNum = feature.shape[0]
    # 假设CSV文件路径为'your_file.csv'

    # 2. 读取已知关联数据
    # 使用pandas读取CSV文件，指定没有列名，并提供列名
    df = pd.read_csv(file_path, header=None, names=['index', 'TF', 'target'],skiprows=1)

    # 由于第一行是列标题，我们需要跳过它，只从第二行开始读取数据
    # 这可以通过使用 `skiprows=1` 参数来实现，但需要在读取文件后删除第一行
    df = df.drop(columns=df.columns[0])

    # 提取行索引和列索引
    rows = df['TF'].values.astype(int)  # 将索引列转换为整数
    cols = df['target'].values.astype(int)  # 使用'TF'列作为列索引
    data = pd.Series([1] * len(df))  # 邻接矩阵中的权重都是1

    # 创建CSR矩阵
    adj_init = csr_matrix((data, (rows, cols)), shape=(geneNum, geneNum))


    train_edges, train_edges_false,  val_edges, val_edges_false, test_edges, test_edges_false = \
        mask_test_edges_general_link_prediction(adj_init)


    # 将边和标签放在一起 【正负样本合并到一个列表，每个元素是tf target label】
    validation_data = [(edge[0], edge[1], 1) for edge in val_edges] + [(edge[0], edge[1], 0) for edge in val_edges_false]

    validation_data = np.array(validation_data)
    test_data = [(edge[0], edge[1], 1) for edge in test_edges] + [(edge[0], edge[1], 0) for edge in test_edges_false]
    test_data = np.array(test_data)
    train_data = [(edge[0], edge[1], 1) for edge in train_edges] + [(edge[0], edge[1], 0) for edge in
                                                                       train_edges_false]

    train_data = np.array(train_data)
    # # 拼接文件名
    # adj_name = data_name + "_adj.csv"
    #
    # # 检查文件是否存在
    # if not os.path.exists(adj_name):
    #     # 将 CSR 矩阵转换为稠密矩阵
    #     dense_matrix = adj_init.toarray()
    #
    #     # 创建 DataFrame 对象
    #     df = pd.DataFrame(dense_matrix)
    #
    #     # 保存 DataFrame 为 CSV 文件
    #     df.to_csv(adj_name, index=False)
    #     print("文件保存成功！")
    # else:
    #     print("文件已存在，无需保存。")
    #
    #     # 保存 DataFrame 为 CSV 文件
    # train_file = pd.read_csv(adj_name)  # .../
    #
    #
    #
    # 假设你的CSV文件名为'gene_expression_matrix.csv'
    # 假设CSV文件路径
    #
    #
    # train_file = 'Specific Dataset hESC TF1000+_adj.csv'  # .../Demo/
    # test_file = 'Data/train_test_val/Non-Specific/hESC500/Test_set.csv'
    # val_file = 'Data/train_test_val/Non-Specific/hESC500/Validation_set.csv'
    # train_data = pd.read_csv(train_file, index_col=0).values
    # validation_data = pd.read_csv(val_file, index_col=0).values
    # test_data = pd.read_csv(test_file, index_col=0).values

    train_data = train_data[np.lexsort(-train_data.T)]    # 按照 train_data 中最后一列的降序对 train_data 进行重新排序，返回重新排序后的数组。
    train_index = np.sum(train_data[:,2])  # 正样本的数量

    validation_data = validation_data[np.lexsort(-validation_data.T)]
    validation_index = np.sum(validation_data[:, 2])

    test_data = test_data[np.lexsort(-test_data.T)]
    test_index = np.sum(test_data[:, 2])


    logits_train = sp.csr_matrix((train_data[0:train_index,2], (train_data[0:train_index,0] , train_data[0:train_index,1])),shape=(geneNum, geneNum)).toarray()
    neg_logits_train = sp.csr_matrix((np.ones(train_data[train_index:, 2].shape), (train_data[train_index:, 0], train_data[train_index:, 1])),
                                 shape=(geneNum, geneNum)).toarray()
    interaction = logits_train
    interaction = interaction + np.eye(interaction.shape[0]) # 加上自环, 6694——7633
    interaction = sp.csr_matrix(interaction)
    logits_train = logits_train.reshape([-1, 1])
    neg_logits_train = neg_logits_train.reshape([-1, 1])

    logits_test = sp.csr_matrix((test_data[0:test_index, 2], (test_data[0:test_index, 0], test_data[0:test_index, 1] )),
                                 shape=(geneNum, geneNum)).toarray()
    neg_logits_test = sp.csr_matrix((np.ones(test_data[test_index:, 2].shape), (test_data[test_index:, 0], test_data[test_index:, 1])),
                                shape=(geneNum, geneNum)).toarray()
    logits_test = logits_test.reshape([-1, 1])
    neg_logits_test = neg_logits_test.reshape([-1, 1])
    logits_validation = sp.csr_matrix((validation_data[0:validation_index, 2], (validation_data[0:validation_index, 0], validation_data[0:validation_index, 1])),
                               shape=(geneNum, geneNum)).toarray()
    neg_logits_validation = sp.csr_matrix(
        (np.ones(validation_data[validation_index:, 2].shape), (validation_data[validation_index:, 0], validation_data[validation_index:, 1])),
        shape=(geneNum, geneNum)).toarray()
    logits_validation = logits_validation.reshape([-1, 1])
    neg_logits_validation = neg_logits_validation.reshape([-1, 1])

    train_mask = np.array(logits_train[:, 0], dtype=np.bool).reshape([-1, 1])
    test_mask = np.array(logits_test[:, 0], dtype=np.bool).reshape([-1, 1])
    validation_mask = np.array(logits_validation[:, 0], dtype=np.bool).reshape([-1, 1])

    return geneName, feature, logits_train, logits_test, logits_validation, train_mask, test_mask, validation_mask, interaction, neg_logits_train, neg_logits_test, neg_logits_validation, train_data, validation_data, test_data


def transform_data(exp_file_name, train_data, validation_data, test_data):
    # 1. 读取基因表达数据，获得基因名称、节点特征、基因数量等信息
    # 读取CSV文件，跳过第一行（标题行）
    df = pd.read_csv(exp_file_name, header=0)  # header=0 表示第一行是列名
    # 删除第一列（列名）
    df = df.iloc[:, 1:]  # 基因表达数据 cell*gene

    # 转置DataFrame
    data_input = df.T  # 基因表达数据
    geneName = data_input.index
    loader = load_data(data_input)
    feature = loader.exp_data()  # 正则化，并转换为 float32
    geneNum = feature.shape[0]

    # 2. 将train_data, validation_data, test_data转换为logits
    train_data = train_data[np.lexsort(-train_data.T)]  # 按照 train_data 中最后一列的降序对 train_data 进行重新排序，返回重新排序后的数组。
    train_index = np.sum(train_data[:, 2])  # 正样本的数量

    validation_data = validation_data[np.lexsort(-validation_data.T)]
    validation_index = np.sum(validation_data[:, 2])

    test_data = test_data[np.lexsort(-test_data.T)]
    test_index = np.sum(test_data[:, 2])

    logits_train = sp.csr_matrix(
        (train_data[0:train_index, 2], (train_data[0:train_index, 0], train_data[0:train_index, 1])),
        shape=(geneNum, geneNum)).toarray()
    neg_logits_train = sp.csr_matrix(
        (np.ones(train_data[train_index:, 2].shape), (train_data[train_index:, 0], train_data[train_index:, 1])),
        shape=(geneNum, geneNum)).toarray()
    interaction = logits_train
    interaction = interaction + np.eye(interaction.shape[0])  # 加上自环, 6694——7633
    interaction = sp.csr_matrix(interaction)
    logits_train = logits_train.reshape([-1, 1])
    neg_logits_train = neg_logits_train.reshape([-1, 1])

    logits_test = sp.csr_matrix((test_data[0:test_index, 2], (test_data[0:test_index, 0], test_data[0:test_index, 1])),
                                shape=(geneNum, geneNum)).toarray()
    neg_logits_test = sp.csr_matrix(
        (np.ones(test_data[test_index:, 2].shape), (test_data[test_index:, 0], test_data[test_index:, 1])),
        shape=(geneNum, geneNum)).toarray()
    logits_test = logits_test.reshape([-1, 1])
    neg_logits_test = neg_logits_test.reshape([-1, 1])
    logits_validation = sp.csr_matrix((validation_data[0:validation_index, 2], (
    validation_data[0:validation_index, 0], validation_data[0:validation_index, 1])),
                                      shape=(geneNum, geneNum)).toarray()
    neg_logits_validation = sp.csr_matrix(
        (np.ones(validation_data[validation_index:, 2].shape),
         (validation_data[validation_index:, 0], validation_data[validation_index:, 1])),
        shape=(geneNum, geneNum)).toarray()
    logits_validation = logits_validation.reshape([-1, 1])
    neg_logits_validation = neg_logits_validation.reshape([-1, 1])

    train_mask = np.array(logits_train[:, 0], dtype=np.bool).reshape([-1, 1])
    test_mask = np.array(logits_test[:, 0], dtype=np.bool).reshape([-1, 1])
    validation_mask = np.array(logits_validation[:, 0], dtype=np.bool).reshape([-1, 1])

    return geneName, feature, logits_train, logits_test, logits_validation, train_mask, test_mask, validation_mask, interaction, neg_logits_train, neg_logits_test, neg_logits_validation


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    # adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    adj_normalized = degree_mat_inv_sqrt.dot(adj_)
    adj_normalized = adj_normalized.dot(degree_mat_inv_sqrt).tocoo()

    return sparse_to_tuple(adj_normalized)

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def mask_test_edges_general_link_prediction(adj, test_percent=20., val_percent=20.):
    """
    Task 1: General Directed Link Prediction: get Train/Validation/Test

    :param adj: complete sparse adjacency matrix of the graph
    :param test_percent: percentage of edges in test set
    :param val_percent: percentage of edges in validation set
    :return: train incomplete adjacency matrix, validation and test sets
    """

    # Remove diagonal elements of adjacency matrix 移除邻接矩阵的对角元素
    adj = adj - sp.dia_matrix((adj.diagonal()[None, :], [0]), shape = adj.shape)
    adj.eliminate_zeros()                                # 将稀疏矩阵中的零元素去除的方法
    edges_positive, _, _ = sparse_to_tuple(adj)
    # edges_positive 包含了矩阵中非零元素的行、列索引和对应的值。另外两个下划线 _ 表示忽略了额外的返回值。
    # Number of positive (and negative) edges in test and val sets:
    num_test = int(np.floor(edges_positive.shape[0] / (100. / test_percent)))
    num_val = int(np.floor(edges_positive.shape[0] / (100. / val_percent)))

    # Sample positive edges for test and val sets:
    edges_positive_idx = np.arange(edges_positive.shape[0])
    np.random.shuffle(edges_positive_idx)
    val_edge_idx = edges_positive_idx[:num_val]
    # positive val edges
    val_edges = edges_positive[val_edge_idx]
    test_edge_idx = edges_positive_idx[num_val:(num_val + num_test)]
    # positive test edges
    test_edges = edges_positive[test_edge_idx]
    # positive train edges
    train_edges = np.delete(edges_positive, np.hstack([test_edge_idx, val_edge_idx]), axis = 0)
    # 初始化 train_edges_false
    num_nodes = adj.shape[0]
    train_edges_false = []

    # 构建 train_edges_false
    for i in range(num_nodes):
        for j in range(num_nodes):
            # 如果边不在 train_edges 中，则将其添加到 train_edges_false 中
            if (i != j) and ([i, j] not in train_edges):
                train_edges_false.append([i, j])

    # 将 train_edges_false 转换为 numpy 数组
    train_edges_false = np.array(train_edges_false)

    # (Text from philipjackson)
    # The above strategy for sampling without replacement will not work for sampling
    # negative edges on large graphs, because the pool of negative edges
    # is much much larger due to sparsity, therefore we'll use the following strategy:
    # 1. sample random linear indices from adjacency matrix WITH REPLACEMENT 1.从具有替换的邻接矩阵中采样随机线性索引
    # (without replacement is super slow). sample more than we need so we'll probably
    # have enough after all the filtering steps.
    # 2. remove any edges that have already been added to the other edge lists  删除已添加到其他边列表中的所有边
    # 3. convert to (i,j) coordinates
    # 4. remove any duplicate elements if there are any
    # 5. remove any diagonal elements
    # 6. if we don't have enough edges, repeat this process until we get enough
    positive_idx, _, _ = sparse_to_tuple(adj) # positive_idx 是一个包含稀疏矩阵 adj 中所有非零元素的行索引和列索引的二维数组。 [i,j] coord pairs for all true edges
    positive_idx = positive_idx[:,0]*adj.shape[0] + positive_idx[:,1] # linear indices  positive_idx[:,0]行索引/列索引
    # Test set
    test_edges_false = np.empty((0,2),dtype='int64')
    idx_test_edges_false = np.empty((0,),dtype='int64')
    while len(test_edges_false) < len(test_edges):
        # step 1:
        idx = np.random.choice(adj.shape[0]**2, 2*(num_test - len(test_edges_false)), replace = True)
        # step 2:
        idx = idx[~np.in1d(idx, positive_idx, assume_unique = True)]
        idx = idx[~np.in1d(idx, idx_test_edges_false, assume_unique = True)]
        # step 3:
        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx, colidx)).transpose()
        # step 4:
        coords = np.unique(coords, axis=0)
        np.random.shuffle(coords)
        # step 5:
        coords = coords[coords[:,0] != coords[:,1]]
        # step 6:
        coords = coords[:min(num_test, len(idx))]
        test_edges_false = np.append(test_edges_false, coords, axis = 0)
        idx = idx[:min(num_test, len(idx))]
        idx_test_edges_false = np.append(idx_test_edges_false, idx)

    # Validation set
    val_edges_false = np.empty((0,2), dtype = 'int64')
    idx_val_edges_false = np.empty((0,), dtype = 'int64')
    while len(val_edges_false) < len(val_edges):
        # step 1:
        idx = np.random.choice(adj.shape[0]**2, 2*(num_val - len(val_edges_false)), replace = True)
        # step 2:
        idx = idx[~np.in1d(idx, positive_idx, assume_unique = True)]
        idx = idx[~np.in1d(idx, idx_test_edges_false, assume_unique = True)]
        idx = idx[~np.in1d(idx, idx_val_edges_false, assume_unique = True)]
        # step 3:
        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx, colidx)).transpose()
        # step 4:
        coords = np.unique(coords, axis = 0)
        np.random.shuffle(coords)
        # step 5:
        coords = coords[coords[:,0] != coords[:,1]]
        # step 6:
        coords = coords[:min(num_val, len(idx))]
        val_edges_false = np.append(val_edges_false, coords, axis=0)
        idx = idx[:min(num_val, len(idx))]
        idx_val_edges_false = np.append(idx_val_edges_false, idx)

    # Sanity checks:
    train_edges_linear = train_edges[:,0]*adj.shape[0] + train_edges[:,1]
    test_edges_linear = test_edges[:,0]*adj.shape[0] + test_edges[:,1]
    assert not np.any(np.in1d(idx_test_edges_false, positive_idx))
    assert not np.any(np.in1d(idx_val_edges_false, positive_idx))
    assert not np.any(np.in1d(val_edges[:,0]*adj.shape[0] + val_edges[:,1], train_edges_linear))
    assert not np.any(np.in1d(test_edges_linear, train_edges_linear))
    assert not np.any(np.in1d(val_edges[:,0]*adj.shape[0] + val_edges[:,1], test_edges_linear))

    # Re-build train adjacency matrix
    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape = adj.shape)
    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false


def get_samples_pair(pos_neg_balanced, subset_TF):
    '''

    :param pos_neg_balanced: DataFrame， 每个TF都有一个正样本+一个hard负样本，tf-靶基因-标签
    :param subset_TF: 训练集/测试集的TF index
    :return: 训练集/测试集的所有正负样本基因对（包含标签）
    '''
    samples_feature = []
    samples_pair = []
    samples_label = []


    all_pos_tf = pos_neg_balanced.iloc[:, 0]
    # print(all_pos_tf.shape)
    # all_pos_tf = all_pos_tf.tolist()
    # print('all_pos_tf', all_pos_tf)

    # 基因表达数据
    # expression = pd.read_csv(exp_file, index_col=0).T  # 779*758

    # 找到 训练集中每个基因对
    for tf in subset_TF:
        # print(tf)
        # tf_expression = expression.iloc[tf,:]
        # print(tf_expression.shape)
        # 返回 tf 在 all_pos_tf中的位置
        target_index_list = all_pos_tf[all_pos_tf.isin([tf[0]])].index.tolist()
        # index_list = all_pos_tf.index(tf)
        # print('target_index_list', target_index_list)

        for i in target_index_list:
            target_index = pos_neg_balanced.iloc[i, 1]
            label = pos_neg_balanced.iloc[i, 2]

            # target_expression = expression.iloc[target_index, :]

            # print(target_index)
            # samples_feature.append([tf_expression, target_expression, label])
            samples_pair.append([tf[0], target_index, label])
            samples_label.append(label)
    # samples_feature = pd.DataFrame(samples_feature)
    samples_pair = pd.DataFrame(samples_pair)
    samples_label = pd.DataFrame(samples_label)



    # print(samples.shape)
    # print(samples)

    return samples_pair,samples_label




def train_val_test_set(label_file,Gene_file,TF_file,train_set_file,val_set_file,test_set_file,GRN_type, dataset_type,p_val=0.5):
    print(GRN_type,dataset_type,p_val)
    # 读取靶基因文件中的靶基因index
    gene_set = pd.read_csv(Gene_file, index_col=0)['index'].values  # 779
    # 读取TF文件中的index，这里是全部TF，可以是基因列表与人类全部TF的交集，也可以是这个交集再加上已知关联对中的TF的并集，在数据预处理部分已经得到
    tf_set = pd.read_csv(TF_file, index_col=0)['index'].values  # 84
    # 读取已知关联文件 tf index，靶基因 index
    label = pd.read_csv(label_file, index_col=0)  #（2478,2）
    # 已知关联文件中出现的所有tf
    tf = label['TF'].values
    # 选出已知关联文件中的tf 列表，不重复
    tf_list = np.unique(tf)   # 14

    # 构建数据集的正样本。找到已知关联中所有的正样本，并存储到字典中，TF为键，靶基因为值
    # 这里生成的结果是没有自环的，所有基因对减少了一些自环的，从2478减少为2469
    pos_dict = {}
    for i in tf_list:
        pos_dict[i] = []
    for i, j in label.values:
        pos_dict[i].append(j)

    # 构建数据集的负样本
    neg_dict = {}
    if GRN_type == 'gene-gene':
        '''
        gene-gene中，负样本是所有基因对，只要除去已知基因对就行, 910*910-4545=823555
        这里是包括自环的
        '''
        gene_set_without_tf = list(set(gene_set)-set(tf_list))

        for i in gene_set:  # 这里的键设置成了所有的基因，因为GRN是没有限制的
            neg_dict[i] = []
        for i in gene_set_without_tf: # 对于没有在已知关联中作为TF的基因，这些基因都可以作为潜在的TF，其靶基因可以是任何基因，包括基因本身（因为加了自环）
            neg_dict[i] = list(range(0,len(gene_set)))
        for i in tf_list: # 对于已经在已知基因对中作为TF的基因，这些基因可以作为TF，其靶基因可以是除了正样本之外的所有基因
            temp_a = list(range(0, len(gene_set)))
            temp_b = pos_dict[i]
            neg_dict[i] = list(set(temp_a)-set(temp_b))
    elif GRN_type=='tf-gene':
        '''
        tf-gene中，正负样本都是TF出发，TF来自TF列表，负样本中每个TF的靶基因是基因列表中除该TF之外的所有基因，好像没有去掉其他TF？
        '''
        for i in tf_set:  # 所有 TF，hESC top 1000数据集中，TF数量为84
            neg_dict[i] = []
        # neg_dict有84个TF，找到所有未知样本
        for i in tf_set:
            if i in pos_dict.keys():
                pos_item = pos_dict[i]  # 取出来tf i的所有调控的靶基因，是一个list
                pos_item.append(i)  # 再append上 tf i自己
                # 去除正样本？？
                neg_item = np.setdiff1d(gene_set, pos_item)  # 找到2个数组中集合元素的差异，返回在第一个数组但不在第二个数组中的唯一值
                neg_dict[i].extend(neg_item)  # 在列表末尾一次性追加另一个序列中的多个值
                pos_dict[i] = np.setdiff1d(pos_dict[i], i)  # 去掉 i，这一步是为啥？？？

            else:
                neg_item = np.setdiff1d(gene_set, i)
                neg_dict[i].extend(neg_item)
    elif GRN_type=='tf-gene-posneg':
        '''
        tf-gene中，正负样本都是TF出发，TF来自正样本TF列表，负样本中每个TF的靶基因是基因列表中除该TF之外的所有基因，好像没有去掉其他TF？
        '''
        for i in tf_list:  # 所有 TF，hESC top 1000数据集中，TF数量为84
            neg_dict[i] = []
        # neg_dict有84个TF，找到所有未知样本
        for i in tf_list:
            if i in pos_dict.keys():
                pos_item = pos_dict[i]  # 取出来tf i的所有调控的靶基因，是一个list
                pos_item.append(i)  # 再append上 tf i自己
                # 去除正样本？？
                neg_item = np.setdiff1d(gene_set, pos_item)  # 找到2个数组中集合元素的差异，返回在第一个数组但不在第二个数组中的唯一值
                neg_dict[i].extend(neg_item)  # 在列表末尾一次性追加另一个序列中的多个值
                pos_dict[i] = np.setdiff1d(pos_dict[i], i)  # 去掉 i，这一步是为啥？？？

            else:
                print('negtive sample tf error')
                # neg_item = np.setdiff1d(gene_set, i)
                # neg_dict[i].extend(neg_item)
    else:
        print("error")
    # num_label0 = 0
    # for i in neg_dict:
    #     num_label0 += len(neg_dict[i])
    # print(num_label0)
    #
    # num_label1 = 0
    # for i in pos_dict:
    #     num_label1 += len(pos_dict[i])
    # print(num_label1)

    print("所有负样本构建完成")

    # 将数据集的正样本按照3 1 1 划分为训练集、验证集、测试集
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
            # 打乱所有正样本，3/5为训练集，1/5为测试集，1/5为验证集
            np.random.shuffle(pos_dict[k])
            train_pos[k] = pos_dict[k][:len(pos_dict[k]) * 3 // 5]
            val_pos[k] = pos_dict[k][len(pos_dict[k]) * 3 // 5:len(pos_dict[k]) * 4 // 5]
            test_pos[k] = pos_dict[k][len(pos_dict[k]) * 4 // 5:]

    # print("正样本划分完成")

    # 将数据集的负样本按照3 1 1 划分为训练集、验证集、测试集
    train_neg = {}
    val_neg = {}
    test_neg = {}
    if dataset_type == 'balanced':
        # 选择与正样本数量相同的样本作为负样本，也是4545个
        neg_dict_balanced = {}
        for i in range(label.shape[0]):
            key, value = get_random_kvpair(neg_dict)
            if key not in neg_dict_balanced:
                neg_dict_balanced[key] = [value]
            else:
                neg_dict_balanced[key].append(value)
        # 3 1 1划分
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
                # 打乱所有正样本，3/5为训练集，1/5为测试集，1/5为验证集
                np.random.shuffle(neg_dict_balanced[k])
                train_neg[k] = neg_dict_balanced[k][:len(neg_dict_balanced[k]) * 3 // 5]
                val_neg[k] = neg_dict_balanced[k][
                              len(neg_dict_balanced[k]) * 3 // 5:len(neg_dict_balanced[k]) * 4 // 5]
                test_neg[k] = neg_dict_balanced[k][len(neg_dict_balanced[k]) * 4 // 5:]

    elif dataset_type == 'unbalanced':
        # 不平衡数据集不需要选择负样本，所有负样本都用来训练测试
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

    # 保存
    save_datasets(train_pos, train_neg, train_set_file)
    save_datasets(val_pos, val_neg, val_set_file)
    save_datasets(test_pos, test_neg, test_set_file)

    print('traing set, validation set, and test set are saved successfully.')

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