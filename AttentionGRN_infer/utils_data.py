import csv
import numpy as np
import pandas as pd
import utils_gz, utils_GRN

def get_origin_expression_data(gene_expression_path):
    # return gene expression matrix, cells list, number of genes, number of cells
    f_expression = open(gene_expression_path, encoding="utf-8")
    expression_reader = list(csv.reader(f_expression))
    cells = expression_reader[0][1:]
    num_cells = len(cells)

    expression_record = {}
    num_genes = 0
    for single_expression_reader in expression_reader[1:]:
        if single_expression_reader[0] in expression_record:
            print('Gene name ' + single_expression_reader[0] + ' repeat!')
        print(single_expression_reader[0])
        expression_record[single_expression_reader[0]] = list(map(float, single_expression_reader[1:]))
        num_genes += 1
    print(str(num_genes) + ' genes and ' + str(num_cells) + ' cells are included in origin expression data.' + '\n')
    return expression_record, cells, num_genes, num_cells


def get_origin_expression_data_mESC(gene_expression_path):
    # return gene expression matrix, cells list, number of genes, number of cells
    exp = pd.read_csv(gene_expression_path).T
    cells = exp.columns[1:]
    gene_name = exp.iloc[:,0]
    expression_record = {}
    for i  in range(exp.shape[0]):
        row = exp.iloc[i, :].to_list()
        genename_row = row[0].upper()
        expression_row = row[1:]
        expression_record[genename_row] = expression_row

    return expression_record, cells, len(gene_name), len(cells)

def get_origin_expression_data_hHEP(gene_expression_path):
    # return gene expression matrix, cells list, number of genes, number of cells
    exp = pd.read_csv(gene_expression_path)
    gene_name = exp.columns[1:]
    cells = exp.iloc[:,0]
    expression_record = {}
    for i  in range(exp.shape[1]-1):
        row = exp.iloc[:, i+1].to_list()
        gene_name_row = gene_name[i]
        expression_record[gene_name_row] = row

    return expression_record, len(gene_name), len(cells)

def get_low_express_gene(origin_expression_record, num_cells):
    # get gene_list who were expressed in fewer than 10% of the cells
    gene_list = []
    threshold = num_cells // 10
    for gene in origin_expression_record:
        num = 0
        for expression in origin_expression_record[gene]:
            if expression != 0:
                num += 1
                if num > threshold:
                    break
        if num <= threshold:
            gene_list.append(gene)
    # up_gene_list = []
    # for gene in gene_list:
    #     up_gene_list.append(gene.upper())
    return gene_list


def get_gene_ranking(gene_order_path, low_express_gene_list, gene_num, flag):  # flag=True:write to output_path
    # 1.delete genes p-value>=0.01
    # 2.delete genes with low expression
    # 3.rank genes in descending order of variance
    # 4.return gene names list of top genes and variance_record of p-value<0.01
    f_order = open(gene_order_path)
    order_reader = list(csv.reader(f_order))
    # if flag:
    #     f_rank = open(output_path, 'w', newline='\n')
    #     f_rank_writer = csv.writer(f_rank)
    variance_record = {}
    variance_list = []
    significant_gene_list = []
    for single_order_reader in order_reader[1:]:
        # column 0:gene name
        # column 1:p value
        # column 2:variance
        if float(single_order_reader[1]) >= 0.01:
            break
            # continue
        if single_order_reader[0] in low_express_gene_list:
            continue
        variance = float(single_order_reader[2])
        if variance not in variance_record:  # 1 variance corresponding to 1 gene

            # print('variance', variance)
            variance_record[variance] = single_order_reader[0]
        else:
            # print(str(variance_record[variance]) + ' and ' + single_order_reader[0] + ' variance repeat!')
            # print(variance)
            variance_record[variance] = [variance_record[variance]]
            # print(variance_record[variance])
            variance_record[variance].append(single_order_reader[0])
            # print(variance_record[variance])

        variance_list.append(variance)
        tstr = single_order_reader[0]
        single_order_reader[0] = tstr.upper()
        significant_gene_list.append(single_order_reader[0])
    # print(variance_record[0.802089040840064])
    print('After delete genes with p-value>=0.01 or low expression, ' + str(len(variance_list)) + ' genes left.')

    variance_list.sort(reverse=True)


    gene_rank = []
    for single_variance_list in variance_list[0:gene_num]:

    #     print('single_variance_list',single_variance_list)
        if type(variance_record[single_variance_list]) is str:
            gene_rank.append(variance_record[single_variance_list])
        else:

            gene_rank.append(variance_record[single_variance_list][0])
            del variance_record[single_variance_list][0]
            if len(variance_record[single_variance_list]) == 1:
                variance_record[single_variance_list] = variance_record[single_variance_list][0]

    f_order.close()
    return gene_rank, significant_gene_list


def get_gene_ranking_2(gene_order_path, low_express_gene_list):  # flag=True:write to output_path
    # 1.delete genes p-value>=0.01
    # 2.delete genes with low expression
    # 3.rank genes in descending order of variance
    # 4.return gene names list of top genes and variance_record of p-value<0.01
    f_order = open(gene_order_path)
    order_reader = list(csv.reader(f_order))
    variance_record = {} #
    significant_gene_list = [] #
    for single_order_reader in order_reader[1:]:
        # column 0:gene name
        # column 1:p value
        # column 2:variance
        tstr = single_order_reader[0]
        single_order_reader[0] = tstr.upper()
        if float(single_order_reader[1]) >= 0.01:
            break  #
            # continue
        if single_order_reader[0].upper() in low_express_gene_list:
            continue
        variance = float(single_order_reader[2])
        if variance not in variance_record:  # 1 variance corresponding to 1 gene
            variance_record[variance] = single_order_reader[0].upper()
        else:
            # print(str(variance_record[variance]) + ' and ' + single_order_reader[0] + ' variance repeat!')
            # print(variance)
            variance_record[variance] = [variance_record[variance]]
            # print(variance_record[variance])
            variance_record[variance].append(single_order_reader[0].upper())
            # print(variance_record[variance])

        # variance_list.append(variance)

        significant_gene_list.append(single_order_reader[0])
    f_order.close()
    return significant_gene_list

def get_topN_gene(gene_order_path, significant_gene_list_no_TF, Rank_num, flag):  # flag=True:write to output_path


    f_order = open(gene_order_path)
    order_reader = list(csv.reader(f_order))
    # if flag:
    #     f_rank = open(output_path, 'w', newline='\n')
    #     f_rank_writer = csv.writer(f_rank)
    variance_record = {}
    variance_list = []
    significant_gene_list = []
    for single_order_reader in order_reader[1:]:
        # column 0:gene name
        # column 1:p value
        # column 2:variance
        gene_name = single_order_reader[0].upper()
        variance = float(single_order_reader[2])

        if gene_name in significant_gene_list_no_TF:
            variance_list.append(variance)
            if variance not in variance_record:
                variance_record[variance] = gene_name
            else:
                variance_record[variance] = [variance_record[variance]]
                variance_record[variance].append(single_order_reader[0])
    variance_list.sort(reverse=True)

    gene_rank = []
    for single_variance_list in variance_list[0:Rank_num]:

        if type(variance_record[single_variance_list]) is str:
            gene_rank.append(variance_record[single_variance_list])
        else:
            gene_rank.append(variance_record[single_variance_list][0])
            del variance_record[single_variance_list][0]
            if len(variance_record[single_variance_list]) == 1:
                variance_record[single_variance_list] = variance_record[single_variance_list][0]

    return gene_rank


def get_filtered_gold(gold_network_path, rank_list):
    # 1.Load origin gold file
    # 2.Delete genes not in rank_list
    # 3.return tf-targets dict and pair-score dict
    # Note: If no score in gold network, score=999
    f_gold = open(gold_network_path, encoding='UTF-8-sig')
    gold_reader = list(csv.reader(f_gold))
    num_origin_goldpair = len(gold_reader)-1

    for i in range(0, len(gold_reader) - 1):
        temp = gold_reader[i]
        s1 = str(temp[0])
        s2 = str(temp[1])

        temp[0] = s1.upper()
        temp[1] = s2.upper()

        gold_reader[i] = temp
    # print("gold_reader",gold_reader)
    # print("rank_list",rank_list)
    # print("gold_reader",gold_reader)
    # print("gold_reader[0]", gold_reader[0])
    has_score = True
    if len(gold_reader[0]) < 3:
        has_score = False

    gold_pair_record = {}
    gold_score_record = {}
    unique_gene_list = []
    #
    for single_gold_reader in gold_reader[1:]:
        # column 0: TF
        # column 1: target gene
        # column 2: regulate score


        if (single_gold_reader[0] not in rank_list) or (single_gold_reader[1] not in rank_list):
            continue
        gene_pair = [single_gold_reader[0], single_gold_reader[1]]
        str_gene_pair = single_gold_reader[0] + ',' + single_gold_reader[1]

        if single_gold_reader[0] not in unique_gene_list: unique_gene_list.append(single_gold_reader[0])
        if single_gold_reader[1] not in unique_gene_list: unique_gene_list.append(single_gold_reader[1])

        if str_gene_pair in gold_score_record:
            print('Gold pair repeat!')
        if has_score:
            # print("single_gold_reader[2]", single_gold_reader[2])
            gold_score_record[str_gene_pair] = float(single_gold_reader[2])
        else:
            gold_score_record[str_gene_pair] = 99999999
        if gene_pair[0] not in gold_pair_record:
            gold_pair_record[gene_pair[0]] = [gene_pair[1]]
        else:
            gold_pair_record[gene_pair[0]].append(gene_pair[1])
    '''
    At this time, the standard network gold_pair_record has been obtained, and the specific information is as follows:
    '''
    # print("gold_pair_record", gold_pair_record)
    # Some statistics of gold_network
    print(str(len(gold_pair_record)) + ' TFs and ' + str(
        len(gold_score_record)) + ' edges in gold_network consisted of genes in rank_list.')
    print(str(len(unique_gene_list)) + ' genes are common in rank_list and gold_network.')
    numTFs = len(gold_pair_record)
    num_gold_pair = len(gold_score_record)
    num_genes = len(unique_gene_list)

    # if flag:
    #     f_unique = open(output_path, 'w', encoding="utf-8", newline='\n')
    #     f_unique_writer = csv.writer(f_unique)
    #     out_unique = np.array(unique_gene_list).reshape(len(unique_gene_list), 1)
    #     f_unique_writer.writerows(out_unique)
    #     f_unique.close()
    return gold_pair_record, gold_score_record, unique_gene_list, numTFs,num_gold_pair,num_genes, num_origin_goldpair


def get_filtered_gold_3(gold_network_path, rank_list, all_gene, all_tfs_human_tf):
    # 1.Load origin gold file
    # 2.Delete genes not in rank_list
    # 3.return tf-targets dict and pair-score dict
    # Note: If no score in gold network, score=999
    f_gold = open(gold_network_path, encoding='UTF-8-sig')
    gold_reader = list(csv.reader(f_gold))
    num_origin_goldpair = len(gold_reader)

    for i in range(0, len(gold_reader) - 1):
        temp = gold_reader[i]
        s1 = str(temp[0])
        s2 = str(temp[1])

        temp[0] = s1.upper()
        temp[1] = s2.upper()

        gold_reader[i] = temp
    # print("gold_reader",gold_reader)
    # print("rank_list",rank_list)
    # print("gold_reader",gold_reader)
    # print("gold_reader[0]", gold_reader[0])
    has_score = True
    if len(gold_reader[0]) < 3:
        has_score = False

    gold_pair_record = {}
    gold_score_record = {}
    unique_gene_list = []
    #
    for single_gold_reader in gold_reader[1:]:
        # column 0: TF
        # column 1: target gene
        # column 2: regulate score


        if (single_gold_reader[0] not in all_tfs_human_tf) or (single_gold_reader[1] not in rank_list):
            continue
        gene_pair = [single_gold_reader[0], single_gold_reader[1]]
        str_gene_pair = single_gold_reader[0] + ',' + single_gold_reader[1]

        if single_gold_reader[0] not in unique_gene_list: unique_gene_list.append(single_gold_reader[0])
        if single_gold_reader[1] not in unique_gene_list: unique_gene_list.append(single_gold_reader[1])

        if str_gene_pair in gold_score_record:
            print('Gold pair repeat!')
        if has_score:
            # print("single_gold_reader[2]", single_gold_reader[2])
            gold_score_record[str_gene_pair] = float(single_gold_reader[2])
        else:
            gold_score_record[str_gene_pair] = 99999999

        if gene_pair[0] not in gold_pair_record:
            gold_pair_record[gene_pair[0]] = [gene_pair[1]]
        else:
            gold_pair_record[gene_pair[0]].append(gene_pair[1])


    print(str(len(gold_pair_record)) + ' TFs and ' + str(
        len(gold_score_record)) + ' edges in gold_network consisted of genes in rank_list.')
    print(str(len(unique_gene_list)) + ' genes are common in rank_list and gold_network.')
    numTFs = len(gold_pair_record)
    num_gold_pair = len(gold_score_record)
    num_genes = len(unique_gene_list)

    return gold_pair_record, gold_score_record, unique_gene_list, numTFs,num_gold_pair,num_genes, num_origin_goldpair


def generate_filtered_gold(gold_pair_record, gold_score_record, output_path):
    # write filtered_gold to output_path
    # print("cnm")
    f_filtered = open(output_path, 'w', encoding="utf-8", newline='\n')
    f_filtered_writer = csv.writer(f_filtered)
    f_filtered_writer.writerow(['TF', 'Target', 'Score'])
    # print("cnm")
    for tf in gold_pair_record:
        once_output = []
        for target in gold_pair_record[tf]:
            # print(tf,target)
            single_output = [tf, target, gold_score_record[tf + ',' + target]]
            once_output.append(single_output)
        f_filtered_writer.writerows(once_output)
    f_filtered.close()


def generate_filtered_gold_2(gold_pair_record, output_path):
    # write filtered_gold to output_path
    f_filtered = open(output_path, 'w', encoding="utf-8", newline='\n')
    f_filtered_writer = csv.writer(f_filtered)
    f_filtered_writer.writerow(['TF', 'Target'])
    for tf in gold_pair_record:
        once_output = []
        for target in gold_pair_record[tf]:
            single_output = [tf, target]
            once_output.append(single_output)
        f_filtered_writer.writerows(once_output)
    f_filtered.close()

def save_expression(final_expression_matrix, output_path_exp, output_path_name):
    # write filtered_gold to output_path
    # print("cnm")
    gene_names = final_expression_matrix.keys()
    gene_index = list(range(len(gene_names)))
    gene_names_index = [gene_names,gene_index]
    gene_names_index = pd.DataFrame(gene_names_index).T
    gene_names_index.columns = ['gene', 'index']
    pd.DataFrame(gene_names_index).to_csv(output_path_name)

    pd.DataFrame(final_expression_matrix).to_csv(output_path_exp)


def get_normalized_expression_data(gene_expression_matirx):
    # expression_record, cells = get_origin_expression_data(gene_expression_path)
    gene_names = gene_expression_matirx.keys()
    Expression_matrix = {}
    # Expression_matrix = np.zeros((len(gene_expression_matirx), len(gene_expression_matirx[0])))
    index_row = 0
    for gene in gene_expression_matirx:
        gene_expression_matirx[gene] = np.log10(np.array(gene_expression_matirx[gene]) + 10 ** -2)
        Expression_matrix[gene] = gene_expression_matirx[gene]
        index_row += 1

    # Heat map
    # plt.figure(figsize=(15,15))
    # sns.heatmap(expression_matrix[0:100,0:100])
    # plt.show()

    return Expression_matrix


def dataset_Info(final_nums, data_sorce):
    head = []
    infos = pd.DataFrame()
    indexss = pd.DataFrame()
    dataset_names = ['hESC', 'hHEP', 'mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L'] #
    for dataset_name in dataset_names:
        network_types = []
        if dataset_name == 'hESC':
            network_types = ['hESC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']

        elif dataset_name == 'hHEP':
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
        for network_type in network_types:
            for Rank_num in final_nums:
                head.append(dataset_name + '/' + network_type + '/' + str(Rank_num))

                path_info = data_sorce + dataset_name + '\\' + network_type + '\\Top' + str(
                    Rank_num) + '\\' + 'Info.csv'
                # path_info = data_path + 'Info.csv'

                a = pd.read_csv(path_info)
                infos = pd.concat([infos, a.iloc[:, 1]], axis=1)

                indexss = a.iloc[:, 0]
                # print(a.iloc[:,1])
    head = pd.DataFrame(head)
    # print(head)
    # print(infos)
    infos.columns = head
    infos.index = indexss
    infos.to_csv(data_sorce  + 'all_dataset_info.csv',encoding="utf_8_sig")
    # head = []


def data_transform(path_gene_list,path_TF_list, path_gene_pair_list, save_TF_list, save_gene_pair_index_list):

    gene_list = pd.read_csv(path_gene_list)
    # print(gene_list.shape)
    # gene_name = gene_list.iloc[:,1]
    # gene_index = gene_list.iloc[:,2]
    # print(gene_name)

    TF_list = pd.read_csv(path_TF_list)
    TF_name = TF_list.iloc[:,1]
    # print(TF_name)

    TF_index = []
    for i in TF_name:
        a = gene_list[gene_list.gene == i].index.tolist()
        TF_index.extend(a)
    # print(TF_index)
    # print(len(TF_index))

    TF_name_index = [TF_name.tolist(),TF_index]
    TF_name_index = pd.DataFrame(TF_name_index).T
    TF_name_index.columns = ['TF','index']
    # print(TF_name_index.shape)
    TF_name_index.to_csv(save_TF_list)

    gene_pair_name = pd.read_csv(path_gene_pair_list)
    TF_pair = gene_pair_name.iloc[:,0]
    target_pair = gene_pair_name.iloc[:,1]
    # print(TF_pair.shape)
    TF_pair_index = []
    for i in TF_pair:
        a = gene_list[gene_list.gene == i].index.tolist()
        TF_pair_index.extend(a)

    target_pair_index = []
    for i in target_pair:
        a = gene_list[gene_list.gene == i].index.tolist()
        target_pair_index.extend(a)
    TF_target_index = [TF_pair_index,target_pair_index]
    TF_target_index = pd.DataFrame(TF_target_index).T
    TF_target_index.columns = ['TF','target']
    # print(TF_target_index.shape)
    TF_target_index.to_csv(save_gene_pair_index_list)



def data_transform_2(path_gene_list, path_gene_pair_list, save_gene_pair_index_list):

    gene_list = pd.read_csv(path_gene_list)
    # print(gene_list.shape)
    # gene_name = gene_list.iloc[:,1]
    # gene_index = gene_list.iloc[:,2]
    # print(gene_name)

    gene_pair_name = pd.read_csv(path_gene_pair_list)
    TF_pair = gene_pair_name.iloc[:,0]
    target_pair = gene_pair_name.iloc[:,1]
    # print(TF_pair.shape)
    TF_pair_index = []
    for i in TF_pair:
        a = gene_list[gene_list.gene == i].index.tolist()
        TF_pair_index.extend(a)

    target_pair_index = []
    for i in target_pair:
        a = gene_list[gene_list.gene == i].index.tolist()
        target_pair_index.extend(a)
    TF_target_index = [TF_pair_index,target_pair_index]
    TF_target_index = pd.DataFrame(TF_target_index).T
    TF_target_index.columns = ['TF','target']
    # print(TF_target_index.shape)
    TF_target_index.to_csv(save_gene_pair_index_list)





def get_nodeid(target_file, save_path):
    node = pd.read_csv(target_file)
    # print(node[1])
    # print(node.iloc[:,1])
    name = node.iloc[:,1]
    ids = node.iloc[:,0]
    data = {}
    for i in range(len(name)):
        data[name[i]] = ids[i]
    # print(data)

    file = open(save_path+ 'gene_list_id.txt', 'w')

    for k, v in data.items():
        # print(k,v)
        file.write(str(k) + ' ' + str(v) + '\n')

    file.close()



def preprocess_DATA1(result_path, gold_network_path, gene_expression_path, dataset_name, gene_order_path, Rank_num, pathHumanTF):
    # save dataset
    final_GRN_pos_file_name = result_path + 'Final_GRNorTRN_pos.csv' # prior GRN
    save_gene_pair_index_list = result_path + 'Final_GRNorTRN_pos_index.csv'  # prior GRN
    final_expression_file_name = result_path + 'Final_expression.csv'
    final_expression_gene_name = result_path + 'Final_gene_list.csv'
    final_TF_common_file_name = result_path + 'Final_TF_common.csv'
    save_TF_index_list = result_path + 'Final_TF_common_index.csv'

    Info_file_name = result_path + 'Info.csv'

    Info = {}

    # get the raw gene expression matrix, cell list, gene number, cell number
    if dataset_name=='mESC':
        origin_expression_record, cells, num_genes, num_cells = get_origin_expression_data_mESC(gene_expression_path)
    else:
        origin_expression_record, cells, num_genes, num_cells = get_origin_expression_data(gene_expression_path)

    Info["original genes"] = [num_genes]
    Info["original cells"] = [num_cells]

    # get the list of genes that are expressed in less than 10% of cells
    low_express_gene_list = get_low_express_gene(origin_expression_record, num_cells)

    print(str(len(low_express_gene_list)) + ' genes in low expression.'+ '\n')
    # Info["low expression genes"] = [len(low_express_gene_list)]


    # get the significant genes, significant genes are genes with p value less than 0.01, and expressed more than 10% cells, highly variable top Rank_num
    print('\n' + "The top 500/1000 genes were selected in order of variance..........")
    rank_list, significant_gene_list = get_gene_ranking(gene_order_path, low_express_gene_list, Rank_num, False)
    # significant genes is genes with p value less than 0.01, and expressed more than 10% cells
    Info["significant genes"] = [len(significant_gene_list)]
    if Rank_num < len(significant_gene_list):
        # ranknum = Rank_num

        for i in range(0, len(rank_list) - 1):
            tstr = str(rank_list[i])
            tstr = tstr.upper()
            rank_list[i] = tstr

        # Based on the original ground-truth network, the ground-truth network with N=500/1000 genes was retrieved
        print('\n' + 'Find the prior network gold_pair_record, gold_score_record of top N genes, and the list of genes after intersection...............')
        gold_pair_record, gold_score_record, unique_gene_list, numTFs,num_gold_pair,num_genes,num_origin_gold_pair = get_filtered_gold(gold_network_path, rank_list)
        Info["original GRN"] = [num_origin_gold_pair]
        Info["filtered GRN"] = [num_gold_pair]
        Info["genes in filtered GRN"] = [num_genes]

        # utils.generate_filtered_gold(gold_pair_record, gold_score_record, FilteredGRN_file_name)

        # Obtain the normalized gene expression matrix according to the new gene list
        label_list = []
        pair_list = []
        total_matrix = []
        num_tf = -1
        num_label1 = 0
        num_label0 = 0
        # miss = 0
        flagFalse = [] # tf and target gene names that were not in the gene expression matrix
        all_gene_name = []
        all_TF_name_gold_pair = []
        final_gold_pair_record_pos = {}
        final_gold_pair_record_neg = {}
        pair_list_name_tf = []
        pair_list_name_target = []

        for i in gold_pair_record: # positve samples, key: TF, value: target genes regulated by this TF
            num_tf += 1
            for j in range(len(unique_gene_list)):
                print('Generating matrix of gene pair ' + str(num_tf) + ' ' + str(j))
                tf_name = i
                target_name = unique_gene_list[j]

                flag = False
                # If the gene expression matrix contains TF and target genes, flag is set to true
                if (origin_expression_record.__contains__(tf_name) & origin_expression_record.__contains__(target_name)):
                    flag = True
                else: # do not have gene expression
                    flagFalse.extend([tf_name,target_name])
                    # print("Do not have gene expression: " + tf_name +'--'+ target_name)
                if (flag):
                    # If the TF is in the prior GRN (gold_pair_record) and the target gene is also regulated by this TF, the label is set to 1, otherwise the label is 0
                    if tf_name in gold_pair_record and target_name in gold_pair_record[tf_name]:
                        label = 1
                        num_label1 += 1
                        if tf_name not in final_gold_pair_record_pos:
                            final_gold_pair_record_pos[tf_name] = [target_name]
                        else:
                            final_gold_pair_record_pos[tf_name].append(target_name)
                    else:
                        label = 0
                        num_label0 += 1
                        if tf_name not in final_gold_pair_record_neg:
                            final_gold_pair_record_neg[tf_name] = [target_name]
                        else:
                            final_gold_pair_record_neg[tf_name].append(target_name)

                    all_TF_name_gold_pair.append(tf_name)
                    all_gene_name.append(tf_name)
                    all_gene_name.append(target_name)
                else:
                    # miss = miss + 1
                    continue
        # save prior GRN
        generate_filtered_gold_2(final_gold_pair_record_pos, final_GRN_pos_file_name)
        # utils.generate_filtered_gold_2(final_gold_pair_record_neg, final_GRN_neg_file_name)
        Info["final GRN "] = [num_label1]

        new_all_gene_name=list(set(all_gene_name))
        Info["final genes "] = [len(new_all_gene_name)]
        new_all_gene_name.sort(key=all_gene_name.index)
        # print(len(all_gene_name))
        # print(len(new_all_gene_name))

        new_all_TF_gold_pair = list(set(all_TF_name_gold_pair))
        Info["all TFs from gold pair"] = [len(new_all_TF_gold_pair)]
        # pd.DataFrame(new_all_TF_gold_pair).to_csv(final_TF_goldpair_file_name)


        human_tf = pd.read_csv(pathHumanTF)
        human_tfs = human_tf['TF'].to_list()
        all_tfs_human_tf = []
        for gene in new_all_gene_name:
            if gene in human_tfs:
                all_tfs_human_tf.append(gene)

        # pd.DataFrame(all_tfs_human_tf).to_csv(final_TF_human_file_name)
        Info["all TFs from human tf"] = [len(all_tfs_human_tf)]

        common_TFs = []
        for tf in new_all_TF_gold_pair:
            if tf not in common_TFs:
                common_TFs.append(tf)
        for tf in all_tfs_human_tf:
            if tf not in common_TFs:
                common_TFs.append(tf)
        pd.DataFrame(common_TFs).to_csv(final_TF_common_file_name)
        Info['common TFs'] = [len(common_TFs)]

        # save gene expression matrix
        new_expression_matrix = {}
        for i in origin_expression_record:
            if i in new_all_gene_name:
                new_expression_matrix[i] = origin_expression_record[i]

        # Normalization of gene expression data
        final_expression_matrix = get_normalized_expression_data(new_expression_matrix)
        save_expression(final_expression_matrix, final_expression_file_name,final_expression_gene_name)

        Info = pd.DataFrame(Info).T
        Info.to_csv(Info_file_name)

        # add index for TF and gene
        data_transform(final_expression_gene_name, final_TF_common_file_name, final_GRN_pos_file_name,
                       save_TF_index_list, save_gene_pair_index_list)


    else:
        print('ranknum is: ' + str(Rank_num) + ' but the number of all gene is ' + str(len(significant_gene_list)))
        # break
        ranknum = 0




def preprocess_DATA_AttentionGRN(result_path, gold_network_path, gene_expression_path, dataset_name, gene_order_path, Rank_num, pathHumanTF):
    # save dataset
    final_GRN_pos_file_name = result_path + 'Final_GRNorTRN_pos.csv' # prior GRN
    save_gene_pair_index_list = result_path + 'Final_GRNorTRN_pos_index.csv'  # prior GRN
    final_expression_file_name = result_path + 'Final_expression.csv'
    final_expression_gene_name = result_path + 'Final_gene_list.csv'
    final_TF_common_file_name = result_path + 'Final_TF_common.csv'
    save_TF_index_list = result_path + 'Final_TF_common_index.csv'

    Info_file_name = result_path + 'Info.csv'

    Info = {}

    # get the raw gene expression matrix, cell list, gene number, cell number
    if dataset_name=='mESC':
        origin_expression_record, cells, num_genes, num_cells = get_origin_expression_data_mESC(gene_expression_path)
    else:
        origin_expression_record, cells, num_genes, num_cells = get_origin_expression_data(gene_expression_path)

    Info["original genes"] = [num_genes]
    Info["original cells"] = [num_cells]

    # get the list of genes that are expressed in less than 10% of cells
    low_express_gene_list = get_low_express_gene(origin_expression_record, num_cells)

    print(str(len(low_express_gene_list)) + ' genes in low expression.'+ '\n')
    # Info["low expression genes"] = [len(low_express_gene_list)]

    # get the significant genes, significant genes are genes with p value less than 0.01, and expressed more than 10% cells, highly variable top Rank_num
    print('\n' + "The top 500/1000 genes were selected in order of variance..........")
    rank_list, significant_gene_list = get_gene_ranking(gene_order_path, low_express_gene_list, Rank_num, False)
    # significant genes is genes with p value less than 0.01, and expressed more than 10% cells
    Info["significant genes"] = [len(significant_gene_list)]
    if Rank_num < len(significant_gene_list):
        # ranknum = Rank_num

        for i in range(0, len(rank_list) - 1):
            tstr = str(rank_list[i])
            tstr = tstr.upper()
            rank_list[i] = tstr

        # Based on the original ground-truth network, the ground-truth network with N=500/1000 genes was retrieved
        print('\n' + 'Find the prior network gold_pair_record, gold_score_record of top N genes, and the list of genes after intersection...............')
        gold_pair_record, gold_score_record, unique_gene_list, numTFs,num_gold_pair,num_genes,num_origin_gold_pair = get_filtered_gold(gold_network_path, rank_list)
        Info["original GRN"] = [num_origin_gold_pair]
        Info["filtered GRN"] = [num_gold_pair]
        Info["genes in filtered GRN"] = [num_genes]

        # utils.generate_filtered_gold(gold_pair_record, gold_score_record, FilteredGRN_file_name)

        # Obtain the normalized gene expression matrix according to the new gene list
        num_tf = -1
        num_label1 = 0
        num_label0 = 0
        # miss = 0
        flagFalse = [] # tf and target gene names that were not in the gene expression matrix
        all_gene_name = []
        all_TF_name_gold_pair = []
        final_gold_pair_record_pos = {}
        final_gold_pair_record_neg = {}
        pair_list_name_tf = []
        pair_list_name_target = []

        for i in gold_pair_record: # positve samples, key: TF, value: target genes regulated by this TF
            num_tf += 1
            for j in range(len(unique_gene_list)):
                print('Generating matrix of gene pair ' + str(num_tf) + ' ' + str(j))
                tf_name = i
                target_name = unique_gene_list[j]

                flag = False
                # If the gene expression matrix contains TF and target genes, flag is set to true
                if (origin_expression_record.__contains__(tf_name) & origin_expression_record.__contains__(target_name)):
                    flag = True
                else: # do not have gene expression
                    flagFalse.extend([tf_name,target_name])
                    # print("Do not have gene expression: " + tf_name +'--'+ target_name)
                if (flag):

                    all_TF_name_gold_pair.append(tf_name)
                    all_gene_name.append(tf_name)
                    all_gene_name.append(target_name)
                else:
                    # miss = miss + 1
                    continue

        new_all_gene_name=list(set(all_gene_name))
        Info["final genes "] = [len(new_all_gene_name)]
        new_all_gene_name.sort(key=all_gene_name.index)
        # print(len(all_gene_name))
        # print(len(new_all_gene_name))

        new_all_TF_gold_pair = list(set(all_TF_name_gold_pair))
        Info["all TFs from gold pair"] = [len(new_all_TF_gold_pair)]
        # pd.DataFrame(new_all_TF_gold_pair).to_csv(final_TF_goldpair_file_name)


        human_tf = pd.read_csv(pathHumanTF)
        human_tfs = human_tf['TF'].to_list()
        all_tfs_human_tf = []
        for gene in new_all_gene_name:
            if gene in human_tfs:
                all_tfs_human_tf.append(gene)

        # pd.DataFrame(all_tfs_human_tf).to_csv(final_TF_human_file_name)
        Info["all TFs from human tf"] = [len(all_tfs_human_tf)]

        common_TFs = []
        for tf in new_all_TF_gold_pair:
            if tf not in common_TFs:
                common_TFs.append(tf)
        for tf in all_tfs_human_tf:
            if tf not in common_TFs:
                common_TFs.append(tf)
        pd.DataFrame(common_TFs).to_csv(final_TF_common_file_name)
        Info['common TFs'] = [len(common_TFs)]

        # # for AttentionGRN
        label_list = []
        pair_list = []
        pair_list_name_tf = []
        pair_list_name_target = []
        # pair_list_id_pos_tf = []
        # pair_list_id_pos_target = []
        # pair_list_id_neg_tf = []
        # pair_list_id_neg_target = []
        total_matrix = []
        for i in gold_pair_record:
            num_tf += 1
            for j in range(len(new_all_gene_name)):
                # print('Generating matrix of gene pair ' + str(num_tf) + ' ' + str(j))
                tf_name = i
                target_name = new_all_gene_name[j]

                flag = False
                # if TF and target gene both have gene expression data, flag is True
                if (origin_expression_record.__contains__(tf_name) & origin_expression_record.__contains__(
                        target_name)) & (tf_name != target_name):
                    flag = True
                else:  # BCL11AC3ORF14
                    flagFalse.extend([tf_name, target_name])
                    # print("flag is flase: " + tf_name +'--'+ target_name)
                if (flag):
                    # If TF is in gold_pair_record and the target gene is also regulated by this TF, the label is set to 1, otherwise the label is 0
                    if tf_name in gold_pair_record and target_name in gold_pair_record[tf_name]:
                        label = 1
                        num_label1 += 1
                        # pair_list_id_pos_tf.append(tf_name)
                        # pair_list_id_pos_target.append(target_name)
                        if tf_name not in final_gold_pair_record_pos:
                            final_gold_pair_record_pos[tf_name] = [target_name]
                        else:
                            final_gold_pair_record_pos[tf_name].append(target_name)
                    else:
                        label = 0
                        num_label0 += 1
                        # pair_list_id_neg_tf.append(tf_name)
                        # pair_list_id_neg_target.append(target_name)
                        if tf_name not in final_gold_pair_record_neg:
                            final_gold_pair_record_neg[tf_name] = [target_name]
                        else:
                            final_gold_pair_record_neg[tf_name].append(target_name)
                    tf_data = origin_expression_record[tf_name]
                    target_data = origin_expression_record[target_name]
                    label_list.append(label)
                    pair_list.append(tf_name + ',' + target_name)
                    # pair_list_id_pos_tf.append()
                    pair_list_name_tf.append(tf_name)
                    pair_list_name_target.append(target_name)

                    all_TF_name_gold_pair.append(tf_name)
                    all_gene_name.append(tf_name)
                    all_gene_name.append(target_name)
                else:
                    # miss = miss + 1
                    continue

                single_tf_list = []
                gap = 100
                for k in range(0, len(tf_data), gap):
                    feature = []
                    a = tf_data[k:k + gap]
                    b = target_data[k:k + gap]
                    feature.extend(a)
                    feature.extend(b)
                    # single_tf_list.append(feature)
                    feature = np.asarray(feature)
                    # print("feature.shape", feature.shape)
                    if (len(feature) == 2 * gap):
                        # print("feature.shape xixihaha", feature.shape)
                        single_tf_list.append(feature)

                single_tf_list = np.asarray(single_tf_list)  # (7,200)

                total_matrix.append(single_tf_list)
        # print('label number')
        # print(num_label1)
        # print(num_label0)
        total_matrix = np.asarray(total_matrix)
        label_list = np.array(label_list)
        # pair_list = np.array(pair_list)

        np.save(result_path + 'matrix.npy', total_matrix)
        np.save(result_path + 'label.npy', label_list)
        np.save(result_path + 'gene_pair.npy', pair_list)

        # save prior GRN
        generate_filtered_gold_2(final_gold_pair_record_pos, final_GRN_pos_file_name)
        # utils.generate_filtered_gold_2(final_gold_pair_record_neg, final_GRN_neg_file_name)
        Info["final GRN "] = [num_label1]

        # save gene expression matrix
        new_expression_matrix = {}
        for i in origin_expression_record:
            if i in new_all_gene_name:
                new_expression_matrix[i] = origin_expression_record[i]

        # Normalization of gene expression data
        final_expression_matrix = get_normalized_expression_data(new_expression_matrix)
        save_expression(final_expression_matrix, final_expression_file_name,final_expression_gene_name)

        Info = pd.DataFrame(Info).T
        Info.to_csv(Info_file_name)

        # add index for TF and gene
        data_transform(final_expression_gene_name, final_TF_common_file_name, final_GRN_pos_file_name,
                       save_TF_index_list, save_gene_pair_index_list)

        # 把 pair_list_name_tf 转换成 id
        gene_list = pd.read_csv(final_expression_gene_name)
        pair_list_id_tf = []
        for i in pair_list_name_tf:
            a = gene_list[gene_list.gene == i].index.tolist()
            pair_list_id_tf.extend(a)

        # gene_list = pd.read_csv(final_expression_gene_name)
        pair_list_id_target = []
        for i in pair_list_name_target:
            a = gene_list[gene_list.gene == i].index.tolist()
            pair_list_id_target.extend(a)

        np.save(result_path + 'pair_list_id_tf.npy', pair_list_id_tf)
        np.save(result_path + 'pair_list_id_target.npy', pair_list_id_target)


    else:
        print('ranknum is: ' + str(Rank_num) + ' but the number of all gene is ' + str(len(significant_gene_list)))
        # break
        ranknum = 0




def preprocess_DATA2_step1(result_path, dataset_name, gene_expression_path, gene_order_path, pathHumanTF, Rank_num, gold_network_path):
    final_GRN_pos_file_name = result_path + 'Final_GRNorTRN_pos.csv'
    final_expression_file_name = result_path + 'Final_expression.csv'
    final_expression_gene_name = result_path + 'Final_gene_list.csv'
    # final_TF_common_file_name = result_path + 'Final_TF_common_index.csv'
    save_TF_index_list = result_path + 'Final_TF_index.csv'
    save_gene_pair_index_list = result_path + 'Final_GRNorTRN_pos_index.csv'

    Info_file_name = result_path + 'Info.csv'

    Info = {}

    # get original gene expression matrix, cell list, cell number, gene number
    if dataset_name=='mESC':
        origin_expression_record, cells, num_genes, num_cells = get_origin_expression_data_mESC(
            gene_expression_path)
    else:
        origin_expression_record, cells, num_genes, num_cells = get_origin_expression_data(gene_expression_path)

    Info["original gene"] = [num_genes]
    Info["original cell"] = [num_cells]

    # get the list of genes that are expressed in less than 10% of cells
    low_express_gene_list_ = get_low_express_gene(origin_expression_record, num_cells)

    low_express_gene_list = []
    for lg in low_express_gene_list_:
        low_express_gene_list.append(lg.upper())

    print(str(len(low_express_gene_list)) + ' genes in low expression.'+ '\n')


    # get the significant genes, significant genes are genes with p value less than 0.01, and expressed more than 10% cells, highly variable top Rank_num
    significant_gene_list = get_gene_ranking_2(gene_order_path, low_express_gene_list)
    # significant genes is genes with p value less than 0.01, and expressed more than 10% cells
    Info["significant genes "] = [len(significant_gene_list)]
    print("significant genes: " + str(len(significant_gene_list)))

    # Get all TFs highly varying
    human_tf = pd.read_csv(pathHumanTF)
    human_tfs_ = human_tf['TF'].to_list()
    human_tfs = []
    for tf_ in human_tfs_:
        human_tfs.append(tf_.upper())

    all_tfs_human_tf = []
    for gene in significant_gene_list:
        if gene in human_tfs:
            all_tfs_human_tf.append(gene)

    # pd.DataFrame(all_tfs_human_tf).to_csv(final_TF_human_file_name)
    Info["all TFs highly varying "] = [len(all_tfs_human_tf)]
    print("all TFs highly varying: " + str(len(all_tfs_human_tf)))


    significant_gene_list_no_TF = list(set(significant_gene_list)-set(all_tfs_human_tf))
    Info["significant genes after removing TF"] = [len(significant_gene_list_no_TF)]
    print("significant genes after removing TF：" + str(len(significant_gene_list_no_TF)))


    # Find the variance of these genes, sort them in descending order, and select top N
    rank_list = get_topN_gene(gene_order_path, significant_gene_list_no_TF, Rank_num, False)

    print("top N additional genes were selected： " + str(len(rank_list)))

    if Rank_num < len(significant_gene_list):

        # Merge the TF list with the top N genes
        all_gene = []
        for i in all_tfs_human_tf:
            if i not in all_gene:
                all_gene.append(i)
        for i in rank_list:
            if i not in all_gene:
                all_gene.append(i)
        print("The number of TF combined with additional top N genes is： " + str(len(all_gene)))
        Info["all TF and genes"] = [len(all_gene)]

        # According to the original ground-truth network, the ground-truth network of all the above genes was retrieved
        print('\n' + 'According to the original ground-truth network, the ground-truth network of all the above genes was retrieved..............')
        gold_pair_record, gold_score_record, unique_gene_list, numTFs,num_gold_pair,num_genes,num_origin_gold_pair = get_filtered_gold_3(gold_network_path, rank_list, all_gene, all_tfs_human_tf)
        Info["original GRN"] = [num_origin_gold_pair]
        Info["filtered GRN"] = [num_gold_pair]
        Info["filtered gene"] = [num_genes]
        Info["filtered TF"] = [numTFs]


        all_genes = pd.DataFrame(unique_gene_list)
        all_genes_index = list(range(len(all_genes)))
        all_gene_name_index = pd.concat([all_genes, pd.DataFrame(all_genes_index)], axis=1)
        all_gene_name_index.columns = ['gene', 'index']
        all_gene_name_index.to_csv(final_expression_gene_name)

        # save TF
        TF_names = pd.DataFrame(gold_pair_record.keys())
        TF_index = []
        TF_name = []
        for i in TF_names.values.tolist():
            tf_name = i[0]
            TF_name.append(tf_name)
            # a = all_gene_name_index[all_gene_name_index.gene == tf_name].index.tolist()

            a = all_gene_name_index[all_gene_name_index['gene'] == tf_name].index.to_list()
            # print(a)
            TF_index.extend(a)

        TF_name_index = [TF_name, TF_index]
        TF_name_index = pd.DataFrame(TF_name_index).T
        TF_name_index.columns = ['TF', 'index']
        TF_name_index.to_csv(save_TF_index_list)

        # The gene expression matrix was obtained and standardized according to the new gene list
        label_list = []
        pair_list = []
        total_matrix = []
        num_label1 = 0
        num_label0 = 0
        # miss = 0
        flagFalse = []
        all_gene_name = []
        all_TF_name_gold_pair = []
        final_gold_pair_record_pos = {}
        final_gold_pair_record_neg = {}

        for i in gold_pair_record: # positive samples, key: TF, value: target genes regulated by this TF
            for j in range(len(unique_gene_list)):
                tf_name = i
                target_name = unique_gene_list[j]

                flag = False

                if (origin_expression_record.__contains__(tf_name) & origin_expression_record.__contains__(target_name)):
                    flag = True
                else:
                    flagFalse.extend([tf_name,target_name])
                    # print("flag is flase: " + tf_name +'--'+ target_name)
                if (flag):
                    # If TF is in gold_pair_record and the target gene is also regulated by this TF, the label is set to 1, otherwise the label is 0
                    if tf_name in gold_pair_record and target_name in gold_pair_record[tf_name]:
                        label = 1
                        num_label1 += 1
                        if tf_name not in final_gold_pair_record_pos:
                            final_gold_pair_record_pos[tf_name] = [target_name]
                        else:
                            final_gold_pair_record_pos[tf_name].append(target_name)
                    else:
                        label = 0
                        num_label0 += 1
                        if tf_name not in final_gold_pair_record_neg:
                            final_gold_pair_record_neg[tf_name] = [target_name]
                        else:
                            final_gold_pair_record_neg[tf_name].append(target_name)

                    all_TF_name_gold_pair.append(tf_name)
                    all_gene_name.append(tf_name)
                    all_gene_name.append(target_name)
                else:
                    # miss = miss + 1
                    continue
        # save the final known gene pair
        generate_filtered_gold_2(final_gold_pair_record_pos, final_GRN_pos_file_name)
        Info["final GRN "] = [num_label1]
        # get the final gene expression matri
        new_all_gene_name=list(set(all_gene_name))
        Info["final genes"] = [len(new_all_gene_name)]
        new_all_gene_name.sort(key=all_gene_name.index)

        # all TFs in the final prior GRN
        new_all_TF_gold_pair = list(set(all_TF_name_gold_pair))
        Info["TFs from known GRN"] = [len(new_all_TF_gold_pair)]


        new_expression_matrix = {}
        for i in origin_expression_record:
            # if i in new_all_gene_name:
            if i in unique_gene_list:
                new_expression_matrix[i] = origin_expression_record[i]


        final_expression_matrix = get_normalized_expression_data(new_expression_matrix)
        save_expression(final_expression_matrix, final_expression_file_name,final_expression_gene_name)

        Info = pd.DataFrame(Info).T
        Info.to_csv(Info_file_name)

        data_transform_2(final_expression_gene_name, final_GRN_pos_file_name, save_gene_pair_index_list)

    else:
        print('ranknum is: ' + str(Rank_num) + ' but the number of all gene is ' + str(len(significant_gene_list)))
        # break


def preprocess_DATA2_step2(data_path):

    target_file = data_path + 'Final_gene_list.csv'
    label_file = data_path + 'Final_GRNorTRN_pos_index.csv'


    gene_set = pd.read_csv(target_file, index_col=0)['index'].values

    label = pd.read_csv(label_file, index_col=0)
    # label_name = pd.read_csv(label_name_file, index_col=0)

    tf = label['TF'].values
    tf_list = np.unique(tf)

    # positive samples
    pos_dict = {}
    for i in tf_list:
        pos_dict[i] = []
    for i, j in label.values:
        pos_dict[i].append(j)

    pos_list = []
    for tf in pos_dict.keys():
        all_target = pos_dict[tf]
        for target in all_target:
            pos_list.append([tf, target])
    # print(len(pos_list))

    # Find a negative sample for each positive sample and group it together
    neg_dict = {}
    neg_list = []
    pos_neg_balanced_id = []  # tf id-- gene id--label

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
            if len(signal_gene_set) == 0:
                break
        # print(pos_tf, pos_target, '1')
        # print(pos_tf, neg_target, '0')

        neg_list.append([pos_tf, neg_target])
        neg_dict[pos_tf].append(neg_target)
        # print('pos: ' + str(pos_tf) + '--' + str(pos_target))
        # print('neg: ' + str(pos_tf) + '--' + str(neg_target))
        # print('\n')
        # positive sample
        pos_neg_balanced_id.append([pos_tf, pos_target, 1])
        # negative sample
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
    pos_neg_balanced_name.to_csv(data_path + 'pos_neg_balanced_name.csv', sep=',', header=0, index=0)

    # return tf_list, pos_neg_balanced_id
    tf_list_save = pd.DataFrame(tf_list)
    tf_list_save.to_csv(data_path + "tf_list.csv", sep=',', header=0, index=0)

    pos_neg_balanced_id = pd.DataFrame(pos_neg_balanced_id)
    pos_neg_balanced_id.to_csv(data_path + 'pos_neg_balanced_id.csv', sep=',', header=0, index=0)


def get_gold_pair_dict(gold_network_path):
    f_gold = open(gold_network_path, encoding='UTF-8-sig')
    gold_reader = list(csv.reader(f_gold))
    num_origin_goldpair = len(gold_reader) - 1

    for i in range(0, len(gold_reader) - 1):
        temp = gold_reader[i]
        s1 = str(temp[0])
        s2 = str(temp[1])

        temp[0] = s1.upper()
        temp[1] = s2.upper()

        gold_reader[i] = temp
    has_score = True
    if len(gold_reader[0]) < 3:
        has_score = False

    gold_pair_record = {}
    gold_score_record = {}
    unique_gene_list = []
    #
    for single_gold_reader in gold_reader[1:]:
        # column 0: TF
        # column 1: target gene
        # column 2: regulate score

        # if (single_gold_reader[0] not in rank_list) or (single_gold_reader[1] not in rank_list):
        #     continue
        gene_pair = [single_gold_reader[0], single_gold_reader[1]]
        str_gene_pair = single_gold_reader[0] + ',' + single_gold_reader[1]

        if single_gold_reader[0] not in unique_gene_list: unique_gene_list.append(single_gold_reader[0])
        if single_gold_reader[1] not in unique_gene_list: unique_gene_list.append(single_gold_reader[1])

        if str_gene_pair in gold_score_record:
            print('Gold pair repeat!')
        if has_score:
            # print("single_gold_reader[2]", single_gold_reader[2])
            gold_score_record[str_gene_pair] = float(single_gold_reader[2])
        else:
            gold_score_record[str_gene_pair] = 99999999
        if gene_pair[0] not in gold_pair_record:
            gold_pair_record[gene_pair[0]] = [gene_pair[1]]
        else:
            gold_pair_record[gene_pair[0]].append(gene_pair[1])

    return gold_pair_record

def get_trainset(gold_pair_record, new_all_gene_name, origin_expression_record, gene_list_path, result_path):
    num_label1 = 0
    num_label0 = 0
    # miss = 0
    flagFalse = []  # tf and target gene names that were not in the gene expression matrix
    all_gene_name = []
    all_TF_name_gold_pair = []
    final_gold_pair_record_pos = {}
    final_gold_pair_record_neg = {}
    label_list = []
    pair_list = []
    pair_list_name_tf = []
    pair_list_name_target = []
    total_matrix = []
    for i in gold_pair_record:
        for j in range(len(new_all_gene_name)):
            # print('Generating matrix of gene pair ' + str(num_tf) + ' ' + str(j))
            tf_name = i
            target_name = new_all_gene_name[j]

            flag = False
            # if TF and target gene both have gene expression data, flag is True
            if (origin_expression_record.__contains__(tf_name) & origin_expression_record.__contains__(
                    target_name)) & (tf_name != target_name):
                flag = True
            else:  # BCL11AC3ORF14
                flagFalse.extend([tf_name, target_name])
                # print("flag is flase: " + tf_name +'--'+ target_name)
            if (flag):
                # If TF is in gold_pair_record and the target gene is also regulated by this TF, the label is set to 1, otherwise the label is 0
                if tf_name in gold_pair_record and target_name in gold_pair_record[tf_name]:
                    label = 1
                    num_label1 += 1
                    # pair_list_id_pos_tf.append(tf_name)
                    # pair_list_id_pos_target.append(target_name)
                    if tf_name not in final_gold_pair_record_pos:
                        final_gold_pair_record_pos[tf_name] = [target_name]
                    else:
                        final_gold_pair_record_pos[tf_name].append(target_name)
                else:
                    label = 0
                    num_label0 += 1
                    # pair_list_id_neg_tf.append(tf_name)
                    # pair_list_id_neg_target.append(target_name)
                    if tf_name not in final_gold_pair_record_neg:
                        final_gold_pair_record_neg[tf_name] = [target_name]
                    else:
                        final_gold_pair_record_neg[tf_name].append(target_name)
                tf_data = origin_expression_record[tf_name]
                target_data = origin_expression_record[target_name]
                label_list.append(label)
                pair_list.append(tf_name + ',' + target_name)
                # pair_list_id_pos_tf.append()
                pair_list_name_tf.append(tf_name)
                pair_list_name_target.append(target_name)

                all_TF_name_gold_pair.append(tf_name)
                all_gene_name.append(tf_name)
                all_gene_name.append(target_name)
            else:
                continue

            single_tf_list = []
            gap = 100
            for k in range(0, len(tf_data), gap):
                feature = []
                a = tf_data[k:k + gap]
                b = target_data[k:k + gap]
                feature.extend(a)
                feature.extend(b)
                # single_tf_list.append(feature)
                feature = np.asarray(feature)
                # print("feature.shape", feature.shape)
                if (len(feature) == 2 * gap):
                    # print("feature.shape xixihaha", feature.shape)
                    single_tf_list.append(feature)

            single_tf_list = np.asarray(single_tf_list)  # (7,200)

            total_matrix.append(single_tf_list)
    total_matrix = np.asarray(total_matrix)
    label_list = np.array(label_list)
    pair_list = np.array(pair_list)

    np.save(result_path + 'train_matrix.npy', total_matrix)
    np.save(result_path + 'train_label.npy', label_list)
    np.save(result_path + 'train_gene_pair.npy', pair_list)
    print('train_matrix.npy saved successfully.')
    print('train_label.npy saved successfully.')
    print('train_gene_pair.npy saved successfully.')


    gene_list = pd.read_csv(gene_list_path)
    pair_list_id_tf = []
    for i in pair_list_name_tf:
        a = gene_list[gene_list.gene == i].index.tolist()
        pair_list_id_tf.extend(a)

    # gene_list = pd.read_csv(final_expression_gene_name)
    pair_list_id_target = []
    for i in pair_list_name_target:
        a = gene_list[gene_list.gene == i].index.tolist()
        pair_list_id_target.extend(a)

    np.save(result_path + 'train_pair_list_id_tf.npy', pair_list_id_tf)
    np.save(result_path + 'train_pair_list_id_target.npy', pair_list_id_target)
    print('train_pair_list_id_tf.npy saved successfully.')
    print('train_pair_list_id_target.npy saved successfully.')

    train_pair_name = {}
    train_pair_id = {}
    for ss in range(len(pair_list_name_tf)):
        tf_name = pair_list_name_tf[ss]
        tf_id = pair_list_id_tf[ss]
        target_name = pair_list_name_target[ss]
        target_id = pair_list_id_target[ss]
        if tf_name not in train_pair_name:
            train_pair_name[tf_name] = [target_name]
            train_pair_id[tf_id] = [target_id]
        else:
            train_pair_name[tf_name].append(target_name)
            train_pair_id[tf_id].append(target_id)

    train_matrix_file = result_path + 'train_matrix.npy'
    train_label_file = result_path + 'train_label.npy'
    train_gene_pair_file = result_path + 'train_gene_pair.npy'
    train_pair_list_id_tf_file = result_path + 'train_pair_list_id_tf.npy'
    train_pair_list_id_target_file = result_path + 'train_pair_list_id_target.npy'

    return train_pair_name,train_matrix_file, train_label_file, train_gene_pair_file, train_pair_list_id_tf_file, train_pair_list_id_target_file


def get_unknownset(tf_list_path, gene_list_path, gold_pair_record, new_all_gene_name, origin_expression_record, train_pair_name, result_path):

    tf_list = pd.read_csv(tf_list_path)
    tf_name, tf_index = tf_list['TF'], tf_list['index']

    flag_exp_False = []  # tf and target gene names that were not in the gene expression matrix
    unknown_pair = {}
    unknown_pair_list = []
    unknown_pair_list_name_tf = []
    unknown_pair_list_name_target = []
    total_matrix = []
    unknown_pair_label_list = []
    train_set = []
    for i in tf_name:
        for j in new_all_gene_name:
            # print('Generating matrix of gene pair ' + str(i) + '--' + str(j))
            tf_name = i
            target_name = j

            flag_exp = False
            # if TF and target gene both have gene expression data, flag is True
            if (origin_expression_record.__contains__(tf_name) & origin_expression_record.__contains__(
                    target_name)) & (tf_name != target_name) :
                flag_exp = True
            else:
                flag_exp_False.extend([tf_name, target_name])

            flag_gold_pair = False
            if tf_name in train_pair_name and target_name in train_pair_name[tf_name]:
                flag_gold_pair = False
                # print('already in train set', tf_name, target_name)
                train_set.append([tf_name, target_name])
            else:
                flag_gold_pair = True


            if (flag_exp and flag_gold_pair):
                if tf_name not in unknown_pair:
                    unknown_pair[tf_name] = [target_name]
                else:
                    unknown_pair[tf_name].append(target_name)

                tf_data = origin_expression_record[tf_name]
                target_data = origin_expression_record[target_name]
                unknown_pair_label_list.append(0)
                unknown_pair_list.append(tf_name + ',' + target_name)
                # pair_list_id_pos_tf.append()
                unknown_pair_list_name_tf.append(tf_name)
                unknown_pair_list_name_target.append(target_name)

                # all_TF_name_gold_pair.append(tf_name)
                # all_gene_name.append(tf_name)
                # all_gene_name.append(target_name)
            else:
                # miss = miss + 1
                continue

            single_tf_list = []
            gap = 100
            for k in range(0, len(tf_data), gap):
                feature = []
                a = tf_data[k:k + gap]
                b = target_data[k:k + gap]
                feature.extend(a)
                feature.extend(b)
                # single_tf_list.append(feature)
                feature = np.asarray(feature)
                # print("feature.shape", feature.shape)
                if (len(feature) == 2 * gap):
                    # print("feature.shape xixihaha", feature.shape)
                    single_tf_list.append(feature)

            single_tf_list = np.asarray(single_tf_list)  # (7,200)

            total_matrix.append(single_tf_list)
    # print('label number')
    # print(num_label1)
    # print(num_label0)
    total_matrix = np.asarray(total_matrix)
    unknown_pair_label_list = np.array(unknown_pair_label_list)
    # pair_list = np.array(pair_list)

    np.save(result_path + 'unknown_matrix.npy', total_matrix)
    np.save(result_path + 'unknown_gene_pair.npy', unknown_pair_list)
    np.save(result_path + 'unknown_pair_label_list.npy', unknown_pair_label_list)

    print('unknown_matrix.npy saved successfully.')
    print('unknown_gene_pair.npy saved successfully.')


    gene_list = pd.read_csv(gene_list_path)
    unknown_pair_list_id_tf = []
    for i in unknown_pair_list_name_tf:
        a = gene_list[gene_list.gene == i].index.tolist()
        unknown_pair_list_id_tf.extend(a)

    # gene_list = pd.read_csv(final_expression_gene_name)
    unknown_pair_list_id_target = []
    for i in unknown_pair_list_name_target:
        a = gene_list[gene_list.gene == i].index.tolist()
        unknown_pair_list_id_target.extend(a)

    np.save(result_path + 'unknown_pair_list_id_tf.npy', unknown_pair_list_id_tf)
    np.save(result_path + 'unknown_pair_list_id_target.npy', unknown_pair_list_id_target)

    print('unknown_pair_list_id_tf.npy saved successfully.')
    print('unknown_pair_list_id_target.npy saved successfully.')

    unknown_matrix_file = result_path + 'unknown_matrix.npy'
    unknown_gene_pair_file = result_path + 'unknown_gene_pair.npy'
    unknown_pair_list_id_tf_file = result_path + 'unknown_pair_list_id_tf.npy'
    unknown_pair_list_id_target_file = result_path + 'unknown_pair_list_id_target.npy'
    unknown_label_file = result_path + 'unknown_pair_label_list.npy'

    return unknown_matrix_file, unknown_label_file, unknown_gene_pair_file, unknown_pair_list_id_tf_file, unknown_pair_list_id_target_file



# def preprocess_DATA_user(result_path, gold_network_path, gene_expression_path, gene_list_path, tf_list_path):
#     '''
#     :param result_path:
#     :param gold_network_path:
#     :param gene_expression_path:
#     :param dataset_name:
#     :param gene_order_path:
#     :param Rank_num:
#     :param pathHumanTF:
#     :return:
#     '''
#     # save dataset
#     # final_GRN_pos_file_name = result_path + 'Final_GRNorTRN_pos.csv' # prior GRN
#     # save_gene_pair_index_list = result_path + 'Final_GRNorTRN_pos_index.csv'  # prior GRN
#     # final_expression_file_name = result_path + 'Final_expression.csv'
#     # final_expression_gene_name = result_path + 'Final_gene_list.csv'
#     # final_TF_common_file_name = result_path + 'Final_TF_common.csv'
#     # save_TF_index_list = result_path + 'Final_TF_common_index.csv'
#     #
#     # Info_file_name = result_path + 'Info.csv'
#     #
#     # Info = {}
#     #
#     # # get the raw gene expression matrix, cell list, gene number, cell number
#     # origin_expression_record, cells, num_genes, num_cells = get_origin_expression_data(gene_expression_path)
#     #
#     #
#     # Info["original genes"] = [num_genes]
#     # Info["original cells"] = [num_cells]
#     #
#     # gold_pair_record, gold_score_record, unique_gene_list, numTFs,num_gold_pair,num_genes,num_origin_gold_pair = get_filtered_gold(gold_network_path, gene_list)
#     #
#     #
#     #
#     # # Based on the original ground-truth network, the ground-truth network with N=500/1000 genes was retrieved
#     # print('\n' + 'Find the prior network gold_pair_record, gold_score_record of top N genes, and the list of genes after intersection...............')
#     # # gold_pair_record, gold_score_record, unique_gene_list, numTFs,num_gold_pair,num_genes,num_origin_gold_pair = get_filtered_gold(gold_network_path, rank_list)
#     # Info["original GRN"] = [num_origin_gold_pair]
#     # Info["filtered GRN"] = [num_gold_pair]
#     # Info["genes in filtered GRN"] = [num_genes]
#     #
#     # # utils.generate_filtered_gold(gold_pair_record, gold_score_record, FilteredGRN_file_name)
#     #
#     # # Obtain the normalized gene expression matrix according to the new gene list
#     # num_tf = -1
#     # num_label1 = 0
#     # num_label0 = 0
#     # # miss = 0
#     # flagFalse = [] # tf and target gene names that were not in the gene expression matrix
#     # all_gene_name = []
#     # all_TF_name_gold_pair = []
#     # final_gold_pair_record_pos = {}
#     # final_gold_pair_record_neg = {}
#     # pair_list_name_tf = []
#     # pair_list_name_target = []
#     #
#     # for i in gold_pair_record: # positve samples, key: TF, value: target genes regulated by this TF
#     #     num_tf += 1
#     #     for j in range(len(unique_gene_list)):
#     #         print('Generating matrix of gene pair ' + str(num_tf) + ' ' + str(j))
#     #         tf_name = i
#     #         target_name = unique_gene_list[j]
#     #
#     #         flag = False
#     #         # If the gene expression matrix contains TF and target genes, flag is set to true
#     #         if (origin_expression_record.__contains__(tf_name) & origin_expression_record.__contains__(target_name)):
#     #             flag = True
#     #         else: # do not have gene expression
#     #             flagFalse.extend([tf_name,target_name])
#     #             # print("Do not have gene expression: " + tf_name +'--'+ target_name)
#     #         if (flag):
#     #
#     #             all_TF_name_gold_pair.append(tf_name)
#     #             all_gene_name.append(tf_name)
#     #             all_gene_name.append(target_name)
#     #         else:
#     #             # miss = miss + 1
#     #             continue
#     #
#     # new_all_gene_name=list(set(all_gene_name))
#     # Info["final genes "] = [len(new_all_gene_name)]
#     # new_all_gene_name.sort(key=all_gene_name.index)
#     # # print(len(all_gene_name))
#     # # print(len(new_all_gene_name))
#     #
#     # new_all_TF_gold_pair = list(set(all_TF_name_gold_pair))
#     # Info["all TFs from gold pair"] = [len(new_all_TF_gold_pair)]
#     # # pd.DataFrame(new_all_TF_gold_pair).to_csv(final_TF_goldpair_file_name)
#     #
#     #
#     # human_tf = pd.read_csv(pathHumanTF)
#     # human_tfs = human_tf['TF'].to_list()
#     # all_tfs_human_tf = []
#     # for gene in new_all_gene_name:
#     #     if gene in human_tfs:
#     #         all_tfs_human_tf.append(gene)
#     #
#     # # pd.DataFrame(all_tfs_human_tf).to_csv(final_TF_human_file_name)
#     # Info["all TFs from human tf"] = [len(all_tfs_human_tf)]
#     #
#     # common_TFs = []
#     # for tf in new_all_TF_gold_pair:
#     #     if tf not in common_TFs:
#     #         common_TFs.append(tf)
#     # for tf in all_tfs_human_tf:
#     #     if tf not in common_TFs:
#     #         common_TFs.append(tf)
#     # pd.DataFrame(common_TFs).to_csv(final_TF_common_file_name)
#     # Info['common TFs'] = [len(common_TFs)]
#
#     # # for AttentionGRN
#
#     gold_pair_record = get_gold_pair_dict(gold_network_path)
#
#     new_all_gene_name = pd.read_csv(gene_list_path)['gene'].to_list()
#
#     origin_expression_record, num_gene, num_cell = get_origin_expression_data_hHEP(gene_expression_path)
#
#     train_pair_name,train_matrix_file, train_label_file, train_gene_pair_file, train_pair_list_id_tf_file, train_pair_list_id_target_file = get_trainset(gold_pair_record, new_all_gene_name, origin_expression_record, gene_list_path, result_path)
#
#     unknown_matrix_file, unknown_gene_pair_file, unknown_pair_list_id_tf_file, unknown_pair_list_id_target_file = get_unknownset(tf_list_path, gene_list_path, gold_pair_record, new_all_gene_name, origin_expression_record, train_pair_name, result_path)
#









# def predict_AttentionGRN(gold_network_path,gene_list_path,gene_expression_path , tf_list_path, result_path, save_folder ):
#     gold_pair_record = get_gold_pair_dict(gold_network_path)
#
#     new_all_gene_name = pd.read_csv(gene_list_path)['gene'].to_list()
#
#     origin_expression_record, num_gene, num_cell = utils_data.get_origin_expression_data_hHEP(gene_expression_path)
#
#     train_pair_name, train_matrix_file, train_label_file, train_gene_pair_file, train_pair_list_id_tf_file, train_pair_list_id_target_file = utils_data.get_trainset(
#         gold_pair_record, new_all_gene_name, origin_expression_record, gene_list_path, result_path)
#
#     unknown_matrix_file, unknown_label_file, unknown_gene_pair_file, unknown_pair_list_id_tf_file, unknown_pair_list_id_target_file = utils_data.get_unknownset(
#         tf_list_path, gene_list_path, gold_pair_record, new_all_gene_name, origin_expression_record, train_pair_name,
#         result_path)
#
#     train_matrix_data = np.load(train_matrix_file)
#     train_label_data = np.load(train_label_file)
#     train_gene_pair_tf = np.load(train_pair_list_id_tf_file)
#     train_gene_pair_target = np.load(train_pair_list_id_target_file)
#
#     # functional related genes
#     corr_g = utils_gz.get_corr(gene_expression_path, gene_list_path, 'cosine',
#                                cutoffbeishu=2.7)  # 'cosine', 'pearson', 'kendall', 'spearman'
#
#     # all data——>train set + val set
#     x_train, x_val, y_train, y_val, gene_pair_tf_train, gene_pair_tf_val, gene_pair_target_train, gene_pair_target_val = train_test_split(
#         train_matrix_data, train_label_data, train_gene_pair_tf, train_gene_pair_target, test_size=0.4,
#         stratify=train_label_data)
#
#     # prepare test set
#     x_test = np.load(unknown_matrix_file)
#     y_test = np.load(unknown_label_file)
#     gene_pair_tf_test = np.load(unknown_pair_list_id_tf_file)
#     gene_pair_target_test = np.load(unknown_pair_list_id_target_file)
#
#     # compute DI
#     train_og_pos, train_og_neg, train_og_pos_T, train_og_neg_T = utils_gz.get_pos_neg_T_2(gene_pair_tf_train,
#                                                                                           gene_pair_target_train,
#                                                                                           y_train, gene_expression_path,
#                                                                                           gene_list_path)
#     data_all_train_pos = utils_gz.add_original_graph(train_og_pos, corr_g, weight=1.0)
#     train_g_pos = utils_GRN.transform_savebinI(train_og_pos, data_all_train_pos, train_og_pos_T, katz_alpha=0.02,
#                                                k_hop=args.k_hop)
#
#     val_og_pos, val_og_neg, val_og_pos_T, val_og_neg_T = utils_gz.get_pos_neg_T_2(gene_pair_tf_val,
#                                                                                   gene_pair_target_val, y_val,
#                                                                                   gene_expression_path, gene_list_path)
#     val_g_pos = utils_GRN.transform_savebinI(val_og_pos, val_og_pos, val_og_pos_T, katz_alpha=0.02, k_hop=args.k_hop)
#
#     test_og_pos, test_og_neg, test_og_pos_T, test_og_neg_T = utils_gz.get_pos_neg_T_2(gene_pair_tf_test,
#                                                                                       gene_pair_target_test, y_test,
#                                                                                       gene_expression_path,
#                                                                                       gene_list_path)
#     test_g_pos = utils_GRN.transform_savebinI(test_og_pos, test_og_pos, test_og_pos_T, katz_alpha=0.02,
#                                               k_hop=args.k_hop)
#
#     X_trainloader, y_trainloader, gene_pair_tf_trainloader, gene_pair_target_trainloader = utils_GRN.numpy2loader(
#         x_train, y_train, gene_pair_tf_train, gene_pair_target_train, args.batch_sizes)
#     X_valloader, y_valloader, gene_pair_tf_valloader, gene_pair_target_valloader = utils_GRN.numpy2loader(x_val, y_val,
#                                                                                                           gene_pair_tf_val,
#                                                                                                           gene_pair_target_val,
#                                                                                                           args.batch_sizes)
#     X_testloader, y_testloader, gene_pair_tf_testloader, gene_pair_target_testloader = utils_GRN.numpy2loader(x_test,
#                                                                                                               y_test,
#                                                                                                               gene_pair_tf_test,
#                                                                                                               gene_pair_target_test,
#                                                                                                               args.batch_sizes)
#
#     X_trainList = utils_GRN.loaderToList(X_trainloader)
#     y_trainList = utils_GRN.loaderToList(y_trainloader)
#     gene_pair_tf_trainList = utils_GRN.loaderToList(gene_pair_tf_trainloader)
#     gene_pair_target_trainList = utils_GRN.loaderToList(gene_pair_target_trainloader)
#
#     X_valList = utils_GRN.loaderToList(X_valloader)
#     y_valList = utils_GRN.loaderToList(y_valloader)
#     gene_pair_tf_valList = utils_GRN.loaderToList(gene_pair_tf_valloader)
#     gene_pair_target_valList = utils_GRN.loaderToList(gene_pair_target_valloader)
#
#     X_testList = utils_GRN.loaderToList(X_testloader)
#     y_testList = utils_GRN.loaderToList(y_testloader)
#     gene_pair_tf_testList = utils_GRN.loaderToList(gene_pair_tf_testloader)
#     gene_pair_target_testList = utils_GRN.loaderToList(gene_pair_target_testloader)
#
#     in_dim_GT = train_g_pos.ndata['x'].shape[1]
#     I_dim = train_g_pos.ndata['I'].shape[1]
#     model = TransformerE.AttentionGRN(in_dim_GT, args.out_dim_GT, args.hidden_dim_GT, args.num_heads_GT, I_dim,
#                                       args.input_dim_exp, num_gtlayers=4, d_model=args.d_models, num_classes=2).to(
#         device)
#
#     criterion = nn.CrossEntropyLoss()
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
#     train_losses = []
#     train_acces = []
#     valid_losses = []
#     valid_acces = []
#     valid_aucs = []
#     test_aucs = []
#
#     min_loss = 100
#
#     for epoch in range(args.epochs):
#         model.train()
#
#         train_loss = []
#         train_accs = []
#
#         for j in range(0, len(X_trainList)):
#             data = X_trainList[j]
#             labels = y_trainList[j]
#             all_tf_train, all_target_train, train_g_batch, k_edge_idx1 = utils_GRN.genepair_to_dgl_I(labels,
#                                                                                                      gene_pair_tf_trainList[
#                                                                                                          j],
#                                                                                                      gene_pair_target_trainList[
#                                                                                                          j],
#                                                                                                      train_g_pos)
#
#             if len(k_edge_idx1) == 0:
#                 flag_DM = False
#             else:
#                 flag_DM = True
#
#             logits = model(train_g_batch.to(device), data.to(device), all_tf_train, all_target_train, device,
#                            flag_DM).to(
#                 'cpu')
#             labels = torch.tensor(labels, dtype=torch.long)
#             loss = criterion(logits, labels)
#             optimizer.zero_grad()
#             loss.backward()
#             grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
#             optimizer.step()
#
#             acc = (logits.argmax(dim=-1) == labels).float().mean()
#             train_loss.append(loss.item())
#             train_accs.append(acc)
#         train_loss = sum(train_loss) / len(train_loss)
#         train_acc = sum(train_accs) / len(train_accs)
#
#         train_acces.append(train_acc)
#         train_losses.append(train_loss)
#
#         print(f"[ Train | {epoch + 1:03d}/{args.epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
#
#         model.eval()
#         predictions = []
#         labelss = []
#         y_val_label = []
#         y_val_predict = []
#         valid_loss = []
#         valid_accs = []
#         for k in range(0, len(X_valList)):
#             val_data = X_valList[k]
#             labels = y_valList[k]
#             labels = torch.tensor(labels, dtype=torch.long)
#             all_tf_val, all_target_val, val_g_batch, k_edge_idx1 = utils_GRN.genepair_to_dgl_I(labels,
#                                                                                                gene_pair_tf_valList[k],
#                                                                                                gene_pair_target_valList[
#                                                                                                    k],
#                                                                                                val_g_pos)
#
#             if len(k_edge_idx1) == 0:
#                 flag_DM = False
#                 print(flag_DM)
#             else:
#                 flag_DM = True
#             with torch.no_grad():
#                 # logits = model(val_data)
#                 logits = model(val_g_batch.to(device), val_data.to(device), all_tf_val, all_target_val, device,
#                                flag_DM).to(
#                     'cpu')
#
#             loss = criterion(logits, labels)
#             valid_loss.append(loss.item())
#
#             acc = (logits.argmax(dim=-1) == labels).float().mean()
#             valid_accs.append(acc)
#
#             if loss.item() < min_loss:
#                 min_loss = loss.item()
#                 print("save model")
#                 torch.save(model.state_dict(), save_folder + "model.pth")
#
#             predt = F.softmax(logits)
#             labelss.extend(labels.cpu().numpy().tolist())
#             y_val_label.extend(labels.cpu().numpy())
#
#             temps = predt.cpu().numpy().tolist()
#             for i in temps:
#                 t = i[1]
#                 y_val_predict.append(t)
#             predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
#         valid_loss = sum(valid_loss) / len(valid_loss)
#         valid_acc = sum(valid_accs) / len(valid_accs)
#
#         valid_acces.append(valid_acc)
#         valid_losses.append(valid_loss)
#
#         AUC_val, AUPR_val, ACC_val, F1_val, SPE_val, MCC_val, Precision_val, Recall_val = utils_gz.metric_scores(
#             y_val_label, y_val_predict, th=0.5)
#
#         print(
#             f"[ Valid | {epoch + 1:03d}/{args.epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.4f}, auc = {AUC_val:.4f}, aupr = {AUPR_val:.4f}")
#
#     epochs_x = [i for i in range(args.epochs)]
#     plt.figure()
#     plt.plot(epochs_x, train_losses, 'bo--', alpha=0.5, linewidth=1, label='train')
#     plt.plot(epochs_x, valid_losses, 'r*--', alpha=0.5, linewidth=1, label='validation')
#     plt.legend()
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#
#     # plt.ylim(-1,1)
#     # plt.show()
#     plt.savefig(save_folder + '/' + dataset_name + '_' + network_type + '_Top' + str(Rank_num) + '_loss.pdf')
#
#     plt.figure()
#     plt.plot(epochs_x, train_acces, 'bo--', alpha=0.5, linewidth=1, label='train')
#     plt.plot(epochs_x, valid_acces, 'r*--', alpha=0.5, linewidth=1, label='validation')
#     plt.legend()
#     plt.xlabel('epoch')
#     plt.ylabel('accuracy')
#
#     # plt.ylim(-1,1)
#     # plt.show()
#     plt.savefig(save_folder + '/' + dataset_name + '_' + network_type + '_Top' + str(Rank_num) + '_accuracy.pdf')
#
#     model = TransformerE.AttentionGRN(in_dim_GT, args.out_dim_GT, args.hidden_dim_GT, args.num_heads_GT, I_dim,
#                                       args.input_dim_exp, num_gtlayers=4, d_model=args.d_models, num_classes=2).to(
#         device)
#     model.load_state_dict(torch.load(save_folder + 'model.pth'))
#     model.eval()
#     # y_test_label = []
#     y_predict = []
#     predictions = []
#     tf_ids = []
#     target_ids = []
#     # labelss = []
#
#     for k in range(0, len(X_testList)):
#         test_data = X_testList[k]
#         labels = y_testList[k]
#         all_tf_test, all_target_test, test_g_batch, k_edge_idx1 = utils_GRN.genepair_to_dgl_I(labels,
#                                                                                               gene_pair_tf_testList[k],
#                                                                                               gene_pair_target_testList[
#                                                                                                   k],
#                                                                                               test_g_pos)
#         if len(k_edge_idx1) == 0:
#             flag_DM = False
#             # print(flag_DM)
#         else:
#             flag_DM = True
#         with torch.no_grad():
#             # logits = model(data)
#             logits = model(test_g_batch.to(device), test_data.to(device), all_tf_test, all_target_test,
#                            device, flag_DM).to('cpu')
#         predt = F.softmax(logits)
#         # labelss.extend(labels.cpu().numpy().tolist())
#         # y_test_label.extend(labels.cpu().numpy())
#
#         # temps = logits.cpu().numpy().tolist()
#         temps = predt.cpu().numpy().tolist()
#         for i in temps:
#             t = i[1]
#             y_predict.append(t)
#         predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())  # [0 1 ]
#         tf_ids.extend(all_tf_test)  # tf_ids[0].item()
#         target_ids.extend(all_target_test)
#
#     new_tf_ids = []
#     new_target_ids = []
#     new_label = []
#     new_scores = []
#     for i in range(len(tf_ids)):
#         tf_id = tf_ids[i].item()
#         target_id = target_ids[i].item()
#         predict_label = predictions[i]
#         score = y_predict[i]
#         if predict_label == 1:
#             new_tf_ids.append(tf_id)
#             new_target_ids.append(target_id)
#             new_label.append(predict_label)
#             new_scores.append(score)
#
#     gene_name = pd.read_csv(gene_list_path)['gene'].to_list()
#     gene_id = pd.read_csv(gene_list_path)['index'].to_list()
#
#     new_tf_names = []
#     for i in new_tf_ids:
#         a = gene_id.index(i)
#         new_tf_names.append(gene_name[a])
#
#     new_target_names = []
#     for i in new_target_ids:
#         a = gene_id.index(i)
#         new_target_names.append(gene_name[a])
#
#     from pandas.core.frame import DataFrame
#     new_tf_names = DataFrame(new_tf_names)
#     new_target_names = DataFrame(new_target_names)
#     new_scores = DataFrame(new_scores)
#
#     predic_GRN = pd.concat([new_tf_names, new_target_names], axis=1)
#     predic_GRN = pd.concat([predic_GRN, new_scores], axis=1)
#     predic_GRN.columns = ['tf', 'target', 'score']
#
#     dfs = predic_GRN
#     dfs.sort_values(by="score", inplace=True, ascending=False)
#
#     dfs.to_csv(save_folder + '/' + dataset_name + '_' + network_type + 'predict_GRN.csv', index=False)
#



