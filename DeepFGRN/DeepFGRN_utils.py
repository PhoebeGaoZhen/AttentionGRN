
import random, csv
import numpy as np
from keras.utils import to_categorical
import pandas as pd
from sklearn.metrics import confusion_matrix,roc_auc_score,matthews_corrcoef,roc_curve,auc,average_precision_score
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score,precision_recall_curve

def get_expression_data(gene_expression_path):
    expression = pd.read_csv(gene_expression_path)
    expression = expression.T
    expression = expression[1:]
    # print(expression)
    return expression

def get_GRN(Ecoli_GRN_known,genename):


    rowNumber = []
    colNumber = []
    TF_name = np.array(Ecoli_GRN_known[0])
    target_name = np.array(Ecoli_GRN_known[1])

    genename2 = genename.tolist()

    for i in range(len(TF_name)):
        rowNumber.append(genename2.index(TF_name[i]))

    for i in range(len(target_name)):
        colNumber.append(genename2.index(target_name[i]))


    num_pos = 0
    num_unknown = 0
    geneNetwork = np.zeros((len(genename2), len(genename2)))

    for i in range(len(TF_name)):
        r = rowNumber[i]
        c = colNumber[i]
        geneNetwork[r][c] = int(1.0)

    for i in range(geneNetwork.shape[0]):
        for j in range(geneNetwork.shape[0]):
            if geneNetwork[i][j] == 1:
                num_pos += 1
            else:
                num_unknown += 1
    return geneNetwork, num_pos, num_unknown




def create_samples_GRN(EXP_cold, Ecoli_GRN,GRN_embedding_s, GRN_embedding_t,num_negative):

    sample_cold_pos_1_tf = []
    sample_cold_neg_0_tf = []
    sample_cold_pos_1_target = []
    sample_cold_neg_0_target = []

    sample_cold_pos_1_net_tf_s = []
    sample_cold_pos_1_net_tf_t = []
    sample_cold_pos_1_net_target_s = []
    sample_cold_pos_1_net_target_t = []
    sample_cold_pos_0_net_tf_s = []
    sample_cold_pos_0_net_tf_t = []
    sample_cold_pos_0_net_target_s = []
    sample_cold_pos_0_net_target_t = []

    labels_pos_1 = []
    labels_neg_0 = []

    positive_1_position = []
    negative_0_positions = []

    for i in range(Ecoli_GRN.shape[0]):
        for j in range(Ecoli_GRN.shape[0]):

            label = int(Ecoli_GRN[i][j])
            # print(label)
            if label == 1:

                tf1 = EXP_cold[i]

                tf_s = GRN_embedding_s[i]

                tf_t = GRN_embedding_t[i]

                target1 = EXP_cold[j]

                target_s = GRN_embedding_s[j]

                target_t = GRN_embedding_t[j]

                sample_cold_pos_1_tf.append(tf1)
                sample_cold_pos_1_target.append(target1)

                sample_cold_pos_1_net_tf_s.append(tf_s)
                sample_cold_pos_1_net_tf_t.append(tf_t)
                sample_cold_pos_1_net_target_s.append(target_s)
                sample_cold_pos_1_net_target_t.append(target_t)

                labels_pos_1.append(label)
                positive_1_position.append((i, j))
            elif label == 0:
                negative_0_positions.append((i, j))
            else:
                print("error")
                print(label)
                print(i, j)
    random.shuffle(negative_0_positions)
    negative_0_position = negative_0_positions[0:num_negative]
    for k in range(len(negative_0_position)):
        i = negative_0_position[k][0]
        j = negative_0_position[k][1]

        tf1 = EXP_cold[i]

        tf_s = GRN_embedding_s[i]

        tf_t = GRN_embedding_t[i]

        target1 = EXP_cold[j]

        target_s = GRN_embedding_s[j]

        target_t = GRN_embedding_t[j]

        sample_cold_neg_0_tf.append(tf1)
        sample_cold_neg_0_target.append(target1)

        sample_cold_pos_0_net_tf_s.append(tf_s)
        sample_cold_pos_0_net_tf_t.append(tf_t)
        sample_cold_pos_0_net_target_s.append(target_s)
        sample_cold_pos_0_net_target_t.append(target_t)

        labels_neg_0.append(0)


    positive_data = list(zip(sample_cold_pos_1_tf, sample_cold_pos_1_target, sample_cold_pos_1_net_tf_s, sample_cold_pos_1_net_tf_t, sample_cold_pos_1_net_target_s, sample_cold_pos_1_net_target_t, labels_pos_1, positive_1_position))  # len
    negative_data = list(zip(sample_cold_neg_0_tf, sample_cold_neg_0_target, sample_cold_pos_0_net_tf_s, sample_cold_pos_0_net_tf_t, sample_cold_pos_0_net_target_s, sample_cold_pos_0_net_target_t, labels_neg_0, negative_0_position))  # len


    feature_size_tf = sample_cold_pos_1_tf[0].shape[0]
    feature_size_target = sample_cold_pos_1_target[0].shape[0]
    feature_size_tf_nets = sample_cold_pos_1_net_tf_s[0].shape[0]

    return positive_data, negative_data, feature_size_tf, feature_size_target, feature_size_tf_nets


def create_samples_TRN(gene_expression_matrix, gold_pair_record, all_tf_list, target_gene_list,GRN_embedding_s, GRN_embedding_t):
    num_tf = 0
    num_label1 = 0
    num_label0 = 0
    # label_list = []
    # pair_list = []
    feature_1 = []
    # feature_tf_1 = []
    # feature_target_1 = []
    label_1 = []
    # feature_tf_0 = []
    # feature_target_0 = []
    feature_0 = []
    label_0 = []
    unknown_pair = []

    for tf_name in gold_pair_record:
        num_tf += 1
        for target_name in target_gene_list:
            # print('Generating sample of TF-gene pair ' + tf_name + '--->' + target_name)

            if tf_name in gold_pair_record and target_name in gold_pair_record[tf_name]:
                label = 1
                num_label1 += 1

                tf_data = gene_expression_matrix.loc[[tf_name]].values.tolist()[0]
                tf_s = GRN_embedding_s[tf_name]
                tf_t = GRN_embedding_t[tf_name]
                target_data = gene_expression_matrix.loc[[target_name]].values.tolist()[0]
                target_s = GRN_embedding_s[target_name]
                target_t = GRN_embedding_t[target_name]
                temp = np.hstack((tf_data, target_data))
                feature_1.append(temp)
                # feature_tf_1.append(tf_data)
                # feature_target_1.append(target_data)
                label_1.append(label)
                # else:

                # label = 0
                # num_label0 += 1
                # unknown_pair.append([tf_name, target_name])
                # unknown_target.append(target_name)
    hello = 0
    for tf_name in all_tf_list:
        hello += 1
        for target_name in target_gene_list:
            if tf_name not in gold_pair_record:
                num_label0 += 1
                unknown_pair.append([tf_name, target_name])
            elif tf_name in gold_pair_record and target_name not in gold_pair_record[tf_name]:
                num_label0 += 1
                unknown_pair.append([tf_name, target_name])
    print("TRN negetive samples: " + str(num_label0))
    print("TRN positive samples: " + str(num_label1))
    print("all_tf_list: " + str(hello))

    random.shuffle(unknown_pair)
    # negative = unknown_pair_balanced[0:num_label1]
    negative = unknown_pair
    for neg_pair in negative:
        tf_name = neg_pair[0]
        target_name = neg_pair[1]
        tf_data = gene_expression_matrix.loc[[tf_name]].values.tolist()[0]
        target_data = gene_expression_matrix.loc[[target_name]].values.tolist()[0]
        temp = np.hstack((tf_data, target_data))
        feature_0.append(temp)
        # feature_tf_0.append(tf_data)
        # feature_target_0.append(target_data)
        label_0.append(0)

    positive_data = list(zip(feature_1, label_1))  # len
    negative_data = list(zip(feature_0, label_0))  # len

    print(len(positive_data))
    print(len(negative_data))
    print("TRN successful")

    feature_size = len(feature_1[0])

    return positive_data, negative_data, feature_size





def transform_data(train_data):
    featuretf_exp = []
    featuretarget_exp = []
    net_tf_s = []
    net_tf_t = []
    net_target_s = []
    net_target_t = []
    label_ = []
    # position = []
    for i in range(len(train_data)):
        featuretf_exp.append(train_data[i][0])
        featuretarget_exp.append(train_data[i][1])
        net_tf_s.append(train_data[i][2])
        net_tf_t.append(train_data[i][3])
        net_target_s.append(train_data[i][4])
        net_target_t.append(train_data[i][5])
        label_.append(train_data[i][6])
        # position.append(train_data[i][7])

    featuretf_exp = np.array(featuretf_exp)
    featuretarget_exp = np.array(featuretarget_exp)
    net_tf_s = np.array(net_tf_s)
    net_tf_t = np.array(net_tf_t)
    net_target_s = np.array(net_target_s)
    net_target_t = np.array(net_target_t)

    dataX_tf = featuretf_exp[: ,np.newaxis ,:]
    dataX_target = featuretarget_exp[: ,np.newaxis ,:]
    net_tf_s = net_tf_s[:,np.newaxis,:]
    net_tf_t = net_tf_t[:,np.newaxis,:]
    net_target_s = net_target_s[:,np.newaxis,:]
    net_target_t = net_target_t[:,np.newaxis,:]
    print("the shape of dataX_tf: " ,dataX_tf.shape)
    print("the shape of dataX_target: " ,dataX_target.shape)
    # print("the shape of net_tf_s: ", net_tf_s.shape)

    label_ = np.array(label_)

    labelY = to_categorical(label_ ,2)

    # position = np.array(position)

    # print(featuretf_exp.shape)
    # print(featuretarget_exp.shape)
    # print(labelY.shape)
    # print(labelY)

    return dataX_tf, dataX_target, net_tf_s, net_tf_t, net_target_s, net_target_t, labelY



def transform_data_noC(train_data):
    featuretf_exp = []
    featuretarget_exp = []
    # net_tf_s = []
    # net_tf_t = []
    # net_target_s = []
    # net_target_t = []
    label_ = []
    # position = []
    for i in range(len(train_data)):
        featuretf_exp.append(train_data[i][0])
        featuretarget_exp.append(train_data[i][1])
        # net_tf_s.append(train_data[i][2])
        # net_tf_t.append(train_data[i][3])
        # net_target_s.append(train_data[i][4])
        # net_target_t.append(train_data[i][5])
        label_.append(train_data[i][2])
        # position.append(train_data[i][7])

    featuretf_exp = np.array(featuretf_exp)
    featuretarget_exp = np.array(featuretarget_exp)
    # net_tf_s = np.array(net_tf_s)
    # net_tf_t = np.array(net_tf_t)
    # net_target_s = np.array(net_target_s)
    # net_target_t = np.array(net_target_t)

    dataX_tf = featuretf_exp[: ,np.newaxis ,:]
    dataX_target = featuretarget_exp[: ,np.newaxis ,:]
    # net_tf_s = net_tf_s[:,np.newaxis,:]
    # net_tf_t = net_tf_t[:,np.newaxis,:]
    # net_target_s = net_target_s[:,np.newaxis,:]
    # net_target_t = net_target_t[:,np.newaxis,:]
    print("the shape of dataX_tf: " ,dataX_tf.shape)
    print("the shape of dataX_target: " ,dataX_target.shape)
    # print("the shape of net_tf_s: ", net_tf_s.shape)

    label_ = np.array(label_)

    labelY = to_categorical(label_ ,2)

    # position = np.array(position)

    # print(featuretf_exp.shape)
    # print(featuretarget_exp.shape)
    # print(labelY.shape)
    # print(labelY)

    return dataX_tf, dataX_target, labelY




def two_scores(y_test,y_pred,th=0.5):
    y_predlabel = [(0 if item<th else 1) for item in y_pred]
    tn,fp,fn,tp = confusion_matrix(y_test,y_predlabel).flatten()
    SPE = tn*1./(tn+fp)
    MCC = matthews_corrcoef(y_test,y_predlabel)
    Recall = recall_score(y_test, y_predlabel)
    Precision = precision_score(y_test, y_predlabel)
    F1 = f1_score(y_test, y_predlabel)
    Acc = accuracy_score(y_test, y_predlabel)
    AUC = roc_auc_score(y_test, y_pred)
    precision_aupr, recall_aupr, _ = precision_recall_curve(y_test, y_pred)
    AUPR1 = auc(recall_aupr, precision_aupr)
    AUPR2 = average_precision_score(y_true=y_test, y_score=y_pred)
    return Recall, SPE, Precision, F1, MCC, Acc, AUC, AUPR1, AUPR2

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
    f_node = open(save_path + 'node.csv', 'w', newline='\n')
    f_node_writer = csv.writer(f_node)
    f_node_writer.writerow(["name","ids"])
    for i in data:
        row = []
        row.append(i)
        row.append(data[i])
        f_node_writer.writerow(row)
    f_node.close()
    new_node = pd.read_csv(save_path + 'node.csv')
    new_node.to_csv(save_path + 'final_node.txt',sep='\t',index=False)

    return new_node


def get_GRN_ids(label_file_name, save_path, new_node):
    gene_pair = pd.read_csv(label_file_name)
    # print(gene_pair)
    tf_ids = []
    target_ids = []
    for i in range(gene_pair.shape[0]):
        tf = gene_pair.iloc[i,0]
        target = gene_pair.iloc[i,1]
        # print(tf)
        # print(target)
        tf_id = new_node[new_node.name == tf].index.tolist()[0]
        target_id = new_node[new_node.name == target].index.tolist()[0]
        tf_ids.append(tf_id)
        target_ids.append(target_id)
        # for j in range(new_node.shape[0]):
        #     if tf==new_node.iloc[j,0]:
        #         tf_id = new_node.iloc[j,1]
        #         tf_ids.append(tf_id)
        #     if target==new_node.iloc[j,0]:
        #         target_id = new_node.iloc[j,1]
        #         target_ids.append(target_id)
    # print(len(tf_ids))
    # print(len(target_ids))
    tf_ids = np.array(tf_ids)
    target_ids = np.array(target_ids)
    Gold_Standard_Network_ids = np.vstack((tf_ids,target_ids))
    Gold_Standard_Network_ids = pd.DataFrame(Gold_Standard_Network_ids)
    Gold_Standard_Network_ids.T.to_csv(save_path + 'Gold_Standard_Network_ids.tsv', sep='\t', header=None, index=False, mode="w")



