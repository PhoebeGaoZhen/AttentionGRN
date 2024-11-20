'''
202409
datasets: DATA1
model: GNNLink
task: TRN inference
evaluation strategy: TF-aware three-fold cross-validation

positive samples: known TF-gene pairs
negative samples: unknown TF-gene pairs, each TF is same as TF in positive samples
balanced datasets
'''

from inits import load_data1, preprocess_graph, load_data_gao, get_samples_pair, transform_data
from Transorfomer import TransorfomerModel
import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import KFold, train_test_split
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import time
import utils


source_path = '../data_preprocess/DATA2/'

dataset_paths = [
    source_path + '/hESC/hESC-ChIP-seq-network/Top500',
    source_path + '/hESC/Non-specific-ChIP-seq-network/Top500',
    source_path + '/hESC/STRING-network/Top500',

    source_path + '/hHep/HepG2-ChIP-seq-network/Top500',
    source_path + '/hHep/Non-specific-ChIP-seq-network/Top500',
    source_path + '/hHep/STRING-network/Top500',

    source_path + '/mDC/mDC-ChIP-seq-network/Top500',
    source_path + '/mDC/Non-Specific-ChIP-seq-network/Top500',
    source_path + '/mDC/STRING-network/Top500',

    source_path + '/mESC/mESC-ChIP-seq-network/Top500',
    source_path + '/mESC/Non-Specific-ChIP-seq-network/Top500',
    source_path + '/mESC/STRING-network/Top500',
    source_path + '/mESC/mESC-lofgof-network/Top500',

    source_path + '/mHSC-E/mHSC-ChIP-seq-network/Top500',
    source_path + '/mHSC-E/Non-Specific-ChIP-seq-network/Top500',
    source_path + '/mHSC-E/STRING-network/Top500',

    source_path + '/mHSC-GM/mHSC-ChIP-seq-network/Top500',
    source_path + '/mHSC-GM/Non-Specific-ChIP-seq-network/Top500',
    source_path + '/mHSC-GM/STRING-network/Top500',

    source_path + '/mHSC-L/mHSC-ChIP-seq-network/Top500',
    source_path + '/mHSC-L/Non-Specific-ChIP-seq-network/Top500',
    source_path + '/mHSC-L/STRING-network/Top500',

    source_path + '/hESC/hESC-ChIP-seq-network/Top1000',
    source_path + '/hESC/Non-specific-ChIP-seq-network/Top1000',
    source_path + '/hESC/STRING-network/Top1000',

    source_path + '/hHep/HepG2-ChIP-seq-network/Top1000',
    source_path + '/hHep/Non-specific-ChIP-seq-network/Top1000',
    source_path + '/hHep/STRING-network/Top1000',

    source_path + '/mDC/mDC-ChIP-seq-network/Top1000',
    source_path + '/mDC/Non-Specific-ChIP-seq-network/Top1000',
    source_path + '/mDC/STRING-network/Top1000',

    source_path + '/mESC/mESC-ChIP-seq-network/Top1000',
    source_path + '/mESC/Non-Specific-ChIP-seq-network/Top1000',
    source_path + '/mESC/STRING-network/Top1000',
    source_path + '/mESC/mESC-lofgof-network/Top1000',

    source_path + '/mHSC-E/mHSC-ChIP-seq-network/Top1000',
    source_path + '/mHSC-E/Non-Specific-ChIP-seq-network/Top1000',
    source_path + '/mHSC-E/STRING-network/Top1000',

    source_path + '/mHSC-GM/mHSC-ChIP-seq-network/Top1000',
    source_path + '/mHSC-GM/Non-Specific-ChIP-seq-network/Top1000',
    source_path + '/mHSC-GM/STRING-network/Top1000',

    source_path + '/mHSC-L/mHSC-ChIP-seq-network/Top1000',
    source_path + '/mHSC-L/Non-Specific-ChIP-seq-network/Top1000',
    source_path + '/mHSC-L/STRING-network/Top1000'
]

resultspath = 'GNNLink_data2_TRN/'
if not os.path.isdir(resultspath):
    os.makedirs(resultspath)
resultfile = 'GNNlink-TRN-data2.csv'

epochs = 100
batch_size = 256
nb_run = 5

def train_main(exp_file_name, train_data, validation_data, test_data):
    geneName, feature, logits_train, logits_test, logits_validation, train_mask, test_mask, validation_mask, interaction, neg_logits_train, neg_logits_test, neg_logits_validation = transform_data(
        exp_file_name, train_data, validation_data, test_data)

    biases = preprocess_graph(interaction)
    # v1 = tf.Variable(5, name='v1')
    # saver = tf.compat.v1.train.Saver([v1])
    model = TransorfomerModel(feature, do_train=False)
    with tf.compat.v1.Session() as sess:
        init_op = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
        sess.run(init_op)
        # saver.restore(sess, tf.train.latest_checkpoint("savemodel/"))
        train_loss_avg = 0
        train_acc_avg = 0
        for epoch in range(epochs):
            t = time.time()
            ######## train #########
            tr_step = 0
            tr_size = 1
            if tr_step * batch_size < tr_size:
                _, loss_value_tr, acc_tr = sess.run([model.train_op, model.loss, model.accuracy],
                                                    feed_dict={
                                                        model.encoded_gene: feature,
                                                        model.bias_in: biases,
                                                        model.lbl_in: logits_train,
                                                        model.msk_in: train_mask,
                                                        model.neg_msk: neg_logits_train
                                                    })
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                tr_step += 1

            score, _, _ = sess.run([model.logits, model.loss, model.accuracy],
                                   feed_dict={
                                       model.encoded_gene: feature,
                                       model.bias_in: biases,
                                       model.lbl_in: logits_validation,
                                       model.msk_in: validation_mask,
                                       model.neg_msk: neg_logits_validation
                                   })
            score = score.reshape((feature.shape[0], feature.shape[0]))
            auc_val, aupr_val = evaluate(validation_data, score)
            print("Epoch:{}".format(epoch + 1), 'train loss:{}'.format(train_loss_avg),
                  'train loss:{}'.format(train_acc_avg), 'val auc:{}'.format(auc_val), 'val aupr:{}'.format(aupr_val),
                  'running time:{}'.format(time.time() - t))

        ###########     test      ############
        ts_size = 1
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        print("Start to test")
        while ts_step * batch_size < ts_size:
            out_come, loss_value_ts, acc_ts = sess.run([model.logits, model.loss, model.accuracy],
                                                       feed_dict={
                                                           model.encoded_gene: feature,
                                                           model.bias_in: biases,
                                                           model.lbl_in: logits_test,
                                                           model.msk_in: test_mask,
                                                           model.neg_msk: neg_logits_test})
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1
        print('Test loss:', ts_loss / ts_step, '; Test accuracy:', ts_acc / ts_step)

        out_come = out_come.reshape((feature.shape[0], feature.shape[0]))

        return geneName, out_come, test_data
        sess.close()


def evaluate(rel_test_label, pre_test_label):
    temp_pre = []
    for i in range(rel_test_label.shape[0]):
        l = []
        m = rel_test_label[i, 0]
        n = rel_test_label[i, 1]
        l.append(m)
        l.append(n)
        l.append(pre_test_label[m, n])
        temp_pre.append(l)
    temp_pre = np.asarray(temp_pre)
    prec, rec, thr = precision_recall_curve(rel_test_label[:, 2], temp_pre[:, 2])
    aupr_val = auc(rec, prec)
    aupr_vec.append(aupr_val)
    fpr, tpr, thr = roc_curve(rel_test_label[:, 2], temp_pre[:, 2])
    auc_val = auc(fpr, tpr)

    return auc_val, aupr_val


# Loop through each dataset path
for dataset_path in dataset_paths:
    file = dataset_path
    exp_file_name = file + '/Final_expression.csv'
    # file_path =  file +'/Final_GRNorTRN_pos_index.csv'
    pos_neg_balanced_id = pd.read_csv(file + '/pos_neg_balanced_id.csv', header=None)
    tf_list = pd.read_csv(file + '/tf_list.csv', header=None).values

    # Lists to collect average results
    mean_roc = []
    mean_ap = []
    mean_time = []
    t_start = time.time()

    for i in range(nb_run):

        print('------------------------------''the ' + str(
            i + 1) + '-th 3CV of ' + dataset_path + '---------------------------------------')
        AUROCs = []
        AUPRs = []
        kf = KFold(n_splits=3, shuffle=True)
        for train_index, test_index in kf.split(tf_list):
            train_TF = tf_list[train_index]
            test_TF = tf_list[test_index]

            train_pair_2fold, train_pair_label_2fold = get_samples_pair(pos_neg_balanced_id, train_TF)
            test_pair_1fold, test_pair_label_1fold = get_samples_pair(pos_neg_balanced_id, test_TF)

            train_pair, val_pair, train_pair_label, val_pair_label = train_test_split(train_pair_2fold,
                                                                                      train_pair_label_2fold,
                                                                                      test_size=0.2, random_state=1,
                                                                                      shuffle=True,
                                                                                      stratify=train_pair_label_2fold)  # , random_state=seed
            train_data = np.array(train_pair)
            validation_data = np.array(val_pair)
            test_data = np.array(test_pair_1fold)

            aupr_vec = []
            auroc_ver = []
            t1 = time.time()
            geneName, pre_test_label, rel_test_label = train_main(exp_file_name, train_data, validation_data, test_data)
            print(time.time() - t1)
            temp_pre = []
            for i in range(rel_test_label.shape[0]):
                l = []
                m = rel_test_label[i, 0]
                n = rel_test_label[i, 1]
                l.append(m)
                l.append(n)
                l.append(pre_test_label[m, n])
                temp_pre.append(l)
            temp_pre = np.asarray(temp_pre)
            prec, rec, thr = precision_recall_curve(rel_test_label[:, 2], temp_pre[:, 2])
            aupr_test = auc(rec, prec)
            aupr_vec.append(aupr_test)
            fpr, tpr, thr = roc_curve(rel_test_label[:, 2], temp_pre[:, 2])
            auc_test = auc(fpr, tpr)

            AUROCs.append(auc_test)
            AUPRs.append(aupr_test)

        avg_AUROC = np.mean(AUROCs)
        avg_AUPR = np.mean(AUPRs)

        mean_time.append(time.time() - t_start)
        mean_roc.append(avg_AUROC)  #
        mean_ap.append(avg_AUPR)
        # plt.figure
        # plt.plot(fpr, tpr)
        # plt.show()
        # plt.figure
        # plt.plot(rec, prec)
        # plt.show()

    AUC_scores = np.mean(mean_roc)
    AUC_std = np.std(mean_roc)
    AP_scores = np.mean(mean_ap)
    AP_std = np.std(mean_ap)
    time_mean = np.mean(mean_time)

    current_time = time.time()
    local_time_struct = time.localtime(current_time)
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time_struct)

    # 列名
    # column_names = [
    #     'date_name',
    #     'AUC scores1', 'AUC scores2', 'AUC scores3', 'AUC scores4', 'AUC scores5',
    #     'AUC scores6', 'AUC scores7', 'AUC scores8', 'AUC scores9', 'AUC scores10',
    #     'AP scores1', 'AP scores2', 'AP scores3', 'AP scores4', 'AP scores5', 'AP scores6',
    #     'AP scores7', 'AP scores8', 'AP scores9', 'AP scores10',
    #     'AUC mean', 'AUC std', 'AP mean', 'AP std', 'Running times'
    # ]
    column_names = [
        'date_name',
        'AUC scores1', 'AUC scores2', 'AUC scores3', 'AUC scores4', 'AUC scores5',
        # 'AUC scores6', 'AUC scores7', 'AUC scores8', 'AUC scores9', 'AUC scores10',
        'AP scores1', 'AP scores2', 'AP scores3', 'AP scores4', 'AP scores5',
        # 'AP scores6',
        # 'AP scores7', 'AP scores8', 'AP scores9', 'AP scores10',
        'AUC mean', 'AUC std', 'AP mean', 'AP std', 'Running times'
    ]

    mean_roc = np.array(mean_roc)
    mean_ap = np.array(mean_ap)

    AUC_scores_mean = np.mean(mean_roc)
    AUC_scores_std = np.std(mean_roc)
    AP_scores_mean = np.mean(mean_ap)
    AP_scores_std = np.std(mean_ap)

    new_data = [file] + list(mean_roc) + list(mean_ap) + \
               [AUC_scores_mean, AUC_scores_std, AP_scores_mean, AP_scores_std, time_mean]


    if not os.path.exists(resultspath + resultfile):
        with open(resultspath + resultfile, mode='w', newline='') as file:
            np.savetxt(file, [column_names], delimiter=',', fmt='%s')

    with open(resultspath + resultfile, mode='a', newline='') as file:
        np.savetxt(file, [new_data], delimiter=',', fmt='%s')


utils.results_summary(resultspath, resultfile)
