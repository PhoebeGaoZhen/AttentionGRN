from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score,average_precision_score
from pandas.core.frame import DataFrame
import pandas as pd
import os

class load_data5():
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


def normalize(expression):
    std = StandardScaler()
    epr = std.fit_transform(expression)

    return epr

# def adj2saprse_tensor(adj):
#     coo = adj.tocoo()
#     i = torch.LongTensor([coo.row, coo.col])
#     v = torch.from_numpy(coo.data).float()
#
#     adj_sp_tensor = torch.sparse_coo_tensor(i, v, coo.shape)
#     return adj_sp_tensor


def Evaluation(y_true, y_pred,flag=False):
    if flag:
        y_p = y_pred[:,-1]
        y_p = y_p.numpy()
        y_p = y_p.flatten()
    else:
        y_p = y_pred.numpy()
        y_p = y_p.flatten()


    y_t = y_true.astype(int)

    AUC = roc_auc_score(y_true=y_t, y_score=y_p)


    AUPR = average_precision_score(y_true=y_t,y_score=y_p)
    AUPR_norm = AUPR/np.mean(y_t)


    return AUC, AUPR, AUPR_norm
def masked_accuracy(preds, labels, mask, negative_mask):
    """Accuracy with masking."""
    preds = tf.cast(preds, tf.float32)
    labels = tf.cast(labels, tf.float32)
    error = tf.square(preds-labels)
    mask += negative_mask
    mask = tf.cast(mask, dtype=tf.float32)
    error *= mask
#     return tf.reduce_sum(error)
    return tf.sqrt(tf.reduce_mean(error))


def ROC(outs, labels, test_arr, label_neg):
    scores = []
    for i in range(len(test_arr)):
        l = test_arr[i]
        scores.append(outs[int(labels[l, 0] - 1), int(labels[l, 1] - 1)])

    for i in range(label_neg.shape[0]):
        scores.append(outs[int(label_neg[i, 0]), int(label_neg[i, 1])])

    test_labels = np.ones((len(test_arr), 1))
    temp = np.zeros((label_neg.shape[0], 1))
    test_labels1 = np.vstack((test_labels, temp))
    test_labels1 = np.array(test_labels1, dtype=np.bool).reshape([-1, 1])

    return test_labels1, scores

def load_data_TF_311(leix, data_path):

    import numpy as np
    xxdata_list = []
    yydata = []
    count_set = [0]
    count_setx = 0
    xdata = np.load(data_path+'/Nxdata_tf7_' + leix + '.npy')
    ydata = np.load(data_path+'/ydata_tf7_' + leix + '.npy')
    for k in range(len(ydata)):
        xxdata_list.append(xdata[k,:,:,:])
        yydata.append(ydata[k])
    count_setx = count_setx + len(ydata)
    count_set.append(count_setx)
    yydata_array = np.array(yydata)
    yydata_x = yydata_array.astype('int')
    print(np.array(xxdata_list).shape)
    return((np.array(xxdata_list),yydata_x,count_set))

def load_data_TF_3CV(indel_list, data_path):
    xxdata_list = []
    yydata = []
    count_set = [0]
    count_setx = 0
    for i in indel_list:  # len(h_tf_sc)):
        xdata = np.load(
            data_path + '/Nxdata_tf7_data_process_success' + str(i) + '.npy')
        ydata = np.load(data_path + '/ydata_tf7_data_process_success' + str(i) + '.npy')
        for k in range(int(len(ydata) / 3)):
            xxdata_list.append(xdata[3 * k, :, :, :])
            xxdata_list.append(xdata[3 * k + 2, :, :, :])
            yydata.append(ydata[3 * k])
            yydata.append(ydata[3 * k + 2])
        count_setx = count_setx + int(len(ydata) * 2 / 3)
        count_set.append(count_setx)
        # print(i, len(ydata))
    yydata_array = np.array(yydata)
    yydata_x = yydata_array.astype('int')
    # print(np.array(xxdata_list).shape)
    return ((np.array(xxdata_list), yydata_x, count_set))


def load_data_TF_3CV_2(indel_list, data_path):
    xxdata_list = []
    yydata = []
    zzdata = []
    count_set = []
    count_setx = 0
    for i in indel_list:  # len(h_tf_sc)):
        xdata = np.load(
            data_path + '/Nxdata_tf7_data_process_TRN' + str(i) + '.npy')
        ydata = np.load(data_path + '/ydata_tf7_data_process_TRN' + str(i) + '.npy')
        zdata = np.load(data_path + '/zdata_tf7_data_process_TRN' + str(i) + '.npy')
        for k in range(int(len(ydata) / 2)):
            xxdata_list.append(xdata[2 * k, :, :, :])
            xxdata_list.append(xdata[2 * k + 1, :, :, :])
            yydata.append(ydata[2 * k])
            yydata.append(ydata[2 * k + 1])
            zzdata.append(zdata[2 * k])
            zzdata.append(zdata[2 * k + 1])
            # print('heeeeeew')

        count_setx = count_setx + int(len(ydata) * 2 / 2)
        count_set.append(count_setx)
        # print(i, len(ydata))
    yydata_array = np.array(yydata)
    yydata_x = yydata_array.astype('int')
    # print(np.array(xxdata_list).shape)
    return ((np.array(xxdata_list), yydata_x, count_set))

def results_summary(resultspath, resultfile):

    df = pd.read_csv(resultspath +resultfile)

    save_path_500 = resultspath + '/Top500/'
    if not os.path.exists(save_path_500):
        os.makedirs(save_path_500)
    save_path_1000 = resultspath + '/Top1000/'
    if not os.path.exists(save_path_1000):
        os.makedirs(save_path_1000)


    top500_specific = np.zeros((5))
    top500_nonspecific = np.zeros((5))
    top500_string = np.zeros((5))
    top500_lofgof = np.zeros((5))

    top1000_specific = np.zeros((5))
    top1000_nonspecific = np.zeros((5))
    top1000_string = np.zeros((5))
    top1000_lofgof = np.zeros((5))
    for i in range(df.shape[0]):

        datapath = df.iloc[:,0][i].split('/')
        # cell, networktype, topN = datapath[2], datapath[3], datapath[4]
        cell, networktype, topN = datapath[4], datapath[5], datapath[6]
        print('cell, networktype, topN: ', cell, networktype, topN)

        # cell, topN = datapath[2], datapath[4]

        auc_mean = df.iloc[i,-5]
        auc_std = df.iloc[i,-4]
        aupr_mean = df.iloc[i,-3]
        aupr_std = df.iloc[i,-2]

        # single_results = [cell, networktype, auc_mean, auc_std, aupr_mean, aupr_std]
        single_results = [cell, auc_mean, auc_std, aupr_mean, aupr_std]
        single_results = np.array(single_results)

        if topN == 'Top500':

            # print(networktype)
            if networktype[0:3] == 'Non':
                top500_nonspecific = np.vstack((top500_nonspecific,single_results ))

            elif networktype[0:3] == 'STR':
                top500_string = np.vstack((top500_string, single_results))

            elif networktype[0:11] == 'mESC-lofgof':
                top500_lofgof = np.vstack((top500_lofgof, single_results))

            else:
                top500_specific = np.vstack((top500_specific, single_results))

        elif topN == 'Top1000':
            if networktype[0:3] == 'Non':
                top1000_nonspecific = np.vstack((top1000_nonspecific, single_results))

            elif networktype[0:3] == 'STR':
                top1000_string = np.vstack((top1000_string, single_results))

            elif networktype[0:11] == 'mESC-lofgof':
                top1000_lofgof = np.vstack((top1000_lofgof, single_results))

            else:
                top1000_specific = np.vstack((top1000_specific, single_results))

        else:
            print('error')

    top500_specific_new = top500_specific[0:2]

    if top500_specific.shape[0] == 7:
        s = ["hHEP", 0,0,0,0]
        s2 = np.array(s).reshape(1,-1)
        # top500_specific_new = top500_specific[0:2]
        top500_specific_new = np.vstack((top500_specific_new, s2))
        top500_specific_new = np.vstack((top500_specific_new, top500_specific[2:7]))

        top500 = np.vstack((top500_specific_new, top500_nonspecific))
        top500 = np.vstack((top500, top500_string))
        top500 = np.vstack((top500, top500_lofgof))
        # print(top500)
        top1000 = np.vstack((top1000_specific, top1000_nonspecific))
        top1000 = np.vstack((top1000, top1000_string))
        top1000 = np.vstack((top1000, top1000_lofgof))

        top500_df = DataFrame(top500)
        top1000_df = DataFrame(top1000)

        top500_df.to_csv(save_path_500  + 'average_results.csv', index=None)
        top1000_df.to_csv(save_path_1000  + 'average_results.csv', index=None)

        print("Top500 results of GRN inference are saved in " + save_path_500)
        print("Top1000 results of GRN inference are saved in " + save_path_1000)
    elif top500_specific.shape[0] == 8:
        top500 = np.vstack((top500_specific, top500_nonspecific))
        top500 = np.vstack((top500, top500_string))
        top500 = np.vstack((top500, top500_lofgof))
        # print(top500)
        top1000 = np.vstack((top1000_specific, top1000_nonspecific))
        top1000 = np.vstack((top1000, top1000_string))
        top1000 = np.vstack((top1000, top1000_lofgof))

        top500_df = DataFrame(top500)
        top1000_df = DataFrame(top1000)

        top500_df.to_csv(save_path_500 + 'average_results.csv', index=None)
        top1000_df.to_csv(save_path_1000 + 'average_results.csv', index=None)

        print("Top500 results of GRN inference are saved in " + save_path_500)
        print("Top1000 results of GRN inference are saved in " + save_path_1000)
    else:
        print('error')

