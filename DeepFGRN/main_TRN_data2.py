
import warnings
warnings.filterwarnings("ignore")
import os,time,argparse
from keras import backend as K
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split

import tensorflow as tf
# from keras.backend import clear_session
import utils_Train_Test_Split as tts
import dggan_embedding_param, DeepFGRN_utils
import corrresnet_pred_224 as corrresnet_pred

parser = argparse.ArgumentParser()
parser.add_argument('--modelname', type=str, default='DeepFGRN', help='DeepFGRN-3CV')
parser.add_argument('--GRN_type', type=str, default='tf-gene', help='tf-gene; gene-gene')
parser.add_argument('--dataset_type', type=str, default='unbalanced', help='balanced; unbalanced')
parser.add_argument('--iteration', type=int, default=5, help='the number of training and test')
parser.add_argument('--nb_classes', type=int, default=2, help='3 or 2')
args = parser.parse_args()


output_directory = '.\\output_directory\\' + args.modelname + '\\'
if not os.path.isdir(output_directory):
    os.makedirs(output_directory)

dataset_names = ['hESC','hHEP','mDC','mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']
processed_data = 'DATA2'
save_folder = 'DeepFGRN_data2_TRN\\'

for Rank_num in [500,1000]:
    for dataset_name in dataset_names:
        # dataset_name = 'hESC' # hESC; hHep; mDC; mESC; mHSC-E; mHSC-GM; mHSC-L
        network_types = []
        if dataset_name=='hESC':
            network_types = ['hESC-ChIP-seq-network','Non-specific-ChIP-seq-network','STRING-network']
        elif dataset_name=='hHEP':
            network_types = ['HepG2-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']
        elif dataset_name=='mDC':
            network_types = ['mDC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']
        elif dataset_name=='mESC':
            network_types = ['mESC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network','mESC-lofgof-network']
        elif dataset_name=='mHSC-E' or dataset_name=='mHSC-GM' or dataset_name=='mHSC-L':
            network_types = ['mHSC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']
        else:
            print("network type error")

        print(dataset_name + ': ' )
        print(network_types)

        for network_type in network_types:
            print('\n\n****************************'+ dataset_name + '————' + network_type + '————' + str(Rank_num) + '****************************')

            network_dict_name = args.modelname + '_3CV_'

            save_path = save_folder + dataset_name + '\\' + network_type + '\\' + 'Top' + str(Rank_num) + '\\'
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            data_path = '..\\data_preprocess\\' + processed_data + '\\' + dataset_name + '\\' + network_type + '\\Top' + str(
            Rank_num) + '\\'
            exp_file = data_path + 'Final_expression.csv'
            tf_file = data_path + 'Final_TF_index.csv'
            target_file = data_path + 'Final_gene_list.csv'
            label_file = data_path + 'Final_GRNorTRN_pos_index.csv'
            label_file_name = data_path + 'Final_GRNorTRN_pos.csv'

            tf_list = pd.read_csv(data_path + 'tf_list.csv', header=None).values
            pos_neg_balanced_id = pd.read_csv(data_path + 'pos_neg_balanced_id.csv', header=None)
            pos_neg_balanced_name = pd.read_csv(data_path + 'pos_neg_balanced_name.csv', header=None)

            new_node = DeepFGRN_utils.get_nodeid(target_file, save_path)
            DeepFGRN_utils.get_GRN_ids(label_file_name, save_path, new_node)

            path_network_ids = save_path + 'Gold_Standard_Network_ids.tsv'
            path_node = save_path + 'final_node.txt'

            columns = []
            kf = KFold(n_splits=3, shuffle=True)
            network_dict = {}
            all_network_dict = {}
            netavgAUROCs = []
            netavgAUPRs = []
            netavgSPEs = []
            netavgRecalls = []
            netavgPrecisions = []
            netavgF1s = []
            netavgMCCs = []
            netavgAccs = []
            for ki in range(args.iteration):
                columns.append(str(ki+1) + '-th 3CV')

                print('\n')
                print("\nthe {}th cross-validation..........\n".format(ki + 1))

                K.clear_session()


                GRN_embedding_s, GRN_embedding_t = dggan_embedding_param.dggan(path_network_ids, path_node)

                # 3CV
                AUROCs = []
                AUPRs = []
                # SPEs = []
                # Recalls = []
                # Precisions = []
                # F1s = []
                # MCCs = []
                # Accs = []

                kf = KFold(n_splits=3, shuffle=True)
                for train_index, test_index in kf.split(tf_list):

                    train_TF = tf_list[train_index]
                    test_TF = tf_list[test_index]

                    train_pair, train_pair_label = tts.get_samples_pair(pos_neg_balanced_id, train_TF, exp_file)
                    test_pair, test_pair_label = tts.get_samples_pair(pos_neg_balanced_id, test_TF, exp_file)

                    # train_pair, validation_pair = train_test_split(train_pair, test_size=0.2, random_state=1, shuffle=True)
                    train_pair, validation_pair, train_label, validation_label = train_test_split(train_pair, train_pair_label, test_size=0.2, random_state=1, shuffle=True, stratify = train_pair_label) # , random_state=seed

                    train_sample = tts.get_samples_DeepFGRN_3CV(exp_file, target_file, train_pair, GRN_embedding_s, GRN_embedding_t)
                    validation_sample = tts.get_samples_DeepFGRN_3CV(exp_file, target_file, validation_pair, GRN_embedding_s, GRN_embedding_t)
                    test_sample = tts.get_samples_DeepFGRN_3CV(exp_file, target_file, test_pair, GRN_embedding_s, GRN_embedding_t)
                    # print('train sample size: ', train_sample.shape)
                    # print('validation sample size: ', validation_sample.shape)
                    # print('test sample size: ' , test_sample.shape)


                    x_train_tf, x_train_target, x_train_net_tf_s, x_train_net_tf_t, x_train_net_target_s, x_train_net_target_t, y_train = DeepFGRN_utils.transform_data(train_sample)
                    x_val_tf, x_val_target, x_val_net_tf_s, x_val_net_tf_t, x_val_net_target_s, x_val_net_target_t, y_val = DeepFGRN_utils.transform_data(validation_sample)
                    x_test_tf, x_test_target, x_test_net_tf_s, x_test_net_tf_t, x_test_net_target_s, x_test_net_target_t, y_test = DeepFGRN_utils.transform_data(test_sample)


                    classifier = corrresnet_pred.Classifier_corrResNET_pred(output_directory, args.nb_classes, x_train_tf, x_train_target,
                                                                                x_train_net_tf_s, x_train_net_tf_t, x_train_net_target_s,
                                                                                x_train_net_target_t, verbose=True, patience=5)
                    score_1, score_int = classifier.fit_5CV(x_train_tf, x_train_target, x_train_net_tf_s, x_train_net_tf_t,
                                                                x_train_net_target_s,x_train_net_target_t, y_train,
                                                                x_val_tf, x_val_target, x_val_net_tf_s, x_val_net_tf_t,
                                                                x_val_net_target_s, x_val_net_target_t, y_val,
                                                                x_test_tf, x_test_target, x_test_net_tf_s, x_test_net_tf_t,
                                                                x_test_net_target_s, x_test_net_target_t)

                    Recall, SPE, Precision, F1, MCC, ACC, AUC, AUPR1, AUPR2 = DeepFGRN_utils.two_scores(y_test[:, 1],
                                                                                               score_1[:, 1],
                                                                                               th=0.5)
                    AUROCs.append(AUC)
                    AUPRs.append(AUPR1)
                    # AUPRs2.append(AUPR2)
                    # SPEs.append(SPE)
                    # Recalls.append(Recall)
                    # Precisions.append(Precision)
                    # F1s.append(F1)
                    # MCCs.append(MCC)
                    # Accs.append(ACC)

                    # print('\n')


                avg_AUROC = np.mean(AUROCs)
                avg_AUPR = np.mean(AUPRs)
                # avg_SPE = np.mean(SPEs)
                # avg_Recalls = np.mean(Recalls)
                # avg_Precisions = np.mean(Precisions)
                # avg_F1s = np.mean(F1s)
                # avg_MCCs = np.mean(MCCs)
                # avg_Accs = np.mean(Accs)

                netavgAUROCs.append(avg_AUROC)
                netavgAUPRs.append(avg_AUPR)
                # netavgSPEs.append(avg_SPE)
                # netavgRecalls.append(avg_Recalls)
                # netavgPrecisions.append(avg_Precisions)
                # netavgF1s.append(avg_F1s)
                # netavgMCCs.append(avg_MCCs)
                # netavgAccs.append(avg_Accs)

            all_network_dict["AUROC"] = netavgAUROCs
            all_network_dict["AUPR"] = netavgAUPRs
            # all_network_dict["SPE"] = netavgSPEs
            # all_network_dict["Recall"] = netavgRecalls
            # all_network_dict["Precision"] = netavgPrecisions
            # all_network_dict["F1"] = netavgF1s
            # all_network_dict["MCC"] = netavgMCCs
            # all_network_dict["Acc"] = netavgAccs

            filename = open(save_path + network_dict_name + '_all.csv', 'w')
            for k, v in all_network_dict.items():
                filename.write(k + ':' + str(v))
                filename.write('\n')
            filename.close()












