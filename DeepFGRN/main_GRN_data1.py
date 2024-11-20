

import warnings
warnings.filterwarnings("ignore")
import os,time,argparse
from keras import backend as K
import pandas as pd
import numpy as np
import tensorflow as tf
# from keras.backend import clear_session
import utils_Train_Test_Split as tts
import dggan_embedding_param, DeepFGRN_utils
import corrresnet_pred_224 as corrresnet_pred

parser = argparse.ArgumentParser()
parser.add_argument('--modelname', type=str, default='DeepFGRN', help='modelname')
parser.add_argument('--hard', type=bool, default=False, help='not hard sample')
parser.add_argument('--GRN_type', type=str, default='tf-gene-posneg', help='tf-gene; gene-gene; tf-gene-posneg')
parser.add_argument('--dataset_type', type=str, default='unbalanced', help='balanced; unbalanced')
parser.add_argument('--iteration', type=int, default=5, help='the number of training and test')
parser.add_argument('--nb_classes', type=int, default=2, help='3 or 2')
args = parser.parse_args()
output_directory = '.\\output_directory\\' + args.modelname + '\\'
if not os.path.isdir(output_directory):
    os.makedirs(output_directory)

processed_data = 'DATA1_AttentionGRN'
save_folder = '.\\DeepFGRN_data1_GRN\\'
dataset_names = [ 'hESC','hHEP','mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']


for Rank_num in [500, 1000]:

    for dataset_name in dataset_names:
        network_types = []
        if dataset_name == 'hESC':
            network_types = ['hESC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']

        elif dataset_name == 'hHEP' and Rank_num == 500:
            network_types = ['Non-specific-ChIP-seq-network', 'STRING-network']

        elif dataset_name == 'hHEP' and Rank_num == 1000:
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

            logTime = time.strftime("%Y-%m-%d-%H%M%S", time.localtime())
            # network_dict_name = args.modelname + '_311_' + logTime
            network_dict_name = args.modelname + '_311_'
            save_index_path = save_folder + dataset_name + '\\' + network_type + '\\' + 'Top' + str(Rank_num) + '\\'
            if not os.path.isdir(save_index_path):
                os.makedirs(save_index_path)

            data_path = '..\\data_preprocess\\' + processed_data + '\\' + dataset_name + '\\' + network_type + '\\Top' + str(
                Rank_num) + '\\'


            exp_file = data_path + 'Final_expression.csv'
            tf_file = data_path + 'Final_TF_common_index.csv'
            target_file = data_path + 'Final_gene_list.csv'
            label_file = data_path + 'Final_GRNorTRN_pos_index.csv'
            label_file_name = data_path + 'Final_GRNorTRN_pos.csv'

            # ---------------transform data--------------------
            new_node = DeepFGRN_utils.get_nodeid(target_file, save_index_path)
            DeepFGRN_utils.get_GRN_ids(label_file_name, save_index_path, new_node)

            path_network_ids = save_index_path + 'Gold_Standard_Network_ids.tsv'
            path_node = save_index_path + 'final_node.txt'

            # ---------------train test split---------------------
            network_dict = {}
            all_network_dict = {}
            AUROCs = []
            AUPRs1 = []
            AUPRs2 = []

            SPEs = []
            Recalls = []
            Precisions = []
            F1s = []
            MCCs = []
            Accs = []
            for ki in range(args.iteration):

                print('-----------------the ' + str(ki+1) + 'th iteration for ' + network_type + ' Top '+ str(Rank_num) + '----------------------------------')
                K.clear_session()

                if args.hard:
                    train_set_file = save_index_path + 'Train_set_hard.csv'
                    test_set_file = save_index_path + 'Test_set_hard.csv'
                    val_set_file = save_index_path + 'Validation_set_hard.csv'
                    metric_file = save_index_path + 'metric_hard.csv'
                    density = tts.Network_Statistic(data_type=dataset_name, net_scale=Rank_num, net_type=network_type)
                    tts.train_val_test_set_hard(label_file, target_file, tf_file, train_set_file, val_set_file, test_set_file, args.GRN_type,
                                                args.dataset_type, density, p_val=0.5)
                else:
                    train_set_file = save_index_path + 'Train_set.csv'
                    test_set_file = save_index_path + 'Test_set.csv'
                    val_set_file = save_index_path + 'Validation_set.csv'
                    metric_file = save_index_path + 'metric.csv'
                    tts.train_val_test_set(label_file, target_file, tf_file, train_set_file, val_set_file, test_set_file, args.GRN_type, args.dataset_type)

                train_data = pd.read_csv(train_set_file, index_col=0).values   # tf ID -- target ID -- label
                validation_data = pd.read_csv(val_set_file, index_col=0).values  # tf ID -- target ID -- label
                test_data = pd.read_csv(test_set_file, index_col=0).values  # tf ID -- target ID -- label

                GRN_embedding_s, GRN_embedding_t = dggan_embedding_param.dggan(path_network_ids, path_node)

                train_data_sample = tts.get_samples_DeepFGRN(exp_file,target_file,train_data,GRN_embedding_s, GRN_embedding_t)
                validation_data_sample = tts.get_samples_DeepFGRN(exp_file,target_file,validation_data,GRN_embedding_s, GRN_embedding_t)
                test_data_sample = tts.get_samples_DeepFGRN(exp_file,target_file,test_data,GRN_embedding_s, GRN_embedding_t)

                x_train_tf, x_train_target, x_train_net_tf_s, x_train_net_tf_t, x_train_net_target_s, x_train_net_target_t, y_train = DeepFGRN_utils.transform_data(train_data_sample)
                x_val_tf, x_val_target, x_val_net_tf_s, x_val_net_tf_t, x_val_net_target_s, x_val_net_target_t, y_val = DeepFGRN_utils.transform_data(validation_data_sample)
                x_test_tf, x_test_target, x_test_net_tf_s, x_test_net_tf_t, x_test_net_target_s, x_test_net_target_t, y_test = DeepFGRN_utils.transform_data(test_data_sample)


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
                AUPRs1.append(AUPR1)
                AUPRs2.append(AUPR2)
                SPEs.append(SPE)
                Recalls.append(Recall)
                Precisions.append(Precision)
                F1s.append(F1)
                MCCs.append(MCC)
                Accs.append(ACC)


            AUROC_mean = np.mean(AUROCs)
            AUROC_std = np.std(AUROCs, ddof=1)
            AUPR_mean1 = np.mean(AUPRs1)
            AUPR_std1 = np.std(AUPRs1)
            AUPR_mean2 = np.mean(AUPRs2)
            AUPR_std2 = np.std(AUPRs2)
            SPE_mean = np.mean(SPEs)
            SPE_std = np.std(SPEs)
            Recall_mean = np.mean(Recalls)
            Recall_std = np.std(Recalls)
            Precision_mean = np.mean(Precisions)
            Precision_std = np.std(Precisions)
            F1_mean = np.mean(F1s)
            F1_std = np.std(F1s)
            MCC_mean = np.mean(MCCs)
            MCC_std = np.std(MCCs)
            Acc_mean = np.mean(Accs)
            Acc_std = np.std(Accs)

            AUROC_mean = float('{:.4f}'.format(AUROC_mean))
            AUROC_std = float('{:.4f}'.format(AUROC_std))
            AUPR_mean1 = float('{:.4f}'.format(AUPR_mean1))
            AUPR_std1 = float('{:.4f}'.format(AUPR_std1))
            AUPR_mean2 = float('{:.4f}'.format(AUPR_mean2))
            AUPR_std2 = float('{:.4f}'.format(AUPR_std2))
            SPE_mean = float('{:.4f}'.format(SPE_mean))
            SPE_std = float('{:.4f}'.format(SPE_std))
            Recall_mean = float('{:.4f}'.format(Recall_mean))
            Recall_std = float('{:.4f}'.format(Recall_std))
            Precision_mean = float('{:.4f}'.format(Precision_mean))
            Precision_std = float('{:.4f}'.format(Precision_std))
            F1_mean = float('{:.4f}'.format(F1_mean))
            F1_std = float('{:.4f}'.format(F1_std))
            MCC_mean = float('{:.4f}'.format(MCC_mean))
            MCC_std = float('{:.4f}'.format(MCC_std))
            Acc_mean = float('{:.4f}'.format(Acc_mean))
            Acc_std = float('{:.4f}'.format(Acc_std))

            network_dict["AUROC mean"] = AUROC_mean
            network_dict["AUROC std"] = AUROC_std
            network_dict["AUPR1 mean"] = AUPR_mean1
            network_dict["AUPR1 std"] = AUPR_std1
            network_dict["AUPR2 mean"] = AUPR_mean2
            network_dict["AUPR2 std"] = AUPR_std2
            network_dict["SPE mean"] = SPE_mean
            network_dict["SPE std"] = SPE_std
            network_dict["Recall mean"] = Recall_mean
            network_dict["Recall std"] = Recall_std
            network_dict["Precision mean"] = Precision_mean
            network_dict["Precision std"] = Precision_std
            network_dict["F1 mean"] = F1_mean
            network_dict["F1 std"] = F1_std
            network_dict["MCC mean"] = MCC_mean
            network_dict["MCC std"] = MCC_std
            network_dict["Acc mean"] = Acc_mean
            network_dict["Acc std"] = Acc_std


            filename = open(save_index_path + network_dict_name + '_avg.csv', 'w')
            for k, v in network_dict.items():
                filename.write(k + ':' + str(v))
                filename.write('\n')
            filename.close()

            all_network_dict["AUROC"] = AUROCs
            all_network_dict["AUPR1"] = AUPRs1
            all_network_dict["AUPR2"] = AUPRs2
            all_network_dict["SPE"] = SPEs
            all_network_dict["Recall"] = Recalls
            all_network_dict["Precision"] = Precisions
            all_network_dict["F1"] = F1s
            all_network_dict["MCC"] = MCCs
            all_network_dict["Acc"] = Accs

            filename = open(save_index_path + network_dict_name + '_all.csv', 'w')
            for k, v in all_network_dict.items():
                filename.write(k + ':' + str(v))
                filename.write('\n')
            filename.close()

















