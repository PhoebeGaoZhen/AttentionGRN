'''
202409
datasets: DATA1
model: STGRNS
task: TRN inference
evaluation strategy: TF-aware three-fold cross-validation

positive samples: known TF-gene pairs
negative samples: unknown TF-gene pairs, each TF is same as TF in positive samples
balanced datasets
'''

import numpy as np
import pandas as pd
from numpy import *
import os, argparse, torch
from sklearn.model_selection import train_test_split, KFold
import torch.nn.functional as F
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
from sklearn import metrics
import utils_data, utils_STGRNS

# 参数
parser = argparse.ArgumentParser()
parser.add_argument('--modelname', type=str, default='STGRNS', help='STGRNS')
parser.add_argument('--iteration', type=int, default=5, help='the number of training and test')
parser.add_argument('--gap', type=int, default=100, help='gap STGRNS')
parser.add_argument('--batch_size', type=int, default='32', help='batch_sizes')
parser.add_argument('--n_epochs', type=int, default='20', help='epochs')
parser.add_argument('--d_models', type=int, default='200', help='epochs')
parser.add_argument('--input_dim', type=int, default='200', help='epochs')

args = parser.parse_args()

dataset_names = ['hESC', 'hHEP', 'mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']

processed_data = '..\\data_preprocess\\DATA2'

for Rank_num in [500, 1000]:
    for dataset_name in dataset_names:
        network_types = []
        if dataset_name == 'hESC':
            network_types = ['hESC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']
        elif dataset_name == 'hHEP':
            network_types = ['HepG2-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']
        elif dataset_name == 'mDC':
            network_types = [ 'mDC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']
        elif dataset_name == 'mESC':
            network_types = ['mESC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network',
                             'mESC-lofgof-network']
        elif dataset_name == 'mHSC-E' or dataset_name == 'mHSC-GM' or dataset_name == 'mHSC-L':
            network_types = ['mHSC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network'] #
        else:
            print("network type error")

        print(dataset_name + ': ')
        print(network_types)
        for network_type in network_types:
            print('\n\n****************************'+ dataset_name + '————' + network_type + '————' + str(Rank_num) + '****************************')

            data_path = processed_data + '\\' + dataset_name + '\\' + network_type + '\\Top' + str(
                Rank_num) + '\\'
            exp_file = data_path + 'Final_expression.csv'
            tf_file = data_path + 'Final_TF_index.csv'
            target_file = data_path + 'Final_gene_list.csv'
            label_file = data_path + 'Final_GRNorTRN_pos_index.csv'
            label_name_file = data_path + 'Final_GRNorTRN_pos.csv'

            save_path = 'STGRNS_data2_TRN\\' + dataset_name + '\\' + network_type + '\\Top' + str(Rank_num) + '\\'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            network_dict_name = args.modelname + '_3CV_'

            gene_pairs_num_dir = save_path + "gene_pairs_num.txt"

            utils_data.get_nodeid(target_file, save_path+ 'gene_list_id.txt')
            sc_gene_list = utils_data.get_gene_list(save_path+ 'gene_list_id.txt')

            tf_list = pd.read_csv(data_path + 'tf_list.csv', header=None).values
            pos_neg_balanced_id = pd.read_csv(data_path + 'pos_neg_balanced_id.csv', header=None)
            pos_neg_balanced_name = pd.read_csv(data_path + 'pos_neg_balanced_name.csv', header=None)

            gene_pair_label_array = np.array(pos_neg_balanced_name)

            expression = pd.read_csv(exp_file, index_col=0)

            x = []
            y = []
            z = []
            for gene_pair in gene_pair_label_array:
                x_tf_name = gene_pair[0]
                x_gene_name = gene_pair[1]
                label = gene_pair[2]
                y.append(label)
                z.append(x_tf_name + ',' + x_gene_name)

                x_tf = expression.loc[:,x_tf_name]
                x_gene = expression.loc[:, x_gene_name]

                single_tf_list = []
                for k in range(0, len(x_gene), args.gap):
                    feature = []
                    a = x_tf[k:k + args.gap]
                    b = x_gene[k:k + args.gap]

                    feature.extend(a)
                    feature.extend(b)

                    feature = np.asarray(feature)

                    if (len(feature) == 2 * args.gap):
                        # print("feature.shape xixihaha", feature.shape)
                        single_tf_list.append(feature)
                    # sample_sizex = len(RPKMs)
                single_tf_list = np.asarray(single_tf_list)
                x.append(single_tf_list)

            save(save_path + 'Nxdata_tf.npy', x)
            save(save_path + 'ydata_tf.npy', array(y))
            save(save_path + 'zdata_tf.npy', array(z))

            # ------------------------3CV-------------------------------------------

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
            ki = 0
            # for ki in range(args.iteration):
            while ki != args.iteration:
            # for ki in range(args.iteration):
                columns.append(str(ki + 1) + '-th 3CV')

                print('\n')
                print("\nthe {}th cross-validation..........\n".format(ki + 1))

                log_dir = "logTF-GENE/relation_nopro/" + dataset_name  + str(ki + 1) + "/"
                if (not os.path.isdir(log_dir)):
                    os.makedirs(log_dir)

                # 3CV
                AUROCs = []
                AUPRs = []
                # SPEs = []
                # Recalls = []
                # Precisions = []
                # F1s = []
                # MCCs = []
                # Accs = []

                for train_index, test_index in kf.split(tf_list):

                    train_TF = tf_list[train_index]
                    test_TF = tf_list[test_index]
                    train_pair_x, train_pair_y, train_pair = utils_data.get_samples_pair(pos_neg_balanced_id, train_TF, save_path)
                    test_pair_x, test_pair_y, test_pair = utils_data.get_samples_pair(pos_neg_balanced_id, test_TF, save_path)

                    train_data, validation_data, train_y, val_y = train_test_split(train_pair_x, train_pair_y, test_size=0.2, random_state=1,
                                                                   shuffle=True, stratify=train_pair_y)  # , random_state=seed
                    x_train, y_train = train_data, train_y
                    x_val, y_val = validation_data, val_y
                    x_test, y_test = test_pair_x, test_pair_y

                    if x_test.shape[0] == 0:
                        print('test sample size is 0')
                        ki = ki
                        break

                    else:

                        X_trainloader, y_trainloader = utils_STGRNS.numpy2loader(x_train, y_train, args.batch_size)
                        X_valloader, y_valloader = utils_STGRNS.numpy2loader(x_val, y_val, args.batch_size)
                        X_testloader, y_testloader = utils_STGRNS.numpy2loader(x_test, y_test, args.batch_size)

                        X_trainList = utils_STGRNS.loaderToList(X_trainloader)
                        y_trainList = utils_STGRNS.loaderToList(y_trainloader)

                        X_valList = utils_STGRNS.loaderToList(X_valloader)
                        y_valList = utils_STGRNS.loaderToList(y_valloader)

                        X_testList = utils_STGRNS.loaderToList(X_testloader)
                        y_testList = utils_STGRNS.loaderToList(y_testloader)

                        model = utils_STGRNS.STGRNS(input_dim=args.input_dim, nhead=2, d_model=args.d_models, num_classes=2)

                        criterion = nn.CrossEntropyLoss()
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)


                        loss_record = {'train': [], 'dev': []}
                        acc_record = {'train': [], 'dev': []}
                        for epoch in range(args.n_epochs):
                            model.train()

                            train_loss = []
                            train_accs = []

                            for j in range(0, len(X_trainList)):
                                data = X_trainList[j]
                                labels = y_trainList[j]
                                logits = model(data)

                                labels = torch.tensor(labels, dtype=torch.long)
                                loss = criterion(logits, labels)
                                optimizer.zero_grad()
                                loss.backward()
                                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                                optimizer.step()
                                acc = (logits.argmax(dim=-1) == labels).float().mean()
                                train_accs.append(acc)

                                train_loss.append(loss.item())

                            train_loss = sum(train_loss) / len(train_loss)
                            train_acc = sum(train_accs) / len(train_accs)

                            loss_record['train'].append(train_loss)
                            acc_record['train'].append(train_acc)

                            print(f"[ Train | {epoch + 1:03d}/{args.n_epochs:03d} ] loss = {train_loss:.5f}")

                            model.eval()
                            valid_loss = []
                            valid_accs = []
                            predictions = []
                            labelss = []
                            y_test = []
                            y_predict = []

                            for k in range(0, len(X_valList)):
                                data = X_valList[k]
                                labels = y_valList[k]
                                labels = torch.tensor(labels, dtype=torch.long)
                                with torch.no_grad():
                                    logits = model(data)
                                loss = criterion(logits, labels)
                                valid_loss.append(loss.item())

                                acc = (logits.argmax(dim=-1) == labels).float().mean()
                                valid_accs.append(acc)

                                predt = F.softmax(logits)
                                labelss.extend(labels.cpu().numpy().tolist())
                                y_test.extend(labels.cpu().numpy())

                                temps = predt.cpu().numpy().tolist()
                                for i in temps:
                                    t = i[1]
                                    y_predict.append(t)
                                predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
                            valid_loss = sum(valid_loss) / len(valid_loss)
                            valid_acc = sum(valid_accs) / len(valid_accs)

                            loss_record['dev'].append(valid_loss)
                            acc_record['dev'].append(valid_acc)

                            print(f"[ Valid | {epoch + 1:03d}/{args.n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

                            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)
                            auc = metrics.auc(fpr, tpr)
                            precision, recall, thresholds_PR = metrics.precision_recall_curve(y_test, y_predict)
                            AUPR = metrics.auc(recall, precision)
                            acc = metrics.accuracy_score(labelss, predictions)
                            bacc = metrics.balanced_accuracy_score(labelss, predictions)
                            f1 = metrics.f1_score(labelss, predictions)

                            print("acc:", acc, "auc:", auc, "aupr:", AUPR, "bacc", bacc, "f1", f1)

                        model_path = log_dir + dataset_name + '.tar'
                        #         print("y_test",y_test)
                        model.eval()
                        y_test = []
                        y_predict = []
                        for k in range(0, len(X_testList)):
                            data = X_testList[k]
                            labels = y_testList[k]
                            labels = torch.tensor(labels, dtype=torch.long)
                            y_test.extend(labels.cpu().numpy())
                            with torch.no_grad():
                                logits = model(data)
                            predt = F.softmax(logits)
                            temps = predt.cpu().numpy().tolist()
                            for i in temps:
                                t = i[1]
                                y_predict.append(t)

                        np.save(log_dir + 'y_test.npy', y_test)
                        np.save(log_dir + 'y_predict.npy', y_predict)
                        torch.save(model.state_dict(), model_path)

                        AUC, AUPR = utils_STGRNS.Evaluation(y_pred=y_predict, y_true=y_test)

                        AUROCs.append(AUC)
                        AUPRs.append(AUPR)
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
                ki += 1
                print(ki)

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










