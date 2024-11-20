
import numpy as np
import pandas as pd
from numpy import *
import os, argparse, torch
from sklearn.model_selection import train_test_split, KFold
import torch.nn.functional as F
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
from model import TransformerE
import utils_gz, utils_GRN

parser = argparse.ArgumentParser()
parser.add_argument('--modelname', type=str, default='AttentionGRN', help='model name')
parser.add_argument('--iteration', type=int, default=5, help='the number of training and test')
parser.add_argument('--batch_sizes', type=int, default='256', help='batch_sizes')
parser.add_argument('--gap', type=int, default=100, help='gap STGRNS')
parser.add_argument('--out_dim_GT', type=int, default='64', help='the output of GT model')  #
parser.add_argument('--hidden_dim_GT', type=int, default='64', help='the hidden of GT model')  #
parser.add_argument('--num_heads_GT', type=int, default='4', help='the heads of GT model')  #
parser.add_argument('--input_dim_exp', type=int, default='200', help='the input of expression Transformer model')  #
parser.add_argument('--d_models', type=int, default='200', help='the input of expression Transformer model')  #
parser.add_argument('--k_hop', type=int, default='1', help='[1, 2]')  #

args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset_names = ['hESC', 'hHEP', 'mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']
# dataset_names = ['mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']

save_folder = 'AttentionGRN_data2_TRN'
processed_data = 'DATA2'

for Rank_num in [500, 1000]:
    for dataset_name in dataset_names:
        network_types = []
        corr_g_cutoffs = []
        # epochs_all = []
        if Rank_num == 500:
            if dataset_name == 'hESC':
                network_types = [ 'hESC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network'] #
                corr_g_cutoffs = [2.9, 3, 2.8] # mean + std*3， top 500
                epochs_all = [20, 10, 20]
            elif dataset_name == 'hHEP':
                network_types = ['HepG2-ChIP-seq-network', 'Non-specific-ChIP-seq-network','STRING-network'] #
                corr_g_cutoffs = [2.5, 2.8, 2.5] # mean + std*3， top 500
                epochs_all = [20, 20, 20]
            elif dataset_name == 'mDC':
                network_types = ['mDC-ChIP-seq-network','Non-specific-ChIP-seq-network', 'STRING-network'] #
                corr_g_cutoffs = [2.6, 2.7, 2.7] # mean + std*3， top 500
                epochs_all = [30, 30, 30]
            elif dataset_name == 'mESC':
                network_types = ['mESC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network', 'mESC-lofgof-network']  #
                corr_g_cutoffs = [2.7, 2.6, 2.7, 2.7] # mean + std*3， top 500
                epochs_all = [30, 30, 20, 30]
            elif dataset_name == 'mHSC-E':
                network_types = ['mHSC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network'] #
                corr_g_cutoffs = [3, 2.6, 2.8] # mean + std*3， top 500
                epochs_all = [20, 30, 20]
            elif dataset_name == 'mHSC-GM':
                network_types = ['mHSC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network'] #
                corr_g_cutoffs = [2.5, 3, 2.5] # mean + std*3， top 500
                epochs_all = [20, 20, 30]
            elif dataset_name == 'mHSC-L':
                network_types = ['mHSC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network'] #
                corr_g_cutoffs = [2.8, 2.5, 2.8] # mean + std*3， top 500
                epochs_all = [20, 30, 30]
            else:
                print("network type error")
        elif Rank_num==1000:
            if dataset_name == 'hESC':
                network_types = ['hESC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']
                corr_g_cutoffs = [2.5, 2.7, 2.9]  # mean + std*3， top 1000
                epochs_all = [20, 10, 20]
            elif dataset_name == 'hHEP':
                network_types = ['HepG2-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']
                corr_g_cutoffs = [2.6, 2.6, 2.7]  # mean + std*3， top 1000
                epochs_all = [10, 10, 30]
            elif dataset_name == 'mDC':
                network_types = ['mDC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']
                corr_g_cutoffs = [3, 2.7, 2.9]  # mean + std*3， top 1000
                epochs_all = [20, 20, 30]
            elif dataset_name == 'mESC':
                network_types = ['mESC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network',
                                 'mESC-lofgof-network']
                corr_g_cutoffs = [3, 2.6, 2.8, 2.7]  # mean + std*3， top 1000
                epochs_all = [20, 10, 10, 20]
            elif dataset_name == 'mHSC-E':
                network_types = ['mHSC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']
                corr_g_cutoffs = [2.6, 2.7, 3]  # mean + std*3， top 1000
                epochs_all = [10, 10, 10]
            elif dataset_name == 'mHSC-GM':
                network_types = ['mHSC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']
                corr_g_cutoffs = [2.5, 2.8, 2.9]  # mean + std*3， top 1000
                epochs_all = [20, 10, 20]
            elif dataset_name == 'mHSC-L':
                network_types = ['mHSC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']
                corr_g_cutoffs = [3, 2.7, 2.9]  # mean + std*3， top 1000
                epochs_all = [10, 20, 30]
            else:
                print("network type error")
        else:
            print('Rank number error')


        print(dataset_name + ': ')
        print(network_types)


        for nt in range(len(network_types)):
            network_type = network_types[nt]
            cutoff = corr_g_cutoffs[nt]
            epochs = epochs_all[nt]
        # for Rank_num in Ranknums:
            print('\n\n****************************'+ dataset_name + '————' + network_type + '————' + str(Rank_num) + '****************************')

            data_path = '..\\data_preprocess\\' + processed_data + '\\' + dataset_name + '\\' + network_type + '\\Top' + str(
                Rank_num) + '\\'

            exp_file = data_path + 'Final_expression.csv'
            tf_file = data_path + 'Final_TF_index.csv'
            gene_list_file = data_path + 'Final_gene_list.csv'
            label_file = data_path + 'Final_GRNorTRN_pos_index.csv'
            label_name_file = data_path + 'Final_GRNorTRN_pos.csv'

            corr_g = utils_gz.get_corr(exp_file, gene_list_file,
                                       'cosine', cutoff)  # 'cosine', 'pearson', 'kendall', 'spearman'

            save_path = save_folder + '\\' + dataset_name + '\\' + network_type + '\\Top' + str(Rank_num) + '\\'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            network_dict_name = args.modelname + '_3CV_'

            tf_list = pd.read_csv(data_path + 'tf_list.csv', header=None).values
            pos_neg_balanced_id = pd.read_csv(data_path + 'pos_neg_balanced_id.csv', header=None) # balanced samples, hard negative sample
            pos_neg_balanced_name = pd.read_csv(data_path + 'pos_neg_balanced_name.csv', header=None)

            gene_pair_label_array = np.array(pos_neg_balanced_name)

            expression = pd.read_csv(exp_file, index_col=0)

            x = []  # gene expression data
            y = []  # labels
            z = []  # tf-target gene
            # for each TF-target gene pair
            for gene_pair in gene_pair_label_array:
                x_tf_name = gene_pair[0]
                x_gene_name = gene_pair[1]
                label = gene_pair[2]
                y.append(label)
                z.append(x_tf_name + ',' + x_gene_name)

                x_tf = expression.loc[:,x_tf_name]
                x_gene = expression.loc[:, x_gene_name]

                # Convert to the input format required by transformer
                single_tf_list = []
                for k in range(0, len(x_gene), args.gap):
                    feature = []
                    a = x_tf[k:k + args.gap]
                    b = x_gene[k:k + args.gap]

                    feature.extend(a)
                    feature.extend(b)

                    feature = np.asarray(feature)

                    if (len(feature) == 2 * args.gap):

                        single_tf_list.append(feature)

                single_tf_list = np.asarray(single_tf_list)
                x.append(single_tf_list)

            save(save_path + 'Nxdata_tf.npy', x)
            save(save_path + 'ydata_tf.npy', array(y))
            save(save_path + 'zdata_tf.npy', array(z))


            # ------------------------TF-aware cross-validation-------------------------------------------


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
            five_auc = []
            five_aupr = []
            ki = 0
            # for ki in range(args.iteration):
            while ki != args.iteration:
                columns.append(str(ki + 1) + '-th 3CV')

                print('\n')
                print("\nthe {}th cross-validation..........\n".format(ki + 1))

                # 3CV
                network_dict = {}
                all_network_dict = {}
                AUROCs = []
                AUPRs = []
                SPEs = []
                Recalls = []
                Precisions = []
                F1s = []
                MCCs = []
                ACCs = []

                for train_index, test_index in kf.split(tf_list):

                    train_TF = tf_list[train_index]
                    test_TF = tf_list[test_index]

                    # Find all positive and negative samples (features and labels) of 2-fold TF as the training set
                    # tf_expression--target_expression--label
                    train_pair_x, train_pair_y, train_pair_tf, train_pair_target = utils_gz.get_samples_pair(pos_neg_balanced_id, train_TF, save_path)
                    # Find all positive and negative samples (features and labels) of 1-fold TF as the test set
                    # tf_expression--target_expression--label
                    x_test, y_test, test_pair_tf, test_pair_target = utils_gz.get_samples_pair(pos_neg_balanced_id, test_TF, save_path)

                    x_train, x_val, y_train, y_val, gene_pair_tf_train, gene_pair_tf_val, gene_pair_target_train, gene_pair_target_val = train_test_split(train_pair_x, train_pair_y, train_pair_tf, train_pair_target, test_size=0.2, random_state=1,
                                                                   shuffle=True, stratify=train_pair_y)

                    if x_test.shape[0] == 0:
                        print('test sample size is 0')
                        ki = ki
                        break

                    else:

                        # Convert the training set validation set test set to dgl respectively

                        train_og_pos, train_og_neg, train_og_pos_T, train_og_neg_T = utils_gz.get_pos_neg_T_2(
                            gene_pair_tf_train, gene_pair_target_train, y_train,
                            exp_file,
                            gene_list_file)

                        # combine the training graph with the function graph
                        original_edges_weight = 1.0
                        data_all_train_pos = utils_gz.add_original_graph(train_og_pos, corr_g, weight=original_edges_weight)
                        # compute DI
                        train_g_pos = utils_GRN.transform_savebinI(train_og_pos, data_all_train_pos, train_og_pos_T,
                                                                   katz_alpha=0.02, k_hop=args.k_hop)
                        # train_g_pos = utils_GRN.transform_savebinI(train_og_pos, train_og_pos, train_og_pos_T,
                        #                                        katz_alpha=0.02, k_hop=args.k_hop)

                        val_og_pos, val_og_neg, val_og_pos_T, val_og_neg_T = utils_gz.get_pos_neg_T_2(
                            gene_pair_tf_val, gene_pair_target_val, y_val,
                            exp_file,
                            gene_list_file)
                        val_g_pos = utils_GRN.transform_savebinI(val_og_pos, val_og_pos, val_og_pos_T,
                                                                 katz_alpha=0.02, k_hop=args.k_hop)

                        test_og_pos, test_og_neg, test_og_pos_T, test_og_neg_T = utils_gz.get_pos_neg_T_2(
                            test_pair_tf, test_pair_target, y_test,
                            exp_file,
                            gene_list_file)
                        test_g_pos = utils_GRN.transform_savebinI(test_og_pos, test_og_pos, test_og_pos_T,
                                                                  katz_alpha=0.02, k_hop=args.k_hop)

                        X_trainloader, y_trainloader, gene_pair_tf_trainloader, gene_pair_target_trainloader = utils_GRN.numpy2loader(
                            x_train, y_train, gene_pair_tf_train, gene_pair_target_train, args.batch_sizes)
                        X_valloader, y_valloader, gene_pair_tf_valloader, gene_pair_target_valloader = utils_GRN.numpy2loader(
                            x_val, y_val, gene_pair_tf_val, gene_pair_target_val, args.batch_sizes)
                        X_testloader, y_testloader, gene_pair_tf_testloader, gene_pair_target_testloader = utils_GRN.numpy2loader(
                            x_test, y_test, test_pair_tf, test_pair_target, args.batch_sizes)

                        X_trainList = utils_GRN.loaderToList(X_trainloader)
                        y_trainList = utils_GRN.loaderToList(y_trainloader)
                        gene_pair_tf_trainList = utils_GRN.loaderToList(gene_pair_tf_trainloader)
                        gene_pair_target_trainList = utils_GRN.loaderToList(gene_pair_target_trainloader)

                        X_valList = utils_GRN.loaderToList(X_valloader)
                        y_valList = utils_GRN.loaderToList(y_valloader)
                        gene_pair_tf_valList = utils_GRN.loaderToList(gene_pair_tf_valloader)
                        gene_pair_target_valList = utils_GRN.loaderToList(gene_pair_target_valloader)

                        X_testList = utils_GRN.loaderToList(X_testloader)
                        y_testList = utils_GRN.loaderToList(y_testloader)
                        gene_pair_tf_testList = utils_GRN.loaderToList(gene_pair_tf_testloader)
                        gene_pair_target_testList = utils_GRN.loaderToList(gene_pair_target_testloader)

                        in_dim_GT = train_g_pos.ndata['x'].shape[1]  # 779*758
                        I_dim = train_g_pos.ndata['I'].shape[1]
                        model = TransformerE.AttentionGRN(in_dim_GT, args.out_dim_GT, args.hidden_dim_GT, args.num_heads_GT,
                                                          I_dim, args.input_dim_exp, d_model=args.d_models, num_classes=2).to(device)

                        criterion = nn.CrossEntropyLoss()

                        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
                        # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)

                        for epoch in range(epochs):
                            model.train()

                            train_loss = []
                            train_accs = []

                            for j in range(0, len(X_trainList)):
                                data = X_trainList[j]
                                labels = y_trainList[j]
                                all_tf_train, all_target_train, train_g_batch, k_edge_idx1 = utils_GRN.genepair_to_dgl_I(labels,
                                                                                                                         gene_pair_tf_trainList[
                                                                                                                             j],
                                                                                                                         gene_pair_target_trainList[
                                                                                                                             j],
                                                                                                                         train_g_pos)

                                if len(k_edge_idx1) == 0:
                                    flag_DM = False
                                    print(flag_DM)
                                else:
                                    flag_DM = True

                                logits = model(train_g_batch.to(device), data.to(device), all_tf_train, all_target_train,
                                               device, flag_DM).to('cpu')
                                labels = torch.tensor(labels, dtype=torch.long)
                                loss = criterion(logits, labels)
                                optimizer.zero_grad()
                                loss.backward()
                                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                                optimizer.step()

                                acc = (logits.argmax(dim=-1) == labels).float().mean()
                                train_loss.append(loss.item())
                                train_accs.append(acc)
                            train_loss = sum(train_loss) / len(train_loss)
                            train_acc = sum(train_accs) / len(train_accs)

                            print(
                                f"[ Train | {epoch + 1:03d}/{epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

                            model.eval()
                            predictions = []
                            labelss = []
                            y_val_label = []
                            y_val_predict = []
                            valid_loss = []
                            valid_accs = []
                            for k in range(0, len(X_valList)):
                                val_data = X_valList[k]
                                labels = y_valList[k]
                                labels = torch.tensor(labels, dtype=torch.long)
                                all_tf_val, all_target_val, val_g_batch, k_edge_idx1 = utils_GRN.genepair_to_dgl_I(labels,
                                                                                                                   gene_pair_tf_valList[
                                                                                                                       k],
                                                                                                                   gene_pair_target_valList[
                                                                                                                       k],
                                                                                                                   val_g_pos)

                                if len(k_edge_idx1) == 0:
                                    flag_DM = False
                                    print(flag_DM)
                                else:
                                    flag_DM = True
                                with torch.no_grad():
                                    # logits = model(val_data)
                                    logits = model(val_g_batch.to(device), val_data.to(device), all_tf_val, all_target_val,
                                                   device, flag_DM).to(
                                        'cpu')

                                loss = criterion(logits, labels)
                                valid_loss.append(loss.item())

                                acc = (logits.argmax(dim=-1) == labels).float().mean()
                                valid_accs.append(acc)

                                predt = F.softmax(logits)
                                labelss.extend(labels.cpu().numpy().tolist())
                                y_val_label.extend(labels.cpu().numpy())

                                temps = predt.cpu().numpy().tolist()
                                for i in temps:
                                    t = i[1]
                                    y_val_predict.append(t)
                                predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
                            valid_loss = sum(valid_loss) / len(valid_loss)
                            valid_acc = sum(valid_accs) / len(valid_accs)
                            AUC_val, AUPR_val = utils_gz.metric_scores(
                                y_val_label, y_val_predict, th=0.5)

                            print(
                                f"[ Valid | {epoch + 1:03d}/{epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.4f}, auc = {AUC_val:.4f}, aupr = {AUPR_val:.4f}")

                        y_test_label = []
                        y_predict = []
                        predictions = []
                        labelss = []
                        model.eval()
                        for k in range(0, len(X_testList)):
                            test_data = X_testList[k]
                            labels = y_testList[k]
                            all_tf_test, all_target_test, test_g_batch, k_edge_idx1 = utils_GRN.genepair_to_dgl_I(labels,
                                                                                                                  gene_pair_tf_testList[
                                                                                                                      k],
                                                                                                                  gene_pair_target_testList[
                                                                                                                      k],
                                                                                                                  test_g_pos)
                            if len(k_edge_idx1) == 0:
                                flag_DM = False
                                print(flag_DM)
                            else:
                                flag_DM = True
                            with torch.no_grad():
                                # logits = model(data)
                                logits = model(test_g_batch.to(device), test_data.to(device), all_tf_test, all_target_test,
                                               device, flag_DM).to('cpu')
                            predt = F.softmax(logits)
                            labelss.extend(labels.cpu().numpy().tolist())
                            y_test_label.extend(labels.cpu().numpy())

                            # temps = logits.cpu().numpy().tolist()
                            temps = predt.cpu().numpy().tolist()
                            for i in temps:
                                t = i[1]
                                y_predict.append(t)
                            predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

                        AUC, AUPR = utils_gz.metric_scores(y_test_label, y_predict, th=0.5)
                        print(f"[Test]--------------------------------------------auc = {AUC:.4f}, aupr = {AUPR:.4f}")

                        AUROCs.append(AUC)
                        AUPRs.append(AUPR)
                        print('AUROCs: ',AUROCs)

                AUROC_mean = np.mean(AUROCs) # The result of a cross-validation
                # AUROC_std = np.std(AUROCs, ddof=1)
                AUPR_mean = np.mean(AUPRs)
                # AUPR_std = np.std(AUPRs)

                AUROC_mean = float('{:.4f}'.format(AUROC_mean))
                # AUROC_std = float('{:.4f}'.format(AUROC_std))
                AUPR_mean = float('{:.4f}'.format(AUPR_mean))
                # AUPR_std = float('{:.4f}'.format(AUPR_std))

                print('one CV results：', AUROC_mean)

                five_auc.append(AUROC_mean) # All results of five or ten cross-validations
                five_aupr.append(AUPR_mean)

                print('all CV results：', five_auc)

                ki += 1
                print(ki)

            five_auc_mean = np.mean(five_auc)  # average results of five or ten cross-validations
            five_aupr_mean = np.mean(five_aupr)
            five_auc_std = np.std(five_auc)
            five_aupr_std = np.std(five_aupr)


            # save
            all_network_dict["AUROC"] = five_auc
            all_network_dict["AUPR"] = five_aupr

            filename = open(save_path + network_dict_name + '_all.csv', 'w')
            for k, v in all_network_dict.items():
                filename.write(k + ':' + str(v))
                filename.write('\n')
            filename.close()