'''
author: Gao Zhen
20240803

GRN reconstruction using Model AttentionGRN, and the prediction performance of the model is evaluated using independent test set.

'''

from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import argparse
import utils_GRN, utils_gz
from model import TransformerE
import warnings
warnings.filterwarnings('ignore')
torch.set_default_tensor_type(torch.DoubleTensor)

parser = argparse.ArgumentParser()
parser.add_argument('--modelname', type=str, default='AttentionGRN', help='model name')
parser.add_argument('--iteration', type=int, default=5, help='the number of training and test')
parser.add_argument('--batch_sizes', type=int, default='256', help='batch_sizes')
parser.add_argument('--out_dim_GT', type=int, default='64', help='the output of GT model')
parser.add_argument('--hidden_dim_GT', type=int, default='64', help='the hidden of GT model')
parser.add_argument('--num_heads_GT', type=int, default='4', help='the heads of GT model')
parser.add_argument('--input_dim_exp', type=int, default='200', help='the input of expression Transformer model')
parser.add_argument('--d_models', type=int, default='200', help='the input of expression Transformer model')
parser.add_argument('--k_hop', type=int, default='1', help='[1, 2]')
args = parser.parse_args()

save_folder = 'AttentionGRN_data1_GRN\\'
processed_data = 'DATA1_AttentionGRN'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset_names = [ 'hESC','hHEP','mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']
for Rank_num in [500, 1000]:
    for dataset_name in dataset_names:
        network_types = []
        corr_g_cutoffs = []
        epochs_all = []
        if Rank_num == 500:
            if dataset_name == 'hESC':
                network_types = ['hESC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']
                corr_g_cutoffs = [2.9, 2.9, 2.7] # mean + std*3， top 500
                epochs_all = [30, 20, 20]

            elif dataset_name == 'hHEP':
                network_types = ['Non-specific-ChIP-seq-network', 'STRING-network']  # top 500
                corr_g_cutoffs = [2.7, 2.5] # mean + std*3， top 500
                epochs_all = [20, 40]

            elif dataset_name == 'mDC':
                network_types = ['mDC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network'] #
                corr_g_cutoffs = [2.8, 2.9, 2.8] # mean + std*3， top 500
                epochs_all = [10, 20, 100]

            elif dataset_name == 'mESC':
                network_types = ['mESC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network', 'mESC-lofgof-network']
                corr_g_cutoffs = [2.6, 2.6, 3, 2.7] # mean + std*3， top 500
                epochs_all = [150, 30, 40, 100]

            elif dataset_name == 'mHSC-E':
                network_types = ['mHSC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']
                corr_g_cutoffs = [2.8, 2.7, 2.9] # mean + std*3， top 500
                epochs_all = [20, 40, 40]

            elif dataset_name == 'mHSC-GM':
                network_types = ['mHSC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']
                corr_g_cutoffs = [3, 2.7, 2.7] # mean + std*3， top 500
                epochs_all = [20, 40, 150]

            elif dataset_name == 'mHSC-L':
                network_types = ['mHSC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']
                corr_g_cutoffs = [2.8, 2.8, 2.7] # mean + std*3， top 500
                epochs_all = [100, 40, 20]

            else:
                print("network type error")

        elif Rank_num == 1000:
            if dataset_name == 'hESC':
                network_types = ['hESC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']
                corr_g_cutoffs = [3, 2.7, 2.8]  # mean + std*3， top 1000
                epochs_all = [30, 30, 30]

            elif dataset_name == 'hHEP':
                network_types = ['HepG2-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']  # top 1000
                corr_g_cutoffs = [3, 2.8, 2.9]  # mean + std*3， top 1000
                epochs_all = [30, 30, 30]

            elif dataset_name == 'mDC':
                network_types = ['mDC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']  #
                corr_g_cutoffs = [2.8, 2.7, 2.9]  # mean + std*3， top 1000
                epochs_all = [40, 40, 40]

            elif dataset_name == 'mESC':
                network_types = ['mESC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network',
                                 'mESC-lofgof-network']
                corr_g_cutoffs = [2.8, 2.7, 2.8, 2.7]  # mean + std*3， top 1000
                epochs_all = [40, 40, 40, 40]

            elif dataset_name == 'mHSC-E':
                network_types = ['mHSC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']
                corr_g_cutoffs = [2.7, 2.8, 2.7]  # mean + std*3， top 1000
                epochs_all = [40, 40, 40]

            elif dataset_name == 'mHSC-GM':
                network_types = ['mHSC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']
                corr_g_cutoffs = [2.8, 3, 2.6]  # mean + std*3， top 1000
                epochs_all = [40, 40, 20]

            elif dataset_name == 'mHSC-L':
                network_types = ['mHSC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']
                corr_g_cutoffs = [3, 2.8, 2.8]  # mean + std*3， top 1000
                epochs_all = [40, 40, 40]

            else:
                print("network type error")
        else:
            print('rank number error')

        print(dataset_name + ': ')
        print(network_types)

        for nt in range(len(network_types)):
            network_type = network_types[nt]
            cutoff = corr_g_cutoffs[nt]
            epochs = epochs_all[nt]

            data_path = '..\\data_preprocess\\' + processed_data + '\\' + dataset_name + '\\' + network_type + '\\Top' + str(
                Rank_num) + '\\'

            log_dir = "log\\" + "\\" + str(Rank_num) + "\\" + network_type + "\\" + dataset_name + "\\"
            if (not os.path.isdir(log_dir)):
                os.makedirs(log_dir)

            matrix_data = np.load(data_path + 'matrix.npy')
            label_data = np.load(data_path + 'label.npy')
            gene_pair_tf = np.load(data_path + 'pair_list_id_tf.npy')
            gene_pair_target = np.load(data_path + 'pair_list_id_target.npy')
            exp_file = data_path + 'Final_expression.csv'
            gene_list_file = data_path + 'Final_gene_list.csv'

            # functional related genes
            corr_g = utils_gz.get_corr(exp_file, gene_list_file,
                                      'cosine', cutoff)  # 'cosine', 'pearson', 'kendall', 'spearman'

            # save path
            save_path = save_folder  + '\\' + dataset_name + '\\' + network_type + '\\Top' + str(Rank_num) + '\\'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            network_dict_name = args.modelname + '_311_'

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

            for ki in range(args.iteration):
                print('\n'+'-----------------the ' + str(ki + 1) + 'th iteration--------------------' + network_type + '--------------')

                x_train, x_t, y_train, y_t, gene_pair_tf_train, gene_pair_tf_t, gene_pair_target_train, gene_pair_target_t = train_test_split(matrix_data, label_data, gene_pair_tf, gene_pair_target, test_size=0.4, stratify=label_data)
                x_val, x_test, y_val, y_test, gene_pair_tf_val, gene_pair_tf_test, gene_pair_target_val, gene_pair_target_test = train_test_split(x_t, y_t, gene_pair_tf_t, gene_pair_target_t, test_size=0.5, stratify=y_t)

                # transform train set into torch_geometric.data.Data
                train_og_pos, train_og_neg, train_og_pos_T, train_og_neg_T = utils_gz.get_pos_neg_T_2(gene_pair_tf_train,gene_pair_target_train, y_train,
                                                                                                    exp_file,
                                                                                                    gene_list_file)

                # merge functional related genes in to train set
                original_edges_weight = 1.0
                data_all_train_pos = utils_gz.add_original_graph(train_og_pos, corr_g, weight=original_edges_weight)
                train_g_pos = utils_GRN.transform_savebinI(train_og_pos, data_all_train_pos, train_og_pos_T,
                                                         katz_alpha=0.02, k_hop=args.k_hop)
                # train_g_pos = utils_GRN.transform_savebinI(train_og_pos, train_og_pos, train_og_pos_T,
                #                                          katz_alpha=0.02, k_hop=args.k_hop)

                val_og_pos, val_og_neg, val_og_pos_T, val_og_neg_T = utils_gz.get_pos_neg_T_2(
                    gene_pair_tf_val, gene_pair_target_val, y_val,
                    exp_file,
                    gene_list_file)

                val_g_pos = utils_GRN.transform_savebinI(val_og_pos, val_og_pos, val_og_pos_T,
                                                           katz_alpha=0.02, k_hop=args.k_hop)

                test_og_pos, test_og_neg, test_og_pos_T, test_og_neg_T = utils_gz.get_pos_neg_T_2(
                    gene_pair_tf_test, gene_pair_target_test, y_test,
                    exp_file,
                    gene_list_file)
                test_g_pos = utils_GRN.transform_savebinI(test_og_pos, test_og_pos, test_og_pos_T,
                                                        katz_alpha=0.02, k_hop=args.k_hop)


                X_trainloader, y_trainloader, gene_pair_tf_trainloader, gene_pair_target_trainloader = utils_GRN.numpy2loader(x_train, y_train, gene_pair_tf_train, gene_pair_target_train, args.batch_sizes)
                X_valloader, y_valloader, gene_pair_tf_valloader, gene_pair_target_valloader = utils_GRN.numpy2loader(x_val, y_val, gene_pair_tf_val, gene_pair_target_val, args.batch_sizes)
                X_testloader, y_testloader, gene_pair_tf_testloader, gene_pair_target_testloader = utils_GRN.numpy2loader(x_test, y_test, gene_pair_tf_test, gene_pair_target_test, args.batch_sizes)

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

                in_dim_GT = train_g_pos.ndata['x'].shape[1]
                I_dim = train_g_pos.ndata['I'].shape[1]
                model = TransformerE.AttentionGRN(in_dim_GT, args.out_dim_GT, args.hidden_dim_GT, args.num_heads_GT, I_dim, args.input_dim_exp, num_gtlayers=4, d_model=args.d_models, num_classes=2).to(device)

                criterion = nn.CrossEntropyLoss()

                optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

                for epoch in range(epochs):
                    model.train()

                    train_loss = []
                    train_accs = []

                    for j in range(0, len(X_trainList)):
                        data = X_trainList[j]
                        labels = y_trainList[j]
                        all_tf_train, all_target_train, train_g_batch, k_edge_idx1 = utils_GRN.genepair_to_dgl_I(labels, gene_pair_tf_trainList[j], gene_pair_target_trainList[j], train_g_pos)

                        if len(k_edge_idx1) == 0:
                            flag_DM = False
                        else:
                            flag_DM = True

                        logits = model(train_g_batch.to(device), data.to(device), all_tf_train, all_target_train, device, flag_DM).to('cpu')
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

                    print(f"[ Train | {epoch + 1:03d}/{epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

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
                        all_tf_val, all_target_val, val_g_batch, k_edge_idx1 = utils_GRN.genepair_to_dgl_I(labels, gene_pair_tf_valList[k], gene_pair_target_valList[k], val_g_pos)

                        if len(k_edge_idx1) == 0:
                            flag_DM = False
                            print(flag_DM)
                        else:
                            flag_DM = True
                        with torch.no_grad():
                            # logits = model(val_data)
                            logits = model(val_g_batch.to(device), val_data.to(device), all_tf_val, all_target_val, device,flag_DM).to(
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
                    AUC_val, AUPR_val, ACC_val, F1_val, SPE_val, MCC_val, Precision_val, Recall_val = utils_gz.metric_scores(y_val_label, y_val_predict, th=0.5)

                    print(f"[ Valid | {epoch + 1:03d}/{epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.4f}, auc = {AUC_val:.4f}, aupr = {AUPR_val:.4f}")


                y_test_label = []
                y_predict = []
                predictions = []
                labelss = []
                model.eval()
                for k in range(0, len(X_testList)):
                    test_data = X_testList[k]
                    labels =y_testList [k]
                    all_tf_test, all_target_test, test_g_batch, k_edge_idx1 = utils_GRN.genepair_to_dgl_I(labels,
                                                                                                          gene_pair_tf_testList[k],
                                                                                                          gene_pair_target_testList[k],
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


                AUC, AUPR, ACC, F1, SPE, MCC, Precision, Recall = utils_gz.metric_scores(y_test_label, y_predict, th=0.5)
                print(f"[Test]--------------------------------------------auc = {AUC:.4f}, aupr = {AUPR:.4f}")

                AUROCs.append(AUC)
                AUPRs.append(AUPR)
                ACCs.append(ACC)
                F1s.append(F1)
                SPEs.append(SPE)
                MCCs.append(MCC)
                Precisions.append(Precision)
                Recalls.append(Recall)

            all_network_dict["AUROC"] = AUROCs
            all_network_dict["AUPR"] = AUPRs
            all_network_dict["Accuracy"] = ACCs
            all_network_dict["F1"] = F1s
            all_network_dict["SPE"] = SPEs
            all_network_dict["MCC"] = MCCs
            all_network_dict["Precision"] = Precisions
            all_network_dict["Recall"] = Recalls


            filename = open(save_path + network_dict_name + '_all.csv', 'w')
            for k, v in all_network_dict.items():
                filename.write(k + ':' + str(v))
                filename.write('\n')
            filename.close()









