'''
202409
datasets: DATA1
model: STGRNS
task: GRN inference
evaluation strategy: independent testing

positive samples: known TF-gene pairs
negative samples: unknown TF-gene pairs, the TF set are the same as TF set in positive samples
unbalanced datasets
'''

from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn import metrics
import os
import argparse
import warnings
warnings.filterwarnings('ignore')
torch.set_default_tensor_type(torch.DoubleTensor)
import utils_STGRNS


parser = argparse.ArgumentParser()
parser.add_argument('--modelname', type=str, default='STGRNS', help='STGRNS')
parser.add_argument('--iteration', type=int, default=5, help='the number of training and test')
parser.add_argument('--batch_sizes', type=int, default='32', help='batch_sizes')
parser.add_argument('--epochs', type=int, default='20', help='epochs')
args = parser.parse_args()

processed_data = '..\\data_preprocess\\DATA1_AttentionGRN'
save_folder = 'STGRNS_data1_GRN\\'
dataset_names = ['hESC', 'hHEP', 'mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']

for Rank_num in  [500, 1000]:
    for dataset_name in dataset_names:
        network_types = []
        if dataset_name == 'hESC':
            network_types = ['hESC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']
        elif Rank_num == 500 and dataset_name == 'hHEP':
            network_types = ['Non-specific-ChIP-seq-network', 'STRING-network']
        elif Rank_num == 1000 and dataset_name == 'hHEP':
            network_types = ['HepG2-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']
        elif dataset_name == 'mDC':
            network_types = ['mDC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']
        elif dataset_name == 'mESC':
            network_types = ['mESC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network','mESC-lofgof-network']
        elif dataset_name == 'mHSC-E' or dataset_name == 'mHSC-GM' or dataset_name == 'mHSC-L':
            network_types = ['mHSC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']
        else:
            print("network type error")

        print(dataset_name + ': ')
        print(network_types)

        for network_type in network_types:

            data_path = processed_data + '\\' + dataset_name + '\\' + network_type + '\\Top' + str(Rank_num) + '\\'

            d_models = 200
            batch_size = args.batch_sizes
            log_dir = "log\\" + "\\" + str(Rank_num) + "\\" + network_type + "\\" + dataset_name + "\\"
            if (not os.path.isdir(log_dir)):
                os.makedirs(log_dir)

            matrix_data = np.load(data_path + 'matrix.npy')
            label_data = np.load(data_path + 'label.npy')

            save_path = save_folder + dataset_name + '\\' + network_type + '\\Top' + str(Rank_num) + '\\'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            network_dict_name = args.modelname + '_311_'

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
                print('\n'+'-----------------the ' + str(ki + 1) + 'th iteration--------------------' + network_type + '--------------')

                x_train, x_t, y_train, y_t = train_test_split(matrix_data, label_data, test_size=0.4, stratify=label_data)
                x_val, x_test, y_val, y_test = train_test_split(x_t, y_t, test_size=0.5, stratify=y_t)

                X_trainloader, y_trainloader = utils_STGRNS.numpy2loader(x_train, y_train, batch_size)
                X_valloader, y_valloader = utils_STGRNS.numpy2loader(x_val, y_val, batch_size)
                X_testloader, y_testloader = utils_STGRNS.numpy2loader(x_test, y_test, batch_size)

                X_trainList = utils_STGRNS.loaderToList(X_trainloader)
                y_trainList = utils_STGRNS.loaderToList(y_trainloader)

                X_valList = utils_STGRNS.loaderToList(X_valloader)
                y_valList = utils_STGRNS.loaderToList(y_valloader)

                X_testList = utils_STGRNS.loaderToList(X_testloader)
                y_testList = utils_STGRNS.loaderToList(y_testloader)

                model = utils_STGRNS.STGRNS(input_dim=200, nhead=2, d_model=d_models, num_classes=2)

                criterion = nn.CrossEntropyLoss()

                optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

                n_epochs = args.epochs

                for epoch in range(n_epochs):
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
                        train_loss.append(loss.item())
                        train_accs.append(acc)
                    train_loss = sum(train_loss) / len(train_loss)
                    train_acc = sum(train_accs) / len(train_accs)
                    # acc_record['train'].append(train_acc)
                    # loss_record['train'].append(train_loss)

                    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

                    model.eval()
                    predictions = []
                    labelss = []
                    y_test = []
                    y_predict = []
                    valid_loss = []
                    valid_accs = []
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

                    # acc_record['dev'].append(valid_acc)
                    # loss_record['dev'].append(valid_loss)

                    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

                    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)
                    auc = metrics.auc(fpr, tpr)
                    precision, recall, thresholds_PR = metrics.precision_recall_curve(y_test, y_predict)
                    AUPR = metrics.auc(recall, precision)
                    acc = metrics.accuracy_score(labelss, predictions)
                    bacc = metrics.balanced_accuracy_score(labelss, predictions)
                    f1 = metrics.f1_score(labelss, predictions)

                    # print("acc:", acc, "auc:", auc, "aupr:", AUPR, "bacc", bacc, "f1", f1)

                model_name = str(Rank_num) + "_" + network_type + "_" + dataset_name

                y_test = []
                y_predict = []
                model.eval()
                for k in range(0, len(X_testList)):
                    data = X_testList[k]
                    labels = y_testList[k]

                    with torch.no_grad():
                        logits = model(data)
                    predt = F.softmax(logits)
                    labelss.extend(labels.cpu().numpy().tolist())
                    y_test.extend(labels.cpu().numpy())

                    # temps = logits.cpu().numpy().tolist()
                    temps = predt.cpu().numpy().tolist()
                    for i in temps:
                        t = i[1]
                        y_predict.append(t)
                    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

                fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)
                auc = metrics.auc(fpr, tpr)
                precision, recall, thresholds_PR = metrics.precision_recall_curve(y_test, y_predict)
                AUPR = metrics.auc(recall, precision)
                acc = metrics.accuracy_score(labelss, predictions)
                bacc = metrics.balanced_accuracy_score(labelss, predictions)
                f1 = metrics.f1_score(labelss, predictions)
                print("acc:", acc, "auc:", auc, "aupr:", AUPR, "bacc", bacc, "f1", f1)
                AUROCs.append(auc)
                AUPRs1.append(AUPR)
                # print('AUROCS')
                # print(AUROCs)
            AUROC_mean = np.mean(AUROCs)
            AUROC_std = np.std(AUROCs, ddof=1)
            AUPR_mean1 = np.mean(AUPRs1)
            AUPR_std1 = np.std(AUPRs1)

            AUROC_mean = float('{:.4f}'.format(AUROC_mean))
            AUROC_std = float('{:.4f}'.format(AUROC_std))
            AUPR_mean1 = float('{:.4f}'.format(AUPR_mean1))
            AUPR_std1 = float('{:.4f}'.format(AUPR_std1))

            network_dict["AUROC mean"] = AUROC_mean
            network_dict["AUROC std"] = AUROC_std
            network_dict["AUPR1 mean"] = AUPR_mean1
            network_dict["AUPR1 std"] = AUPR_std1

            filename = open(save_path + network_dict_name + '_avg.csv', 'w')
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

            filename = open(save_path + network_dict_name + '_all.csv', 'w')
            for k, v in all_network_dict.items():
                filename.write(k + ':' + str(v))
                filename.write('\n')
            filename.close()


