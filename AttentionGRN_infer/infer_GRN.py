'''
author: Gao Zhen
20241010
GRN reconstruction based on AttentionGRN using hHEP.
'''
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import utils_gz, utils_GRN, utils_data
from model import TransformerE
import warnings
warnings.filterwarnings('ignore')
torch.set_default_tensor_type(torch.DoubleTensor)

parser = argparse.ArgumentParser()
parser.add_argument('--k_hop', type=int, default='1', help='[1, 2]')
parser.add_argument('--epochs', type=int, default='50', help='training epochs 30')
parser.add_argument('--batch_sizes', type=int, default='256', help='batch_sizes')
parser.add_argument('--out_dim_GT', type=int, default='64', help='the output of GT model')
parser.add_argument('--hidden_dim_GT', type=int, default='64', help='the hidden of GT model')
parser.add_argument('--num_heads_GT', type=int, default='4', help='the heads of GT model')
parser.add_argument('--input_dim_exp', type=int, default='200', help='the input of expression Transformer model')
parser.add_argument('--d_models', type=int, default='200', help='the input of expression Transformer model')
args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def predict_AttentionGRN(gold_network_path,gene_list_path,gene_expression_path , tf_list_path, result_path, save_folder ):
    gold_pair_record = utils_data.get_gold_pair_dict(gold_network_path)

    new_all_gene_name = pd.read_csv(gene_list_path)['gene'].to_list()

    origin_expression_record, num_gene, num_cell = utils_data.get_origin_expression_data_hHEP(gene_expression_path)

    train_pair_name, train_matrix_file, train_label_file, train_gene_pair_file, train_pair_list_id_tf_file, train_pair_list_id_target_file = utils_data.get_trainset(
        gold_pair_record, new_all_gene_name, origin_expression_record, gene_list_path, result_path)

    unknown_matrix_file, unknown_label_file, unknown_gene_pair_file, unknown_pair_list_id_tf_file, unknown_pair_list_id_target_file = utils_data.get_unknownset(
        tf_list_path, gene_list_path, gold_pair_record, new_all_gene_name, origin_expression_record, train_pair_name,
        result_path)

    train_matrix_data = np.load(train_matrix_file)
    train_label_data = np.load(train_label_file)
    train_gene_pair_tf = np.load(train_pair_list_id_tf_file)
    train_gene_pair_target = np.load(train_pair_list_id_target_file)

    # functional related genes
    corr_g = utils_gz.get_corr(gene_expression_path, gene_list_path, 'cosine',
                               cutoffbeishu=2.7)  # 'cosine', 'pearson', 'kendall', 'spearman'

    # all data——>train set + val set
    x_train, x_val, y_train, y_val, gene_pair_tf_train, gene_pair_tf_val, gene_pair_target_train, gene_pair_target_val = train_test_split(
        train_matrix_data, train_label_data, train_gene_pair_tf, train_gene_pair_target, test_size=0.4,
        stratify=train_label_data)

    # prepare test set
    x_test = np.load(unknown_matrix_file)
    y_test = np.load(unknown_label_file)
    gene_pair_tf_test = np.load(unknown_pair_list_id_tf_file)
    gene_pair_target_test = np.load(unknown_pair_list_id_target_file)

    # compute DI
    train_og_pos, train_og_neg, train_og_pos_T, train_og_neg_T = utils_gz.get_pos_neg_T_2(gene_pair_tf_train,
                                                                                          gene_pair_target_train,
                                                                                          y_train, gene_expression_path,
                                                                                          gene_list_path)
    data_all_train_pos = utils_gz.add_original_graph(train_og_pos, corr_g, weight=1.0)
    train_g_pos = utils_GRN.transform_savebinI(train_og_pos, data_all_train_pos, train_og_pos_T, katz_alpha=0.02,
                                               k_hop=args.k_hop)

    val_og_pos, val_og_neg, val_og_pos_T, val_og_neg_T = utils_gz.get_pos_neg_T_2(gene_pair_tf_val,
                                                                                  gene_pair_target_val, y_val,
                                                                                  gene_expression_path, gene_list_path)
    val_g_pos = utils_GRN.transform_savebinI(val_og_pos, val_og_pos, val_og_pos_T, katz_alpha=0.02, k_hop=args.k_hop)

    test_og_pos, test_og_neg, test_og_pos_T, test_og_neg_T = utils_gz.get_pos_neg_T_2(gene_pair_tf_test,
                                                                                      gene_pair_target_test, y_test,
                                                                                      gene_expression_path,
                                                                                      gene_list_path)
    test_g_pos = utils_GRN.transform_savebinI(test_og_pos, test_og_pos, test_og_pos_T, katz_alpha=0.02,
                                              k_hop=args.k_hop)

    X_trainloader, y_trainloader, gene_pair_tf_trainloader, gene_pair_target_trainloader = utils_GRN.numpy2loader(
        x_train, y_train, gene_pair_tf_train, gene_pair_target_train, args.batch_sizes)
    X_valloader, y_valloader, gene_pair_tf_valloader, gene_pair_target_valloader = utils_GRN.numpy2loader(x_val, y_val,
                                                                                                          gene_pair_tf_val,
                                                                                                          gene_pair_target_val,
                                                                                                          args.batch_sizes)
    X_testloader, y_testloader, gene_pair_tf_testloader, gene_pair_target_testloader = utils_GRN.numpy2loader(x_test,
                                                                                                              y_test,
                                                                                                              gene_pair_tf_test,
                                                                                                              gene_pair_target_test,
                                                                                                              args.batch_sizes)

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
    model = TransformerE.AttentionGRN(in_dim_GT, args.out_dim_GT, args.hidden_dim_GT, args.num_heads_GT, I_dim,
                                      args.input_dim_exp, num_gtlayers=4, d_model=args.d_models, num_classes=2).to(
        device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
    train_losses = []
    train_acces = []
    valid_losses = []
    valid_acces = []
    valid_aucs = []
    test_aucs = []

    min_loss = 100

    for epoch in range(args.epochs):
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
            else:
                flag_DM = True

            logits = model(train_g_batch.to(device), data.to(device), all_tf_train, all_target_train, device,
                           flag_DM).to(
                'cpu')
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

        train_acces.append(train_acc)
        train_losses.append(train_loss)

        print(f"[ Train | {epoch + 1:03d}/{args.epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

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
                                                                                               gene_pair_tf_valList[k],
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
                logits = model(val_g_batch.to(device), val_data.to(device), all_tf_val, all_target_val, device,
                               flag_DM).to(
                    'cpu')

            loss = criterion(logits, labels)
            valid_loss.append(loss.item())

            acc = (logits.argmax(dim=-1) == labels).float().mean()
            valid_accs.append(acc)

            if loss.item() < min_loss:
                min_loss = loss.item()
                print("save model")
                torch.save(model.state_dict(), save_folder + "model.pth")

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

        valid_acces.append(valid_acc)
        valid_losses.append(valid_loss)

        AUC_val, AUPR_val, ACC_val, F1_val, SPE_val, MCC_val, Precision_val, Recall_val = utils_gz.metric_scores(
            y_val_label, y_val_predict, th=0.5)

        print(
            f"[ Valid | {epoch + 1:03d}/{args.epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.4f}, auc = {AUC_val:.4f}, aupr = {AUPR_val:.4f}")

    epochs_x = [i for i in range(args.epochs)]
    plt.figure()
    plt.plot(epochs_x, train_losses, 'bo--', alpha=0.5, linewidth=1, label='train')
    plt.plot(epochs_x, valid_losses, 'r*--', alpha=0.5, linewidth=1, label='validation')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    # plt.ylim(-1,1)
    # plt.show()
    plt.savefig(save_folder +  'loss.pdf')

    plt.figure()
    plt.plot(epochs_x, train_acces, 'bo--', alpha=0.5, linewidth=1, label='train')
    plt.plot(epochs_x, valid_acces, 'r*--', alpha=0.5, linewidth=1, label='validation')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')

    # plt.ylim(-1,1)
    # plt.show()
    plt.savefig(save_folder + 'accuracy.pdf')

    model = TransformerE.AttentionGRN(in_dim_GT, args.out_dim_GT, args.hidden_dim_GT, args.num_heads_GT, I_dim,
                                      args.input_dim_exp, num_gtlayers=4, d_model=args.d_models, num_classes=2).to(
        device)
    model.load_state_dict(torch.load(save_folder + 'model.pth'))
    model.eval()
    # y_test_label = []
    y_predict = []
    predictions = []
    tf_ids = []
    target_ids = []
    # labelss = []

    for k in range(0, len(X_testList)):
        test_data = X_testList[k]
        labels = y_testList[k]
        all_tf_test, all_target_test, test_g_batch, k_edge_idx1 = utils_GRN.genepair_to_dgl_I(labels,
                                                                                              gene_pair_tf_testList[k],
                                                                                              gene_pair_target_testList[
                                                                                                  k],
                                                                                              test_g_pos)
        if len(k_edge_idx1) == 0:
            flag_DM = False
            # print(flag_DM)
        else:
            flag_DM = True
        with torch.no_grad():
            # logits = model(data)
            logits = model(test_g_batch.to(device), test_data.to(device), all_tf_test, all_target_test,
                           device, flag_DM).to('cpu')
        predt = F.softmax(logits)
        # labelss.extend(labels.cpu().numpy().tolist())
        # y_test_label.extend(labels.cpu().numpy())

        # temps = logits.cpu().numpy().tolist()
        temps = predt.cpu().numpy().tolist()
        for i in temps:
            t = i[1]
            y_predict.append(t)
        predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())  # [0 1 ]
        tf_ids.extend(all_tf_test)  # tf_ids[0].item()
        target_ids.extend(all_target_test)

    new_tf_ids = []
    new_target_ids = []
    new_label = []
    new_scores = []
    for i in range(len(tf_ids)):
        tf_id = tf_ids[i].item()
        target_id = target_ids[i].item()
        predict_label = predictions[i]
        score = y_predict[i]
        if predict_label == 1:
            new_tf_ids.append(tf_id)
            new_target_ids.append(target_id)
            new_label.append(predict_label)
            new_scores.append(score)

    gene_name = pd.read_csv(gene_list_path)['gene'].to_list()
    gene_id = pd.read_csv(gene_list_path)['index'].to_list()

    new_tf_names = []
    for i in new_tf_ids:
        a = gene_id.index(i)
        new_tf_names.append(gene_name[a])

    new_target_names = []
    for i in new_target_ids:
        a = gene_id.index(i)
        new_target_names.append(gene_name[a])

    from pandas.core.frame import DataFrame
    new_tf_names = DataFrame(new_tf_names)
    new_target_names = DataFrame(new_target_names)
    new_scores = DataFrame(new_scores)

    predic_GRN = pd.concat([new_tf_names, new_target_names], axis=1)
    predic_GRN = pd.concat([predic_GRN, new_scores], axis=1)
    predic_GRN.columns = ['tf', 'target', 'score']

    dfs = predic_GRN
    dfs.sort_values(by="score", inplace=True, ascending=False)

    dfs.to_csv(save_folder + 'predict_GRN.csv', index=False)
    print('save predicted GRN successfully.')



save_folder = 'predict_results/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

gene_expression_path = 'data\\Final_expression.csv'
gold_network_path = 'data\\Final_GRNorTRN_pos.csv'
gene_list_path = 'data\\Final_gene_list.csv'
tf_list_path = 'data\\Final_TF_common_index.csv'
result_path = 'transform_data\\'
if not os.path.isdir(result_path):
    os.makedirs(result_path)

predict_AttentionGRN(gold_network_path,gene_list_path,gene_expression_path , tf_list_path, result_path, save_folder)