'''
20240405
datasets: DATA1
model: GENELink
task: TRN inference
evaluation strategy: TF-aware three-fold cross-validation

positive samples: known TF-gene pairs
negative samples: unknown TF-gene pairs, each TF is same as TF in positive samples
balanced datasets
'''
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import numpy as np
import os
import argparse
from sklearn.model_selection import KFold, train_test_split
import utils_Train_Test_Split as tts
from scGNN import GENELink
from utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation,  Network_Statistic, get_samples, get_samples_pair



parser = argparse.ArgumentParser()
parser.add_argument('--modelname', type=str, default='GENELink', help='GENELink')

parser.add_argument('--lr', type=float, default=3e-3, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default= 20, help='Number of epoch.')
parser.add_argument('--num_head', type=list, default=[3,3], help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--hidden_dim', type=int, default=[128,64,32], help='The dimension of hidden layer')
parser.add_argument('--output_dim', type=int, default=16, help='The dimension of latent layer')
parser.add_argument('--batch_size', type=int, default=32, help='The size of each batch')
parser.add_argument('--seed', type=int, default=8, help='Random seed')
parser.add_argument('--reduction',type=str,default='concate', help='how to integrate multihead attention')
parser.add_argument('--iteration', type=int, default=5, help='the number of 3CV')
parser.add_argument('--direction', type=bool, default=True, help='the direction of the adjacent matrix')
parser.add_argument('--loop', type=bool, default=True, help='whether to add self-loop in adjacent matrix')
parser.add_argument('--Type',type=str,default='MLP', help='score metric')
parser.add_argument('--flag', type=bool, default=True, help='the identifier whether to conduct causal inference')
args = parser.parse_args()
seed = args.seed
torch.manual_seed(args.seed)

dataset_names = ['hESC', 'hHEP', 'mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']
processed_data = '..\\data_preprocess\\DATA2'
save_folder = 'GENELink_data2_TRN\\'


for Rank_num in [500, 1000]:
    for dataset_name in dataset_names:
        network_types = []
        if dataset_name=='hESC':
            network_types = ['hESC-ChIP-seq-network','Non-specific-ChIP-seq-network','STRING-network']
        elif dataset_name=='hHEP':
            network_types = ['HepG2-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']
        elif dataset_name=='mDC':
            network_types = ['mDC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']
        elif dataset_name=='mESC':
            network_types = ['mESC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network', 'mESC-lofgof-network']
        elif dataset_name=='mHSC-E' or dataset_name=='mHSC-GM' or dataset_name=='mHSC-L':
            network_types = ['mHSC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']
        else:
            print("network type error")

        print(dataset_name + ': ' )
        print(network_types)

        for network_type in network_types:

            print('\n\n****************************'+ dataset_name + '————' + network_type + '————' + str(Rank_num) + '****************************')

            data_path = processed_data + '\\' + dataset_name + '\\' + network_type + '\\Top' + str(
                Rank_num) + '\\'
            exp_file = data_path + 'Final_expression.csv'
            tf_file = data_path + 'Final_TF_index.csv'
            target_file = data_path + 'Final_gene_list.csv'
            # label_file = data_path + 'Final_GRNorTRN_pos_index.csv'

            save_path = save_folder + dataset_name + '\\' + network_type + '\\Top' + str(Rank_num) + '\\'

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            network_dict_name = args.modelname + '_3CV_'

            expression = pd.read_csv(exp_file,index_col=0).T

            tf_list = pd.read_csv(data_path + 'tf_list.csv', header=None).values
            pos_neg_balanced_id = pd.read_csv(data_path + 'pos_neg_balanced_id.csv', header=None)
            pos_neg_balanced_name = pd.read_csv(data_path + 'pos_neg_balanced_name.csv', header=None)

            columns = []
            kf = KFold(n_splits=3, shuffle=True)
            network_dict = {}
            all_network_dict = {}
            netavgAUROCs = []
            netavgAUPRs = []
            # netavgSPEs = []
            # netavgRecalls = []
            # netavgPrecisions = []
            # netavgF1s = []
            # netavgMCCs = []
            # netavgAccs = []
            for ki in range(args.iteration):

                columns.append(str(ki+1) + '-th 3CV')

                print('\n')
                print("\nthe {}th cross-validation..........\n".format(ki + 1))

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
                    train_pair_2fold, train_pair_label_2fold = get_samples_pair(pos_neg_balanced_id, train_TF)
                    test_pair_1fold, test_pair_label_1fold = get_samples_pair(pos_neg_balanced_id, test_TF)

                    train_pair, val_pair, train_pair_label, val_pair_label = train_test_split(train_pair_2fold, train_pair_label_2fold, test_size=0.2, random_state=1, shuffle=True, stratify = train_pair_label_2fold) # , random_state=seed

                    train_pair = np.array(train_pair)
                    val_pair = np.array(val_pair)
                    test_pair = np.array(test_pair_1fold)
                    feature = expression.values

                    feature = torch.from_numpy(feature).to(torch.float32)

                    tf = pd.read_csv(tf_file, index_col=0)['index'].values.astype(np.int64)
                    tf = torch.from_numpy(tf)

                    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                    data_feature = feature.to(device)
                    tf = tf.to(device)

                    train_load = scRNADataset(train_pair, feature.shape[0], flag=args.flag)
                    adj = train_load.Adj_Generate(tf, direction=args.flag, loop=args.loop)

                    adj = adj2saprse_tensor(adj)

                    train_pair = torch.from_numpy(train_pair)
                    val_pair = torch.from_numpy(val_pair)
                    test_pair = torch.from_numpy(test_pair)

                    model = GENELink(input_dim=feature.size()[1],
                                     hidden1_dim=args.hidden_dim[0],
                                     hidden2_dim=args.hidden_dim[1],
                                     hidden3_dim=args.hidden_dim[2],
                                     output_dim=args.output_dim,
                                     num_head1=args.num_head[0],
                                     num_head2=args.num_head[1],
                                     alpha=args.alpha,
                                     device=device,
                                     type=args.Type,
                                     reduction=args.reduction
                                     )

                    adj = adj.to(device)
                    model = model.to(device)
                    train_pair = train_pair.to(device)
                    val_pair = val_pair.to(device)
                    test_pair = test_pair.to(device)

                    optimizer = Adam(model.parameters(), lr=args.lr)
                    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

                    model_path = 'model/'
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)

                    for epoch in range(args.epochs):
                        key = 'epoch' + str(epoch+1)
                        running_loss = 0.0

                        for train_x, train_y in DataLoader(train_load, batch_size=args.batch_size, shuffle=True):
                            model.train()
                            optimizer.zero_grad()

                            if args.flag:
                                train_y = train_y.to(device).to(torch.long)
                                # train_y = train_y.to(device)[:,1].view(-1, 1)
                            else:
                                train_y = train_y.to(device).view(-1, 1)

                            pred = model(data_feature, adj, train_x)

                            if args.flag:
                                loss_BCE = torch.nn.CrossEntropyLoss()(pred, train_y)
                            else:
                                pred = torch.sigmoid(pred)
                                loss_BCE = F.binary_cross_entropy(pred, train_y)


                            loss_BCE.backward()
                            optimizer.step()
                            scheduler.step()

                            running_loss += loss_BCE.item()


                        model.eval()
                        score = model(data_feature, adj, val_pair)
                        if args.flag:
                            score = torch.softmax(score, dim=1)
                        else:
                            score = torch.sigmoid(score)

                        # score = torch.sigmoid(score)

                        AUC, AUPR = Evaluation(y_pred=score, y_true=val_pair[:, -1],flag=args.flag)
                        print('Epoch:{}'.format(epoch + 1),
                                'train loss:{}'.format(running_loss),
                                'AUC:{:.4F}'.format(AUC),
                                'AUPR:{:.4F}'.format(AUPR))

                    torch.save(model.state_dict(), model_path +'model.pkl')

                    model.load_state_dict(torch.load(model_path+'model.pkl'))
                    model.eval()
                    score_test = model(data_feature, adj, test_pair)
                    if args.flag:
                        score_test = torch.softmax(score_test, dim=1)
                    else:
                        score_test = torch.sigmoid(score_test)

                    AUC, AUPR = Evaluation(y_pred=score_test, y_true=test_pair[:, -1], flag=args.flag)


                    print('Test data：',
                          'AUC:{:.4F}'.format(AUC),
                          'AUPR:{:.4F}'.format(AUPR))

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
            print('all auroc: --------------------------------------------')
            print(netavgAUROCs)

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

