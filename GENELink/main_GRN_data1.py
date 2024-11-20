'''
20240405
datasets: DATA1
model: GENELink
task: GRN inference
evaluation strategy: independent testing

positive samples: known TF-gene pairs
negative samples: unknown TF-gene pairs, the TF set are the same as TF set in positive samples
unbalanced datasets
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
import utils_Train_Test_Split as tts
from scGNN import GENELink
from utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation,  Network_Statistic

parser = argparse.ArgumentParser()
parser.add_argument('--modelname', type=str, default='GENELink', help='GENELink')
parser.add_argument('--lr', type=float, default=3e-3, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default= 50, help='Number of epoch.')
parser.add_argument('--num_head', type=list, default=[3,3], help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--hidden_dim', type=int, default=[128,64,32], help='The dimension of hidden layer')
parser.add_argument('--output_dim', type=int, default=16, help='The dimension of latent layer')
parser.add_argument('--batch_size', type=int, default=256, help='The size of each batch')
parser.add_argument('--seed', type=int, default=8, help='Random seed')
parser.add_argument('--reduction',type=str,default='concate', help='how to integrate multihead attention')
parser.add_argument('--iteration', type=int, default=5, help='the number of training and test')
parser.add_argument('--loop', type=bool, default=False, help='whether to add self-loop in adjacent matrix')
# parser.add_argument('--Type',type=str,default='dot', help='score metric')
parser.add_argument('--Type',type=str,default='MLP', help='score metric, causal inference')
parser.add_argument('--flag', type=bool, default=True, help='the identifier whether to conduct causal inference')
parser.add_argument('--hard', type=bool, default=False)
args = parser.parse_args()
seed = args.seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)


GRN_type = 'tf-gene-posneg' # tf-gene；  gene-gene; tf-gene-posneg
dataset_type = 'unbalanced' # balanced；  unbalanced
dataset_names = [ 'hESC','hHEP','mDC', 'mESC', 'mHSC-E', 'mHSC-GM', 'mHSC-L']
processed_data = '..\\data_preprocess\\DATA1_AttentionGRN\\'
save_folder = 'GENELink_data1_GRN\\'


for Rank_num in [500, 1000]:
    for dataset_name in dataset_names:
        network_types = []
        if dataset_name == 'hESC':
            network_types = ['hESC-ChIP-seq-network', 'Non-specific-ChIP-seq-network', 'STRING-network']

        elif dataset_name == 'hHEP' and Rank_num==500:
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
            data_path = processed_data + '\\' + dataset_name + '\\' + network_type + '\\Top' + str(Rank_num) + '\\'

            exp_file = data_path + 'Final_expression.csv'
            tf_file = data_path + 'Final_TF_common_index.csv'
            target_file = data_path + 'Final_gene_list.csv'
            label_file = data_path + 'Final_GRNorTRN_pos_index.csv'

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

                print('-----------------the ' + str(ki+1) + 'th iteration--------------------' +network_type + '--------------')
                if args.hard:
                    train_set_file = save_path + 'Train_set_hard.csv'
                    test_set_file = save_path + 'Test_set_hard.csv'
                    val_set_file = save_path + 'Validation_set_hard.csv'
                    metric_file = save_path + 'metric_hard.csv'
                    density = Network_Statistic(data_type=dataset_name, net_scale=Rank_num, net_type=network_type)
                    tts.train_val_test_set_hard(label_file, target_file, tf_file, train_set_file, val_set_file, test_set_file, density, p_val=0.5)
                else:
                    train_set_file = save_path + 'Train_set.csv'
                    test_set_file = save_path + 'Test_set.csv'
                    val_set_file = save_path + 'Validation_set.csv'
                    metric_file = save_path + 'metric.csv'
                    tts.train_val_test_set(label_file, target_file, tf_file, train_set_file, val_set_file, test_set_file, GRN_type,dataset_type)

                data_input = pd.read_csv(exp_file,index_col=0).T
                loader = load_data(data_input)
                feature = loader.exp_data()

                tf = pd.read_csv(tf_file,index_col=0)['index'].values.astype(np.int64)

                target = pd.read_csv(target_file,index_col=0)['index'].values.astype(np.int64)

                feature = torch.from_numpy(feature)
                tf = torch.from_numpy(tf)

                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                data_feature = feature.to(device)
                tf = tf.to(device)


                train_data = pd.read_csv(train_set_file, index_col=0).values  # tf ID -- target ID -- label
                validation_data = pd.read_csv(val_set_file, index_col=0).values  # tf ID -- target ID -- label
                test_data = pd.read_csv(test_set_file, index_col=0).values  # tf ID -- target ID -- label

                train_load = scRNADataset(train_data, feature.shape[0], flag=args.flag)

                adj = train_load.Adj_Generate(tf, direction=args.flag, loop=args.loop)

                adj = adj2saprse_tensor(adj)

                train_data = torch.from_numpy(train_data)
                val_data = torch.from_numpy(validation_data)
                test_data = torch.from_numpy(test_data)

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
                train_data = train_data.to(device)
                validation_data = val_data.to(device)
                test_data = test_data.to(device)


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
                    score = model(data_feature, adj, validation_data)
                    if args.flag:
                        score = torch.softmax(score, dim=1)
                    else:
                        score = torch.sigmoid(score)

                    # score = torch.sigmoid(score)
                    y_true = validation_data[:, -1]
                    AUC, AUPR = Evaluation(y_pred=score, y_true=validation_data[:, -1],flag=args.flag)
                    print('Epoch:{}'.format(epoch + 1),
                            'train loss:{}'.format(running_loss),
                            'AUC:{:.4F}'.format(AUC),
                            'AUPR:{:.4F}'.format(AUPR))

                torch.save(model.state_dict(), model_path +'model.pkl')

                model.load_state_dict(torch.load(model_path+'model.pkl'))
                model.eval()
                score_test = model(data_feature, adj, test_data)
                if args.flag:
                    score_test = torch.softmax(score_test, dim=1)
                else:
                    score_test = torch.sigmoid(score_test)

                AUC, AUPR = Evaluation(y_pred=score_test, y_true=test_data[:, -1], flag=args.flag)

                AUROCs.append(AUC)
                AUPRs1.append(AUPR)
                # SPEs.append(SPE)
                # Recalls.append(Recall)
                # Precisions.append(Precision)
                # F1s.append(F1)
                # MCCs.append(MCC)
                # Accs.append(ACC)

                print('Test data：',
                      'AUC:{:.4F}'.format(AUC),
                      'AUPR:{:.4F}'.format(AUPR))

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



