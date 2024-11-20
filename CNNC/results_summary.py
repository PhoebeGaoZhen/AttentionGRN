import numpy as np
import pandas as pd
import os
from pandas.core.frame import DataFrame

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


    if top500_specific.shape[0] == 7:
        s = ["hHEP", 0,0,0,0]
        s2 = np.array(s).reshape(1,-1)
        top500_specific_new = top500_specific[0:2]
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

    print("Top500 results of GRN inference via CNNC are saved in " + save_path_500)
    print("Top1000 results of GRN inference via CNNC are saved in " + save_path_1000)



resultspath = 'CNNC_data1_GRN/'
resultfile = 'CNNC-GRN.csv'
results_summary(resultspath, resultfile)

resultspath = 'CNNC_data2_TRN/'
resultfile = 'CNNC-TRN.csv'
results_summary(resultspath, resultfile)


