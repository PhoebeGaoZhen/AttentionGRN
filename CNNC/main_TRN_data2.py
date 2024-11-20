
from __future__ import print_function

import random

# Usage  python train_with_labels_three_fold.py number_of_data_parts_divided NEPDF_pathway number_of_category
# command line in developer's linux machine :
# module load cuda-8.0 using GPU
#srun -p gpu --gres=gpu:1 -c 2 --mem=20Gb python train_with_labels_three_foldx.py 9 /home/yey3/cnn_project/code3/NEPDF_data 3 > results.txt
#######################OUTPUT
# it will generate three-fold cross validation results
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from keras.callbacks import EarlyStopping,ModelCheckpoint
from data_process_3fold import preprocess_dataset
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import os,sys
import time, utils
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
from scipy.interpolate import interp1d

batch_size = 256
epochs = 30
model_name = 'keras_cnn_trained_model_shallow.h5'
iteration = 5
num_classes = 2
###################################################

source_path = '../data_preprocess/DATA2/'
resultspath = './CNNC_data2_TRN/'
if not os.path.isdir(resultspath):
    os.makedirs(resultspath)
resultfile = 'CNNC-TRN.csv'

dataset_paths = [
    source_path + '/hESC/hESC-ChIP-seq-network/Top500',
    source_path + '/hESC/Non-specific-ChIP-seq-network/Top500',
    source_path + '/hESC/STRING-network/Top500',

    source_path + '/hHep/HepG2-ChIP-seq-network/Top500',
    source_path + '/hHep/Non-specific-ChIP-seq-network/Top500',
    source_path + '/hHep/STRING-network/Top500',

    source_path + '/mDC/mDC-ChIP-seq-network/Top500',
    source_path + '/mDC/Non-Specific-ChIP-seq-network/Top500',
    source_path + '/mDC/STRING-network/Top500',

    source_path + '/mESC/mESC-ChIP-seq-network/Top500',
    source_path + '/mESC/Non-Specific-ChIP-seq-network/Top500',
    source_path + '/mESC/STRING-network/Top500',
    source_path + '/mESC/mESC-lofgof-network/Top500',

    source_path + '/mHSC-E/mHSC-ChIP-seq-network/Top500',
    source_path + '/mHSC-E/Non-Specific-ChIP-seq-network/Top500',
    source_path + '/mHSC-E/STRING-network/Top500',

    source_path + '/mHSC-GM/mHSC-ChIP-seq-network/Top500',
    source_path + '/mHSC-GM/Non-Specific-ChIP-seq-network/Top500',
    source_path + '/mHSC-GM/STRING-network/Top500',

    source_path + '/mHSC-L/mHSC-ChIP-seq-network/Top500',
    source_path + '/mHSC-L/Non-Specific-ChIP-seq-network/Top500',
    source_path + '/mHSC-L/STRING-network/Top500',

    source_path + '/hESC/hESC-ChIP-seq-network/Top1000',
    source_path + '/hESC/Non-specific-ChIP-seq-network/Top1000',
    source_path + '/hESC/STRING-network/Top1000',

    source_path + '/hHep/HepG2-ChIP-seq-network/Top1000',
    source_path + '/hHep/Non-specific-ChIP-seq-network/Top1000',
    source_path + '/hHep/STRING-network/Top1000',

    source_path + '/mDC/mDC-ChIP-seq-network/Top1000',
    source_path + '/mDC/Non-Specific-ChIP-seq-network/Top1000',
    source_path + '/mDC/STRING-network/Top1000',

    source_path + '/mESC/mESC-ChIP-seq-network/Top1000',
    source_path + '/mESC/Non-Specific-ChIP-seq-network/Top1000',
    source_path + '/mESC/STRING-network/Top1000',
    source_path + '/mESC/mESC-lofgof-network/Top1000',

    source_path + '/mHSC-E/mHSC-ChIP-seq-network/Top1000',
    source_path + '/mHSC-E/Non-Specific-ChIP-seq-network/Top1000',
    source_path + '/mHSC-E/STRING-network/Top1000',

    source_path + '/mHSC-GM/mHSC-ChIP-seq-network/Top1000',
    source_path + '/mHSC-GM/Non-Specific-ChIP-seq-network/Top1000',
    source_path + '/mHSC-GM/STRING-network/Top1000',

    source_path + '/mHSC-L/mHSC-ChIP-seq-network/Top1000',
    source_path + '/mHSC-L/Non-Specific-ChIP-seq-network/Top1000',
    source_path + '/mHSC-L/STRING-network/Top1000'
]



for dataset_path in dataset_paths:
    tf_list = pd.read_csv(dataset_path + '/tf_list.csv', header=None).values
    data_path = dataset_path + '/NEPDF_data'
    length_TF =len(tf_list)
    t_start1 = time.time()

    print(f"Paths for dataset {dataset_path} do not exist. Preprocessing data...")
    preprocess_dataset(dataset_path,tf_list)

    # all samples
    file_path = dataset_path + '/pos_neg_balanced_id.csv'
    df = pd.read_csv(file_path, header=None, names=['TF', 'target', 'label'], skiprows=1)
    samples_labels = df.to_numpy()

    t_end1 = time.time()

    execution_time = t_end1 - t_start1

    cell, network_type, ranknum = dataset_path.split('/')[4], dataset_path.split('/')[5], dataset_path.split('/')[6][3:]

    mean_ap = []
    mean_roc = []
    mean_time = []
    for i in range(iteration):

        print('\n'+'-----------the ' + str(i+1) + '-th test------' + cell + '----' + network_type + '----' + ranknum + '-----------')

        whole_data_TF = [i for i in range(length_TF)]
        # random.shuffle(whole_data_TF)
        # auc_scores = []
        # aupr_scores = []
        t_start = time.time()
        AUROCs = []
        AUPRs = []
        for test_indel in range(1, 4):  ################## three fold cross validation
            test_TF = [i for i in range(int(np.ceil((test_indel - 1) * 0.333333 * length_TF)),
                                        int(np.ceil(test_indel * 0.333333 * length_TF)))]

            train_TF = [i for i in whole_data_TF if i not in test_TF]
            (x_train, y_train, count_set_train) = utils.load_data_TF_3CV_2(train_TF, data_path)
            (x_test, y_test, count_set) = utils.load_data_TF_3CV_2(test_TF, data_path)
            # print(x_train.shape, 'x_train samples')
            # print(x_test.shape, 'x_test samples')
            save_dir = os.path.join(os.getcwd(),
                                    str(test_indel) + '_xxjust_two_3fold_db_lr002_YYYY_saved_models_T_32-32-64-64-128-128-512_e' + str(
                                        epochs))
            if num_classes > 2:
                y_train = keras.utils.to_categorical(y_train, num_classes)
                y_test = keras.utils.to_categorical(y_test, num_classes)
            # print(y_train.shape, 'y_train samples')
            # print(y_test.shape, 'y_test samples')
            ############
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            ############
            model = Sequential()
            model.add(Conv2D(32, (3, 3), padding='same',
                             input_shape=x_train.shape[1:]))
            model.add(Activation('relu'))
            model.add(Conv2D(32, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.5))

            model.add(Conv2D(64, (3, 3), padding='same'))
            model.add(Activation('relu'))
            model.add(Conv2D(64, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.5))

            model.add(Conv2D(128, (3, 3), padding='same'))
            model.add(Activation('relu'))
            model.add(Conv2D(128, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.5))

            model.add(Flatten())
            model.add(Dense(512))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            if num_classes < 2:
                print('no enough categories')
                sys.exit()
            elif num_classes == 2:
                model.add(Dense(1, activation='sigmoid'))
                sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)
                model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
            else:
                model.add(Dense(num_classes))
                model.add(Activation('softmax'))
                sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)
                model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

            early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', patience=200, verbose=0, mode='auto')
            checkpoint1 = ModelCheckpoint(filepath=save_dir + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                          monitor='val_loss',
                                          verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
            checkpoint2 = ModelCheckpoint(filepath=save_dir + '/weights.hdf5', monitor='val_acc', verbose=1,
                                          save_best_only=True, mode='auto', period=1)
            callbacks_list = [checkpoint2, early_stopping]
            # if not data_augmentation:
            #     print('Not using data augmentation.')
            history = model.fit(x_train, y_train,
                                batch_size=batch_size,
                                epochs=epochs, validation_split=0.2,
                                shuffle=True, callbacks=callbacks_list)

            # Save model and weights
            model_path = os.path.join(save_dir, model_name)
            model.save(model_path)
            print('Saved trained model at %s ' % model_path)
            # Score trained model.
            scores = model.evaluate(x_test, y_test, verbose=1)
            # print('Test loss:', scores[0])
            # print('Test accuracy:', scores[1])
            y_predict = model.predict(x_test)
            # np.save(save_dir + '/end_y_test.npy', y_test)
            # np.save(save_dir + '/end_y_predict.npy', y_predict)

            AUC = roc_auc_score(y_test, y_predict)
            precision_aupr, recall_aupr, _ = precision_recall_curve(y_test, y_predict)
            AUPR = auc(recall_aupr, precision_aupr)

            AUROCs.append(AUC)
            AUPRs.append(AUPR)

        avg_AUROC = np.mean(AUROCs)
        avg_AUPR = np.mean(AUPRs)

        mean_roc.append(avg_AUROC)
        mean_ap.append(avg_AUPR)

        mean_time.append(time.time() - t_start)

    time_mean = np.mean(mean_time)
    time_mean_total = np.mean(mean_time) + execution_time

    current_time = time.time()
    local_time_struct = time.localtime(current_time)
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time_struct)

    column_names = [
        'date_name',
        'AUC scores1', 'AUC scores2', 'AUC scores3', 'AUC scores4', 'AUC scores5',
        'AP scores1', 'AP scores2', 'AP scores3', 'AP scores4', 'AP scores5',
        'AUC mean', 'AUC std', 'AP mean', 'AP std', 'Running times'
    ]

    mean_roc = np.array(mean_roc)
    mean_ap = np.array(mean_ap)

    AUC_scores_mean = np.mean(mean_roc)
    AUC_scores_std = np.std(mean_roc)
    AP_scores_mean = np.mean(mean_ap)
    AP_scores_std = np.std(mean_ap)

    new_data = [dataset_path] + list(mean_roc) + list(mean_ap) + \
               [AUC_scores_mean, AUC_scores_std, AP_scores_mean, AP_scores_std, time_mean_total]

    if not os.path.exists(resultspath+resultfile):
        with open(resultspath+resultfile, mode='w', newline='') as file:
            np.savetxt(file, [column_names], delimiter=',', fmt='%s')

    with open(resultspath+resultfile, mode='a', newline='') as file:
        np.savetxt(file, [new_data], delimiter=',', fmt='%s')

utils.results_summary(resultspath, resultfile)






