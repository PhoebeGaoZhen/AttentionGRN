import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizer_v2.gradient_descent import SGD
from data_process import preprocess_dataset, preprocess_dataset_GRN
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, auc
from scipy.interpolate import interp1d
import utils

source_path = '../data_preprocess/DATA1_AttentionGRN/'
dataset_paths = [
    source_path + '/hESC/hESC-ChIP-seq-network/Top500',
    source_path + '/hESC/Non-specific-ChIP-seq-network/Top500',
    source_path + '/hESC/STRING-network/Top500',

    # source_path + '/hHep/HepG2-ChIP-seq-network/Top500',
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
resultspath = "./CNNC_data1_GRN/"
if not os.path.isdir(resultspath):
    os.makedirs(resultspath)
resultfile = 'CNNC-GRN.csv'

iteration = 5
batch_size = 256
epochs = 50
num_classes = 2

for dataset_path in dataset_paths:
    data_augmentation = False
    model_name = 'keras_cnn_trained_model_shallow.h5'
    data_path = dataset_path +'/NEPDF_data'
    execution_time = 0
    train_path = data_path+'/Nxdata_tf7_train_edges_data' + '.npy'
    cell, network_type, ranknum = dataset_path.split('/')[4], dataset_path.split('/')[5], dataset_path.split('/')[
                                                                                              6][3:]
    if os.path.exists(train_path) :
        print(f"Paths for dataset {dataset_path} already exist. Skipping data preprocessing.")
    else:
        t_start1 = time.time()

        print(f"Paths for dataset {dataset_path} do not exist. Preprocessing data...")
        # preprocess_dataset(dataset_path) # balanced
        print("preprocess data..........")
        save_path = source_path + cell + '/' + network_type + '/Top' + str(ranknum) + '/'
        preprocess_dataset_GRN(dataset_path, save_path) # unbalanced

        t_end1 = time.time()

        execution_time = t_end1 - t_start1

    (x_train, y_train, count_set_train) = utils.load_data_TF_311(leix='train_edges_data', data_path =data_path)
    (x_val, y_val, count_set_val) = utils.load_data_TF_311(leix='validation_edges', data_path =data_path)
    (x_test, y_test, count_set) = utils.load_data_TF_311(leix='test_edges_data', data_path =data_path)
    # print(x_train.shape, 'x_train samples')
    # print(x_val.shape, 'x_val samples')
    # print(x_test.shape, 'x_test samples')

    mean_ap = []
    mean_roc = []
    mean_time = []
    for i in range(iteration):
        print('-----------the ' + str(i+1) + '-th test------' + cell + '----' + network_type + '----' + ranknum + '-----------')

        t_start = time.time()
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        if num_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
            loss_function = 'binary_crossentropy'
        else:
            model.add(Dense(num_classes))
            model.add(Activation('softmax'))
            loss_function = 'categorical_crossentropy'

        sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss=loss_function, metrics=['accuracy'])

        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(x_val, y_val),
                            shuffle=True)

        model.save(model_name)

        scores = model.evaluate(x_test, y_test, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

        y_predict = model.predict(x_test)
        # np.save('end_y_test.npy', y_test)
        # np.save('end_y_predict.npy', y_predict)
        auc11 = roc_auc_score(y_test, y_predict)
        mean_roc.append(auc11)
        precision, recall, _ = precision_recall_curve(y_test, y_predict)
        auprc = auc(recall, precision)
        mean_ap.append(auprc)
        mean_time.append(time.time() - t_start)

    print("AUC scores\n", mean_roc)
    AUC_scores = np.mean(mean_roc)
    AUC_std = np.std(mean_roc)
    print("Mean AUC score: ", np.mean(mean_roc),
          "\nStd of AUC scores: ", np.std(mean_roc), "\n \n")
    print("AP scores \n", mean_ap)
    AP_scores = np.mean(mean_ap)
    AP_std = np.std(mean_ap)
    print("Mean AP score: ", np.mean(mean_ap),
          "\nStd of AP scores: ", np.std(mean_ap), "\n \n")
    print("Running times\n", mean_time)
    time_mean = np.mean(mean_time)
    print("Mean running time: ", np.mean(mean_time),
          "\nStd of running time: ", np.std(mean_time), "\n \n")
    time_mean_total = np.mean(mean_time) + execution_time
    print("The time taken for data preprocessing and the total runtime: ", time_mean_total)

    current_time = time.time()
    local_time_struct = time.localtime(current_time)
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time_struct)

    #
    # column_names = [
    #     'date_name',
    #     'AUC scores1', 'AUC scores2', 'AUC scores3', 'AUC scores4', 'AUC scores5',
    #     'AUC scores6', 'AUC scores7', 'AUC scores8', 'AUC scores9', 'AUC scores10',
    #     'AP scores1', 'AP scores2', 'AP scores3', 'AP scores4', 'AP scores5', 'AP scores6',
    #     'AP scores7', 'AP scores8', 'AP scores9', 'AP scores10',
    #     'AUC mean', 'AUC std', 'AP mean', 'AP std', 'Running times'
    # ]
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

    if not os.path.exists(resultspath + resultfile):
        with open(resultspath + resultfile, mode='w', newline='') as file:
            np.savetxt(file, [column_names], delimiter=',', fmt='%s')

    with open(resultspath + resultfile, mode='a', newline='') as file:
        np.savetxt(file, [new_data], delimiter=',', fmt='%s')


utils.results_summary(resultspath, resultfile)
