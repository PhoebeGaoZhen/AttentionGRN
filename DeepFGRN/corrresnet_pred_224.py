
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from sklearn.utils import class_weight


class Classifier_corrResNET_pred:

    def __init__(self, output_directory, nb_classes, x1, x2, net_emd_tf_s, net_emd_tf_t, net_emd_target_s,net_emd_target_t, verbose=False, build=True, load_weights=False, patience=5):
        keras.backend.clear_session()
        self.output_directory = output_directory
        self.patience = patience
        if build == True:
            self.model = self.build_model(nb_classes, x1, x2, net_emd_tf_s, net_emd_tf_t, net_emd_target_s,net_emd_target_t)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            if load_weights == True:
                # keras.backend.clear_session()
                self.model.load_weights(self.output_directory
                                        .replace('resnet_augment', 'resnet')
                                        .replace('TSC_itr_augment_x_10', 'TSC_itr_10')
                                        + '/model_init.hdf5')
            else:
                # keras.backend.clear_session()
                self.model.save_weights(self.output_directory + 'model_init.hdf5')
        return

    def reduceDimension(self, tensorx):

        tensorx = tf.squeeze(tensorx,[1])

        return tensorx

    def myreshape(self, tensorx):

        tensorx = tf.expand_dims(tensorx, axis=1)

        return tensorx

    def build_feature_model(self, x_train, nb_classes):
        n_feature_maps = 16

        # input_layer = keras.layers.Input(input_shape)
        input_layer = keras.layers.Input(shape=x_train.shape[1:])

        # BLOCK 1
        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=9, padding='same')(input_layer)
        conv_x = keras.layers.BatchNormalization(momentum=0.8)(conv_x)
        conv_x = keras.layers.MaxPooling1D(2, padding="same")(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)


        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization(momentum=0.8)(conv_y)
        conv_y = keras.layers.MaxPooling1D(2, padding="same")(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = keras.layers.BatchNormalization(momentum=0.8)(shortcut_y)
        shortcut_y = keras.layers.MaxPooling1D(2, padding="same")(shortcut_y)


        # output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.add([shortcut_y, conv_y])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)
        # C1 = output_block_1

        # BLOCK 2

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=9, padding='same')(output_block_1)
        conv_x = keras.layers.BatchNormalization(momentum=0.8)(conv_x)
        conv_x = keras.layers.MaxPooling1D(2, padding="same")(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization(momentum=0.8)(conv_y)
        conv_y = keras.layers.MaxPooling1D(2, padding="same")(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)


        # conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        # conv_z = keras.layers.BatchNormalization(momentum=0.8)(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = keras.layers.BatchNormalization(momentum=0.8)(shortcut_y)
        shortcut_y = keras.layers.MaxPooling1D(2, padding="same")(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_y])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)
        # C2 = output_block_2

        # BLOCK 3

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=9, padding='same')(output_block_2)
        conv_x = keras.layers.BatchNormalization(momentum=0.8)(conv_x)
        conv_x = keras.layers.MaxPooling1D(2, padding="same")(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization(momentum=0.8)(conv_y)
        conv_y = keras.layers.MaxPooling1D(2, padding="same")(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        # conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        # conv_z = keras.layers.BatchNormalization(momentum=0.8)(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization(momentum=0.8)(output_block_2)
        shortcut_y = keras.layers.MaxPooling1D(2, padding="same")(shortcut_y)

        output_block_3 = keras.layers.add([shortcut_y, conv_y])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)

        # FINAL

        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)

        output_layer = keras.layers.Dense(224, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        return model


    def build_model(self, nb_classes, x_tarin1, x_tarin2,net_emd_tf_s, net_emd_tf_t, net_emd_target_s,net_emd_target_t):

        feature_network1 = self.build_feature_model(x_tarin1,nb_classes)
        input1_fe_network1 = keras.layers.Input(shape=x_tarin1.shape[1:])
        output1_feature_network1 = feature_network1(input1_fe_network1)

        feature_network2 = self.build_feature_model(x_tarin2, nb_classes)
        input2_fe_network2 = keras.layers.Input(shape=x_tarin2.shape[1:])
        output2_feature_network2 = feature_network2(input2_fe_network2)

        input_layer_net_tf_s = keras.layers.Input(shape=net_emd_tf_s.shape[1:])
        input_layer_net_tf_t = keras.layers.Input(shape=net_emd_tf_t.shape[1:])
        input_layer_net_target_s = keras.layers.Input(shape=net_emd_target_s.shape[1:])
        input_layer_net_target_t = keras.layers.Input(shape=net_emd_target_t.shape[1:])

        fci = keras.layers.Concatenate()([output1_feature_network1, output2_feature_network2])
        # fci = keras.layers.Concatenate()([feature_network1.output, feature_network2.output])
        fc0 = keras.layers.Flatten()(fci)
        fc1 = keras.layers.Dense(256, activation='relu')(fc0)
        fc1 = keras.layers.Dropout(0.3)(fc1)

        fc2 = keras.layers.Dense(224, activation='relu')(fc1)

        input_layer_net_tf_s_ = keras.layers.Lambda(self.reduceDimension)(input_layer_net_tf_s)
        input_layer_net_tf_t_ = keras.layers.Lambda(self.reduceDimension)(input_layer_net_tf_t)
        input_layer_net_target_s_ = keras.layers.Lambda(self.reduceDimension)(input_layer_net_target_s)
        input_layer_net_target_t_ = keras.layers.Lambda(self.reduceDimension)(input_layer_net_target_t)

        allfeature = keras.layers.Concatenate()([fc2, input_layer_net_tf_s_,input_layer_net_tf_t_,input_layer_net_target_s_,input_layer_net_target_t_]) # net_emd_s, net_emd_t需要转变为layer吗
        # allfeature = keras.layers.Concatenate()([output1_feature_network1, output2_feature_network2, fc2, input_layer_net_tf_s_,input_layer_net_tf_t_,input_layer_net_target_s_,input_layer_net_target_t_]) # net_emd_s, net_emd_t需要转变为layer吗

        allfeature = keras.layers.Lambda(self.myreshape)(allfeature)

        conv_x = keras.layers.Conv1D(filters=16, kernel_size=16, padding='same')(allfeature)
        conv_x = keras.layers.BatchNormalization(momentum=0.8)(conv_x)
        conv_x = keras.layers.MaxPooling1D(5, padding="same")(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        gap_layer = keras.layers.GlobalAveragePooling1D()(conv_x)

        fc_pred2 = keras.layers.Dense(128, activation='relu')(gap_layer)
        fc_pred2 = keras.layers.Dropout(0.3)(fc_pred2)
        fc_pred4 = keras.layers.Dense(nb_classes, activation='softmax')(fc_pred2)
        classifiers_model = keras.models.Model(inputs=[input1_fe_network1, input2_fe_network2,input_layer_net_tf_s,input_layer_net_tf_t,input_layer_net_target_s,input_layer_net_target_t], outputs=fc_pred4)
        # classifiers_model = keras.models.Model(inputs=[feature_network1.input, feature_network2.input], outputs=fc3)

        return classifiers_model

    def fit(self, x_train_1,x_train_2, y_train, x_val_1, x_val_2, y_val):
        # if not tf.test.is_gpu_available:
        #     print('error')
        #     exit()

        self.model.compile(loss='categorical_crossentropy',
                         optimizer=keras.optimizers.Adam(),
                         metrics=['acc'])
        # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.001, patience=int(self.patience / 2), min_lr=0.0001)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.001, patience=int(self.patience / 2), verbose=1)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_acc',
                                                           save_best_only=True)
        early_stop = keras.callbacks.EarlyStopping('val_acc', 0.0001, patience=self.patience)
        self.callbacks = [reduce_lr, early_stop, model_checkpoint]

        # x_val and y_val are only used to monitor the test loss and NOT for training
        batch_size = 32
        nb_epochs = 150

        mini_batch_size = int(min(x_train_1.shape[0] / 10, batch_size))
        hist = self.model.fit([x_train_1,x_train_2], y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=([x_val_1, x_val_2], y_val), callbacks=self.callbacks)

        keras.backend.clear_session()


    def fit_5CV(self, x_train_1,x_train_2, net_emd_tf_s_train, net_emd_tf_t_train, net_emd_target_s_train,net_emd_target_t_train, y_train,
                x_val_1, x_val_2,net_emd_tf_s_val, net_emd_tf_t_val, net_emd_target_s_val,net_emd_target_t_val, y_val,
                x_test_1, x_test_2,net_emd_tf_s_test, net_emd_tf_t_test, net_emd_target_s_test,net_emd_target_t_test):
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=keras.optimizers.Adam(),
                           metrics=['acc'])
        # reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.001, patience=int(self.patience / 2), min_lr=0.0001)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.001, patience=int(self.patience / 2),
                                                      verbose=1)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_acc',
                                                           save_best_only=True)
        early_stop = keras.callbacks.EarlyStopping('val_acc', 0.0001,
                                                   patience=self.patience)
        self.callbacks = [reduce_lr, early_stop, model_checkpoint]

        batch_size = 32
        nb_epochs = 150

        mini_batch_size = int(min(x_train_1.shape[0] / 10, batch_size))
        hist = self.model.fit([x_train_1,x_train_2, net_emd_tf_s_train, net_emd_tf_t_train, net_emd_target_s_train,net_emd_target_t_train], y_train, batch_size=batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=([x_val_1, x_val_2,net_emd_tf_s_val, net_emd_tf_t_val, net_emd_target_s_val,net_emd_target_t_val], y_val),
                              callbacks=self.callbacks)

        # keras.backend.clear_session()

        self.model.save(self.output_directory + 'last_model.hdf5')

        y_pred = self.model.predict([x_test_1, x_test_2,net_emd_tf_s_test, net_emd_tf_t_test, net_emd_target_s_test,net_emd_target_t_test])
        yy_pred = np.argmax(y_pred, axis=1)
        return y_pred, yy_pred


    def predict(self, x_test_1, x_test_2,net_emd_tf_s_test, net_emd_tf_t_test, net_emd_target_s_test,net_emd_target_t_test):
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict([x_test_1, x_test_2,net_emd_tf_s_test, net_emd_tf_t_test, net_emd_target_s_test,net_emd_target_t_test])

        return y_pred



