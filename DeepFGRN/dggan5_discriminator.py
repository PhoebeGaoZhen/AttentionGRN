import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="0,1"

import tensorflow as tf


def discriminator_CNN3_model():
    model = tf.keras.Sequential()

    # (64,1) ---(32,8)
    model.add(tf.keras.layers.Conv1D(8, kernel_size = 3, strides=2, padding="same",use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    # (32,8)---(16,16)
    model.add(tf.keras.layers.Conv1D(16, kernel_size=3, strides=2, padding="same",use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    # (16,16)---(8,32)
    model.add(tf.keras.layers.Conv1D(32, kernel_size=3, strides=2, padding="same",use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Dense(1))

    return model

def discriminator_CNN2_model():
    model = tf.keras.Sequential()

    # (64,1) ---(32,8)
    model.add(tf.keras.layers.Conv1D(8, kernel_size = 3, strides=2, padding="same",use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    # (32,8)---(16,16)
    model.add(tf.keras.layers.Conv1D(16, kernel_size=3, strides=2, padding="same",use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Dense(1))

    return model

def discriminator_CNN1_model():
    model = tf.keras.Sequential()

    # (64,1) ---(32,8)
    model.add(tf.keras.layers.Conv1D(8, kernel_size = 3, strides=2, padding="same",use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Dense(1))

    return model




class Discriminator():
    def __init__(self, n_node, node_emd_init, config):
        self.n_node = n_node
        self.emd_dim = config.n_emb
        self.node_emd_init = node_emd_init

        with tf.variable_scope('disciminator'):

            if node_emd_init:
                self.node_embedding_matrix = tf.get_variable(name = 'dis_node_embedding',
                                                             shape = self.node_emd_init.shape,
                                                             initializer = tf.constant_initializer(self.node_emd_init),
                                                             trainable = True)
            else:
                self.node_embedding_matrix = tf.get_variable(name = 'dis_node_embedding',
                                                             shape = [2, self.n_node, self.emd_dim],
                                                             initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                                                             trainable = True)

            self.pos_node_ids = tf.placeholder(tf.int32, shape = [None])
            self.pos_node_neighbor_ids = tf.placeholder(tf.int32, shape = [None])
            self.fake_node_embedding = tf.placeholder(tf.float32, shape = [2, 2, None, self.emd_dim])

            _node_embedding_matrix = []
            for i in range(2):
                _node_embedding_matrix.append(tf.reshape(tf.nn.embedding_lookup(self.node_embedding_matrix, tf.constant([i])), [-1, self.emd_dim]))

            self.pos_node_embedding_1 = tf.nn.embedding_lookup(_node_embedding_matrix[0], self.pos_node_ids)
            self.pos_node_neighbor_embedding_1 = tf.nn.embedding_lookup(_node_embedding_matrix[1], self.pos_node_neighbor_ids)
            discriminator_keras = discriminator_CNN2_model()
            self.pos_node_embedding_2 = tf.reshape(self.pos_node_embedding_1, [-1, self.emd_dim, 1])
            self.pos_node_neighbor_embedding_2 = tf.reshape(self.pos_node_neighbor_embedding_1, [-1, self.emd_dim, 1])
            self.pos_node_embedding = discriminator_keras(self.pos_node_embedding_2, training=True)
            self.pos_node_neighbor_embedding = discriminator_keras(self.pos_node_neighbor_embedding_2, training=True)
            pos_score = tf.matmul(self.pos_node_embedding, self.pos_node_neighbor_embedding, transpose_b=True)
            self.pos_loss = -tf.reduce_mean(pos_score)

            _neg_loss = [0, 0, 0, 0]
            node_id = [self.pos_node_ids, self.pos_node_neighbor_ids]
            for i in range(2):
                for j in range(2):
                    node_embedding = tf.nn.embedding_lookup(_node_embedding_matrix[j], node_id[i])
                    ''' 
                    fake_node_embedding = [[us,ut],[vs,vt]]
                    i=0 j=0 
                        node_embedding = lookup(_node_embedding_matrix[0], node_id[0])             [u]        
                        _fake_node_embedding = fake_node_embedding[] row 0，reshape (2,-1,128);  [us,ut]
                        _fake_node_embedding = _fake_node_embedding row 0，reshape (-1,128);   [us]
                        neg_score = tf.matmul(node_embedding, _fake_node_embedding, transpose_b=True) (us,u)
                    i=0 j=1
                        node_embedding = lookup(_node_embedding_matrix[1], node_id[0])              [u] 
                        _fake_node_embedding = fake_node_embedding row 0，reshape (2,-1,128);   [us,ut]
                        _fake_node_embedding = _fake_node_embedding row 1，reshape (-1,128);    [ut]
                        neg_score = tf.matmul(node_embedding, _fake_node_embedding, transpose_b=True)  (u,ut)
                    i=1 j=0
                        node_embedding = lookup(_node_embedding_matrix[0], node_id[1])              [v] 
                        _fake_node_embedding = fake_node_embedding row 1，reshape (2,-1,128);   [vs,vt]
                        _fake_node_embedding = _fake_node_embedding row 0，reshape (-1,128);    [vs]
                        neg_score = tf.matmul(node_embedding, _fake_node_embedding, transpose_b=True) (vs,v)
                    i=1 j=1 
                        node_embedding = lookup(_node_embedding_matrix[1], node_id[1])              [v] 
                        _fake_node_embedding = fake_node_embedding row 1，reshape (2,-1,128);   [vs,vt]
                        _fake_node_embedding = _fake_node_embedding row 1，reshape (-1,128);    [vt]
                        neg_score = tf.matmul(node_embedding, _fake_node_embedding, transpose_b=True)  (v,vt)
                    '''

                    _fake_node_embedding = tf.reshape(tf.nn.embedding_lookup(self.fake_node_embedding, tf.constant([i])), [2, -1, self.emd_dim])
                    _fake_node_embedding = tf.reshape(tf.nn.embedding_lookup(_fake_node_embedding, tf.constant([j])),[-1, self.emd_dim])
                    neg_score = tf.matmul(node_embedding, _fake_node_embedding, transpose_b=True)
                    # _neg_loss[i * 2 + j] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_score), logits=neg_score))
                    _neg_loss[i * 2 + j] = tf.reduce_mean(neg_score)


            self.neg_loss = _neg_loss
            self.loss = self.neg_loss[0] * config.neg_weight[0] + self.neg_loss[1] * config.neg_weight[1] + \
                    self.neg_loss[2] * config.neg_weight[2] + self.neg_loss[3] * config.neg_weight[3] + self.pos_loss
            '''
            config.neg_weight: [1, 1, 1, 1]
            '''
            # optimizer = tf.train.AdamOptimizer(config.lr_dis)
            #optimizer = tf.train.GradientDescentOptimizer(config.lr_dis)
            optimizer = tf.train.RMSPropOptimizer(config.lr_dis)
            self.d_updates = optimizer.minimize(self.loss)
            #self.reward = tf.log(1 + tf.exp(tf.clip_by_value(self.score, clip_value_min=-10, clip_value_max=10)))
            # print('here is discriminator-----------------------')
