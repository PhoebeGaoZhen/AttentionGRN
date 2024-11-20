import os
os.environ['CUDA_VISIBLE_DEVICES']="0"
import tensorflow as tf


def generator_CNN2_model():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(56*16, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    # (16,16)
    # (32,16)
    model.add(tf.keras.layers.Reshape((56,16)))

    # (16,16)-- (32,16)
    # (32,16) ---(64,16)
    model.add(tf.keras.layers.UpSampling1D(2))
    # (32,16) ---(32,16)
    # (64,16)---(64,16)
    model.add(tf.keras.layers.Conv1D(16, 5, padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    # (32,16)---(64,16)
    # (64,16)---(128,16)
    model.add(tf.keras.layers.UpSampling1D(2))
    # (64,16)---(64,1)
    # (128,16)---(128,1)
    model.add(tf.keras.layers.Conv1D(1, 5, padding='same', use_bias=False,
                                              activation='tanh'))

    return model

def generator_CNN1_model():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(112*16, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    # (64, 16)
    model.add(tf.keras.layers.Reshape((112, 16)))

    # (32,16)---(64,16)
    # (64,16)---(128,16)
    model.add(tf.keras.layers.UpSampling1D(2))
    # (64,16)---(64,1)
    # (128,16)---(128,1)
    model.add(tf.keras.layers.Conv1D(1, 5, padding='same', use_bias=False,
                                              activation='tanh'))  # (64, 64, 1)

    # model.add(tf.keras.layers.LeakyReLU())
    # model.add(tf.keras.layers.Dropout(0.3))

    return model



class Generator():

    def __init__(self, n_node, node_emd_init, config):

        self.n_node = n_node
        self.node_emd_init = node_emd_init
        self.emd_dim = config.n_emb

        with tf.variable_scope('generator'):

            self.node_embedding_matrix = tf.get_variable(name = 'gen_node_embedding',
                                                       shape = [self.n_node, self.emd_dim],
                                                       initializer = tf.contrib.layers.xavier_initializer(uniform = False),
                                                       trainable = True)

            self.node_ids = tf.placeholder(tf.int32, shape = [None])
            self.noise_embedding = tf.placeholder(tf.float32, shape = [2, None, self.emd_dim])
            self.dis_node_embedding = tf.placeholder(tf.float32, shape = [2, None, self.emd_dim])


            self.node_embedding = tf.nn.embedding_lookup(self.node_embedding_matrix, self.node_ids)

            _noise_embedding= []
            for i in range(2):

                _noise_embedding.append(tf.reshape(tf.nn.embedding_lookup(self.noise_embedding, tf.constant([i])),
                                                   [-1, self.emd_dim]))
            _dis_node_embedding = []
            for i in range(2):

                _dis_node_embedding.append(tf.reshape(tf.nn.embedding_lookup(self.dis_node_embedding, tf.constant([i])),
                                                      [-1, self.emd_dim]))

            _neg_loss = [0.0, 0.0]
            _fake_node_embedding_list = []
            _score = [0, 0]


            for i in range(2):

                _fake_node_embedding = self.generate_node(self.node_embedding, _noise_embedding[i])

                _fake_node_embedding = tf.reshape(_fake_node_embedding, [-1,self.emd_dim])

                _fake_node_embedding_list.append(_fake_node_embedding)

                _score[i] = tf.reduce_sum(tf.multiply(_dis_node_embedding[i], _fake_node_embedding), axis=1)

                _neg_loss[i] = -tf.reduce_mean(_score[i])

            self.fake_node_embedding = _fake_node_embedding_list

            self.neg_loss = _neg_loss
            self.loss = self.neg_loss[0] + self.neg_loss[1]
            # optimizer = tf.train.AdamOptimizer(config.lr_gen)
            optimizer = tf.train.RMSPropOptimizer(config.lr_gen)
            self.g_updates = optimizer.minimize(self.loss)

    def generate_node(self, node_embedding, noise_embedding):

        input = tf.reshape(node_embedding, [-1, self.emd_dim])
        input = tf.add(input, noise_embedding)

        generator_keras = generator_CNN1_model()
        output = generator_keras(input, training=True)
        # generator_keras.summary()

        return output


