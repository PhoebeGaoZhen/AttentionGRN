import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()

def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random.uniform([input_dim, output_dim], minval = -init_range, maxval = init_range, dtype = tf.float32)
    return tf.Variable(initial, name=name)
def glorot(shape, name = None):
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random.uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

class GraphConvolution():
    def __init__(self, n_node, input_dim, output_dim, dropout=0., act = tf.nn.relu, norm = False, is_train=False):
        self.name = 'Convolution'
        self.var = {}
        self.issparse = True
        self.n_node = n_node
        with tf.compat.v1.variable_scope(self.name+'_vars'):
            self.var['weights1'] = glorot([input_dim, output_dim])#, name='weights1'
            self.var['weights2'] = weight_variable_glorot(output_dim, 64, name = 'weights2')
        self.dropout = dropout
        self.act = act
        self.norm = norm
        self.is_train = is_train

    def encoder(self, inputs, adj):
        with tf.name_scope(self.name):
            x = inputs
            x = tf.matmul(x, self.var['weights1'])
            x = tf.sparse.sparse_dense_matmul(adj, x)
            outputs = self.act(x)

            x2 = tf.matmul(outputs, self.var['weights2'])
            x2 = tf.sparse.sparse_dense_matmul(adj, x2)
            outputs = self.act(x2)
        if self.norm:
            outputs = tf.layers.batch_normalization(outputs, training = self.is_train)
            # adj = tf.cast(adj, tf.float32)
            # x = inputs
            # x = tf.nn.dropout(x, 1- self.dropout)
            # x = tf.matmul(x, self.var['weights'])
            # x = tf.matmul(adj, x)
            # output = self.act(x)
            # if self.norm:
            #     output = tf.layers.batch_normalization(output)

        return outputs

    def decoder(self, embed, nd):
        embed_size = embed.shape[1]
        logits = tf.matmul(embed, tf.transpose(embed))
        logits = tf.reshape(logits, [-1,1])
        return tf.nn.relu(logits)

    def training(self, loss, lr, l2_cof):
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)

        train_op = opt.minimize(loss)
        return train_op

