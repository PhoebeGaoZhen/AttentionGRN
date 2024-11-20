from gcn import GraphConvolution
import tensorflow as tf
from utils import masked_accuracy

tf.compat.v1.disable_eager_execution()

class TransorfomerModel:
    def __init__(self, exp, do_train = False):
        self.batch_size = 1
        self.lr = 0.001 #0.005
        self.l2_coef = 0.0005
        self.weight_decay = 5e-04
        self.dim_embedding = 100
        self.do_train = do_train
        self.num_filter = 1
        self.num_nodes = exp.shape[0]
        self.num_Cells = exp.shape[1]
        self.entry_size = self.num_nodes ** 2
        self.exp = exp

        if self.do_train:
            print(1)
        else:
            with tf.name_scope('input_train'):
                self.encoded_gene = tf.compat.v1.placeholder(dtype=tf.float32,
                                                                shape=(self.num_nodes, self.num_Cells))
                self.bias_in = tf.compat.v1.sparse_placeholder(dtype=tf.float32, name='graph_train')
                self.lbl_in = tf.compat.v1.placeholder(dtype=tf.int32, shape=(self.entry_size, self.batch_size))
                self.msk_in = tf.compat.v1.placeholder(dtype=tf.int32, shape=(self.entry_size, self.batch_size))
                self.neg_msk = tf.compat.v1.placeholder(dtype=tf.int32, shape=(self.entry_size, self.batch_size))



        # self.instantiate_embeddings()
        self.logits = self.inference()
        self.loss, self.accuracy = self.loss_func()
        self.train_op = self.train()

    def inference(self):
        embedding_genes = self.exp ####已有数据，无需提取
        embedding_genes = tf.nn.l2_normalize(embedding_genes, 1)

        self.model = GraphConvolution(
            n_node= embedding_genes.shape[0],
            input_dim= embedding_genes.shape[1],
            output_dim= 128,
            act= tf.nn.leaky_relu,
            dropout= 0.25
        )
        if self.do_train:
           return 1
        else:
            self.final_embedding = self.model.encoder(embedding_genes, self.bias_in)
            logits = self.model.decoder(self.final_embedding, self.num_nodes)
            return logits

    def loss_func(self):
        if self.do_train:
            loss = 0
            accuracy = 0
            print(1)
        else:
           loss = masked_accuracy(self.logits, self.lbl_in, self.msk_in, self.neg_msk)
           accuracy  =  loss
        return loss, accuracy
    def train(self):
        train_op = self.model.training(self.loss, self.lr, self.l2_coef)
        return train_op

    # def instantiate_embeddings(self):
    #     """define all embeddings here"""
    #     with tf.name_scope("token_embedding"):
    #        self.embedding_tokens = tf.get_variable("embedding", shape=[self.token_size, self.dim_embedding],initializer=tf.random_normal_initializer(stddev=0.1))





