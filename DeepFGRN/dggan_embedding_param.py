
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="0"
import tensorflow as tf
import time
import numpy as np
import math
import dggan_utils
from dggan5_discriminator import Discriminator
from dggan5_generator_224 import Generator
import dggan_config_224 as dggan_config

class Model():
    def __init__(self,pathnetwork, pathnode):
        tf.reset_default_graph()
        t = time.time()
        print('reading graph...')

        self.graph, self.n_node, self.node_list, self.node_list_s, self.egs = dggan_utils.read_graph_Ecoli(pathnetwork, pathnode)
        print()
        self.node_emd_shape = [2, self.n_node, dggan_config.n_emb]
        print('[%.2f] reading graph finished. #node = %d' % (time.time() - t, self.n_node))

        self.dis_node_embed_init = None
        self.gen_node_embed_init = None

        self.discriminator = None
        self.generator = None

        self.build_generator()
        self.build_discriminator()

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = self.config)
        self.saver = tf.train.Saver(max_to_keep=0)

        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init_op)

    def build_discriminator(self):

        self.discriminator = Discriminator(n_node = self.n_node,
                                           node_emd_init = self.dis_node_embed_init,
                                           config = dggan_config)
    def build_generator(self):

        self.generator = Generator(n_node = self.n_node,
                                   node_emd_init = self.gen_node_embed_init,
                                   config = dggan_config)
    '''
    dis_loss = 0.0
    dis_pos_loss = 0.0
    dis_neg_loss = [0.0, 0.0, 0.0, 0.0]
    dis_cnt = 0
    '''
    def train_dis(self, dis_loss, pos_loss, neg_loss, dis_cnt):
        np.random.shuffle(self.egs)

        info = ''

        for index in range(math.floor(len(self.egs) / dggan_config.d_batch_size)):

            pos_node_ids, pos_node_neighbor_ids, fake_node_embedding = self.prepare_data_for_d(index, self.egs)


            '''
            fake_node_embedding, len=2, [[emb1, emb2],[emb1, emb2]]  
            
            fake_node_embedding[0][0]  source node, direction 0 , shape= (16, 64)
            fake_node_embedding[0][1]  source node, direction 1, shape= (16, 64)
            fake_node_embedding[1][0]  target node, direction 0, shape= (16, 64)
            fake_node_embedding[1][1]  target node, direction 1, shape= (16, 64)
            '''
            _, _loss, _pos_loss, _neg_loss = self.sess.run([self.discriminator.d_updates, self.discriminator.loss, \
                                            self.discriminator.pos_loss, self.discriminator.neg_loss],
                                            feed_dict = {self.discriminator.pos_node_ids : np.array(pos_node_ids),
                                                         self.discriminator.pos_node_neighbor_ids : np.array(pos_node_neighbor_ids),
                                                         self.discriminator.fake_node_embedding : np.array(fake_node_embedding)})

            dis_loss += _loss
            pos_loss += _pos_loss
            for i in range(4):
                neg_loss[i] += _neg_loss[i]
            dis_cnt += 1
            info = 'dis_loss=%.4f pos_loss=%.4f neg_loss_0=%.4f neg_loss_1=%.4f neg_loss_2=%.4f neg_loss_3=%.4f' % \
                (dis_loss / dis_cnt, pos_loss / dis_cnt, neg_loss[0] / dis_cnt, neg_loss[1] / dis_cnt, \
                 neg_loss[2] / dis_cnt, neg_loss[3] / dis_cnt)
            avg_dis_loss = dis_loss / dis_cnt
            self.my_print(info, True, 1)

        return (dis_loss, pos_loss, neg_loss, dis_cnt, avg_dis_loss)

    def train_gen(self, gen_loss, neg_loss, gen_cnt):

        np.random.shuffle(self.node_list)

        info = ''

        for index in range(math.floor(len(self.node_list) / dggan_config.g_batch_size)):

            node_ids, noise_embedding, dis_node_embedding = self.prepare_data_for_g(index, self.node_list)
            _, _loss, _neg_loss = self.sess.run([self.generator.g_updates, self.generator.loss, self.generator.neg_loss],
                                                 feed_dict = {self.generator.node_ids : np.array(node_ids),
                                                              self.generator.noise_embedding : np.array(noise_embedding),
                                                              self.generator.dis_node_embedding : np.array(dis_node_embedding)})

            gen_loss += _loss
            for i in range(2):
                neg_loss[i] += _neg_loss[i]
            gen_cnt += 1

            avg_gen_loss = gen_loss / gen_cnt
            self.my_print(info, True, 1)

        return (gen_loss, neg_loss, gen_cnt,avg_gen_loss)

    def train(self):
        for epoch in range(dggan_config.n_epoch):
            info = 'epoch %d' % epoch
            self.my_print(info, False, 1)

            dis_loss = 0.0
            dis_pos_loss = 0.0
            dis_neg_loss = [0.0, 0.0, 0.0, 0.0]
            dis_cnt = 0

            gen_loss = 0.0
            gen_neg_loss = [0.0, 0.0]
            gen_cnt = 0

            epochs_d = []
            dis_loss_c = []
            epochs_g = []
            gen_loss_c = []

            # D-step
            for d_epoch in range(dggan_config.d_epoch):
                dis_loss, dis_pos_loss, dis_neg_loss, dis_cnt,avg_dis_loss = self.train_dis(dis_loss, dis_pos_loss, dis_neg_loss, dis_cnt)

                self.my_print('', False, 1)
                print("dis_loss=%.4f " % avg_dis_loss)
                dis_loss_c.append(avg_dis_loss)
                epochs_d.append(d_epoch)

            # G-step
            for g_epoch in range(dggan_config.g_epoch):
                gen_loss, gen_neg_loss, gen_cnt,avg_gen_loss = self.train_gen(gen_loss, gen_neg_loss, gen_cnt)
                print("gen_loss=%.4f " % avg_gen_loss)
                gen_loss_c.append(avg_gen_loss)
                epochs_g.append(g_epoch)
                self.my_print('', False, 1)


        embedding_matrix = self.sess.run(self.discriminator.node_embedding_matrix)
        self.sess.close()
        return embedding_matrix


    def prepare_data_for_d(self, index, egs):

        pos_node_ids = []  # true source node
        pos_node_neighbor_ids = []  # true target node

        for eg in egs[index * dggan_config.d_batch_size : (index + 1) * dggan_config.d_batch_size]:
            node_id, node_neighbor_id = eg
            pos_node_ids.append(node_id)
            pos_node_neighbor_ids.append(node_neighbor_id)

        fake_node_embedding = []

        noise_embedding = np.random.normal(0.0, dggan_config.sig, (2, len(pos_node_ids), dggan_config.n_emb))

        fake_node_embedding.append(self.sess.run(self.generator.fake_node_embedding,
                                            feed_dict = {self.generator.node_ids : np.array(pos_node_ids),
                                                         self.generator.noise_embedding : np.array(noise_embedding)}))


        noise_embedding = np.random.normal(0.0, dggan_config.sig, (2, len(pos_node_ids), dggan_config.n_emb))

        fake_node_embedding.append(self.sess.run(self.generator.fake_node_embedding,
                                            feed_dict = {self.generator.node_ids : np.array(pos_node_neighbor_ids),
                                                         self.generator.noise_embedding : np.array(noise_embedding)}))


        return pos_node_ids, pos_node_neighbor_ids, fake_node_embedding


    def prepare_data_for_g(self, index, node_list):

        node_ids = []

        for node_id in node_list[index * dggan_config.g_batch_size : (index + 1) * dggan_config.g_batch_size]:
            node_ids.append(node_id)

        noise_embedding = np.random.normal(0.0, dggan_config.sig, (2, len(node_ids), dggan_config.n_emb))


        dis_node_embedding = []
        dis_node_embedding1 = self.sess.run([self.discriminator.pos_node_embedding_1],
                                             feed_dict = {self.discriminator.pos_node_ids : np.array(node_ids)})
        dis_node_embedding2 = self.sess.run([self.discriminator.pos_node_neighbor_embedding_1],
                                             feed_dict = {self.discriminator.pos_node_neighbor_ids : np.array(node_ids)})

        dis_node_embedding = np.vstack([dis_node_embedding1, dis_node_embedding2])

        return node_ids, noise_embedding, dis_node_embedding

    def my_print(self, info, r_flag, verbose):
        if verbose == 1 and dggan_config.verbose == 0:
            return
        if r_flag:
            print('\r%s' % info, end='')
        else:
            print('%s' % info)


def dggan(pathnetwork, pathnode):

    model = Model(pathnetwork, pathnode)

    node_embedding_matrix = model.train()
    node_embedding_matrix_0 = dggan_utils.minmaxstandard(node_embedding_matrix[0])
    node_embedding_matrix_1 = dggan_utils.minmaxstandard(node_embedding_matrix[1])
    print('successfully saved.')
    return node_embedding_matrix_0, node_embedding_matrix_1



