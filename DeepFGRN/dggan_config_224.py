g_batch_size = 4
d_batch_size = 4
n_emb = 224

lambda_gen = 1e-5
lambda_dis = 1e-5
lr_gen = 1e-4
lr_dis = 1e-4
sig = 1.0
label_smooth = 0.0

n_epoch = 4
d_epoch = 6
g_epoch = 3
# n_epoch = 1
# d_epoch = 1
# g_epoch = 1

pre_d_epoch = 0
pre_g_epoch = 0
neg_weight = [1, 1, 1, 1]

pretrain_ckpt = ''
pretrain_dis_node_emb = []
pretrain_gen_node_emb = []
save = True
save_last = True
verbose = 1
log = True

