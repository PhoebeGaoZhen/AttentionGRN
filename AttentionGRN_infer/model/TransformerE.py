import torch
import torch.nn.functional as F
import dgl.function as fn
import torch.nn as nn
import dgl
import numpy as np
import math


"""
	Util functions
"""

def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-10, 10))}

    return func


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}

    return func

def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-10, 10))}

    return func

# Improving implicit attention scores with explicit edge features, if available
def scaling(field, scale_constant):
    def func(edges):
        return {field: (((edges.data[field])) / scale_constant)}

    return func

class GT(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim, num_heads, I_dim, num_gtlayers):
        super().__init__()
        self.h = None
        self.embedding_h = nn.Linear(in_dim, hidden_dim, bias=False)  # node feat is an integer
        self.D_dim = I_dim
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.layergt = GraphTransformerLayer(hidden_dim, out_dim, num_heads, I_dim)

        # self.layers = nn.ModuleList(
        #     [GraphTransformerLayer(hidden_dim, out_dim, num_heads, I_dim) for _ in range(num_gtlayers)])
        #
        # self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, I_dim))

        self.MLP_layer_x = Reconstruct_X(out_dim, in_dim)

    def extract_features(self, g):

        X = g.ndata['x'].to(torch.float64) # (779,758)
        I = g.ndata['I'].to(torch.float64)  # (779,5)


        h = self.embedding_h(X)

        # for layer in self.layers:
        #     h = layer(h, g, I)
        #     h = F.dropout(h, 0.3, training=self.training)

        h1 = self.layergt(h, g, I)
        h1 = F.dropout(h1, 0.5, training=self.training)
        h1 = self.layergt(h1, g, I)
        h1 = F.dropout(h1, 0.5, training=self.training)

        h2 = self.layergt(h1, g, I)
        h2 = F.dropout(h2, 0.3, training=self.training)
        h2 = self.layergt(h2, g, I)
        h2 = F.dropout(h2, 0.3, training=self.training)

        h2 = self.layergt(h2, g, I)
        h2 = F.dropout(h2, 0.3, training=self.training)
        h2 = self.layergt(h2, g, I)
        h2 = F.dropout(h2, 0.3, training=self.training)

        h2 = self.layergt(h2, g, I)
        h2 = F.dropout(h2, 0.3, training=self.training)
        h2 = self.layergt(h2, g, I)
        h2 = F.dropout(h2, 0.3, training=self.training)

        h2 = self.layergt(h2, g, I)
        h2 = F.dropout(h2, 0.3, training=self.training)
        h2 = self.layergt(h2, g, I)
        h2 = F.dropout(h2, 0.3, training=self.training)


        # h2 = self.layergt(h2, g, I)
        # h2 = F.dropout(h2, 0.3, training=self.training)
        # h2 = self.layergt(h2, g, I)
        # h2 = F.dropout(h2, 0.3, training=self.training)


        return h2


    def forward(self, g):

        h = self.extract_features(g)

        # compute X
        x_hat = self.MLP_layer_x(h)
        self.h = h

        return h, x_hat


class GraphTransformerLayer(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, I_dim):
        super().__init__()

        self.in_channels = in_dim # 16
        self.out_channels = out_dim # 16
        self.num_heads = num_heads # 4

        self.attention = MultiHeadAttentionLayer(in_dim, out_dim // num_heads, num_heads, I_dim)

        self.O = nn.Linear(out_dim, out_dim)

        self.batchnorm1 = nn.BatchNorm1d(out_dim)
        self.batchnorm2 = nn.BatchNorm1d(out_dim)
        self.layer_norm1 = nn.LayerNorm(out_dim)
        self.layer_norm2 = nn.LayerNorm(out_dim)

        self.FFN_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_layer2 = nn.Linear(out_dim * 2, out_dim)

        self.proj_i = nn.Linear(I_dim, out_dim) # (33/59, 128)

    def forward(self, h, g, I):
        h_in1 = h  # for first residual connection

        attn_out = self.attention(h, g) # (779,4,32)


        h = attn_out.view(-1, self.out_channels)

        h = F.dropout(h, 0.5, training=self.training) # 779*128


        h = h + self.proj_i(I)

        h = self.O(h)

        h = h_in1 + h  # residual connection

        h = self.layer_norm1(h)

        h_in2 = h  # for second residual connection

        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)

        h = F.dropout(h, 0.5, training=self.training)
        h = self.FFN_layer2(h)
        h = h_in2 + h  # residual connection
        # h = F.dropout(h, 0.3, training=self.training)
        h = self.layer_norm2(h)

        return h


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, D_dim):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)

        self.hidden_size = in_dim  # 80 -- 16
        self.num_heads = num_heads  # 8  -- 4
        self.head_dim = out_dim // num_heads  # 10 --- 16/4
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(in_dim, in_dim)
        self.k_proj = nn.Linear(in_dim, in_dim)
        self.v_proj = nn.Linear(in_dim, in_dim)


    def propagate_attention(self, g):
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))

        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))

        # softmax
        g.apply_edges(exp('score'))

        g.apply_edges(scaling('score', 2))

        # Send weighted values to target nodes
        eids = g.edges()

        g.send_and_recv(eids, dgl.function.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))

        g.send_and_recv(eids, dgl.function.copy_e('score', 'score'), fn.sum('score', 'z'))


    def forward(self, h, g):
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)

        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(g)


        h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))


        return h_out


class Reconstruct_X(torch.nn.Module):
    def __init__(self, inp, outp, dims=128):
        super().__init__()

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(inp, dims),
            torch.nn.SELU(),
            torch.nn.Linear(dims, outp))

    def forward(self, x):
        x = self.mlp(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        a = self.pe[:x.size(0), :]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class EXP_Transformer(nn.Module):
    def __init__(self, input_dim, d_model=200, num_classes=2, dropout=0.1):
        super().__init__()
        self.prenet = nn.Linear(input_dim, d_model)
        self.positionalEncoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=256, nhead=2, dropout=dropout
        )

        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, exp_data):
        out = exp_data.permute(1, 0, 2)
        out = self.positionalEncoding(out)
        out = self.encoder_layer(out)
        out = out.transpose(0, 1)
        stats = out.mean(dim=1)
        # out = self.pred_layer(stats)
        return stats


class AttentionGRN(nn.Module):
    def __init__(self, in_dim_GT, out_dim_GT, hidden_dim, num_heads, I_dim, in_dim_exp, num_gtlayers, d_model=200, num_classes=2):
        super().__init__()
        self.topo_net = GT(in_dim_GT, out_dim_GT, hidden_dim, num_heads, I_dim, num_gtlayers)
        self.exp_net = EXP_Transformer(in_dim_exp, d_model=d_model, num_classes=2, dropout=0.2)

        self.pred_layer = nn.Sequential(
            nn.Linear(d_model + out_dim_GT + out_dim_GT, d_model),
            # nn.Linear(out_dim_GT+out_dim_GT, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes),
        )


    def forward(self, g, exp_data, all_tf, all_target,device, flag_DM):
        if flag_DM:
            A, A_hat = self.topo_net(g) # A 779*64
            B = self.exp_net(exp_data)  # B 32*200


            AA = []
            for i in range(len(all_tf)):
                tf = all_tf[i]
                target = all_target[i]
                feature = torch.cat((A[tf], A[target]), dim = 0, out = None).to('cpu')
                AA.append(feature.detach().numpy())
            AA = np.array(AA)
            AA = torch.from_numpy(AA).to(device)

            C = torch.cat((AA, B), dim = 1, out = None)
            out = self.pred_layer(C)
            # out = self.pred_layer(AA)

        else:
            B = self.exp_net(exp_data)  # B 32*200


            AA = np.zeros((B.shape[0], 128))
            AA = torch.from_numpy(AA).to(device)

            C = torch.cat((AA, B), dim=1, out=None)
            out = self.pred_layer(C)
            # out = self.pred_layer(AA)


        return out


















