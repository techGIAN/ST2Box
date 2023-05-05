import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_geometric.nn import GCNConv
from DModel import Date2VecConvert
import time
import datetime
import numpy as np

class GCN(nn.Module):
    def __init__(self, feature_size, embedding_size):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(feature_size, embedding_size, cached=True)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x)
        return x

class TrajEmbedding(nn.Module):
    def __init__(self, feature_size, embedding_size, device):
        super(TrajEmbedding, self).__init__()
        self.feature_size = feature_size
        self.embedding_size = embedding_size
        self.device = device
        self.gcn = GCN(feature_size, embedding_size).to(self.device)

    def forward(self, network, traj_seqs):

        batch_size = len(traj_seqs)
        seq_lengths = list(map(len, traj_seqs))

        for traj_one in traj_seqs:
            traj_one += [0]*(max(seq_lengths)-len(traj_one))

        embedded_seq_tensor = torch.zeros((batch_size, max(seq_lengths), self.embedding_size), dtype=torch.float32)

        seq_lengths = torch.LongTensor(seq_lengths).to(self.device)
        traj_seqs = torch.tensor(traj_seqs).to(self.device)

        node_embeddings = self.gcn(network)

        for idx, (seq, seqlen) in enumerate(zip(traj_seqs, seq_lengths)):
            embedded_seq_tensor[idx, :seqlen] = node_embeddings.index_select(0, seq[:seqlen])

        seq_lengths = seq_lengths.cpu()
        embedded_seq_tensor = embedded_seq_tensor.to(self.device)

        return embedded_seq_tensor, seq_lengths

class TimeEmbedding(nn.Module):
    def __init__(self, date2vec_size, device):
        super(TimeEmbedding, self).__init__()
        self.device = device
        self.date2vec_size = date2vec_size

    def forward(self, time_seqs):

        batch_size = len(time_seqs)
        seq_lengths = list(map(len, time_seqs))

        for time_one in time_seqs:
            time_one += [[0 for i in range(self.date2vec_size)]]*(max(seq_lengths)-len(time_one))

        embedded_seq_tensor = torch.zeros((batch_size, max(seq_lengths), self.date2vec_size), dtype=torch.float32)

        seq_lengths = torch.LongTensor(seq_lengths).to(self.device)
        vec_time_seqs = torch.tensor(time_seqs).to(self.device)

        for idx, (seq, seqlen) in enumerate(zip(vec_time_seqs, seq_lengths)):
            embedded_seq_tensor[idx, :seqlen] = seq[:seqlen]

        seq_lengths = seq_lengths.cpu()
        embedded_seq_tensor = embedded_seq_tensor.to(self.device)

        return embedded_seq_tensor

class Co_Att(nn.Module):
    def __init__(self, dim):
        super(Co_Att, self).__init__()
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.temperature = dim ** 0.5
        self.FFN = nn.Sequential(
            nn.Linear(dim, int(dim*0.5)),
            nn.ReLU(),
            nn.Linear(int(dim*0.5), dim),
            nn.Dropout(0.1)
        )
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, seq_s, seq_t):
        h = torch.stack([seq_s, seq_t], 2)  
        q = self.Wq(h)
        k = self.Wk(h)
        v = self.Wv(h)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = F.softmax(attn, dim=-1)
        attn_h = torch.matmul(attn, v)

        attn_o = self.FFN(attn_h) + attn_h
        attn_o = self.layer_norm(attn_o)

        att_s = attn_o[:, :, 0, :]
        att_t = attn_o[:, :, 1, :]

        return att_s, att_t

class ST_LSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, dropout_rate, device):
        super(ST_LSTM, self).__init__()
        self.device = device
        self.bi_lstm = nn.LSTM(input_size=embedding_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout_rate,
                               bidirectional=True)
        self.w_omega = nn.Parameter(torch.Tensor(hidden_size * 2, hidden_size * 2))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_size * 2, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def getMask(self, seq_lengths):

        max_len = int(seq_lengths.max())
        mask = torch.ones((seq_lengths.size()[0], max_len)).to(self.device)

        for i, l in enumerate(seq_lengths):
            if l < max_len:
                mask[i, l:] = 0

        return mask

    def forward(self, packed_input):

        packed_output, _ = self.bi_lstm(packed_input)
        outputs, seq_lengths = pad_packed_sequence(packed_output, batch_first=True)

        mask = self.getMask(seq_lengths)
        u = torch.tanh(torch.matmul(outputs, self.w_omega))
        att = torch.matmul(u, self.u_omega).squeeze()
        att = att.masked_fill(mask == 0, -1e10)

        att_score = F.softmax(att, dim=1).unsqueeze(2)
        scored_outputs = outputs * att_score

        out = torch.sum(scored_outputs, dim=1)
        return out


class ST_Encoder(nn.Module):
    def __init__(self, feature_size, date2vec_size, embedding_size, hidden_size,
                                    num_layers, dropout_rate, device):
        super(ST_Encoder, self).__init__()
        self.embedding_S = TrajEmbedding(feature_size, embedding_size, device)
        self.embedding_T = TimeEmbedding(date2vec_size, device)
        self.co_attention = Co_Att(date2vec_size).to(device)
        self.encoder_ST = ST_LSTM(embedding_size+date2vec_size, hidden_size, num_layers, dropout_rate, device)

    def forward(self, network, traj_seqs, time_seqs):
        s_input, seq_lengths = self.embedding_S(network, traj_seqs)
        t_input = self.embedding_T(time_seqs)
        att_s, att_t = self.co_attention(s_input, t_input)

        st_input = torch.cat((att_s, att_t), dim=2)

        packed_input = pack_padded_sequence(st_input, seq_lengths, batch_first=True, enforce_sorted=False)  

        att_output = self.encoder_ST(packed_input)

        return att_output

class STTrajSimEncoder(nn.Module):
    def __init__(self, feature_size, embedding_size, date2vec_size, hidden_size, num_layers, dropout_rate, concat, device):
        super(STTrajSimEncoder, self).__init__()
        self.stEncoder = ST_Encoder(feature_size, date2vec_size, embedding_size, hidden_size,
                                    num_layers, dropout_rate, device)
        self.concat = concat

    def forward(self, network, traj_seqs, time_seqs):

        st_emb = self.stEncoder(network, traj_seqs, time_seqs)
        return st_emb

