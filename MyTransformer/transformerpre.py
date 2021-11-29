import math
import torch
import numpy as np
import torch.nn as nn

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

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
        x = x + self.pe[:x.size(0), :]

        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequence_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        self.d_k = d_k
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_k
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, self.d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, self.d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, self.d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * self.d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)
        input_Q = nn.LayerNorm(self.d_model).cuda()(input_Q)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        context, attn = ScaledDotProductAttention(self.d_k)(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        return output + residual, attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_model, bias=False)
        )

    def forward(self, inputs):
        residual = inputs
        inputs = nn.LayerNorm(self.d_model).cuda()(inputs)
        output = self.fc(inputs)
        return output + residual


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, n_heads):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_k = d_k
        self.n_heads = n_heads
        self.enc_self_attn = MultiHeadAttention(self.d_model, self.d_k, self.n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(self.d_model, self.d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, n_heads):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_k = d_k
        self.n_heads = n_heads
        self.dec_self_attn = MultiHeadAttention(self.d_model, self.d_k, self.n_heads)
        self.dec_enc_attn = MultiHeadAttention(self.d_model, self.d_k, self.n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(self.d_model, self.d_ff)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, d_k, n_layers, n_heads, src_vocab_size):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_k = d_k
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.src_vocab_size = src_vocab_size
        self.src_emb = Embedding(self.src_vocab_size, self.d_model, 0)
        self.pos_emb = PositionalEncoding(self.d_model)
        self.layers = nn.ModuleList([EncoderLayer(self.d_model, self.d_ff, self.d_k, self.n_heads) for _ in range(self.n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        enc_outputs = nn.LayerNorm(self.d_model).cuda()(enc_outputs)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        enc_outputs = nn.LayerNorm(self.d_model).cuda()(enc_outputs)
        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, d_k, n_layers, n_heads, tgt_vocab_size):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_k = d_k
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.tgt_vocab_size = tgt_vocab_size
        self.tgt_emb = nn.Embedding(self.tgt_vocab_size, self.d_model)
        self.pos_emb = PositionalEncoding(self.d_model)
        self.layers = nn.ModuleList([DecoderLayer(self.d_model, self.d_ff, self.d_k, self.n_heads) for _ in range(self.n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        dec_outputs = self.tgt_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).cuda()
        dec_outputs = nn.LayerNorm(self.d_model).cuda()(dec_outputs)
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).cuda()
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).cuda()
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0).cuda()

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        dec_outputs = nn.LayerNorm(self.d_model).cuda()(dec_outputs)
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self, d_model = 512, d_ff = 2048 , d_k = 64, n_layers = 6, n_heads = 8, src_vocab_size = 10000, tgt_vocab_size = 10000):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_k = d_k
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.encoder = Encoder(self.d_model, self.d_ff, self.d_k, self.n_layers, self.n_heads, self.src_vocab_size).cuda()
        self.decoder = Decoder(self.d_model, self.d_ff, self.d_k, self.n_layers, self.n_heads, self.tgt_vocab_size).cuda()
        self.projection = nn.Linear(self.d_model, self.tgt_vocab_size, bias=False).cuda()

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns