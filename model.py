# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from kan_layer import KANLinear
from config import KAN_GRID_SIZE, KAN_SPLINE_ORDER, TRANS_HEADS, TRANS_D_K, TRANS_D_V, TRANS_D_FF, TRANS_DROPOUT, DEVICE

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return ctx.lambd * grad_output.neg(), None

def grad_reverse(x, lambd):
    return GradReverse.apply(x, lambd)

class SimpleGCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, adj):
        h = self.proj(x)            
        out = torch.matmul(adj, h)
        return out

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V) 
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.ScaledDotProductAttention = ScaledDotProductAttention(d_k, dropout)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.xavier_normal_(self.W_Q.weight)
        nn.init.xavier_normal_(self.W_K.weight)
        nn.init.xavier_normal_(self.W_V.weight)
        nn.init.xavier_normal_(self.fc.weight)
     
    def forward(self, input_Q, input_K, input_V):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        context = self.ScaledDotProductAttention(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        output = self.dropout(output)
        return self.norm(output + residual)

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ELU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_normal_(self.fc[0].weight)
        nn.init.xavier_normal_(self.fc[2].weight)
    
    def forward(self, inputs):
        residual = inputs
        output = self.dropout(self.fc(inputs))
        return self.norm(output + residual)
    
class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)

    def forward(self, enc_inputs):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs

class EEG_Transformer_KAN(nn.Module):
    """
    EEG model using GCN for spatial, Transformer for temporal, and KAN for classification.
    """
    def __init__(self,
                 num_nodes=62,
                 in_channels=5,
                 gcn_out=16,
                 num_classes=4,
                 adj=None):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.num_classes = num_classes

        # GCN layers (3-layer)
        self.use_gcn = adj is not None
        if self.use_gcn:
            self.register_buffer('adj', torch.from_numpy(adj).float())
            self.gcn1 = SimpleGCN(in_channels, gcn_out)
            self.gcn2 = SimpleGCN(gcn_out, gcn_out)
            self.gcn3 = SimpleGCN(gcn_out, gcn_out)
            self.gcn_act = nn.ELU()
            model_dim = num_nodes * gcn_out
        else:
            model_dim = num_nodes * in_channels

        # Transformer layer
        self.transformer = EncoderLayer(
            n_heads=TRANS_HEADS,
            d_model=model_dim,
            d_k=TRANS_D_K,
            d_v=TRANS_D_V,
            d_ff=TRANS_D_FF,
            dropout=TRANS_DROPOUT
        )

        self.fc_emotion = KANLinear(
            in_features=model_dim,
            out_features=num_classes,
            grid_size=KAN_GRID_SIZE,
            spline_order=KAN_SPLINE_ORDER
        )
        
        self.fc_domain = nn.Linear(model_dim, 2)

    def kan_reg(self):
        reg = 0
        reg += self.fc_emotion.regularisation_loss()
        return reg

    def extract_features(self, X):
        B, T, N, F_in = X.shape
        
        if self.use_gcn:
            X_reshaped = X.reshape(B * T, N, F_in)
            H = self.gcn_act(self.gcn1(X_reshaped, self.adj))
            H = self.gcn_act(self.gcn2(H, self.adj))
            H = self.gcn3(H, self.adj)
            X_flat = H.reshape(B, T, N * H.shape[-1])
        else:
            X_flat = X.reshape(B, T, N * F_in)

        trans_out = self.transformer(X_flat) 
        feat_final = torch.mean(trans_out, dim=1)
        return feat_final

    def forward(self, Xs, Xt, lambda_grl=1.0):
        feat_s = self.extract_features(Xs)
        feat_t = self.extract_features(Xt)

        emo_logits = self.fc_emotion(feat_s)

        feat_all = torch.cat([feat_s, feat_t], dim=0)
        feat_rev = grad_reverse(feat_all, lambda_grl)
        domain_logits = self.fc_domain(feat_rev)

        return emo_logits, domain_logits

    def predict_emotion(self, X):
        feat = self.extract_features(X)
        return self.fc_emotion(feat)