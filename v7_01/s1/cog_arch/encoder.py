# from Representation Learning for Conversational Data using Discourse Mutual Information Maximization

import math
import numpy as np
import torch
from torch import nn as nn
from transformers import RobertaModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        if d_model % 2 == 0:
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            div_term = torch.exp(torch.arange(0, d_model + 1, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :, :].repeat(1, x.size(1), 1)
        return self.dropout(x)


class Embedding(nn.Module):
    def __init__(self, vocab_size=9000, d_model=512):
        super(Embedding, self).__init__()
        self.emb = nn.Embedding(vocab_size, d_model)

    def forward(self, ids):
        return self.emb(ids)


class Projection(nn.Module):
    def __init__(self, input_size, proj_size, dropout=0.1):
        super(Projection, self).__init__()
        self.W = nn.Linear(input_size, input_size)

    def forward(self, x):
        return self.W(x)


class Transformer(nn.Module):
    def __init__(self, d_model=512, vocab_size=9000, num_layers=2, heads=4, dim_feedforward=2048):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=dim_feedforward
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

    def forward(self, x, mask):
        x = x.permute(1, 0, 2)
        pos_x = self.pos_encoder(x)
        encoder_output = self.encoder(pos_x, src_key_padding_mask=mask)  # (T, B, d_model)
        encoder_output = encoder_output.permute(1, 0, 2)                 # (B, T, d_model)
        return encoder_output


class WrappedSMI(nn.Module):
    def __init__(self, model):
        super(WrappedSMI, self).__init__()
        self.module = model

    def forward(self, x):
        return self.module(x)


def identity(x):
    return x


def recursive_init(module):
    try:
        for c in module.children():
            if hasattr(c, "reset_parameters"):
                print("Reset:", c)
                c.reset_parameters()
            else:
                recursive_init(c)
    except Exception as e:
        print(module)
        print(e)


class SMI(nn.Module):
    def __init__(self, vocab_size=9000, d_model=512, projection_size=512,
                 encoder_layers=4, encoder_heads=4, dim_feedforward=2048,
                 symmetric_loss=False, roberta_init=False, roberta_name="roberta-base"):
        super(SMI, self).__init__()
        self.d_model = d_model
        self.invert_mask = False
        self.roberta_init = roberta_init

        if not roberta_init:
            self.encoder  = Transformer(d_model, vocab_size, encoder_layers,
                                        encoder_heads, dim_feedforward=dim_feedforward)
            self.embedding = Embedding(vocab_size, d_model)
        else:
            self.invert_mask = True
            self.embedding   = identity
            self.encoder     = RobertaModel.from_pretrained(roberta_name, add_pooling_layer=False)
            # NOTE: recursive_init removed — it destroys pretrained weights

        self.proj = Projection(d_model, projection_size)

        if symmetric_loss:
            self.lsoftmax0 = nn.LogSoftmax(dim=0)
            self.lsoftmax1 = nn.LogSoftmax(dim=1)
        else:
            self.lsoftmax1 = nn.LogSoftmax(dim=1)
        self.symmetric_loss = symmetric_loss

    def _reset_parameters(self):
        for n, p in self.named_parameters():
            if p.dim() > 1:
                if 'embedding' in n:
                    nn.init.normal_(p, mean=0.0, std=0.02)  # init estándar para embeddings
                else:
                    nn.init.xavier_normal_(p)

    def _mean_pool(self, encoder_output, mask):
        """
        encoder_output : (B, T, d_model)
        mask           : (B, T) bool — True at padding positions
        returns        : (B, d_model)
        """
        token_mask = (~mask).float().unsqueeze(-1)      # (B, T, 1)
        summed     = (encoder_output * token_mask).sum(1)  # (B, d_model)
        counts     = token_mask.sum(1).clamp(min=1e-9)     # (B, 1)
        return summed / counts

    def forward_context_only(self, context, mask_ctx):
        if self.invert_mask:
            mask_ctx = (mask_ctx == 0) * 1
        context_enc = self.embedding(context)
        c_t = self.encoder(context_enc, mask_ctx)
        if self.roberta_init:
            c_t = c_t.last_hidden_state
        return self._mean_pool(c_t, mask_ctx.bool())    # (B, d_model)

    def forward(self, context, response, mask_ctx, mask_rsp):
        if self.invert_mask:
            mask_ctx = (mask_ctx == 0) * 1
            mask_rsp = (mask_rsp == 0) * 1

        context_enc  = self.embedding(context)
        response_enc = self.embedding(response)

        c_t = self.encoder(context_enc, mask_ctx)
        r_t = self.encoder(response_enc, mask_rsp)

        if self.roberta_init:
            c_t = c_t.last_hidden_state
            r_t = r_t.last_hidden_state

        c_t = self._mean_pool(c_t, mask_ctx.bool())     # (B, d_model)
        z_t = self._mean_pool(r_t, mask_rsp.bool())     # (B, d_model)
        z_t = self.proj(z_t)
        return c_t, z_t

    def _compute_loss(self, c_t, z_t):
        c_t = torch.nn.functional.normalize(c_t, dim=1)
        z_t = torch.nn.functional.normalize(z_t, dim=1)
        score = torch.mm(c_t, z_t.T)
        if self.symmetric_loss:
            loss = - 0.5 * torch.mean(torch.diag(self.lsoftmax1(score))) \
                   - 0.5 * torch.mean(torch.diag(self.lsoftmax0(score)))
        else:
            loss = -torch.mean(torch.diag(self.lsoftmax1(score)))
        mi = torch.log(torch.tensor(float(c_t.shape[0]))) - loss.item()
        return score, loss, mi

    def compute_loss_custom(self, c_t, z_t, estimator_name=None):
        if estimator_name == "infonce":
            return self._compute_loss(c_t, z_t)
        elif estimator_name == "jsd":
            estimator = JSD
        elif estimator_name == "nwj":
            estimator = NWJ
        elif estimator_name == "tuba":
            estimator = TUBA
        elif estimator_name == "dv":
            estimator = DV
        elif estimator_name == "smile":
            estimator = SMILE

        score = torch.mm(c_t, torch.transpose(z_t, 0, 1))
        mi    = estimator(score)
        loss  = -mi
        return score, loss, mi


class SMIClassifier(nn.Module):
    """
    Thin classification wrapper around SMI.
    Routes forward() through forward_context_only() and adds a linear head.
    """
    def __init__(self, vocab_size=9000, d_model=512, encoder_layers=4,
                 encoder_heads=4, dim_feedforward=2048,
                 num_classes=4, pad_token_id=1):
        super().__init__()
        self.smi = SMI(
            vocab_size=vocab_size,
            d_model=d_model,
            encoder_layers=encoder_layers,
            encoder_heads=encoder_heads,
            dim_feedforward=dim_feedforward,
        )
        self.classifier   = nn.Linear(d_model, num_classes)
        self.pad_token_id = pad_token_id

    def forward(self, input_ids):
        mask = (input_ids == self.pad_token_id).bool()
        c_t  = self.smi.forward_context_only(input_ids, mask)
        return self.classifier(c_t)