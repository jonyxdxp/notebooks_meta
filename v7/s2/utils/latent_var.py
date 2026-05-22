

# from https://github.com/d1024choi/HLSTrajForecast/blob/main/models/HLS.py







class ModeSelectionNetwork(nn.Module):

    def __init__(self, args):
        super(ModeSelectionNetwork, self).__init__()

        self.batch = args.batch_size
        self.num_max_paths = args.num_max_paths
        self.lane_feat_dim = args.lane_feat_dim
        self.traj_enc_h_dim = args.traj_enc_h_dim

        self.att_op = AdditiveAttention(args.lane_feat_dim, args.traj_enc_h_dim)

        self.embedder = make_mlp([2*args.lane_feat_dim + 2*args.traj_enc_h_dim, args.lane_feat_dim], [True], ['relu'], [0])
        self.classifier = make_mlp([args.lane_feat_dim * args.num_max_paths, args.num_max_paths], [True], ['none'], [0])

    def forward(self, agent_context, lane_contexts, ngh_lane_context, lane_label, ngh_contexts, isTrain=True):

        '''
        agent_context : batch x dim
        lane_contexts : batch x num_max_paths x dim
        lane_label : batch x num_max_paths
        ngh_contexts : batch x num_max_paths x dim

        '''

        batch = agent_context.size(0)

        '''
        repeat_interleave : 0 0 0 1 1 1 2 2 2
        repeat            : 0 1 2 0 1 2 0 1 2
        
        tensor.repeat_interleave(num_vids_cur, dim=0)
        '''
        # agent context
        agent_context_repeat = agent_context.reshape(batch, 1, -1).repeat_interleave(self.num_max_paths, dim=1) #  batch x num_max_paths x dim
        context_cat = torch.cat((agent_context_repeat, lane_contexts, ngh_lane_context, ngh_contexts), dim=2) #  batch x num_max_paths x dim
        context_emb = self.embedder(context_cat.view(-1, 2*self.lane_feat_dim+2*self.traj_enc_h_dim)).view(batch, self.num_max_paths, self.lane_feat_dim)
        logits = self.classifier(context_emb.view(batch, -1))

        # the best-matched lane by gt-label
        best_lane_contexts = torch.zeros(size=(batch, self.lane_feat_dim)).to(lane_contexts)
        best_ngh_lane_contexts = torch.zeros(size=(batch, self.lane_feat_dim)).to(lane_contexts)
        best_ngh_contexts = torch.zeros(size=(batch, self.traj_enc_h_dim)).to(lane_contexts)
        if (isTrain):
            for b in range(batch):
                best_lane_idx = np.argwhere(toNP(lane_label[b, :]) == 1)[0][0]

                cur_lanes = lane_contexts[b] # num_max_paths x dim
                best_lane_contexts[b, :] += cur_lanes[best_lane_idx, :]

                cur_ngh_lanes = ngh_lane_context[b]
                best_ngh_lane_contexts[b, :] += cur_ngh_lanes[best_lane_idx, :]

                cur_nghs = ngh_contexts[b]
                best_ngh_contexts[b, :] += cur_nghs[best_lane_idx, :]

        return logits, best_lane_contexts, best_ngh_lane_contexts, best_ngh_contexts









# ----------------------





# from https://github.com/uber-research/PPLM/blob/master/run_pplm_discrim_train.py











#! /usr/bin/env python3
# coding=utf-8

# This code is licensed under a non-commercial license.

import argparse
import csv
import json
import math
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
import torch.optim
import torch.optim as optim
import torch.utils.data as data
from nltk.tokenize.treebank import TreebankWordDetokenizer
from torchtext import data as torchtext_data
from torchtext import datasets
from tqdm import tqdm, trange
from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from pplm_classification_head import ClassificationHead

torch.manual_seed(0)
np.random.seed(0)
EPSILON = 1e-10
example_sentence = "This is incredible! I love it, this is the best chicken I have ever had."
max_length_seq = 100









import torch

class ClassificationHead(torch.nn.Module):
    """Classification Head for  transformer encoders"""

    def __init__(self, class_size, embed_size):
        super(ClassificationHead, self).__init__()
        self.class_size = class_size
        self.embed_size = embed_size
        # self.mlp1 = torch.nn.Linear(embed_size, embed_size)
        # self.mlp2 = (torch.nn.Linear(embed_size, class_size))
        self.mlp = torch.nn.Linear(embed_size, class_size)

    def forward(self, hidden_state):
        # hidden_state = F.relu(self.mlp1(hidden_state))
        # hidden_state = self.mlp2(hidden_state)
        logits = self.mlp(hidden_state)
        return logits







class Discriminator(torch.nn.Module):
    """Transformer encoder followed by a Classification Head"""

    def __init__(
            self,
            class_size=None,
            pretrained_model="gpt2-medium",
            classifier_head=None,
            cached_mode=False,
            device='cpu'
    ):
        super(Discriminator, self).__init__()
        if pretrained_model.startswith("gpt2"):
            self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
            self.encoder = GPT2LMHeadModel.from_pretrained(pretrained_model)
            self.embed_size = self.encoder.transformer.config.hidden_size
        elif pretrained_model.startswith("bert"):
            self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
            self.encoder = BertModel.from_pretrained(pretrained_model)
            self.embed_size = self.encoder.config.hidden_size
        else:
            raise ValueError(
                "{} model not yet supported".format(pretrained_model)
            )
        if classifier_head:
            self.classifier_head = classifier_head
        else:
            if not class_size:
                raise ValueError("must specify class_size")
            self.classifier_head = ClassificationHead(
                class_size=class_size,
                embed_size=self.embed_size
            )
        self.cached_mode = cached_mode
        self.device = device

    def get_classifier(self):
        return self.classifier_head

    def train_custom(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier_head.train()

    def avg_representation(self, x):
        mask = x.ne(0).unsqueeze(2).repeat(
            1, 1, self.embed_size
        ).float().to(self.device).detach()
        if hasattr(self.encoder, 'transformer'):
            # for gpt2
            hidden, _ = self.encoder.transformer(x)
        else:
            # for bert
            hidden, _ = self.encoder(x)
        masked_hidden = hidden * mask
        avg_hidden = torch.sum(masked_hidden, dim=1) / (
                torch.sum(mask, dim=1).detach() + EPSILON
        )
        return avg_hidden

    def forward(self, x):
        if self.cached_mode:
            avg_hidden = x.to(self.device)
        else:
            avg_hidden = self.avg_representation(x.to(self.device))

        logits = self.classifier_head(avg_hidden)
        probs = F.log_softmax(logits, dim=-1)

        return probs

    def predict(self, input_sentence):
        input_t = self.tokenizer.encode(input_sentence)
        input_t = torch.tensor([input_t], dtype=torch.long, device=self.device)
        if self.cached_mode:
            input_t = self.avg_representation(input_t)

        log_probs = self(input_t).data.cpu().numpy().flatten().tolist()
        prob = [math.exp(log_prob) for log_prob in log_probs]
        return prob












# -------------------------------------------------------------------





# from https://github.com/edenton/svg/blob/master/models/lstm.py










import torch
import torch.nn as nn
from torch.autograd import Variable

class lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super(lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.output = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                #nn.BatchNorm1d(output_size),
                nn.Tanh())
        self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()),
                           Variable(torch.zeros(self.batch_size, self.hidden_size).cuda())))
        return hidden

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]

        return self.output(h_in)

class gaussian_lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super(gaussian_lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(self.batch_size, self.hidden_size).cuda()),
                           Variable(torch.zeros(self.batch_size, self.hidden_size).cuda())))
        return hidden

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, input):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar













# ------------------------------------------------------




# from https://github.com/YongchaoHuang/VJEPA/blob/main/vjepa_Yongchao.py






import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from copy import deepcopy
import time

# Set global seeds for reproducibility
seed = 111
torch.manual_seed(seed)
np.random.seed(seed)




class LinearBJEPA(nn.Module):
    """
    Bayesian JEPA with Product of Experts.

    Training: Soft fusion via KL regularization (dynamics learns independently)
    Inference: Hard fusion via PoE (combines dynamics + prior)

    From the paper (Section 6.3):
    - Training phase uses soft fusion where prior acts as regularizer
    - Inference phase uses hard fusion (PoE) to intersect dynamics and prior manifolds
    """
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)
        self.target_enc_mu = deepcopy(self.encoder)
        self.target_enc_logvar = nn.Linear(input_dim, latent_dim, bias=False)

        for p in self.target_enc_mu.parameters():
            p.requires_grad = False
        for p in self.target_enc_logvar.parameters():
            p.requires_grad = False

        # Dynamics predictor (likelihood expert)
        self.pred_mu = nn.Linear(latent_dim, latent_dim, bias=False)
        self.pred_logvar = nn.Linear(latent_dim, latent_dim, bias=False)

        # Learnable static prior (constraint expert)
        self.prior_mu = nn.Parameter(torch.zeros(latent_dim))
        self.prior_logvar = nn.Parameter(torch.zeros(latent_dim))

    def update_target(self, tau=0.99):
        for p, tp in zip(self.encoder.parameters(), self.target_enc_mu.parameters()):
            tp.data = tau * tp.data + (1 - tau) * p.data

    def product_of_experts(self, mu1, logvar1, mu2, logvar2):
        """
        Combine two Gaussian experts via Product of Experts.
        Returns the posterior mean and logvar.

        posterior_precision = precision_1 + precision_2
        posterior_mean = (precision_1 * mu1 + precision_2 * mu2) / posterior_precision
        """
        prec1 = torch.exp(-logvar1)
        prec2 = torch.exp(-logvar2)
        prec_post = prec1 + prec2
        var_post = 1.0 / prec_post
        mu_post = (mu1 * prec1 + mu2 * prec2) * var_post
        logvar_post = torch.log(var_post)
        return mu_post, logvar_post

    def forward(self, x_t, x_next):
        """
        Training forward pass.
        Returns dynamics prediction (for soft fusion training) and target.
        """
        z_t = self.encoder(x_t)

        # Dynamics expert prediction
        dyn_mu = self.pred_mu(z_t)
        dyn_logvar = self.pred_logvar(z_t)

        # Expand prior for batch
        batch_size = x_t.size(0)
        prior_mu = self.prior_mu.unsqueeze(0).expand(batch_size, -1)
        prior_logvar = self.prior_logvar.unsqueeze(0).expand(batch_size, -1)

        # Target distribution
        with torch.no_grad():
            t_mu = self.target_enc_mu(x_next)
            t_logvar = self.target_enc_logvar(x_next)

        # Sample from target for training loss
        std = torch.exp(0.5 * t_logvar)
        z_target_sample = t_mu + torch.randn_like(std) * std

        return z_target_sample, (dyn_mu, dyn_logvar), (prior_mu, prior_logvar), (t_mu, t_logvar)

    def get_latent_for_probe(self, x_t, x_next=None):
        """
        For BJEPA at inference: use HARD FUSION via Product of Experts.

        This combines:
        - Dynamics expert: p_like(Z_{t+1} | Z_t) - what the model predicts will happen
        - Prior expert: p_prior(Z_{t+1}) - structural constraints

        Returns the fused posterior mean, which should be evaluated against s_{t+1}.
        """
        z_t = self.encoder(x_t)

        # Dynamics expert
        dyn_mu = self.pred_mu(z_t)
        dyn_logvar = self.pred_logvar(z_t)

        # Expand prior for batch
        batch_size = x_t.size(0)
        prior_mu = self.prior_mu.unsqueeze(0).expand(batch_size, -1)
        prior_logvar = self.prior_logvar.unsqueeze(0).expand(batch_size, -1)

        # HARD FUSION: Product of Experts
        post_mu, post_logvar = self.product_of_experts(dyn_mu, dyn_logvar, prior_mu, prior_logvar)

        return post_mu
    








    
def bjepa_loss(z_sample, dyn_params, prior_params, t_params, beta=0.01, gamma=0.1):
    """
    BJEPA training loss with SOFT FUSION.

    From paper Eq. 16:
    L_BJEPA = L_VJEPA (dynamics fitting) + γ * KL(p_like || p_prior) (structural regularization)

    This is soft fusion: dynamics learns to fit the data while being regularized toward prior.
    """
    # Standard VJEPA loss for dynamics
    loss_vjepa = vjepa_prob_loss(z_sample, dyn_params, t_params, beta)

    # KL between dynamics and prior (soft fusion regularization)
    d_mu, d_logvar = dyn_params
    pr_mu, pr_logvar = prior_params

    var_rat = torch.exp(d_logvar - pr_logvar)
    kl_prior = 0.5 * torch.mean(torch.sum(
        var_rat + (pr_mu - d_mu) ** 2 / torch.exp(pr_logvar) - 1 - (d_logvar - pr_logvar),
        dim=1
    ))

    return loss_vjepa + gamma * kl_prior




















# ----------------------------------------------------------------------



# from https://github.com/yangkevin2/naacl-2021-fudge-controlled-generation/blob/master/model.py





import math

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline, set_seed, GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, GPT2Config, GPT2ForSequenceClassification, GPT2LMHeadModel, MarianTokenizer

from constants import *
from util import pad_mask

class Model(nn.Module):
    def __init__(self, args, gpt_pad_id, vocab_size, rhyme_group_size=None, glove_embeddings=None, verbose=True):
        super(Model, self).__init__()

        self.topic = args.task == 'topic'
        self.formality = args.task == 'formality'
        self.iambic = args.task == 'iambic'
        self.rhyme = args.task == 'rhyme'
        self.newline = args.task == 'newline'
        if self.topic:
            self.gpt_embed = nn.Embedding(gpt_pad_id + 1, HIDDEN_DIM, padding_idx=gpt_pad_id) # these are subwords, not words
            if glove_embeddings is None:
                if verbose:
                    print('initializing word embeddings from scratch')
                self.word_embed = nn.Embedding(vocab_size, GLOVE_DIM, padding_idx=0)
            else:
                if verbose:
                    print('initializing word embeddings from glove')
                self.word_embed = nn.Embedding.from_pretrained(glove_embeddings, padding_idx=0)
            self.rnn = nn.LSTM(HIDDEN_DIM, RNN_DIM, num_layers=3, bidirectional=True)
            self.attention_linear = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            large_hidden_dim = HIDDEN_DIM
            self.embed_key_linear = nn.Linear(large_hidden_dim, HIDDEN_DIM)
            self.attention_value_linear = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            self.out_embed_linear = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            self.out_linear = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            self.out_linear2 = nn.Linear(HIDDEN_DIM + large_hidden_dim, HIDDEN_DIM)
            self.out_linear3 = nn.Linear(HIDDEN_DIM, 1)
            self.nonlinear = nn.ReLU()
        elif self.formality:
            self.marian_embed = nn.Embedding(gpt_pad_id + 1, HIDDEN_DIM, padding_idx=0) # 0 in marian is ''
            self.rnn = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, num_layers=3, bidirectional=False, dropout=0.5) # want it to be causal so we can learn all positions
            self.out_linear = nn.Linear(HIDDEN_DIM, 1)
        elif self.iambic:
            self.gpt_embed = nn.Embedding(gpt_pad_id + 1, HIDDEN_DIM, padding_idx=gpt_pad_id)
            self.rnn = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, num_layers=3, bidirectional=False, dropout=0) # want it to be causal so we can learn all positions
            self.out_linear = nn.Linear(HIDDEN_DIM, 1)
        elif self.rhyme:
            self.gpt_embed = nn.Embedding(gpt_pad_id + 1, HIDDEN_DIM, padding_idx=gpt_pad_id) # these are subwords, not words
            self.word_embed = nn.Embedding(rhyme_group_size+1, GLOVE_DIM, padding_idx=0) # this embedding for future words will actually embed the rhyme group idx
            self.rnn = nn.LSTM(HIDDEN_DIM, RNN_DIM, num_layers=3, bidirectional=True)
            self.attention_linear = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            large_hidden_dim = HIDDEN_DIM + COUNT_SYLLABLE_DIM
            self.embed_key_linear = nn.Linear(large_hidden_dim, HIDDEN_DIM)
            self.attention_value_linear = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            self.out_embed_linear = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            self.out_linear = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            self.out_linear2 = nn.Linear(HIDDEN_DIM + large_hidden_dim, HIDDEN_DIM)
            self.out_linear3 = nn.Linear(HIDDEN_DIM, 1)
            self.count_syllable_embed = nn.Embedding(MAX_COUNT_SYLLABLE_DIST+1, COUNT_SYLLABLE_DIM)
            self.nonlinear = nn.ReLU()
        elif self.newline:
            self.gpt_embed = nn.Embedding(gpt_pad_id + 1, HIDDEN_DIM, padding_idx=gpt_pad_id) # these are subwords, not words
            self.rnn = nn.LSTM(HIDDEN_DIM, HIDDEN_DIM, num_layers=3, bidirectional=False)
            self.count_syllable_embed = nn.Embedding(MAX_COUNT_SYLLABLE_DIST+1, COUNT_SYLLABLE_DIM)
            self.out_linear = nn.Linear(HIDDEN_DIM + COUNT_SYLLABLE_DIM, HIDDEN_DIM)
            self.out_linear2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            self.out_linear3 = nn.Linear(HIDDEN_DIM, 1)
            self.nonlinear = nn.ReLU()
        else:
            raise NotImplementedError # TODO honestly this can/should be refactored into different models


    def forward(self, inputs, lengths=None, future_words=None, log_probs=None, syllables_to_go=None, future_word_num_syllables=None, rhyme_group_index=None, run_classifier=False):
        """
        inputs: token ids, batch x seq, right-padded with 0s
        lengths: lengths of inputs; batch
        future_words: batch x N words to check if not predict next token, else batch
        log_probs: N
        syllables_to_go: batch
        """
        if self.topic:
            inputs = self.gpt_embed(inputs) # batch x seq x 300
            inputs = pack_padded_sequence(inputs.permute(1, 0, 2), lengths.cpu(), enforce_sorted=False)
            rnn_output, _ = self.rnn(inputs)
            rnn_output, _ = pad_packed_sequence(rnn_output)
            rnn_output = rnn_output.permute(1, 0, 2) # batch x seq x 300
            hidden = rnn_output
            attention_mask = pad_mask(lengths).permute(1, 0) # batch x seq
            embed = self.word_embed(future_words) # batch x N x 300
            embed_query = self.embed_key_linear(embed)
            attention_tensor = self.attention_linear(hidden).unsqueeze(2) * embed_query.unsqueeze(1) # batch x seq x N x 300
            attention_weights = F.softmax(attention_tensor.sum(dim=3), dim=1) # batch x seq x N
            attention_weights = attention_weights * attention_mask.unsqueeze(2)
            hidden = self.attention_value_linear(hidden)
            weighted_hidden = (hidden.unsqueeze(2) * attention_weights.unsqueeze(3)).sum(dim=1) # batch x seq x N x 768 -> batch x N x 768
            unnormalized_scores = (self.out_linear(weighted_hidden) * self.out_embed_linear(embed)) # batch x N x 300
            unnormalized_scores = torch.cat([unnormalized_scores, embed], dim=2)
            unnormalized_scores = self.nonlinear(self.out_linear2(self.nonlinear(unnormalized_scores)))
            unnormalized_scores = self.out_linear3(unnormalized_scores)
            scores = unnormalized_scores.squeeze(2) - log_probs.unsqueeze(0) 
            return scores # batch x N of normalized scores or batch x 
        elif self.formality:
            inputs = self.marian_embed(inputs)
            inputs = pack_padded_sequence(inputs.permute(1, 0, 2), lengths.cpu(), enforce_sorted=False)
            rnn_output, _ = self.rnn(inputs)
            rnn_output, _ = pad_packed_sequence(rnn_output)
            rnn_output = rnn_output.permute(1, 0, 2) # batch x seq x 300
            return self.out_linear(rnn_output).squeeze(2)
        elif self.iambic:
            inputs = self.gpt_embed(inputs)
            inputs = pack_padded_sequence(inputs.permute(1, 0, 2), lengths.cpu(), enforce_sorted=False)
            rnn_output, _ = self.rnn(inputs)
            rnn_output, _ = pad_packed_sequence(rnn_output)
            rnn_output = rnn_output.permute(1, 0, 2) # batch x seq x 300
            return self.out_linear(rnn_output).squeeze(2)
        elif self.rhyme:
            inputs = self.gpt_embed(inputs) # batch x seq x 300
            inputs = pack_padded_sequence(inputs.permute(1, 0, 2), lengths.cpu(), enforce_sorted=False)
            rnn_output, _ = self.rnn(inputs)
            rnn_output, _ = pad_packed_sequence(rnn_output)
            rnn_output = rnn_output.permute(1, 0, 2) # batch x seq x 300
            hidden = rnn_output
            attention_mask = pad_mask(lengths).permute(1, 0) # batch x seq
            embed = self.word_embed(future_words) # batch x N x 300
            embedded_syllables_to_go = self.count_syllable_embed(syllables_to_go).unsqueeze(1).expand(-1, embed.shape[1], -1) # batch x N x 100
            auxiliary_embed = embedded_syllables_to_go
            embed_query = self.embed_key_linear(torch.cat([embed, auxiliary_embed], dim=2))
            attention_tensor = self.attention_linear(hidden).unsqueeze(2) * embed_query.unsqueeze(1) # batch x seq x N x 300
            attention_weights = F.softmax(attention_tensor.sum(dim=3), dim=1) # batch x seq x N
            attention_weights = attention_weights * attention_mask.unsqueeze(2)
            hidden = self.attention_value_linear(hidden)
            weighted_hidden = (hidden.unsqueeze(2) * attention_weights.unsqueeze(3)).sum(dim=1) # batch x seq x N x 768 -> batch x N x 768
            unnormalized_scores = (self.out_linear(weighted_hidden) * self.out_embed_linear(embed)) # batch x N x 300
            unnormalized_scores = torch.cat([unnormalized_scores, embed, auxiliary_embed], dim=2)
            unnormalized_scores = self.nonlinear(self.out_linear2(self.nonlinear(unnormalized_scores)))
            unnormalized_scores = self.out_linear3(unnormalized_scores)
            scores = unnormalized_scores.squeeze(2) - log_probs.unsqueeze(0) 
            return scores # batch x N of normalized scores or batch x 
        elif self.newline:
            inputs = self.gpt_embed(inputs) # batch x seq x 300
            inputs = pack_padded_sequence(inputs.permute(1, 0, 2), lengths.cpu(), enforce_sorted=False)
            rnn_output, _ = self.rnn(inputs)
            rnn_output, _ = pad_packed_sequence(rnn_output)
            rnn_output = rnn_output.permute(1, 0, 2) # batch x seq x 300
            hidden = torch.cat([rnn_output, self.count_syllable_embed(syllables_to_go).unsqueeze(1).expand(-1, rnn_output.shape[1], -1)], dim=2)
            return self.out_linear3(self.nonlinear(self.out_linear2(self.nonlinear(self.out_linear(hidden))))).squeeze(2)
        else: 
            raise NotImplementedError
            








# from https://github.com/yangkevin2/naacl-2021-fudge-controlled-generation/blob/master/predict_topic.py








import os
import random
import time
import pickle
import math
from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline, set_seed, GPT2Tokenizer, GPT2Model

from data import Dataset
from model import Model
from util import save_checkpoint, ProgressMeter, AverageMeter, num_params
from constants import *

def main(args):
    with open(args.dataset_info, 'rb') as rf:
        dataset_info = pickle.load(rf)
    for cw in args.condition_words.split():
        assert cw in dataset_info.word2index
    gpt_tokenizer = AutoTokenizer.from_pretrained(args.model_string)
    gpt_tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
    gpt_pad_id = gpt_tokenizer.encode(PAD_TOKEN)[0]
    gpt_model = AutoModelWithLMHead.from_pretrained(args.model_string).to(args.device)
    gpt_model.eval()

    checkpoint = torch.load(args.ckpt, map_location=args.device)
    model_args = checkpoint['args']
    conditioning_model = Model(model_args, gpt_pad_id, len(dataset_info.index2word)) # no need to get the glove embeddings when reloading since they're saved in model ckpt anyway
    conditioning_model.load_state_dict(checkpoint['state_dict'])
    conditioning_model = conditioning_model.to(args.device)
    conditioning_model.eval()
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(args.ckpt, checkpoint['epoch']))
    print('num params', num_params(conditioning_model))

    while True:
        results = predict(gpt_model, 
                        gpt_tokenizer, 
                        conditioning_model, 
                        [args.input_text], 
                        args.condition_words, 
                        dataset_info, 
                        args.precondition_topk,
                        args.topk, 
                        args.length_cutoff,
                        condition_lambda=args.condition_lambda,
                        device=args.device)
        print(results)
        import pdb; pdb.set_trace()

def predict(gpt_model, gpt_tokenizer, conditioning_model, input_text, condition_words, dataset_info, precondition_topk, postcondition_topk, length_cutoff, condition_lambda=1.0, device='cuda'):
    with torch.no_grad():
        batch_size = len(input_text)

        condition_words = condition_words.split()
        future_words = torch.LongTensor([dataset_info.word2index[cw] for cw in condition_words]).to(device) # N
        log_probs = torch.Tensor([math.log(dataset_info.vocab[cw] / dataset_info.total_words) for cw in condition_words]).to(device) # N

        # assumes initially all same length.
        encoded_input = [gpt_tokenizer.encode(it, return_tensors='pt').to(device) for it in input_text] # batch x seq
        encoded_input = torch.cat(encoded_input, dim=0)
        lengths = torch.LongTensor([encoded_input.shape[1]]).to(device)

        gpt_encoded_future_words = [gpt_tokenizer.encode(' ' + cw, return_tensors='pt')[0].to(device) for cw in condition_words]
        while lengths.max() < length_cutoff:
            tokens_left = torch.LongTensor([length_cutoff - lengths.max() for _ in range(batch_size)]).to(device)
            gpt_logits = gpt_model(encoded_input)[0][:, -1, :] # batch x vocab
            top_logits, top_indices = gpt_logits.topk(precondition_topk, dim=1) # batch x topk
            new_input_candidates = torch.cat([encoded_input.unsqueeze(1).expand(-1, precondition_topk, -1), top_indices.unsqueeze(2)], dim=2) # batch x topk x seq+1
            expanded_lengths = (lengths + 1).unsqueeze(1).expand(batch_size, precondition_topk) # batch x topk
            expanded_future_words = future_words.unsqueeze(0).unsqueeze(1).expand(batch_size, precondition_topk, -1) # batch x topk x N
            expanded_tokens_left = tokens_left.unsqueeze(1).expand(-1, precondition_topk) # batch x topk
            if condition_lambda == 0:
                condition_logits = torch.zeros_like(expanded_future_words).float()
            else:
                condition_logits = conditioning_model(new_input_candidates.flatten(0, 1), # batch*topk x seq+1
                                                    expanded_lengths.flatten(0, 1), # batch*topk
                                                    expanded_future_words.flatten(0, 1), # batch*topk x N
                                                    log_probs, # N
                                                    expanded_tokens_left.flatten(0, 1)) # batch*topk
                condition_logits = condition_logits.view(batch_size, precondition_topk, -1) # batch x topk x N
                condition_logits = condition_logits - torch.log(1 + torch.exp(condition_logits)) # get correct log probs

            condition_logits = torch.mean(condition_logits, dim=2)
            full_logits = top_logits + condition_logits * condition_lambda # batch x topk
            post_logits, post_indices = full_logits.topk(postcondition_topk, dim=1)
            post_probs = F.softmax(post_logits, dim=1)
            index_into_top_indices = post_indices[torch.arange(batch_size).to(post_indices.device), torch.multinomial(post_probs, 1).flatten()] # batch
            next_indices = top_indices[torch.arange(batch_size).to(top_indices.device), index_into_top_indices] # batch
            encoded_input = torch.cat([encoded_input, next_indices.unsqueeze(1)], dim=1) # batch x seq+1
            lengths = lengths + 1 # batch
        return [gpt_tokenizer.decode(s) for s in encoded_input]
        

if __name__=='__main__':
    parser = ArgumentParser()

    # DATA
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--dataset_info', type=str, required=True, help='saved dataset info')
    parser.add_argument('--model_string', type=str, default='gpt2-medium')

    parser.add_argument('--input_text', type=str, default=None, required=True, help='initial text')
    parser.add_argument('--condition_words', type=str, default=None, required=True, help='word(s) to optimize for')

    parser.add_argument('--precondition_topk', type=int, default=200, help='consider top k outputs from gpt at each step before conditioning and re-pruning')
    parser.add_argument('--topk', type=int, default=10, help='consider top k outputs from gpt at each step')
    parser.add_argument('--condition_lambda', type=float, default=1.0, help='lambda weight on conditioning model')
    parser.add_argument('--length_cutoff', type=int, default=80, help='max length')

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--debug', action='store_true', default=False)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)


















# --------------------------------------------------------------






# ══════════════════════════════════════════════════════════════════════════════
# 1. DIALOG ACT CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

class DialogActClassifier(nn.Module):
    """Lightweight MLP classifier over S1 mean-pooled representations."""
    def __init__(self, input_dim=256, hidden_dim=128, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, z):
        """z: (B, D) → logits (B, num_classes)"""
        return self.net(z)
















# ------------------------------------------------------------------




