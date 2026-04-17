

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




