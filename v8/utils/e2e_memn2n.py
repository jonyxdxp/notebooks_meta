
# from https://github.com/nmhkahn/MemN2N-pytorch/blob/master/memn2n/model.py










import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

def position_encoding(sentence_size, embedding_dim):
    encoding = np.ones((embedding_dim, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_dim + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (embedding_dim+1)/2) * (j - (sentence_size+1)/2)
    encoding = 1 + 4 * encoding / embedding_dim / sentence_size
    # Make position encoding of time words identity to avoid modifying them
    encoding[:, -1] = 1.0
    return np.transpose(encoding)

class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class MemN2N(nn.Module):
    def __init__(self, settings):
        super(MemN2N, self).__init__()

        use_cuda = settings["use_cuda"]
        num_vocab = settings["num_vocab"]
        embedding_dim = settings["embedding_dim"]
        sentence_size = settings["sentence_size"]
        self.max_hops = settings["max_hops"]

        for hop in range(self.max_hops+1):
            C = nn.Embedding(num_vocab, embedding_dim, padding_idx=0)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")

        self.softmax = nn.Softmax()
        self.encoding = Variable(torch.FloatTensor(
            position_encoding(sentence_size, embedding_dim)), requires_grad=False)

        if use_cuda:
            self.encoding = self.encoding.cuda()

    def forward(self, story, query):
        story_size = story.size()

        u = list()
        query_embed = self.C[0](query)
        # weired way to perform reduce_dot
        encoding = self.encoding.unsqueeze(0).expand_as(query_embed)
        u.append(torch.sum(query_embed*encoding, 1))
        
        for hop in range(self.max_hops):
            embed_A = self.C[hop](story.view(story.size(0), -1))
            embed_A = embed_A.view(story_size+(embed_A.size(-1),))
       
            encoding = self.encoding.unsqueeze(0).unsqueeze(1).expand_as(embed_A)
            m_A = torch.sum(embed_A*encoding, 2)
       
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob   = self.softmax(torch.sum(m_A*u_temp, 2))
        
            embed_C = self.C[hop+1](story.view(story.size(0), -1))
            embed_C = embed_C.view(story_size+(embed_C.size(-1),))
            m_C     = torch.sum(embed_C*encoding, 2)
       
            prob = prob.unsqueeze(2).expand_as(m_C)
            o_k  = torch.sum(m_C*prob, 1)
       
            u_k = u[-1] + o_k
            u.append(u_k)
       
        a_hat = u[-1]@self.C[self.max_hops].weight.transpose(0, 1)
        return a_hat, self.softmax(a_hat)

















# --------------------------------------------






# from https://github.com/jojonki/MemoryNetworks/blob/master/memnn.py





import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import to_var
import copy
import math


class MemNN(nn.Module):
    def __init__(self, vocab_size, embd_size, ans_size, max_story_len, hops=3, dropout=0.2, te=True, pe=True):
        super(MemNN, self).__init__()
        self.hops = hops
        self.embd_size = embd_size
        self.temporal_encoding = te
        self.position_encoding = pe

        init_rng = 0.1
        self.dropout = nn.Dropout(p=dropout)
        self.A = nn.ModuleList([nn.Embedding(vocab_size, embd_size) for _ in range(hops+1)])
        for i in range(len(self.A)):
            self.A[i].weight.data.normal_(0, init_rng)
            self.A[i].weight.data[0] = 0 # for padding index
        self.B = self.A[0] # query encoder

        # Temporal Encoding: see 4.1
        if self.temporal_encoding:
            self.TA = nn.Parameter(torch.Tensor(1, max_story_len, embd_size).normal_(0, 0.1))
            self.TC = nn.Parameter(torch.Tensor(1, max_story_len, embd_size).normal_(0, 0.1))

    def forward(self, x, q):
        # x (bs, story_len, s_sent_len)
        # q (bs, q_sent_len)

        bs = x.size(0)
        story_len = x.size(1)
        s_sent_len = x.size(2)

        # Position Encoding
        if self.position_encoding:
            J = s_sent_len
            d = self.embd_size
            pe = to_var(torch.zeros(J, d)) # (s_sent_len, embd_size)
            for j in range(1, J+1):
                for k in range(1, d+1):
                    l_kj = (1 - j / J) - (k / d) * (1 - 2 * j / J)
                    pe[j-1][k-1] = l_kj
            pe = pe.unsqueeze(0).unsqueeze(0) # (1, 1, s_sent_len, embd_size)
            pe = pe.repeat(bs, story_len, 1, 1) # (bs, story_len, s_sent_len, embd_size)

        x = x.view(bs*story_len, -1) # (bs*s_sent_len, s_sent_len)

        u = self.dropout(self.B(q)) # (bs, q_sent_len, embd_size)
        u = torch.sum(u, 1) # (bs, embd_size)

        # Adjacent weight tying
        for k in range(self.hops):
            m = self.dropout(self.A[k](x))            # (bs*story_len, s_sent_len, embd_size)
            m = m.view(bs, story_len, s_sent_len, -1) # (bs, story_len, s_sent_len, embd_size)
            if self.position_encoding:
                m *= pe # (bs, story_len, s_sent_len, embd_size)
            m = torch.sum(m, 2) # (bs, story_len, embd_size)
            if self.temporal_encoding:
                m += self.TA.repeat(bs, 1, 1)[:, :story_len, :]

            c = self.dropout(self.A[k+1](x))           # (bs*story_len, s_sent_len, embd_size)
            c = c.view(bs, story_len, s_sent_len, -1)  # (bs, story_len, s_sent_len, embd_size)
            c = torch.sum(c, 2)                        # (bs, story_len, embd_size)
            if self.temporal_encoding:
                c += self.TC.repeat(bs, 1, 1)[:, :story_len, :] # (bs, story_len, embd_size)

            p = torch.bmm(m, u.unsqueeze(2)).squeeze() # (bs, story_len)
            p = F.softmax(p, -1).unsqueeze(1)          # (bs, 1, story_len)
            o = torch.bmm(p, c).squeeze(1)             # use m as c, (bs, embd_size)
            u = o + u # (bs, embd_size)

        W = torch.t(self.A[-1].weight) # (embd_size, vocab_size)
        out = torch.bmm(u.unsqueeze(1), W.unsqueeze(0).repeat(bs, 1, 1)).squeeze() # (bs, ans_size)

        return F.log_softmax(out, -1)













# --------------------------------------------------------------





# from https://github.com/jojonki/key-value-memory-networks/blob/master/net/memnn_kv.py





from keras import backend as K
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Lambda, Permute, Dropout, add, multiply, dot
from keras.layers.normalization import BatchNormalization
from keras import regularizers

def MemNNKV(mem_key_len, mem_val_len, mem_size, query_maxlen, vocab_size, embd_size, answer_size):
    print('mem_size:', mem_size)
    print('q_max', query_maxlen)
    print('embd_size', embd_size)
    print('vocab_size', vocab_size)
    print('-----------')

    # placeholders
    key = Input((mem_size, mem_key_len,), name='Key_Input')
    val = Input((mem_size, mem_val_len,), name='Val_Input')
    question = Input((query_maxlen,), name='Question_Input')

    shared_embd_A = Embedding(input_dim=vocab_size, output_dim=embd_size)

    key_encoded = shared_embd_A(key) # (None, mem_size, mem_len, embd_size)
    key_encoded = BatchNormalization()(key_encoded)
#     key_encoded = Dropout(.3)(key_encoded)
    key_encoded = Lambda(lambda x: K.sum(x, axis=2)) (key_encoded) #(None, mem_size, embd_size)
    val_encoded = shared_embd_A(val) # (None, mem_size, embd_size)
    val_encoded = BatchNormalization()(val_encoded)
#     val_encoded = Dropout(.3)(val_encoded)
    val_encoded = Lambda(lambda x: K.sum(x, axis=2)) (val_encoded)
    
    question_encoded = shared_embd_A(question) # (None, query_max_len, embd_size)
    question_encoded = BatchNormalization()(question_encoded)
#     question_encoded = Dropout(.3)(question_encoded)
    question_encoded = Lambda(lambda x: K.sum(x, axis=1)) (question_encoded) #(None, embd_size)
    # print('q_encoded', question_encoded.shape)
    q= question_encoded
    for h in range(2):
        ph = dot([q, key_encoded], axes=(1, 2))  # (None, mem_size)
        ph = Activation('softmax')(ph)
        o = dot([ph, val_encoded], axes=(1, 1)) # (None, embd_size)
        print('o', o.shape)
        # R = Dense(embd_size, input_shape=(embd_size,), kernel_regularizer=regularizers.l2(1e-4), name='R_Dense_h' + str(h+1))
        R = Dense(embd_size, input_shape=(embd_size,), name='R_Dense_h' + str(h+1))
        q = R(add([q,  o])) # (None, embd_size)
        q = BatchNormalization()(q)

#     answer = Dense(vocab_size, name='last_Dense', kernel_regularizer=regularizers.l2(0.01))(q) #(None, vocab_size)
    # answer = Dense(answer_size, kernel_regularizer=regularizers.l2(1e-4), name='last_Dense')(q) #(None, vocab_size)
    answer = Dense(answer_size, name='last_Dense')(q) #(None, vocab_size)
    answer = BatchNormalization()(answer)
    # print('answer.shape', answer.shape)
    preds = Activation('softmax')(answer)
    
    # build the final model
    model = Model([key, val, question], preds)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
    return model
