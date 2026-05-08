

# from

















import os 
import sys
sys.path.append( './' )
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import argparse
import torch
import torch.nn as nn

from models.Transformers import PSCBert, PSCRoberta, PSCDistilBERT
from training import PSCTrainer
from dataloader.dataloader import pair_loader_csv, pair_loader_txt
from utils.utils import set_global_random_seed, setup_path
from utils.optimizer import get_optimizer, get_bert_config_tokenizer, MODEL_CLASS
import subprocess
    
def run(args):
    args.resPath, args.tensorboard = setup_path(args)
    set_global_random_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_id = torch.cuda.device_count()
    print("\t {} GPUs available to use!".format(device_id))

    '''
    We assume paired training data (e.g., NLI data) is always saved in csv/tsv format,
    and single training data (e.g., wiki) is always saved in txt format.
    '''
    if args.dataname.endswith(".csv") or args.dataname.endswith(".tsv"):
        train_loader = pair_loader_csv(args)
    elif args.dataname.endswith(".txt"):
        train_loader = pair_loader_txt(args)
    else:
        return ValueError()
    
    # model & optimizer
    config, tokenizer = get_bert_config_tokenizer(args.bert)
    if 'roberta' in args.bert:
        model = PSCRoberta.from_pretrained(MODEL_CLASS[args.bert], feat_dim=args.feat_dim)
    elif 'distilbert' in args.bert:
        model = PSCDistilBERT.from_pretrained(MODEL_CLASS[args.bert], feat_dim=args.feat_dim)
    else:
        model = PSCBert.from_pretrained(MODEL_CLASS[args.bert], feat_dim=args.feat_dim)

    optimizer = get_optimizer(model, args)
    
    model = nn.DataParallel(model)
    model.to(device)
    
    # set up the trainer
    trainer = PSCTrainer(model, tokenizer, optimizer, train_loader, args)
    trainer.train()
    return None

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_instance', type=str, default='local')
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0], help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--seed', type=int, default=0, help="")
    parser.add_argument('--resdir', type=str, default='./results')
    parser.add_argument('--logging_step', type=int, default=250, help="")
    # Dataset
    parser.add_argument('--datapath', type=str, default='')
    parser.add_argument('--dataname', type=str, default='tod_single_pos_3.tsv', help="")
    # Training parameters
    parser.add_argument('--max_length', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=5e-06, help="")
    parser.add_argument('--lr_scale', type=int, default=100, help="")
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max_iter', type=int, default=100000000)
    # Contrastive learning
    parser.add_argument('--mode', type=str, default='contrastive', help="")
    parser.add_argument('--bert', type=str, default='distilbert', help="")
    parser.add_argument('--contrast_type', type=str, default="HardNeg")
    parser.add_argument('--feat_dim', type=int, default=128, help="dimension of the projected features for instance discrimination loss")
    parser.add_argument('--decay_rate', type=float, default=1, help="the decay rate when modeling multi-turn dialogue")
    parser.add_argument('--num_turn', type=int, default=1, help="number of previous turn used in model training and response selection")
    parser.add_argument('--temperature', type=float, default=0.05, help="temperature required by contrastive loss")
    parser.add_argument('--save_model_every_epoch', action='store_true', default=True, help="Whether to save model at every epoch")

    
    args = parser.parse_args(argv)
    args.use_gpu = args.gpuid[0] >= 0
    args.resPath = None
    args.tensorboard = None
    return args

if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    run(args)
















# from https://github.com/amazon-science/dse/blob/main/pretrain/training.py







import os
import sys
import csv
import numpy as np

import torch
import torch.nn as nn
from utils.contrastive_utils import HardConLoss
from utils.utils import statistics_log 


from torch.utils.data import DataLoader, SequentialSampler
from sklearn.preprocessing import normalize
from tqdm import tqdm







class PSCTrainer(nn.Module):
    def __init__(self, model, tokenizer, optimizer, train_loader, args):
        super(PSCTrainer, self).__init__()
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.task_type = self.args.mode
        self.gstep = 0
        self.dev_objective = -1
        
        self.psc_loss = HardConLoss(temperature=self.args.temperature, contrast_type=self.args.contrast_type).cuda()
        self.classify_loss = nn.CrossEntropyLoss().cuda()
        print("\nUsing PSC_Trainer, {}\n".format(self.args.contrast_type))
        

    def get_batch_token(self, text, max_length=-1):
        if max_length == -1:
            max_length = self.args.max_length

        token_feat = self.tokenizer.batch_encode_plus(
            text, 
            max_length=max_length, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True
        )
        return token_feat
        

    def prepare_pairwise_input(self, batch):
        text1, text2, pairsimi = batch['text1'], batch['text2'], batch['pairsimi'].cuda()
        feat1 = self.get_batch_token(text1)
        feat2 = self.get_batch_token(text2)

        
        input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1)], dim=1)
        attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1)], dim=1)
        return input_ids.cuda(), attention_mask.cuda(), pairsimi.detach()
    


    def prepare_pairwise_input_multiturn_concatenate(self, batch):
        text1, text2, pairsimi = batch['text1'], batch['text2'], batch['pairsimi'].cuda()
        max_query_length = self.args.num_turn * self.args.max_length
        num_keeped_words = int(max_query_length*0.9)
        text1 = [" ".join(t.split()[-num_keeped_words:]) for t in text1]
        feat1 = self.get_batch_token(text1, max_length=max_query_length)
        feat2 = self.get_batch_token(text2, max_length=32)


        batch_size = feat2['input_ids'].shape[0]
        seq_length = feat2['input_ids'].shape[1]



        input_ids = torch.cat([feat1['input_ids'].reshape(batch_size, -1, seq_length), feat2['input_ids'].unsqueeze(1)], dim=1)
        attention_mask = torch.cat([feat1['attention_mask'].reshape(batch_size, -1, seq_length), feat2['attention_mask'].unsqueeze(1)], dim=1)
        return input_ids.cuda(), attention_mask.cuda(), pairsimi.detach()



    def save_model(self, epoch, best_dev=False):
        if best_dev:
            save_dir = os.path.join(self.args.resPath, 'dev')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.model.module.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)
        else:
            save_dir = os.path.join(self.args.resPath, str(epoch+1))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.model.module.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)


    def train(self):

        all_iter = self.args.epochs * len(self.train_loader)
        print('\n={}/{}=Iterations/Batches'.format(all_iter, len(self.train_loader)))

        self.model.train()
        epoch_iterator = tqdm(self.train_loader, desc="Iteration")
        for epoch in range(self.args.epochs):
            for j, batch in enumerate(epoch_iterator):
                if self.args.num_turn > 1:
                    input_ids, attention_mask, pairsimi = self.prepare_pairwise_input_multiturn_concatenate(batch)
                else:
                    input_ids, attention_mask, pairsimi = self.prepare_pairwise_input(batch)
                    
                losses = self.train_step(input_ids, attention_mask, pairsimi)


                if (self.gstep%self.args.logging_step==0) or (self.gstep==all_iter) or (self.gstep==self.args.max_iter):
                    statistics_log(self.args.tensorboard, losses=losses, global_step=self.gstep)
                        
                elif self.gstep > self.args.max_iter:
                    break
                    
                self.gstep += 1
                
            print("Finish Epoch: ", epoch)
            if self.args.save_model_every_epoch:
                self.save_model(epoch, best_dev=False)
        return None
        

    def train_step(self, input_ids, attention_mask, pairsimi, speaker_query_labels=None, speaker_response_labels=None):         
        feat1, feat2, _, _ = self.model(input_ids, attention_mask, task_type='contrastive')
        losses = self.psc_loss(feat1, feat2, pairsimi)
        loss = losses["instdisc_loss"]

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return losses
    

    




    

