


# from Representation Learning for Conversational Data using Discourse Mutual Information Maximization













import json.decoder
import shutil
import os
import queue
import argparse
import random
import time
import logging

from datautils.data_dialog import WoWData
logger = logging.getLogger(__name__)
import multiprocessing

import numpy as np
from tqdm import tqdm
import wandb
from sklearn.metrics import roc_auc_score

import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import BlenderbotTokenizer, BertTokenizerFast, RobertaTokenizerFast, GPT2TokenizerFast
from transformers import get_linear_schedule_with_warmup
# from rezero.transformer import RZTXEncoderLayer

from models import SMI, is_ddp_module, WrappedSMI
from utils import GEN_UNIQ_RUN_ID, pprint_args
from datautils import DialogData, RMaxData

# =============================== DDP ====================================
# DDP Guides:
# https://spell.ml/blog/pytorch-distributed-data-parallel-XvEaABIAAB8Ars0e
# https://pytorch.org/tutorials/intermediate/dist_tuto.html
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
# ========================================================================







def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(random_seed)
    random.seed(random_seed)


set_random_seeds(random_seed=1234)









def train(rank, context, response, model, model_opt, pad_token_id, args=None, accumulate_grad=False, zero_grad=True):
    if zero_grad:
        model_opt.zero_grad()

    mask_ctx = (context == pad_token_id)
    mask_rsp = (response == pad_token_id)

    c_t, z_t = model(context, response, mask_ctx, mask_rsp)
    
    # =================== DDP Custom Sync ============================
    # Partial derivative chain rule applied to get gradient of combined
    # batch.(https://amsword.medium.com/gradient-backpropagation-with-torch-distributed-all-gather-9f3941a381f8)
    
    # Context
    with torch.no_grad():
        all_c_t = [torch.zeros_like(c_t) for _ in range(args.world_size)]
        dist.all_gather(all_c_t, c_t)
    all_c_t[rank] = c_t
    c_t = torch.cat(all_c_t, dim=0)
    
    # Response
    with torch.no_grad():
        all_z_t = [torch.zeros_like(z_t) for _ in range(args.world_size)]
        dist.all_gather(all_z_t, z_t)
    all_z_t[rank] = z_t
    z_t = torch.cat(all_z_t, dim=0)
    # ================================================================

    # score, loss = model.module.compute_loss(c_t, z_t)
    score, loss, mi = model.module.compute_loss_custom(c_t, z_t, estimator_name=args.estimator)

    if torch.isnan(loss):
        raise Exception("NaN in loss!")
    else:
        # Reverse DDP average as we are using partial derivative trick
        (args.world_size*loss).backward() 
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        if not accumulate_grad:
            model_opt.step()
    # print(loss)
    return loss.item()












def get_g_norm(net):
    total_norm = 0.
    for p in list(filter(lambda p: p.grad is not None, net.parameters())):
        x = p.grad.data.norm(2).item()
        total_norm += x ** 2
    total_norm = total_norm ** (1. / 2)
    print(total_norm)










def evaluate(rank, context, response, model, model_opt, pad_token_id, choice=0, args=None):
    with torch.no_grad():

        mask_ctx = (context == pad_token_id)
        mask_rsp = (response == pad_token_id)

        c_t, z_t = model(context, response, mask_ctx, mask_rsp)
        
        # =================== DDP Custom Sync ============================
        # Context
        all_c_t = [torch.zeros_like(c_t) for _ in range(args.world_size)]
        dist.all_gather(all_c_t, c_t)
        c_t = torch.cat(all_c_t, dim=0)

        # Response
        all_z_t = [torch.zeros_like(z_t) for _ in range(args.world_size)]
        dist.all_gather(all_z_t, z_t)
        z_t = torch.cat(all_z_t, dim=0)
        # ================================================================
        
        # score, loss = model.module.compute_loss(c_t, z_t)
        score, loss, mi = model.module.compute_loss_custom(c_t, z_t, estimator_name=args.estimator)
        

        if choice:
#             score = model.lsoftmax1(score)
            return score.detach().cpu().numpy()
        else:
#             loss = model.compute_loss(score)
#             score = model.lsoftmax1(score)
            return loss.item(), score.detach().cpu().numpy(), mi.item()


def get_rmax_data(rank, json_path, tokenizer, args):
    print_msg = lambda x: print(f"[RANK {rank}]: {x}")
    print_msg(f"<{args.dataset}> --- Load next shard of data.")
    print_msg(f"<{args.dataset}> --- {json_path}\n\n")
    train_data = RMaxData(data_path=json_path, tokenizer=tokenizer, 
                          reddit_filter_enabled=args.reddit_filter_enabled, unsupervised_adrl=args.unsupervised_discourse_losses)

    BS = args.batch_size * args.world_size  # data will be split between workers equally
    print(f"*** Effective batch size {BS}")
    train_loader = DataLoader(train_data, batch_size=BS, num_workers=8, collate_fn=train_data.collate_fn, pin_memory=False)
    return train_loader















def trainIters(rank, model, epochs, train_loader, test_loader, valid_loader, tokenizer, learning_rate=0.0001, args=None):
    print_msg = lambda x: print(f"[RANK {rank}]: {x}")
    start = time.time()
    # model_opt = optim.SGD(model.parameters(), lr=learning_rate)
    if args.roberta_init:
        """
        Roberta has
        - embeddings
        - encoder
            - layer # module list of length 12
        """
        model_opt = optim.Adam([
            # Slow weights
            {"params": model.module.encoder.embeddings.parameters(), 'lr': learning_rate/10},
            {"params": model.module.encoder.encoder.layer[:11].parameters(), 'lr': learning_rate/10},
            # Fast weights
            {"params": model.module.encoder.encoder.layer[11].parameters(), 'lr': learning_rate},
            {"params": model.module.proj.parameters(), 'lr': learning_rate}
        ])
    else:
        if args.learning_rate_control:
            model_opt = optim.Adam([
                # Slow weights
                {"params": model.module.embedding.parameters(), 'lr': learning_rate/10},
                {"params": model.module.encoder.encoder.layers[:-1].parameters(), 'lr': learning_rate/10},
                # Fast weights
                {"params": model.module.encoder.encoder.layers[-1].parameters(), 'lr': learning_rate},
                {"params": model.module.proj.parameters(), 'lr': learning_rate}
            ])
        else:
            model_opt = optim.Adam(model.parameters(), lr=learning_rate)
    
    print_msg(model)
    print_msg(model_opt)
    
    if args.use_scheduler:
        scdl = get_linear_schedule_with_warmup(model_opt, num_warmup_steps=1000, num_training_steps=epochs*len(train_loader))
        # NOTE: we will just continue to use r1M/cc to start training of rMax
        # if args.dataset not in ["rMax", "rMax++"]:
        #     scdl = get_linear_schedule_with_warmup(model_opt, num_warmup_steps=1000, num_training_steps=epochs*len(train_loader))
        # else:
        #     avg_dataset_length = 1786000/?? (batch size x world size)
        #     print_msg(f"Overriding dataset length for LR scheduler: {avg_dataset_length}")
        #     scdl = get_linear_schedule_with_warmup(model_opt, num_warmup_steps=1000, num_training_steps=epochs*avg_dataset_length)

    # criterion = nn.BCEWithLogitsLoss(reduction='none')
    print_msg(f"Initialised optimisers")

    valid_score = []
    best_valid_loss = 100000.0  # any large number
    best_auc = 0.  # any large number
    
    gstep = 0
    for epoch in range(epochs):
        if rank == 0:
            print(f"\n\n========================== NEW EPOCH: {epoch} ===============================\n\n")
        if args.dataset in ["rMax", "rMax++"]:
            rMax_num_files = 1000  # There are 1000 training shards for rMax

            try:
                if args.dataset == "rMax":
                    # replacing data from the very first epoch
                    if epoch > -1:
                        f2load = os.path.join(args.rmax_path, f"train-{(epoch % rMax_num_files):05d}-of-01000.json")
                        train_loader = get_rmax_data(rank, f2load, tokenizer, args)
                else:
                    # replacing data from second epoch
                    if epoch > 2:
                        f2load = os.path.join(args.rmax_path, f"train-{((epoch-3) % rMax_num_files):05d}-of-01000.json")
                        train_loader = get_rmax_data(rank, f2load, tokenizer, args)
            except json.decoder.JSONDecodeError as e:
                print_msg("Failed to load data for epoch, skipping...")
                continue

            print_msg("Data reset barrier reached... Waiting...")
            dist.barrier()

        print_loss_total = 0.
        model.train()

        train_losses = []
        valid_loss_mean = np.inf

        # Pbar
        if rank == 0:
            pbar = tqdm(train_loader, disable=args.no_tqdm)
        else:
            pbar = train_loader

        for batch_idx, entry in enumerate(pbar):
            # gstep = epoch*len(train_loader) + batch_idx
            gstep += 1

            #print_msg(f"{batch_idx} -> {entry[0][0, :5]}")
            #if batch_idx == 10:
            #    break

            # validation step
            if batch_idx % args.val_interval == 0 and batch_idx > 0:
                if rank == 0:
                    pbar.set_description(f"Running validation ...")

                # VALIDATE
                model.eval()

                auc, valid_loss_mean, valid_mi_mean = combined_validation(rank, valid_loader, model, model_opt, tokenizer, args=args)
                if rank == 0:
                    best_auc, best_valid_loss = create_checkpoint(rank, args, model.module, model_opt, epoch + batch_idx/len(train_loader), train_losses,
                                                                  auc, best_auc, valid_loss_mean, best_valid_loss)
                if args.tracking:
                    wandb.log({
                        "valid_loss": valid_loss_mean,
                        "auc": auc,
                        "mutual_info": valid_mi_mean
                    }, step=gstep)  # use moving average

                # Print current LR also
                if args.use_scheduler:
                    print_msg(f"STEP: {gstep}, LR: {scdl.get_last_lr()}")

                # Reset training flag
                model.train()

            # Train step
            # ============================ DDP =======================================
            if args.unsupervised_discourse_losses or args.supervised_discourse_losses:
                # Unsupervised ADRLs
                # if args.unsupervised_discourse_losses:
                for rel_index, (rel_name, discourse) in enumerate(entry.items()):
                    # print(rel_name)
                    # Do zero grad once at the begining
                    zero_grad = True if (rel_index == 0) else False
                    
                    # Backprop after all ADRLs have been processed
                    backprop = True if (rel_index == len(entry) - 1) else False
                    
                    eff_batch_size = len(discourse[0])//args.world_size
                    if eff_batch_size >= 1:
                        batch_context = discourse[0][eff_batch_size*rank:eff_batch_size*(rank+1)].to(rank)
                        batch_response = discourse[1][eff_batch_size*rank:eff_batch_size*(rank+1)].to(rank)
                        loss = train(rank, batch_context, batch_response, model, model_opt, tokenizer.pad_token_id, args=args, 
                                    accumulate_grad=not backprop, zero_grad=zero_grad)
                    else:
                        continue
                
                # Then supervised ADRLs
                # if args.supervised_discourse_losses:
                #     raise NotImplementedError("supervised-ADRL")
            
            else:
                # Otherwise standard CR-DRL
                eff_batch_size = len(entry[0])//args.world_size
                if eff_batch_size >= 1:
                    batch_context = entry[0][eff_batch_size*rank:eff_batch_size*(rank+1)].to(rank)
                    batch_response = entry[1][eff_batch_size*rank:eff_batch_size*(rank+1)].to(rank)
                    loss = train(rank, batch_context, batch_response, model, model_opt, tokenizer.pad_token_id, args=args)
                else:
                    continue
                    
            # =======================================================================
            print_loss_total = print_loss_total + loss
            train_losses.append(loss)
            if rank == 0:
                pbar.set_description(f"[E{epoch}] T.loss: {loss:0.2f}, V.Loss: {valid_loss_mean:0.2f}")

            # Update LR
            if args.use_scheduler:
                scdl.step()

            # Log step
            if (batch_idx+1) % args.log_interval == 0:
                if args.tracking:
                    # 4. Log metrics to visualize performance
                    # + avoid the last batch, may be smaller in the end
                    wandb.log({"train_loss": np.mean(train_losses[-11:-1])}, step=gstep)  # use moving average

        # End of epoch
        model.eval()
        print_msg("TEST AUC")
        _, _, _ = combined_validation(rank, test_loader, model, model_opt, tokenizer, args=args)

        # if args.tracking:
        #     print_msg("Transer .pth files to wandb/...")
        #     if best_auc > 0.7:
        #         print_msg("****** Copy AUC pth")
        #         shutil.copy(f'model_best_auc[{args.UNIQ_RUN_ID}].pth', os.path.join(wandb.run.dir, 'model_best_auc.pth'))
        #         wandb.save(os.path.join(wandb.run.dir, '*.pth'), policy="live")
        #     if np.log(args.batch_size * args.world_size) - best_valid_loss > 0.1:
        #         print_msg("****** Copy LOSS pth")
        #         shutil.copy(f'model_best_loss[{args.UNIQ_RUN_ID}].pth', os.path.join(wandb.run.dir, 'model_best_loss.pth'))
















def create_checkpoint(rank, args, model_pt, model_opt, epoch, train_losses, auc, best_auc, valid_loss_mean, best_valid_loss):
    print_msg = lambda x: print(f"[RANK {rank}]: {x}")
    try:
        # Wandb config works with this
        writable_args = args.as_dict()
    except AttributeError:
        # If it fails then we have argparse output
        writable_args = vars(args)
    print_msg(f"Validation Loss: {valid_loss_mean}")

    # Current Model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_pt.state_dict(),
        'optim_state_dict': model_opt.state_dict(),
        'loss': valid_loss_mean,
        'auc': auc,
        'args': writable_args
    }, os.path.join(args.output_path, f'model_current[{args.UNIQ_RUN_ID}].pth'))
    print_msg(f"[CURRENT] Model saved, V.Loss = {valid_loss_mean}")

    if valid_loss_mean < best_valid_loss:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_pt.state_dict(),
            'optim_state_dict': model_opt.state_dict(),
            'loss': valid_loss_mean,
            'auc': auc,
            'args': writable_args
        }, os.path.join(args.output_path, f'model_best_loss[{args.UNIQ_RUN_ID}].pth'))
        print_msg(f"[Loss] Model saved for current epoch, V.Loss = {valid_loss_mean}")
        best_valid_loss = valid_loss_mean
    print_msg(f'Epoch {epoch} Finished...')
    print_msg(f'Train loss: {np.mean(train_losses)}')
    if best_auc < auc:
        # Save the latest model also:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_pt.state_dict(),
            'optim_state_dict': model_opt.state_dict(),
            'loss': valid_loss_mean,
            'auc': auc,
            'args': writable_args
        }, os.path.join(args.output_path, f'model_best_auc[{args.UNIQ_RUN_ID}].pth'))
        print_msg(f"[AUC] Model saved for current epoch, V.AUC = {auc}")
        best_auc = auc
    return best_auc, best_valid_loss















def combined_validation(rank, test_loader, model, model_opt, tokenizer, args=None):
    print_msg = lambda x: print(f"[RANK {rank}]: {x}")
    with torch.no_grad():
        # auc-roc calculation
        valid_losses = []
        valid_mi = []
        y_pred = []
        y_test = []
        for entry in test_loader:
            # ============================ DDP =======================================
            eff_batch_size = len(entry[0])//args.world_size
            if eff_batch_size >= 1:
                batch_context = entry[0][eff_batch_size*rank:eff_batch_size*(rank+1)].to(rank)
                batch_response = entry[1][eff_batch_size*rank:eff_batch_size*(rank+1)].to(rank)
            else:
                continue
            # =======================================================================
            vloss, score, mi = evaluate(rank, batch_context, batch_response, model, model_opt, tokenizer.pad_token_id, args=args)
            y_pred.extend(score.ravel())
            y_test.extend(np.eye(score.shape[0]).ravel())
            valid_losses.append(vloss)
            # valid_mi.append(np.log(eff_batch_size*args.world_size) - vloss)
            valid_mi.append(mi)

        # print_msg(f"Samples: {len(y_pred)}")
        auc = roc_auc_score(y_test, y_pred)
        print_msg(f"\n*** Eval AUC: {auc} | Eval AUC / Num positives: {np.sum(y_test)} | Eval Dataset: {len(test_loader.dataset)}\n")
    return auc, np.mean(valid_losses[:-1]), np.mean(valid_mi)











# ============================= DDP ===============================
def model_launcher(rank, size, container):
    print_msg = lambda x: print(f"[RANK {rank}]: {x}")
    
    dataload = container['train']
    dataload_valid = container['valid']
    dataload_test = container['test']
    args = container['args']
    tokenizer = container['tokenizer']
    
    args.tracking = args.tracking and (rank == 0)
    
    print_msg(f"Tracking: {args.tracking}")
    
    # WANDB
    if args.tracking:
        # 1. Start a new run
        if args.resume:
            if args.estimator == "infonce":
                wandb.init(project='infonce-stage2', entity='c2ai', config=args)
            else:
                wandb.init(project='estimators-dev-stage2', entity='c2ai', config=args)
        else:
            if args.estimator == "infonce":
                wandb.init(project='infonce-dialog', entity='c2ai', config=args)
            else:
                wandb.init(project='estimators-dev', entity='c2ai', config=args)

        # 2. Save model inputs and hyperparameters
        # Access all hyperparameter values through wandb.config
        args = wandb.config

        # 3. Log gradients and model parameters
        # wandb.watch(model)
        # for batch_idx, (data, target) in enumerate(train_loader):
        #     ...

    print(
        f"SPLIT: train ({len(dataload.dataset)}), "
        f"valid ({len(dataload_valid.dataset)}), "
        f"test ({len(dataload_test.dataset)})")

    # Set random seed within each process to make sure (ddp-)model initializations are same
    set_random_seeds(random_seed=1234)

    # MODEL
    if not args.resume:
        pt_model = SMI(vocab_size=len(tokenizer), d_model=args.d_model, projection_size=args.projection,
                       encoder_layers=args.encoder_layers, encoder_heads=args.encoder_heads,
                       dim_feedforward=args.dim_feedforward,
                       symmetric_loss=args.symmetric_loss,
                       roberta_init=args.roberta_init,
                       roberta_name=args.roberta_name).to(rank)
    else:
        print_msg("Loading model for stage--2 pretraining...")
        print_msg(f"RESUME from: {args.resume_model_path}")
        device = torch.device("cpu")  # Load to cpu first
        checkpoint = torch.load(args.resume_model_path, map_location=device)
        ckpt_args = checkpoint['args']
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        auc = checkpoint['auc']
        state_dict = checkpoint['model_state_dict']

        print_msg("Detect model size from checkpoint:")
        arg_dict = dict(vocab_size=len(tokenizer),
                        d_model=ckpt_args['d_model'],
                        projection_size=ckpt_args['projection'],
                        encoder_layers=ckpt_args['encoder_layers'],
                        encoder_heads=ckpt_args['encoder_heads'],
                        symmetric_loss=ckpt_args['symmetric_loss']
                        )
        print_msg(f"{arg_dict}")
        
        # HACK: Old models did not have a 'dim_feedforward' parameter...
        if 'dim_feedforward' in ckpt_args:
            pt_model = SMI(vocab_size=len(tokenizer), d_model=ckpt_args['d_model'],
                           projection_size=ckpt_args['projection'],
                           encoder_layers=ckpt_args['encoder_layers'], encoder_heads=ckpt_args['encoder_heads'],
                           dim_feedforward=ckpt_args['dim_feedforward'],
                           symmetric_loss=ckpt_args['symmetric_loss'],
                           roberta_init=ckpt_args['roberta_init'],
                           roberta_name=ckpt_args['roberta_name'])
        else:
            # OLD Config (For DMI-Base-RoB of 2021-Sep-06)
            # Missing dim_feedforward and roberta_name
            pt_model = SMI(vocab_size=len(tokenizer), d_model=ckpt_args['d_model'],
                           projection_size=ckpt_args['projection'],
                           encoder_layers=ckpt_args['encoder_layers'], encoder_heads=ckpt_args['encoder_heads'],
                           symmetric_loss=ckpt_args['symmetric_loss'],
                           roberta_init=ckpt_args['roberta_init'])

        if is_ddp_module(state_dict):
            print("*** WARNING: Model was saved as ddp module. Extracting self.module...")
            _wsmi = WrappedSMI(pt_model)
            _wsmi.load_state_dict(state_dict)
            pt_model = _wsmi.module
        else:
            pt_model.load_state_dict(state_dict)

        pt_model = pt_model.to(rank)

    # Wrap inside ddp
    pt_model.train()
    ddp_model = DDP(pt_model, device_ids=[rank])
    
    if args.tracking:
        wandb.watch(ddp_model)
            
    num_params = sum(p.numel() for p in pt_model.parameters() if p.requires_grad)
    print_msg(f'Total number of trainable parameters:  {str(num_params/float(1000000))}M')

    # Check if on cuda
#     print_msg("CUDA:", use_cuda)
#     if use_cuda:
#         model.cuda()

    # TRAIN
    # print_msg(len(train_data.word2idx))
    print_msg("Training starts")
    trainIters(rank, ddp_model, args.epochs, dataload, dataload_test, dataload_valid, tokenizer, learning_rate=args.learning_rate, args=args)
    
    # WANDB
    if args.tracking:
        wandb.finish()












def init_training_process(rank, size, container, proc_entry_fn=model_launcher, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    proc_entry_fn(rank, size, container)
    












    
# ==================================================================

def cmdline_args():
    # Make parser object
    p = argparse.ArgumentParser()
    p.add_argument("-dd", "--dataset", type=str, choices=['dd', 'r5k', 'r100k', 'r1M', 'r1M/cc', 'rMax', 'rMax++', 'paa', 'WoW'],
                   default='dd', help="which dataset to use for pretraining.")
    p.add_argument("-rf", "--reddit_filter_enabled", action="store_true", help="Enable reddit data filter for removing low quality dialogs.")
    p.add_argument("-rmp", "--rmax_path", type=str, help="path to dir for r727m (.json) data files.")
    p.add_argument("-dp", "--data_path", type=str, default=f'./data', help="path to the root data folder.")
    p.add_argument("-op", "--output_path", type=str, default=f'./', help="Path to store the output ``model.pth'' files")
    p.add_argument("-voc", "--vocab", type=str, choices=["bert", "blender", "roberta", "dgpt-m"], required=True,
                   help="mention which tokenizer was used for pretraining? bert or blender")

    p.add_argument("-rob", "--roberta_init", action="store_true", help="Initialize transformer-encoder with roberta weights?")
    p.add_argument("-robname", "--roberta_name", type=str, default="roberta-base",
                   help="name of checkpoint from huggingface")
    p.add_argument("-d", "--d_model", type=int, default=512, help="size of transformer encoders' hidden representation")
    p.add_argument("-d_ff", "--dim_feedforward", type=int, default=2048, help="dim_feedforward for transformer encoder. ")
    p.add_argument("-p", "--projection", type=int, default=512, help="size of projection layer output")
    p.add_argument("-el", "--encoder_layers", type=int, default=4, help="number of layers in transformer encoder")
    p.add_argument("-eh", "--encoder_heads", type=int, default=4, help="number of heads in tformer enc")
    p.add_argument("-sym", "--symmetric_loss", action="store_true", help="whether to train using symmetric infonce")
    p.add_argument("-udrl", "--unsupervised_discourse_losses", action="store_true", help="Additional unsupervised discourse-relation loss components")
    p.add_argument("-sdrl", "--supervised_discourse_losses", action="store_true", help="Additional supervised discourse-relation loss components")
    p.add_argument("-es", "--estimator", type=str,
                   choices=["infonce", "jsd", "nwj", "tuba", "dv", "smile", "infonce/td"], default="infonce",
                   help="which MI estimator is used as the loss function.")
    p.add_argument("-bs", "--batch_size", type=int, default=128, help="batch size during pretraining")

    p.add_argument("-ep", "--epochs", type=int, default=10, help="epochs for pretraining")
    p.add_argument("-vi", "--val_interval", type=int, default=1000, help="validation interval during training")
    p.add_argument("-li", "--log_interval", type=int, default=100, help="logging interval during training")
    p.add_argument("-lr", "--learning_rate", type=float, default=1e-3, help="set learning rate")
    p.add_argument("-lrc", "--learning_rate_control", action="store_true", help="LRC: outer layer and projection layer will have faster LR and rest will be LR/10")
    p.add_argument("-t", "--tracking", default=0, type=int, choices=[0, 1], help="whether to track training+validation loss wandb")
    p.add_argument("-scdl", "--use_scheduler", action="store_true", help="whether to use a warmup+decay schedule for LR")
    p.add_argument("-ntq", "--no_tqdm", action="store_true", help="disable tqdm to create concise log files!")
    
    # ======================== DDP =====================
    p.add_argument("-ddp", "--distdp", action="store_true", help="Should it use pytorch Distributed dataparallel?")
    p.add_argument("-ws", "--world_size", type=int, default=1, help="world size when using DDP with pytorch.")

    # ======================== RESUME ==================
    p.add_argument("-re", "--resume", action="store_true", help="2-stage pretrain: Resume training from a previous checkpoint?")
    p.add_argument("-rept", "--resume_model_path", type=str, help="If ``Resuming'', path to ckpt file.")

    return (p.parse_args())

if __name__ == '__main__':
    # ===================== DDP ====================================
    mp.set_start_method("spawn")
    # ============================================================

    args = cmdline_args()

    if args.roberta_init:
        print(f"[WARNING] Initializing from Roberta-base. This will OVERRIDE all arg config parameters...")
        print("..........................................................................................\n")
        if args.roberta_name == "roberta-base":
            args.d_model = 768
            args.projection = 768
            args.encoder_layers = 12
            args.encoder_heads = 12
            args.dim_feedforward = 3072
        elif args.roberta_name == "roberta-large":
            args.d_model = 1024
            args.projection = 1024
            args.encoder_layers = 24
            args.encoder_heads = 16
            args.dim_feedforward = 4096
        args.vocab = "roberta"

    UNIQ_RUN_ID = GEN_UNIQ_RUN_ID()
    print(f"*** RUN ID: [{UNIQ_RUN_ID}] ***")
    args.UNIQ_RUN_ID = UNIQ_RUN_ID
    pprint_args(args)

    # CONSTRAINTS
    if args.resume:
        assert args.resume_model_path is not None, "resume_model_path is required."
        assert os.path.isfile(args.resume_model_path), "resume_model_path is invalid: No such file"
        
    if args.supervised_discourse_losses and args.dataset != "WoW":
        raise NotImplementedError("s-ADRL is only available for WoW dataset.")

    # Tokenizer
    if args.vocab == "roberta":
        tokenizer = RobertaTokenizerFast.from_pretrained(args.roberta_name)
    elif args.vocab == "dgpt-m":
        mname = "microsoft/DialoGPT-medium"
        tokenizer = GPT2TokenizerFast.from_pretrained(mname)
        tokenizer.add_special_tokens({'pad_token': '<pad/>'})
        tokenizer.add_special_tokens({'cls_token': '<cls/>'})
    else:
        if args.vocab == "blender":
            mname = 'facebook/blenderbot-3B'
            tokenizer = BlenderbotTokenizer.from_pretrained(mname)
        else:
            mname = 'bert-base-uncased'
            tokenizer = BertTokenizerFast.from_pretrained(mname)
        tokenizer.add_special_tokens({'sep_token': '__eou__'})

    print(f"\nVocab Size: {len(tokenizer)}")

    # DATA
    if args.dataset == "dd":
        train_data_path = os.path.join(args.data_path, "dailydialog/dialogues_train.txt")
    elif args.dataset == "r5k":
        train_data_path = os.path.join(args.data_path, "reddit_5k/train_dialogues.txt")
    elif args.dataset == 'r100k':
        train_data_path = os.path.join(args.data_path, "reddit_100k/train_dialogues.txt")
    elif args.dataset == 'r1M':
        train_data_path = os.path.join(args.data_path, "reddit_1M/train_dialogues.txt")
    elif args.dataset == 'r1M/cc':
        train_data_path = os.path.join(args.data_path, "reddit_1M_cc/train_dialogues.txt")
    elif args.dataset == 'rMax':
        # rMax = r5M/cc + r727
        assert args.rmax_path is not None, "rmax_path missing. Provide location of Reddit-727M data."
        # if rMax++: Training will start with r1m/cc or r5m/cc data (see trainIters())
        # NOTE _1: Using r1M_cc as it has the closes length of 1.5M to the length of rMax (1.78M)
        # NOTE _2: the length of the dataset should be such that lr with scdl never goes to zero.
        #       this mainly happens with rMax and if you set the initial dialog data to be DD!
        train_data_path = os.path.join(args.data_path, "reddit_1M_cc/train_dialogues.txt")
    elif args.dataset == 'rMax++':
        raise Exception(f"Yet to implement: {args.dataset}")
    elif args.dataset == 'paa':
        train_data_path = os.path.join(args.data_path, "PAA/train_dialogues.txt")
    elif args.dataset == 'WoW':
        train_data_path = os.path.join(args.data_path, "WoW/data.json")
    else:
        raise Exception(f"Not ready yet: {args.dataset}")

    if args.dataset != 'paa':
        valid_data_path = os.path.join(args.data_path, "dailydialog/dialogues_valid.txt")
        test_data_path = os.path.join(args.data_path, "dailydialog/dialogues_test.txt")
    else:
        valid_data_path = os.path.join(args.data_path, "PAA/valid_dialogues.txt")
        test_data_path = os.path.join(args.data_path, "PAA/test_dialogues.txt")

    # READ DATA
    if args.dataset == 'WoW':
        train_data = WoWData(data_path=train_data_path, tokenizer=tokenizer, unsupervised_adrl=args.unsupervised_discourse_losses, supervised_adrl=args.supervised_discourse_losses)
    else:
        train_data = DialogData(data_path=train_data_path, tokenizer=tokenizer, unsupervised_adrl=args.unsupervised_discourse_losses)
    valid_data = DialogData(data_path=valid_data_path, tokenizer=tokenizer)
    test_data = DialogData(data_path=test_data_path, tokenizer=tokenizer)

    BS = args.batch_size * args.world_size # data will be split between workers equally
    print(f"*** Effective batch size {BS}")

    # SHUFFLE with DDP will need special care
    cpu_count = min(4, multiprocessing.cpu_count())
    print("CPU Count:", cpu_count)
    dataload = DataLoader(train_data, batch_size=BS, num_workers=cpu_count, collate_fn=train_data.collate_fn, pin_memory=False, shuffle=True)
    dataload_valid = DataLoader(valid_data, batch_size=BS, num_workers=cpu_count, collate_fn=valid_data.collate_fn, pin_memory=False)
    dataload_test = DataLoader(test_data, batch_size=BS, num_workers=cpu_count, collate_fn=test_data.collate_fn, pin_memory=False)

    print('Data loaded...')

    if args.world_size <= 1 or args.distdp == False:
        logger.warning("DDP disabled >> Launcing single process training.")
        args.distdp = False
        args.world_size = 1
        container = {
            'train': dataload,
            'valid': dataload_valid,
            'test': dataload_test,
            'tokenizer': tokenizer,
            'args': args
        }
        init_training_process(0, args.world_size, container, model_launcher, "nccl")
    else:
        assert torch.cuda.device_count() >= args.world_size
        # ===================== DDP ====================================
        # Data is Loaded: Need to be split across ranks properly
        processes = []
        container = {
            'train': dataload,
            'valid': dataload_valid,
            'test': dataload_test,
            'tokenizer': tokenizer,
            'args': args
        }
        process_q = queue.Queue()
        for rank in range(args.world_size):
            p = mp.Process(target=init_training_process, args=(rank, args.world_size, container, model_launcher, "nccl"))
            p.start()
            processes.append(p)
            process_q.put(p)
        # ===============================================================


        # ===================== DDP ====================================
        while not process_q.empty():
        #for p in processes:
            p = process_q.get()
            p.join(60)
            print(f"[exitcode] {p.pid} --> {p.exitcode}")
            if p.exitcode is None:
                # Not yet terminated
                process_q.put(p)
            elif p.exitcode != 0:
                print(f"[WARNING] Child process-{p.pid} is dead. \n\tTraining will not progress further.\n\tTrying to kill processes.")
                while not process_q.empty():
                    p = process_q.get()
                    print(f"[KILL] process-{p.pid}")
                    p.kill()

    #     dist.destroy_process_group()
    # ===============================================================

