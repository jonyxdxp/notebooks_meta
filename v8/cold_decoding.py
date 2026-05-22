


# from https://github.com/qkaren/COLD_decoding/blob/main/cold_decoding.py






# ("lexical contraint generation" logic only)




#!/usr/bin/env python
# coding: utf-8

import os
import json
import numpy as np
import time
import argparse

import sys
sys.path.insert(0, './GPT2ForwardBackward')

import nltk
nltk.download('punkt')
nltk.download('stopwords')

from util import *
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from bleuloss import batch_log_bleulosscnn_ae
from modeling_opengpt2 import OpenGPT2LMHeadModel








def options():
    parser = argparse.ArgumentParser()
    ## setting
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--print-every", type=int, default=200)
    parser.add_argument("--pretrained_model", type=str, default="gpt2-large")
    parser.add_argument("--straight-through", action="store_true")
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--rl-topk", type=int, default=0)
    parser.add_argument("--lexical", type=str, default='max', choices=['max', 'ppl_max', 'all', 'bleu'])
    parser.add_argument("--if-zx", action="store_true")
    ## experiment
    parser.add_argument("--input-file", type=str,
                        default="./data/lexical/commongen_data/test.multi.constraint.json")
    parser.add_argument("--output-dir", type=str, default="./data/commongen/")
    parser.add_argument("--fwd-model", type=str,
                        default="/var/karen/workspace/GPT2ForwardBackward/opengpt2_pytorch_forward")
    parser.add_argument("--back-model", type=str,
                        default="danyaljj/opengpt2_pytorch_backward")
    parser.add_argument("--version", type=str, default="")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=10)
    parser.add_argument("--repeat-batch", type=int, default=1)
    ## model
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--length", type=int, default=15)
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--frozen-length", type=int, default=0)
    parser.add_argument("--constraint-weight", type=float, default=0.1)
    parser.add_argument("--abductive-c2-weight", type=float, default=0.05)
    parser.add_argument("--lr-nll-portion", type=float, default=1)
    parser.add_argument("--prefix-length", type=int, default=0)
    parser.add_argument("--no-loss-rerank", action="store_true")
    # temperature
    parser.add_argument("--input-lgt-temp", type=float, default=1)
    parser.add_argument("--output-lgt-temp", type=float, default=1)
    parser.add_argument("--rl-output-lgt-temp", type=float, default=1)
    parser.add_argument("--init-temp", type=float, default=0.1)
    parser.add_argument("--init-mode", type=str, default='random', choices=['random', 'original'])
    # lr
    parser.add_argument("--stepsize", type=float, default=0.1)
    parser.add_argument("--stepsize-ratio", type=float, default=1)
    parser.add_argument("--stepsize-iters", type=int, default=1000)
    # iterations
    parser.add_argument("--num-iters", type=int, default=1000)
    parser.add_argument("--min-iters", type=int, default=0)
    parser.add_argument("--noise-iters", type=int, default=1)
    parser.add_argument("--win-anneal-iters", type=int, default=-1)
    parser.add_argument("--constraint-iters", type=int, default=1000)
    # gaussian noise
    parser.add_argument("--gs_mean", type=float, default=0.0)
    parser.add_argument("--gs_std", type=float, default=0.01)
    parser.add_argument("--large-noise-iters", type=str, default="-1")
    parser.add_argument("--large_gs_std", type=str, default="1")

    args = parser.parse_args()
    return args










def decode(model, tokenizer, device, x="", z="", constraints=None, args=None, model_back=None, zz=None):
    '''
    x:   left context / prompt
    z:   constraint words (as a sentence, e.g. ". pet couch cat")
    zz:  same as z (keywords repeated for the 1-gram similarity constraint)
    constraints: unused in lexical mode, kept for API compatibility
    '''

    # ------------------------------------------------------------------ #
    #  Encode left context x
    # ------------------------------------------------------------------ #
    x_ = tokenizer.encode(x)
    x_t = torch.tensor(x_, device=device, dtype=torch.long)
    x_onehot = one_hot(x_t, dimension=tokenizer.vocab_size)

    x_t      = x_t.unsqueeze(0).repeat(args.batch_size, 1)
    x_onehot = x_onehot.repeat(args.batch_size, 1, 1)

    length = args.length

    # ------------------------------------------------------------------ #
    #  Encode z (constraint words) and zz (keyword mask)
    # ------------------------------------------------------------------ #
    z_ = tokenizer.encode(z)[1:]          # drop leading "." token
    z_t = torch.tensor(z_, device=device, dtype=torch.long)
    z_onehot = one_hot(z_t, dimension=tokenizer.vocab_size)
    z_t      = z_t.unsqueeze(0).repeat(args.batch_size, 1)
    z_onehot = z_onehot.repeat(args.batch_size, 1, 1)

    zz_ = tokenizer.encode(zz)[1:]        # drop leading "." token
    zz_t = torch.tensor(zz_, device=device, dtype=torch.long)
    zz_t = zz_t.unsqueeze(0).repeat(args.batch_size, 1)

    # z_mask: binary vocab mask — 1 for each keyword token, used to force
    # keywords into the top-k candidate set during discretization
    z_mask = np.zeros([tokenizer.vocab_size])
    z_mask[zz_] = 1.0
    z_mask = torch.tensor(z_mask, device=device)
    z_mask = z_mask.unsqueeze(0).unsqueeze(0).repeat(args.batch_size, length, 1)

    if args.verbose:
        print("x:\t|%s|\nz:\t|%s|\nzz:\t|%s|" % (
            tokenizer.decode(x_), tokenizer.decode(z_), tokenizer.decode(zz_)))

    # ------------------------------------------------------------------ #
    #  Initialization — greedy forward pass through GPT2 gives ỹ⁽⁰⁾
    # ------------------------------------------------------------------ #
    model.eval()

    if args.init_mode == 'random':
        init_logits = initialize(model, x_t, length, args.init_temp, device)
    else:
        init_logits = z_onehot / 0.1
        init_logits = init_logits[:, :length, :]
        if length > init_logits.shape[1]:
            init_logits = torch.cat(
                [init_logits,
                 torch.zeros([args.batch_size, length - init_logits.shape[1],
                              tokenizer.vocab_size], device=device)],
                dim=1)

    text, _, _ = get_text_from_logits(init_logits, tokenizer)
    for bi in range(args.batch_size):
        print("[initial]: %s" % text[bi])

    # ------------------------------------------------------------------ #
    #  Cache x KV-pairs for efficient soft forward passes
    # ------------------------------------------------------------------ #
    assert args.prefix_length <= 0, "prefix_length > 0 not supported"
    soft_forward_x = x_onehot[:, -1:, :]   # only the last token of x
    if x_t.shape[1] == 1:
        x_model_past = None
    else:
        x_model_outputs = model(x_t[:, :-1])
        x_model_past = x_model_outputs.past_key_values
        x_model_past = [_.detach() for _ in x_model_past]

    # ------------------------------------------------------------------ #
    #  Optimisation setup
    #  epsilon is the learnable perturbation on top of y_logits
    # ------------------------------------------------------------------ #
    y_logits = init_logits
    epsilon  = torch.nn.Parameter(torch.zeros_like(y_logits))
    optim    = torch.optim.Adam([epsilon], lr=args.stepsize)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optim, step_size=args.stepsize_iters, gamma=args.stepsize_ratio)

    frozen_len = args.frozen_length
    rl_reverse_index = torch.arange(y_logits.shape[1] - 1, -1, -1)

    mask_t    = None
    y_logits_ = None
    noise_std = 0.0

    # ------------------------------------------------------------------ #
    #  Langevin dynamics loop
    # ------------------------------------------------------------------ #
    for iter in range(args.num_iters):
        optim.zero_grad()

        y_logits_ = y_logits + epsilon  # current soft sequence

        # ---- soft forward pass through GPT2 (forward fluency constraint) ----
        soft_forward_y = y_logits_ / 0.001
        if args.straight_through:
            if mask_t is None:
                soft_forward_y = (y_logits_.detach() / 0.001 - y_logits_).detach() + y_logits_
            else:
                soft_forward_y = top_k_filter_3d(
                    y_logits_, args.topk, mask=mask_t, extra_mask=z_mask) / 0.001

        y_logits_t = soft_forward(model, soft_forward_x, soft_forward_y, x_past=x_model_past)

        if args.topk == 0:
            mask_t = None
        else:
            _, indices_t = torch.topk(y_logits_t, args.topk)
            mask_t = torch.zeros_like(y_logits_t).scatter_(2, indices_t, 1)

        # ---- forward (left-to-right) fluency loss ----
        lr_nll_loss = soft_nll(
            top_k_filter_3d(y_logits_t / args.output_lgt_temp, args.topk, extra_mask=z_mask),
            y_logits_ / args.input_lgt_temp)

        # ---- reverse (right-to-left) fluency loss ----
        yz_logits_rev   = torch.flip(torch.cat([y_logits_, z_onehot], dim=1), [1])
        yz_logits_rev_t = soft_backward(model_back, yz_logits_rev / 0.00001)
        yz_logits_rev_rev_t = torch.flip(yz_logits_rev_t, [1])
        yz_logits_rev_rev_t = yz_logits_rev_rev_t[:, :, 1:y_logits_.shape[-1] + 1]
        yz_logits_rev_rev_t_ = yz_logits_rev_rev_t[:, :y_logits_.shape[1], :]

        # repetition penalty
        repetition_mask = torch.cat(
            [F.softmax(yz_logits_rev_rev_t_[:, 1:, :], dim=-1),
             torch.zeros_like(yz_logits_rev_rev_t_[:, -1:, :])], dim=1)
        yz_logits_rev_rev_t_ = (yz_logits_rev_rev_t_ - repetition_mask * 1e4).detach()

        rl_nll_loss = soft_nll(
            top_k_filter_3d(yz_logits_rev_rev_t_ / args.rl_output_lgt_temp, args.rl_topk),
            y_logits_ / args.input_lgt_temp)

        # ---- constraint losses ----
        # c_loss_1: future-token prediction (cross-entropy on z given y)
        soft_forward_y_ = (y_logits_.detach() / 0.3 - y_logits_).detach() + y_logits_
        xyz_logits, xy_length = soft_forward_xyz(model, soft_forward_x, soft_forward_y_, z_onehot)

        bz = args.batch_size
        lg = xyz_logits.shape[1]
        st = xy_length - 1
        ed = xyz_logits.shape[1] - 1
        xyz_logits = xyz_logits.view(-1, xyz_logits.shape[-1])
        z_logits = torch.cat(
            [xyz_logits[bi * lg + st:bi * lg + ed, :] for bi in range(bz)], dim=0)

        c_loss_1 = torch.nn.CrossEntropyLoss(reduction='none')(z_logits, z_t.view(-1))
        c_loss_1 = c_loss_1.view(args.batch_size, -1).mean(-1)

        # c_loss_2: 1-gram similarity — keyword coverage (BLEU-1 against keyword list)
        c_loss_2 = batch_log_bleulosscnn_ae(
            decoder_outputs=y_logits_.transpose(0, 1),
            target_idx=zz_t,
            ngram_list=[1])

        c_loss = c_loss_1 + args.abductive_c2_weight * c_loss_2

        # ---- total loss ----
        loss = (
            (1.0 - args.constraint_weight) * args.lr_nll_portion       * lr_nll_loss
          + (1.0 - args.constraint_weight) * (1 - args.lr_nll_portion) * rl_nll_loss
          + args.constraint_weight                                      * c_loss
        ).mean()

        if iter < args.num_iters - 1:
            loss.backward()
            optim.step()
            scheduler.step()
            last_lr = scheduler.get_last_lr()[0]

        if args.verbose and ((iter + 1) % args.print_every == 0 or iter == 0 or iter + 1 == args.num_iters):
            text, _, _ = decode_with_model_topk(
                model, y_logits_, args.topk, soft_forward_x, x_model_past, tokenizer, extra_mask=z_mask)
            for bi in range(args.batch_size):
                print("%d, loss: %.4f, lr_nll: %.4f, rl_nll: %.4f, c2: %.4f, lr: %.4f, |%s|" % (
                    iter + 1, loss.item(),
                    lr_nll_loss[bi].item(), rl_nll_loss[bi].item(),
                    c_loss_2[bi].item(), last_lr, text[bi]))
            print()

        # ---- noise injection (Langevin dynamics annealing schedule) ----
        if iter < args.num_iters - 1:
            large_noise_iters = [int(_) for _ in args.large_noise_iters.split(',')]
            large_gs_stds     = [float(_) for _ in args.large_gs_std.split(',')]
            noise_std = 0.0
            if iter % args.noise_iters == 0:
                noise_last = True
                for ni in range(len(large_noise_iters)):
                    if iter < large_noise_iters[ni]:
                        noise_last = False
                        break
                noise_std = args.gs_std if noise_last else large_gs_stds[ni]

                noise = torch.normal(mean=args.gs_mean, std=noise_std,
                                     size=epsilon.size(), device='cuda', requires_grad=False)
                if args.win_anneal_iters >= 0 and iter >= args.win_anneal_iters:
                    zeros     = torch.zeros_like(noise)
                    noise_mix = torch.cat([zeros[:, :frozen_len], noise[:, frozen_len:]], dim=1)
                    y_logits  = y_logits + noise_mix
                else:
                    y_logits = y_logits + noise

    # ------------------------------------------------------------------ #
    #  Discretization — top-k filtering with keyword force-inclusion
    # ------------------------------------------------------------------ #
    text, _, last_text_ids = decode_with_model_topk(
        model, y_logits_, args.topk, soft_forward_x, x_model_past, tokenizer, extra_mask=z_mask)

    last_rank_loss = model(input_ids=last_text_ids, labels=last_text_ids).loss
    last_rank_loss = last_rank_loss.detach().clone().data.cpu().numpy()
    ppl_last       = np.exp(last_rank_loss)

    text_post = post_process(last_text_ids, model, args.max_length, args.length, tokenizer, device)

    if args.verbose:
        for bi in range(args.batch_size):
            print("[final]: %s\n%.4f" % (text[bi], ppl_last))
            print("[final complete sentence]: %s\n" % text_post[bi])

    return ppl_last, text, text_post














def lexical_generation(model, tokenizer, device, args, model_back=None):
    with open(args.input_file, 'r') as f:
        data = [json.loads(l.strip()) for l in f.readlines()]

    outfile = '%if_zx%s_seed%d_%d_%d_lexical_cw%.3f_c2w%.3f_lrnllp%.3f_len%d_topk%d' \
              '_niter%d_frozlen%d_winiter%d_noiseiter%d_gsstd%.4f_lr%.3f' \
              '_lrratio%.2f_lriter%d_%s_%s_output.json' % (
                  args.if_zx, args.version, args.seed, args.start, args.end,
                  args.constraint_weight, args.abductive_c2_weight, args.lr_nll_portion,
                  args.length, args.topk, args.num_iters, args.frozen_length,
                  args.win_anneal_iters, args.noise_iters, args.gs_std, args.stepsize,
                  args.stepsize_ratio, args.stepsize_iters,
                  args.large_noise_iters, args.large_gs_std)

    print("outputs: %s" % outfile)
    fw        = open(os.path.join(args.output_dir, outfile), 'w')
    fw_pretty = open(os.path.join(args.output_dir, 'pretty_' + outfile), 'w')

    for i, d in enumerate(data):
        if i < args.start or i > args.end:
            continue

        # Keywords come from the concept set, e.g. "pet#couch#cat"
        constraints  = d["concept_set"].split("#")
        constraints  = ' '.join(constraints)
        x            = "<|endoftext|>"   # no left context for CommonGen
        z            = ". " + constraints
        z_keywords   = ". " + constraints

        print("%d / %d  |  concepts: %s" % (i, len(data), constraints))

        text_candidates          = []
        text_complete_candidates = []
        for _ in range(args.repeat_batch):
            ppl_last, text, text_post = decode(
                model, tokenizer, device, x, z, None, args,
                model_back=model_back, zz=z_keywords)
            text_candidates.extend(text)
            text_complete_candidates.extend(text_post)

        out = {
            'x':                   x,
            'constraints':         constraints,
            'generation':          text_candidates,
            'generation_complete': text_complete_candidates,
        }
        print(out)

        fw.write(json.dumps(out) + '\n')
        fw.flush()
        fw_pretty.write(json.dumps(out, indent=4) + '\n')
        fw_pretty.flush()

    print("outputs: %s" % outfile)















def main():
    args   = options()
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    if args.seed != -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # ---- Load frozen GPT2 (forward LM) ----
    model = GPT2LMHeadModel.from_pretrained(
        args.pretrained_model, output_hidden_states=True,
        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, summary_first_dropout=0)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_model)

    # ---- Load frozen reverse LM ----
    model_back = OpenGPT2LMHeadModel.from_pretrained(
        args.back_model,
        hidden_dropout_prob=0, attention_probs_dropout_prob=0, summary_first_dropout=0)
    model_back.to(device)
    model_back.eval()
    for param in model_back.parameters():
        param.requires_grad = False

    lexical_generation(model, tokenizer, device, args, model_back)


if __name__ == "__main__":
    main()