



# from








import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.nn.parameter import Parameter
from collections import OrderedDict
from utils import *
import random
from net import Net

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MetaSGD(nn.Module):

    def __init__(self, outer_lr=1e-3):
        super(MetaSGD, self).__init__()
        self.net = Net()
        init_lr = random.uniform(5e-3, 0.1)

        self.w1_lr = Parameter(init_lr * torch.ones_like(self.net.w1, requires_grad=True))
        self.w2_lr = Parameter(init_lr * torch.ones_like(self.net.w2, requires_grad=True))
        self.w3_lr = Parameter(init_lr * torch.ones_like(self.net.w3, requires_grad=True))

        self.b1_lr = Parameter(init_lr * torch.ones_like(self.net.b1, requires_grad=True))
        self.b2_lr = Parameter(init_lr * torch.ones_like(self.net.b2, requires_grad=True))
        self.b3_lr = Parameter(init_lr * torch.ones_like(self.net.b3, requires_grad=True))

        self.task_lr = OrderedDict(
            [(name, p) for name, p in self.named_parameters() if 'lr' in name]
        )

        self.optimizer = optim.Adam(list(self.net.parameters()) + list(self.task_lr.values()), lr=outer_lr)

    def forward(self, k_x, k_y, q_x, q_y):
        task_num = k_x.size(0)
        losses = 0

        for i in range(task_num):
            pred_k = self.net(k_x[i])
            loss_k = F.mse_loss(pred_k, k_y[i])
            grad = torch.autograd.grad(loss_k, self.net.parameters())
            fast_weights = OrderedDict(
                [(name, p - lr * g) for (name, p), g, lr in zip(self.net.named_parameters(), grad, self.task_lr.values())]
            )

            pred_q = self.net(q_x[i], fast_weights)
            loss_q = F.mse_loss(pred_q, q_y[i])

            losses = losses + loss_q

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        return losses.item() / task_num

    def test(self, k_x, k_y, q_x, q_y):
        load_state = torch.load('ckpt/meta_sgd.ckpt', map_location='cpu')
        self.load_state_dict(load_state['model_state_dict'])

        batch_size = k_x.size(0)
        losses = 0

        for i in range(batch_size):
            pred_k = self.net(k_x[i])
            loss_k = F.mse_loss(pred_k, k_y[i])
            grad = torch.autograd.grad(loss_k, self.net.parameters())
            fast_weights = OrderedDict(
                [(name, p - lr * g) for (name, p), g, lr in zip(self.net.named_parameters(), grad, self.task_lr.values())]
            )

            pred_q = self.net(q_x[i], fast_weights)
            loss_q = F.mse_loss(pred_q, q_y[i])
            losses = losses + loss_q

        losses /= batch_size

        return losses.item()

class MAML(nn.Module):

    def __init__(self, inner_lr, outer_lr):
        super(MAML, self).__init__()
        self.net = Net()
        self.inner_lr = inner_lr
        self.optimizer = optim.Adam(list(self.net.parameters()), lr=outer_lr)

    def forward(self, k_x, k_y, q_x, q_y):
        task_num = k_x.size(0)
        losses = 0

        for i in range(task_num):
            pred_k = self.net(k_x[i])
            loss_k = F.mse_loss(pred_k, k_y[i])

            grad = torch.autograd.grad(loss_k, list(self.net.parameters()))

            fast_weights = OrderedDict(
                [(name, p - self.inner_lr * g) for (name, p), g in zip(self.net.named_parameters(), grad)]
            )

            pred_q = self.net(q_x[i], fast_weights)
            loss_q = F.mse_loss(pred_q, q_y[i])

            losses = losses + loss_q

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        return losses.item() / task_num

    def test(self, k_x, k_y, q_x, q_y):
        load_state = torch.load('ckpt/maml.ckpt', map_location='cpu')
        self.load_state_dict(load_state['model_state_dict'])

        batch_size = k_x.size(0)
        losses = 0

        for i in range(batch_size):
            pred_k = self.net(k_x[i])
            loss_k = F.mse_loss(pred_k, k_y[i])
            grad = torch.autograd.grad(loss_k, self.net.parameters())
            fast_weights = OrderedDict(
                [(name, p - self.inner_lr * g) for (name, p), g in zip(self.net.named_parameters(), grad)]
            )

            pred_q = self.net(q_x[i], fast_weights)
            loss_q = F.mse_loss(pred_q, q_y[i])
            losses = losses + loss_q

        losses /= batch_size

        return losses.item()
















# --------------------------------------------------















# from https://github.com/lmzintgraf/cavia/blob/master/classification/main.py










import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import arguments
import utils
from dataset_miniimagenet import MiniImagenet
from logger import Logger
from models import CondConvNet


def run(args, num_workers=1, log_interval=100, verbose=True, save_path=None):
    utils.set_seed(args.seed)

    # ---------------------------------------------------------
    # -------------------- training ---------------------------

    # initialise model
    model = CondConvNet(num_context_params=args.num_context_params,
                        context_in=args.context_in,
                        num_classes=args.n_way,
                        num_filters=args.num_filters,
                        max_pool=not args.no_max_pool,
                        num_film_hidden_layers=args.num_film_hidden_layers,
                        imsize=args.imsize,
                        initialisation=args.nn_initialisation,
                        device=args.device
                        )
    model.train()

    # set up meta-optimiser for model parameters
    meta_optimiser = torch.optim.Adam(model.parameters(), args.lr_meta)
    scheduler = torch.optim.lr_scheduler.StepLR(meta_optimiser, 5000, args.lr_meta_decay)

    # initialise logger
    logger = Logger(log_interval, args, verbose=verbose)

    # initialise the starting point for the meta gradient (it's faster to copy this than to create new object)
    meta_grad_init = [0 for _ in range(len(model.state_dict()))]

    iter_counter = 0
    while iter_counter < args.n_iter:

        # batchsz here means total episode number
        dataset_train = MiniImagenet(mode='train',
                                     n_way=args.n_way,
                                     k_shot=args.k_shot,
                                     k_query=args.k_query,
                                     batchsz=10000,
                                     imsize=args.imsize,
                                     data_path=args.data_path)
        # fetch meta_batchsz num of episode each time
        dataloader_train = DataLoader(dataset_train, args.tasks_per_metaupdate, shuffle=True,
                                      num_workers=num_workers,
                                      pin_memory=False)

        # initialise dataloader
        dataset_valid = MiniImagenet(mode='val',
                                     n_way=args.n_way,
                                     k_shot=args.k_shot,
                                     k_query=args.k_query,
                                     batchsz=500,
                                     imsize=args.imsize,
                                     data_path=args.data_path)
        dataloader_valid = DataLoader(dataset_valid, batch_size=num_workers, shuffle=True, num_workers=num_workers,
                                      pin_memory=True)

        logger.print_header()

        for step, batch in enumerate(dataloader_train):

            scheduler.step()

            support_x = batch[0].to(args.device)
            support_y = batch[1].to(args.device)
            query_x = batch[2].to(args.device)
            query_y = batch[3].to(args.device)

            # skip batch if we don't have enough tasks in the current batch (might happen in last batch)
            if support_x.shape[0] != args.tasks_per_metaupdate:
                continue

            # initialise meta-gradient
            meta_grad = copy.deepcopy(meta_grad_init)

            logger.prepare_inner_loop(iter_counter)

            for inner_batch_idx in range(args.tasks_per_metaupdate):

                # reset context parameters
                model.reset_context_params()

                # -------------- inner update --------------

                logger.log_pre_update(iter_counter, support_x[inner_batch_idx], support_y[inner_batch_idx],
                                      query_x[inner_batch_idx], query_y[inner_batch_idx], model)

                for _ in range(args.num_grad_steps_inner):
                    # forward train data through net
                    pred_train = model(support_x[inner_batch_idx])

                    # compute loss
                    task_loss_train = F.cross_entropy(pred_train, support_y[inner_batch_idx])

                    # compute gradient for context parameters
                    task_grad_train = torch.autograd.grad(task_loss_train,
                                                          model.context_params,
                                                          create_graph=True)[0]

                    # set context parameters to their updated values
                    model.context_params = model.context_params - args.lr_inner * task_grad_train

                # -------------- get meta gradient --------------

                # forward test data through updated net
                pred_test = model(query_x[inner_batch_idx])

                # compute loss on test data
                task_loss_test = F.cross_entropy(pred_test, query_y[inner_batch_idx])

                # compute gradient for shared parameters
                task_grad_test = torch.autograd.grad(task_loss_test, model.parameters())

                # add to meta-gradient
                for g in range(len(task_grad_test)):
                    meta_grad[g] += task_grad_test[g].detach()

                # ------------------------------------------------

                logger.log_post_update(iter_counter, support_x[inner_batch_idx], support_y[inner_batch_idx],
                                       query_x[inner_batch_idx], query_y[inner_batch_idx], model)

            # reset context parameters
            model.reset_context_params()

            # summarise inner loop and get validation performance
            logger.summarise_inner_loop(mode='train')

            if iter_counter % log_interval == 0:
                # evaluate how good the current model is (*before* updating so we can compare better)
                evaluate(iter_counter, args, model, logger, dataloader_valid, save_path)
                if save_path is not None:
                    np.save(save_path, [logger.training_stats, logger.validation_stats])
                    # save model to CPU
                    save_model = model
                    if args.device == 'cuda:0':
                        save_model = copy.deepcopy(model).to(torch.args.device('cpu'))
                    torch.save(save_model, save_path)

            logger.print(iter_counter, task_grad_train, meta_grad)
            iter_counter += 1
            if iter_counter > args.n_iter:
                break

            # -------------- meta update --------------

            meta_optimiser.zero_grad()

            # set gradients of parameters manually
            for c, param in enumerate(model.parameters()):
                param.grad = meta_grad[c] / float(args.tasks_per_metaupdate)
                param.grad.data.clamp_(-10, 10)

            # the meta-optimiser only operates on the shared parameters, not the context parameters
            meta_optimiser.step()

    model.reset_context_params()
    return logger, model


def evaluate(iter_counter, args, model, logger, dataloader, save_path):
    logger.prepare_inner_loop(iter_counter, mode='valid')

    for c, batch in enumerate(dataloader):

        support_x = batch[0].to(args.device)
        support_y = batch[1].to(args.device)
        query_x = batch[2].to(args.device)
        query_y = batch[3].to(args.device)

        for inner_batch_idx in range(support_x.shape[0]):

            # reset context parameters
            model.reset_context_params()

            # -------------- inner update --------------

            logger.log_pre_update(iter_counter,
                                  support_x[inner_batch_idx], support_y[inner_batch_idx],
                                  query_x[inner_batch_idx], query_y[inner_batch_idx],
                                  model, mode='valid')

            for _ in range(args.num_grad_steps_eval):
                # forward train data through net
                pred_train = model(support_x[inner_batch_idx])

                # compute loss
                loss_inner = F.cross_entropy(pred_train, support_y[inner_batch_idx])

                # compute gradient for context parameters
                grad_inner = torch.autograd.grad(loss_inner,
                                                 model.context_params,
                                                 create_graph=True)[0]

                # set context parameters to their updated values
                model.context_params = model.context_params - args.lr_inner * grad_inner

            logger.log_post_update(iter_counter,
                                   support_x[inner_batch_idx], support_y[inner_batch_idx],
                                   query_x[inner_batch_idx], query_y[inner_batch_idx],
                                   model, mode='valid')

    # reset context parameters
    model.reset_context_params()

    # this will take the mean over the batches
    logger.summarise_inner_loop(mode='valid')

    # keep track of best models
    logger.update_best_model(model, save_path)


if __name__ == '__main__':

    args = arguments.parse_args()

    # --- settings ---

    if not os.path.exists(os.path.join(utils.get_base_path(), 'result_files')):
        os.mkdir(os.path.join(utils.get_base_path(), 'result_files'))
    if not os.path.exists(os.path.join(utils.get_base_path(), 'result_plots')):
        os.mkdir(os.path.join(utils.get_base_path(), 'result_plots'))

    path = os.path.join(utils.get_base_path(), 'result_files', utils.get_path_from_args(args))
    log_interval = 100

    if (not os.path.exists(path + '.npy')) or args.rerun:
        print('Starting experiment. Logging under filename {}'.format(path + '.npy'))
        run(args, num_workers=1, log_interval=log_interval, save_path=path)
    else:
        print('Found results in {}. If you want to re-run, use the argument --rerun'.format(path))

    # -------------- plot -----------------

    plt.switch_backend('agg')
    training_stats, validation_stats = np.load(path + '.npy', allow_pickle=True)

    plt.figure(figsize=(10, 5))
    x_ticks = np.arange(1, log_interval * len(training_stats['train_accuracy_pre_update']), log_interval)

    # training set
    plt.subplot(1, 2, 1)
    p = plt.plot(x_ticks, training_stats['train_accuracy_pre_update'], '--', label='[train] pre-update')
    plt.plot(x_ticks, training_stats['train_accuracy_post_update'], color=p[-1].get_color(),
             label='[train] post-update')
    p = plt.plot(x_ticks, training_stats['test_accuracy_pre_update'], '--', label='[test] pre-update')
    plt.plot(x_ticks, training_stats['test_accuracy_post_update'], color=p[-1].get_color(), linewidth=1,
             label='[test] post-update')
    plt.ylim([0, 1.01])
    plt.xlim([0, 60000])

    # validation set
    plt.subplot(1, 2, 2)
    p = plt.plot(x_ticks, validation_stats['train_accuracy_pre_update'], '--', label='[train] pre-update')
    plt.plot(x_ticks, validation_stats['train_accuracy_post_update'], color=p[-1].get_color(),
             label='[train] post-update')
    p = plt.plot(x_ticks, validation_stats['test_accuracy_pre_update'], '--', label='[test] pre-update')
    plt.plot(x_ticks, validation_stats['test_accuracy_post_update'], color=p[-1].get_color(), linewidth=1,
             label='[test] post-update')
    plt.ylim([0, 1.01])
    plt.xlim([0, 60000])

    title = 'k={}, cfilt={}, init={}, #t={}, lr={}-{}, ' \
            'grad={}-{} phi={} ({}) #f={} i={} seed={}'.format(args.k_shot,
                                                               args.num_filters,
                                                               args.nn_initialisation,
                                                               args.tasks_per_metaupdate,
                                                               args.lr_inner,
                                                               args.lr_meta,
                                                               args.num_grad_steps_inner,
                                                               args.num_grad_steps_eval,
                                                               args.num_context_params,
                                                               args.context_in,
                                                               args.num_film_hidden_layers,
                                                               args.n_iter,
                                                               args.seed
                                                               )
    plt.suptitle(title)
    plt.title(' ')
    plt.xlabel('num iter', fontsize=10)
    plt.ylabel('accuracy', fontsize=10)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(utils.get_base_path(), 'result_plots', '{}'.format(title.replace('.', ''))))
    plt.close()









