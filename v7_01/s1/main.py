

# from https://github.com/ruizhaoz/MOML/blob/main/main.py


import warnings
warnings.filterwarnings('ignore', category=UserWarning)


import sys, os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import copy
from torch.utils.tensorboard import SummaryWriter

# higher library requires math attention — flash/efficient kernels lack backward support
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from v7_01.s1.data.dataset import *
from v7_01.s1.cog_arch.encoder import *
from v7_01.s1.utils_general import *

# ── Paths ────────────────────────────────────────────────────────────────────
data_path      = '/content/drive/MyDrive/data/cache'          # directory, not a file
dialog_train   = '/content/Discourse-Mutual-Information-DMI-main/data/dailydialog/dialogues_train.txt'
dialog_valid   = '/content/Discourse-Mutual-Information-DMI-main/data/dailydialog/dialogues_valid.txt'
dialog_test    = '/content/Discourse-Mutual-Information-DMI-main/data/dailydialog/dialogues_test.txt'


# ── Dataset ──────────────────────────────────────────────────────────────────
data_obj = DialogPairDatasetObject(        # ← replaces DatasetObject
    train_file=dialog_train,
    valid_file=dialog_valid,
    test_file=dialog_test,
    n_tasks=10,
)

# ── Model ─────────────────────────────────────────────────────────────────────
model_name = 'SMI_DMI_MOML_v7_01'

model_func = lambda: SMI(                  # ← back to raw SMI, no classifier head
    vocab_size=50265,
    d_model=512,
    encoder_layers=4,
    encoder_heads=4,
)


# ── Hyperparameters ───────────────────────────────────────────────────────────
weight_decay          = 1e-4
batch_size            = 64
learning_rate    = 1e-3
learning_rate_ft = 1e-4    # slower inner loop for stable MI optimisation
lr_decay_per_round    = 1

sch_step              = 1
sch_gamma             = 1

K_list                = [50, 100]
alpha_list            = [1, 5, 10]
num_grad_step_list    = [1, 5]
learning_rate_ft_list = [0.1, 0.01]

save_models           = True
save_performance      = True
save_tensorboard      = True
suffix                = model_name

# ── Init model ────────────────────────────────────────────────────────────────
torch.manual_seed(17)
init_model = model_func()

model_dir = '%s/Model/%s' % (data_path, data_obj.name)
os.makedirs(model_dir, exist_ok=True)

init_mdl_path = '%s/%s_init_mdl.pt' % (model_dir, model_name)
if not os.path.exists(init_mdl_path):
    torch.save(init_model.state_dict(), init_mdl_path)
else:
    init_model.load_state_dict(torch.load(init_mdl_path))












# MOML function


def train_MOML(data_obj, alpha, learning_rate, learning_rate_ft, batch_size, K, num_grad_step,
               print_per, weight_decay, model_func, init_model, sch_step, sch_gamma, lr_decay_per_round,
               save_models, save_performance, save_tensorboard, suffix='', data_path=''):
    suffix = 'MOML_' + suffix
    suffix += '_alpha%f_Lr%f_LrT%f_B%d_K%d_GS_%d_W%f' % (
    alpha, learning_rate, learning_rate_ft, batch_size, K, num_grad_step, weight_decay)
    suffix += '_lrdecay%f_%d_%f' % (lr_decay_per_round, sch_step, sch_gamma)

    task_x = data_obj.trn_x
    task_y = data_obj.trn_y
    dataset_name = data_obj.dataset
    n_tasks = len(task_x)

    if (save_models or save_performance) and (
    not os.path.exists('%s/Model/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%s/Model/%s/%s' % (data_path, data_obj.name, suffix))

    online_mdls = [None] * n_tasks
    writer = SummaryWriter('%s/Runs/%s/%s' % (data_path, data_obj.name, suffix)) if save_tensorboard else None

    tst_before_perf = np.zeros((n_tasks, n_tasks, 2))
    trn_before_perf = np.zeros((n_tasks, 2))

    tst_after_perf = np.zeros((n_tasks, n_tasks, 2))
    trn_after_perf = np.zeros((n_tasks, 2))

    CTM_perf = np.zeros((n_tasks, 2))
    LTM_perf = np.zeros((n_tasks, 2))

    # Initialize model
    meta_model = model_func().to(device)
    meta_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())), strict=False)

    omega_model = get_mdl_params([init_model])[0]
    n_par = omega_model.shape[0]
    lambda_model = torch.zeros(n_par, dtype=torch.float32, device=device)  # Start from all 0s

    if not os.path.exists('%s/Model/%s/%s/%d_tst_before_perf.npy' % (data_path, data_obj.name, suffix, n_tasks)):
        # Train
        for task in range(n_tasks):
            print('---- Round %2d' % (task + 1))

            ### Evaluation
            # Test all seen tasks including the current one before training.
            for tt in range(task):
                tst_before_perf[task][tt] = tst_after_perf[task - 1][
                    tt]  # The model is not updated, no need to calculate twice

            tst_before_perf[task][task] = get_maml_mi_loss(
                    task_x[task], task_y[task], meta_model, model_func,
                    learning_rate_ft, num_grad_step,
                    tst_ctx=data_obj.tst_x[task], tst_rsp=data_obj.tst_y[task])

            trn_before_perf[task] = get_maml_mi_loss(
                    task_x[task], task_y[task], meta_model, model_func,
                    learning_rate_ft, num_grad_step)

            # Train only get the current task
            trn_x = task_x[task]
            trn_y = task_y[task]
            decay = lr_decay_per_round ** task
            meta_model = train_MOML_smi_model(
                meta_model, model_func, trn_x, trn_y,
                alpha, omega_model, lambda_model,
                learning_rate * decay, learning_rate_ft, num_grad_step,
                batch_size, K, print_per, weight_decay,
                pad_token_id=1,          # RoBERTa pad
                sch_step=sch_step, sch_gamma=sch_gamma)
            curr_par = get_mdl_params([meta_model], n_par=n_par)[0]
            # updating the lambda model and omega model
            lambda_model = lambda_model - alpha * (curr_par - omega_model)
            omega_model = 1 / 2 * (curr_par + omega_model) - 1 / 2 * 1 / alpha * lambda_model

            ### Evaluation
            trn_after_perf[task] = get_maml_mi_loss(
                        task_x[task], task_y[task], meta_model, model_func,
                        learning_rate_ft, num_grad_step)

            # Test all seen tasks.
            for tt in range(task + 1):
                    tst_after_perf[task][tt] = get_maml_mi_loss(
                            task_x[tt], task_y[tt], meta_model, model_func,
                            learning_rate_ft, num_grad_step,
                            tst_ctx=data_obj.tst_x[tt], tst_rsp=data_obj.tst_y[tt])

            ### CTM and LTM
            CTM_perf[task] = tst_before_perf[task][task]
            LTM_perf[task] = np.mean(tst_after_perf[task, :task + 1, :], axis=0)

            print('\n*** Task %2d, Training   data, MI: %.4f, Loss: %.4f'
                % (task + 1, trn_after_perf[task][1], trn_after_perf[task][0]))
            print('*** Task %2d, Test       data, MI: %.4f, Loss: %.4f\n'
                % (task + 1, tst_after_perf[task, task, 1], tst_after_perf[task, task, 0]))

            if save_tensorboard:
                ## Loss
                writer.add_scalar('Loss/Train_Before', trn_before_perf[task, 0], task + 1)
                writer.add_scalar('Loss/Train_After', trn_after_perf[task, 0], task + 1)

                writer.add_scalar('Loss/Train_Before_Avg', np.mean(trn_before_perf[:task + 1, 0]), task + 1)
                writer.add_scalar('Loss/Train_After_Avg', np.mean(trn_after_perf[:task + 1, 0]), task + 1)

                writer.add_scalar('Loss/CTM', CTM_perf[task, 0], task + 1)
                writer.add_scalar('Loss/CTM_Avg', np.mean(CTM_perf[:task + 1, 0]), task + 1)

                writer.add_scalar('Loss/LTM', LTM_perf[task, 0], task + 1)
                writer.add_scalar('Loss/LTM_Avg', np.mean(LTM_perf[:task + 1, 0]), task + 1)

                writer.add_scalar('Loss/Task_1', tst_after_perf[task, 0, 0], task + 1)

                ## Accuracy
                writer.add_scalar('Accuracy/Train_Before', trn_before_perf[task, 1], task + 1)
                writer.add_scalar('Accuracy/Train_After', trn_after_perf[task, 1], task + 1)

                writer.add_scalar('Accuracy/Train_Before_Avg', np.mean(trn_before_perf[:task + 1, 1]), task + 1)
                writer.add_scalar('Accuracy/Train_After_Avg', np.mean(trn_after_perf[:task + 1, 1]), task + 1)

                writer.add_scalar('Accuracy/CTM', CTM_perf[task, 1], task + 1)
                writer.add_scalar('Accuracy/CTM_Avg', np.mean(CTM_perf[:task + 1, 1]), task + 1)

                writer.add_scalar('Accuracy/LTM', LTM_perf[task, 1], task + 1)
                writer.add_scalar('Accuracy/LTM_Avg', np.mean(LTM_perf[:task + 1, 1]), task + 1)

                writer.add_scalar('Accuracy/Task_1', tst_after_perf[task, 0, 1], task + 1)

            online_mdls[task] = meta_model

            if save_models:
                torch.save(meta_model.state_dict(), '%s/Model/%s/%s/%d_meta_model.pt'
                           % (data_path, data_obj.name, suffix, task + 1))

        if save_performance:
            # Save results
            path_ = '%s/Model/%s/%s/%d' % (data_path, data_obj.name, suffix, n_tasks)
            np.save(path_ + '_tst_before_perf.npy', tst_before_perf)
            np.save(path_ + '_trn_before_perf.npy', trn_before_perf)

            np.save(path_ + '_tst_after_perf.npy', tst_after_perf)
            np.save(path_ + '_trn_after_perf.npy', trn_after_perf)

            np.save(path_ + '_CTM_perf.npy', CTM_perf)
            np.save(path_ + '_LTM_perf.npy', LTM_perf)


    else:
        # Load
        if save_models:
            for task in range(n_tasks):
                meta_model = model_func().to(device)
                meta_model.load_state_dict(torch.load('%s/Model/%s/%s/%d_meta_model.pt'
                                                      % (data_path, data_obj.name, suffix, task + 1)))
                online_mdls[task] = meta_model

        path_ = '%s/Model/%s/%s/%d' % (data_path, data_obj.name, suffix, n_tasks)
        tst_before_perf = np.load(path_ + '_tst_before_perf.npy')
        trn_before_perf = np.load(path_ + '_trn_before_perf.npy')

        tst_after_perf = np.load(path_ + '_tst_after_perf.npy')
        trn_after_perf = np.load(path_ + '_trn_after_perf.npy')

        CTM_perf = np.load(path_ + '_CTM_perf.npy')
        LTM_perf = np.load(path_ + '_LTM_perf.npy')

    return online_mdls, tst_before_perf, trn_before_perf, tst_after_perf, trn_after_perf, CTM_perf, LTM_perf



















### Method


# ── Train ─────────────────────────────────────────────────────────────────────
print('Train MOML')
for K in K_list:
    for num_grad_step in num_grad_step_list:
        for alpha in alpha_list:
            print('K %3d, GS %3d, alpha %f' % (K, num_grad_step, alpha))
            print_per = K // 4 if K > 4 else 1
            _ = train_MOML(
                data_obj=data_obj,
                alpha=alpha,
                learning_rate=learning_rate,
                learning_rate_ft=learning_rate_ft,
                batch_size=batch_size,
                K=K,
                num_grad_step=num_grad_step,
                print_per=print_per,
                weight_decay=weight_decay,
                model_func=model_func,
                init_model=init_model,
                sch_step=sch_step,
                sch_gamma=sch_gamma,
                lr_decay_per_round=lr_decay_per_round,
                save_models=save_models,
                save_performance=save_performance,
                save_tensorboard=save_tensorboard,
                suffix=suffix,
                data_path=data_path,
            )





# ── Sanity check ─────────────────────────────────────────────────────────────
_test = model_func()
_vocab = _test.smi.embedding.emb.num_embeddings
print(f"[check] embedding vocab size = {_vocab}")
assert _vocab == 50265, f"Wrong vocab size {_vocab} — stale .pyc still loaded"
del _test









