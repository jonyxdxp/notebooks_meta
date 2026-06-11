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

from data.dataset import *
from cog_arch.encoder import *
from utils_general import *

# ── Paths ────────────────────────────────────────────────────────────────────
data_path      = '/content/drive/MyDrive/data/cache'          # directory, not a file
dialog_train   = '/content/Discourse-Mutual-Information-DMI-main/data/dailydialog/dialogues_train.txt'
dialog_valid   = '/content/Discourse-Mutual-Information-DMI-main/data/dailydialog/dialogues_valid.txt'
dialog_test    = '/content/Discourse-Mutual-Information-DMI-main/data/dailydialog/dialogues_test.txt'







# --- Configuración y Ejecución ---
if __name__ == '__main__':
    # Parámetros del modelo DMI
    VOCAB_SIZE = 30522 # Ejemplo para BertTokenizerFast, ajustar si se usa otro
    D_MODEL = 256 # Tamaño del embedding
    PROJECTION_SIZE = 256
    ENCODER_LAYERS = 2 # Capas del Transformer
    ENCODER_HEADS = 4 # Cabezas de atención
    DIM_FEEDFORWARD = 512

    # Parámetros de MOML
    ALPHA = 0.1
    LEARNING_RATE = 0.001 # Meta-learning rate
    LEARNING_RATE_FT = 0.01 # Inner loop learning rate
    BATCH_SIZE = 16 # Tamaño de batch para support/query sets
    K_META_UPDATES = 5 # Número de meta-actualizaciones por tarea (K en MOML)
    NUM_GRAD_STEP_INNER = 1 # Número de pasos de gradiente en el inner loop
    PRINT_PER = 1 # Frecuencia de impresión en el inner loop
    WEIGHT_DECAY = 1e-4
    SCH_STEP = 1
    SCH_GAMMA = 1
    LR_DECAY_PER_ROUND = 1 # No decay por defecto

    # Parámetros del Dataset
    NUM_TASKS = 50 # Número de tareas para MOML
    SAMPLES_PER_TASK = 100 # Muestras por tarea (se dividirán en support/query)
    MAX_CTX_LEN = 128
    MAX_RESP_LEN = 64
    
    # Ruta al dataset DailyDialog
    DAILYDIALOG_PATH = os.path.join(repo_dmi_dir, 'data', 'dailydialog', 'dialogues_train.txt')

    # Inicializar Tokenizador (usaremos BertTokenizerFast como ejemplo)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    # Añadir tokens especiales si es necesario, como '__eou__'
    if '__eou__' not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'additional_special_tokens': ['__eou__']})
        # Actualizar VOCAB_SIZE si se añaden tokens
        VOCAB_SIZE = len(tokenizer)

    # Crear el objeto de dataset para MOML
    meta_dataset_obj = MetaDialogDataset(
        DAILYDIALOG_PATH, tokenizer, NUM_TASKS, SAMPLES_PER_TASK,
        max_ctx_len=MAX_CTX_LEN, max_resp_len=MAX_RESP_LEN
    )

    # Definir la función que devuelve el modelo DMI envuelto
    model_func_dmi = lambda: MetaDMIModel(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, projection_size=PROJECTION_SIZE,
        encoder_layers=ENCODER_LAYERS, encoder_heads=ENCODER_HEADS, dim_feedforward=DIM_FEEDFORWARD
    )

    # Inicializar el modelo para MOML
    init_dmi_model = model_func_dmi()

    # Ejecutar el entrenamiento MOML-DMI
    final_meta_model, all_losses = train_MOML_DMI(
        data_obj=meta_dataset_obj, alpha=ALPHA, learning_rate=LEARNING_RATE,
        learning_rate_ft=LEARNING_RATE_FT, batch_size=BATCH_SIZE, K=K_META_UPDATES,
        num_grad_step=NUM_GRAD_STEP_INNER, print_per=PRINT_PER, weight_decay=WEIGHT_DECAY,
        model_func=model_func_dmi, init_model=init_dmi_model, sch_step=SCH_STEP,
        sch_gamma=SCH_GAMMA, lr_decay_per_round=LR_DECAY_PER_ROUND,
        save_models=False, save_performance=False, save_tensorboard=False
    )

    print("\nMeta-entrenamiento completado. El modelo final está en 'final_meta_model'.")
    print("Pérdidas por tarea:", all_losses)

    # Opcional: Guardar el modelo final
    # torch.save(final_meta_model.state_dict(), '/content/drive/MyDrive/final_dmi_moml_encoder.pt')
    # print("Modelo final guardado en /content/drive/MyDrive/final_dmi_moml_encoder.pt")














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
        torch.cuda.empty_cache()
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
            meta_model = train_MOML_DMI_model(
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

            torch.cuda.empty_cache()

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
            torch.cuda.empty_cache()
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
 


















if __name__ == "__main__":
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
    
    
    
    
    
    
    
    
    


# ── Added for MOML+DMI notebook ───────────────────────────────────────────────
import os, random, math
from dataclasses import dataclass, asdict

import torch
import higher
from tqdm.auto import tqdm


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _flat(model, device):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).to(device)


@dataclass
class MOMLConfig:
    # paths
    data_root:       str   = '/content/Discourse-Mutual-Information-DMI-main/data/dailydialog'
    output_path:     str   = '/content/drive/MyDrive/dmi_moml_ckpts'
    # tokeniser
    max_ctx_len:     int   = 150
    max_resp_len:    int   = 60
    # model
    d_model:         int   = 256
    dim_feedforward: int   = 1024
    encoder_layers:  int   = 4
    encoder_heads:   int   = 4
    projection_size: int   = 256
    symmetric_loss:  bool  = False
    estimator:       str   = 'infonce'
    # MOML
    alpha:           float = 5.0
    n_tasks:         int   = 300
    pairs_per_task:  int   = 240
    # inner loop
    num_inner_steps: int   = 3
    lr_inner:        float = 5e-4
    # outer loop
    lr_outer:        float = 1e-4
    batch_size:      int   = 32
    grad_clip:       float = 1.0
    # misc
    seed:            int   = 42
    log_every:       int   = 10
    val_every:       int   = 50
    val_batches:     int   = 40
    save_best:       bool  = True


def _moml_step(meta_model, spt_pairs, qry_pairs, collate_fn,
               omega, lam, cfg, device, outer_opt):
    """One MOML task: inner adapt on support → outer InfoNCE+proximal on query."""
    from cog_arch.encoder import compute_loss

    meta_model.train()
    bs = cfg.batch_size

    def batches(pairs):
        random.shuffle(pairs)
        return [pairs[i:i+bs] for i in range(0, len(pairs), bs)
                if len(pairs[i:i+bs]) >= 8]

    spt_b, qry_b = batches(spt_pairs), batches(qry_pairs)
    if not spt_b or not qry_b:
        return None

    inner_opt = torch.optim.SGD(meta_model.parameters(), lr=cfg.lr_inner)
    outer_opt.zero_grad()
    total_ql, total_mi, n = torch.tensor(0., device=device), 0., 0

    with higher.innerloop_ctx(meta_model, inner_opt,
                              copy_initial_weights=False) as (fnet, diffopt):
        for _ in range(cfg.num_inner_steps):
            for b in spt_b:
                ctx, rsp, mc, mr = collate_fn(b)
                ctx, rsp, mc, mr = ctx.to(device), rsp.to(device), mc.to(device), mr.to(device)
                c_t, z_t = fnet(ctx, rsp, mc, mr)
                _, loss, _ = compute_loss(c_t, z_t, cfg.estimator, cfg.symmetric_loss)
                diffopt.step(loss)

        for b in qry_b:
            ctx, rsp, mc, mr = collate_fn(b)
            ctx, rsp, mc, mr = ctx.to(device), rsp.to(device), mc.to(device), mr.to(device)
            c_t, z_t = fnet(ctx, rsp, mc, mr)
            _, ql, mi = compute_loss(c_t, z_t, cfg.estimator, cfg.symmetric_loss)
            total_ql = total_ql + ql
            total_mi += mi
            n += 1

        if n == 0:
            return None
        avg_ql = total_ql / n
        theta   = torch.cat([p.reshape(-1) for p in meta_model.parameters()])
        total_loss = (avg_ql
                      - torch.sum(theta * lam)
                      - cfg.alpha * torch.sum(theta * omega)
                      + cfg.alpha / 2 * torch.sum(theta * theta))
        total_loss.backward()

    if cfg.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(meta_model.parameters(), cfg.grad_clip)
    outer_opt.step()
    return {'total_loss': total_loss.item(),
            'qry_loss': avg_ql.item(), 'qry_mi': total_mi / n}


@torch.no_grad()
def validate(meta_model, valid_ds, cfg, device):
    from cog_arch.encoder import compute_loss
    meta_model.eval()
    pairs = list(valid_ds.cr_pairs)
    random.shuffle(pairs)
    bs = cfg.batch_size
    batches = [pairs[i:i+bs] for i in range(0, len(pairs), bs)
               if len(pairs[i:i+bs]) >= 8][:cfg.val_batches]
    tl, tm, n = 0., 0., 0
    for b in batches:
        ctx, rsp, mc, mr = valid_ds.collate(b)
        ctx, rsp, mc, mr = ctx.to(device), rsp.to(device), mc.to(device), mr.to(device)
        c_t, z_t = meta_model(ctx, rsp, mc, mr)
        _, loss, mi = compute_loss(c_t, z_t, cfg.estimator, cfg.symmetric_loss)
        tl += loss.item(); tm += mi; n += 1
    meta_model.train()
    return {'val_loss': tl/n if n else 0., 'val_mi': tm/n if n else 0.}


def train_moml_dmi(cfg: MOMLConfig, train_ds, valid_ds,
                   vocab_size: int, device: str):
    from cog_arch.encoder import DMIScratchEncoder
    from data.dataset import OnlineTaskStream

    torch.manual_seed(cfg.seed); random.seed(cfg.seed)
    os.makedirs(cfg.output_path, exist_ok=True)

    meta_model = DMIScratchEncoder(
        vocab_size=vocab_size, d_model=cfg.d_model,
        projection_size=cfg.projection_size,
        encoder_layers=cfg.encoder_layers, encoder_heads=cfg.encoder_heads,
        dim_feedforward=cfg.dim_feedforward,
        symmetric_loss=cfg.symmetric_loss,
    ).to(device)
    print(f'Parameters: {count_params(meta_model)/1e6:.2f}M')

    omega = _flat(meta_model, device)
    lam   = torch.zeros_like(omega)
    outer_opt = torch.optim.Adam(meta_model.parameters(), lr=cfg.lr_outer)

    stream = OnlineTaskStream(train_ds.cr_pairs, cfg.pairs_per_task,
                              cfg.n_tasks, cfg.seed)
    history = {'tasks': [], 'qry_loss': [], 'qry_mi': [],
               'val_tasks': [], 'val_mi': []}
    best_val_mi = -float('inf')

    for t, (spt, qry) in enumerate(tqdm(stream, total=cfg.n_tasks, desc='Tasks')):
        m = _moml_step(meta_model, spt, qry, train_ds.collate,
                       omega, lam, cfg, device, outer_opt)
        if m is None:
            continue

        curr = _flat(meta_model, device)
        lam   = lam   - cfg.alpha * (curr - omega)
        omega = 0.5 * (curr + omega) - (0.5 / cfg.alpha) * lam

        if (t + 1) % cfg.log_every == 0:
            history['tasks'].append(t + 1)
            history['qry_loss'].append(m['qry_loss'])
            history['qry_mi'].append(m['qry_mi'])
            tqdm.write(f"[{t+1:4d}] qry_loss={m['qry_loss']:.4f}  MI≈{m['qry_mi']:.3f}")

        if (t + 1) % cfg.val_every == 0:
            vm = validate(meta_model, valid_ds, cfg, device)
            history['val_tasks'].append(t + 1)
            history['val_mi'].append(vm['val_mi'])
            tqdm.write(f"  ▶ val_MI={vm['val_mi']:.4f}")
            if cfg.save_best and vm['val_mi'] > best_val_mi:
                best_val_mi = vm['val_mi']
                torch.save({'task': t+1,
                            'model_state_dict': meta_model.state_dict(),
                            'omega': omega.cpu(), 'lambda': lam.cpu(),
                            'val_mi': best_val_mi, 'cfg': asdict(cfg),
                            'vocab_size': vocab_size},
                           os.path.join(cfg.output_path, 'dmi_moml_best.pt'))
                tqdm.write(f"  ✓ saved (val_MI={best_val_mi:.4f})")

    torch.save({'task': cfg.n_tasks, 'model_state_dict': meta_model.state_dict(),
                'omega': omega.cpu(), 'lambda': lam.cpu(),
                'cfg': asdict(cfg), 'vocab_size': vocab_size},
               os.path.join(cfg.output_path, 'dmi_moml_final.pt'))
    return meta_model, history


def load_checkpoint(path: str, device: str):
    from cog_arch.encoder import DMIScratchEncoder
    ckpt = torch.load(path, map_location=device)
    c    = ckpt['cfg']
    model = DMIScratchEncoder(
        vocab_size=ckpt['vocab_size'], d_model=c['d_model'],
        projection_size=c['projection_size'],
        encoder_layers=c['encoder_layers'], encoder_heads=c['encoder_heads'],
        dim_feedforward=c['dim_feedforward'],
        symmetric_loss=c['symmetric_loss'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    return model, ckpt['omega'].to(device), ckpt['lambda'].to(device), c, ckpt['task']