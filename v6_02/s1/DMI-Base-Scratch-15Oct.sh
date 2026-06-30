# 1. DMI-Base-Scratch - 7F04DE1E
LINK="https://iitkgpacin-my.sharepoint.com/:u:/g/personal/bishal_santra_iitkgp_ac_in/EaKSsHWJbcVLtyhR1kz1VI4BdO3MuPIRmnrQyWrbG5n0Wg?e=8oj102"
wget "$LINK&download=1" -O model_best_auc.pth
mkdir -p checkpoints/DMI-B-15Oct/
mv model_best_auc.pth checkpoints/DMI-B-15Oct/

# 1
(python -u run_finetune.py -task banking77 -voc roberta -lr 8e-4 -scdl -bs 64 -ep 35 -ckpt checkpoints/DMI-B-15Oct/model_best_auc.pth -t 1 --no_tqdm;
python -u run_finetune.py -task mutual -voc roberta -lr 8e-6 -wtl -scdl -bs 64 -ep 70 -ckpt checkpoints/DMI-B-15Oct/model_best_auc.pth -t 1 --no_tqdm) &

(python -u run_finetune.py -task mutual_plus -voc roberta -lr 5e-6 -wtl -bs 64 -ep 70 -ckpt checkpoints/DMI-B-15Oct/model_best_auc.pth -t 1 --no_tqdm;
python -u run_finetune.py -task e/intent -voc roberta -lr 8e-4 -scdl -bs 256 -ep 32 -ckpt checkpoints/DMI-B-15Oct/model_best_auc.pth -t 1 --no_tqdm)


# 2
python -u run_finetune.py -task swda -voc roberta -lr 1e-3 -scdl -ep 50 -bs 64 -ckpt checkpoints/DMI-B-15Oct/model_best_auc.pth --no_tqdm -t 1

# 3
python -u run_finetune.py -task dd++/full -voc roberta -lr 1.2e-5 -scdl -bs 64 -ep 40 -ckpt checkpoints/DMI-B-15Oct/model_best_auc.pth -t 1 --no_tqdm

# 4
python -u run_finetune.py -task dd++/cross -voc roberta -lr 1.2e-5 -scdl -bs 64 -ep 36 -ckpt checkpoints/DMI-B-15Oct/model_best_auc.pth -t 1 --no_tqdm

# 5
python -u run_finetune.py -task dd++/adv -voc roberta -lr 5e-5 -scdl -bs 64 -ep 36 -ckpt checkpoints/DMI-B-15Oct/model_best_auc.pth -t 1 --no_tqdm

# 6
python -u run_finetune.py -task dd++ -voc roberta -lr 5e-5 -scdl -bs 64 -ep 36 -ckpt checkpoints/DMI-B-15Oct/model_best_auc.pth -t 1 --no_tqdm
