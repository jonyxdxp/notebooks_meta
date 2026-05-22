
from v6.s2.config  import CKPT_DIR, DEVICE, EPOCHS, LR, WEIGHT_DECAY
from v6.s2.data.data import make_dataloaders
from v6.s2.model   import DialogueJEPAPredictor
from v6.s2.train   import train
import os

os.makedirs(CKPT_DIR, exist_ok=True)

if __name__ == "__main__":
    train_loader, valid_loader = make_dataloaders()
    predictor = DialogueJEPAPredictor().to(DEVICE)
    train(predictor, train_loader, valid_loader,
          save_path=f"{CKPT_DIR}/jepa_predictor_best.pth")
