


# from https://github.com/jordiclive/Convert-PolyAI-Torch/blob/master/src/model.py















import sys
sys.path.insert(0, '/content/notebooks_meta/v5_5/s1')

import logging
import random
from collections import OrderedDict

import numpy as np
import pytorch_lightning as pl
import torch

import math


from config import ConveRTModelConfig, ConveRTTrainConfig
from losses import LossFunction

from cog_arch.encoder import FeedForward2, TransformerLayers

import argparse
from sentencepiece import SentencePieceProcessor
from s1.data.dataset import DataModule, RedditData, load_instances_from_reddit_json


logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_subword_params(model):
    """Long winded helper fn to return Subword Embedding Params for clipping, as they are the only parameters that
    are gradient clipped in the paper, only calculated once after model instantiation, but before training"""
    embeds = set()
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            if mn.startswith("transformer_layers.subword_embedding"):
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                embeds.add(fpn)
    param_dict = {pn: p for pn, p in model.named_parameters()}

    return [param_dict[pn] for pn in sorted(list(embeds))], embeds


# todo  need to write own
# lightning optimizer step to include torch.nn.utils.clip_grad_norm_(find_subword_params(model), config.grad_norm_clip),


class SingleContextConvert(pl.LightningModule):
    def __init__(
            self, model_config: ConveRTModelConfig, train_config: ConveRTTrainConfig
    ):
        super().__init__()

        self.model_config = model_config
        self.train_config = train_config
        self.transformer_layers = TransformerLayers(model_config)
        self.ff2_context = FeedForward2(model_config)
        self.ff2_reply = FeedForward2(model_config)
        self.loss_function = LossFunction()

        self.weight_decay = train_config.l2_weight_decay

        self.hparams = self.train_config._field_defaults
        self.hparams.update(self.model_config._field_defaults)
        self.subword_params = None

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )
    def register_subword_params(self):
        self.subword_params = find_subword_params(self)[0]

    def forward(self, x):
        return self.transformer_layers(x)

    def backward(self, trainer, loss, optimizer, optimizer_idx):
        """override hook of lightning as want specific grad norm clip of only subword embedding parameters, after loss.backward()
        but before optimizer step"""
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.subword_params, self.train_config.grad_norm_clip)


    def configure_optimizers(self):
        """
        here I did not implement weight decay on bias and Layernorm layers as is typical in modern  NLP papers.
        I do not think the paper specified params to avoid weight decay on
        :return:
        :rtype:
        """
        # create the optimizer, here I did not implement weight decay on bias and weight as is customary in modern
        # NLP papers.
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [
            p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)
        ]
        params_nodecay = [
            p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)
        ]
        optim_groups = [
            {"params": params_decay, "weight_decay": self.hparams.l2_weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr = self.hparams.learning_rate
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        batch_context = batch.context
        batch_reply = batch.reply
        rx = self(batch_context)
        ry = self(batch_reply)
        hx = self.ff2_context(rx, batch_context.attention_mask)
        hy = self.ff2_reply(ry, batch_reply.attention_mask)

        loss = self.loss_function(hx, hy)

        tqdm_dict = {"train_loss": loss}
        output = OrderedDict(
            {"loss": loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
        )
        # result = pl.TrainResult(minimize=loss, checkpoint_on=loss)
        # result.log("train_loss", loss)
        return output

    def validation_step(self, batch, batch_idx):
        output = self.training_step(batch, batch_idx)
        val_output = {"val_loss": output["loss"]}
        return val_output
    







# Really not clear from paper, paper starts talking about cosine annealing when discussing
# the cosine similarity measure. Needs clarification
# I assume 0.1 to 1 linear warm up over first 10000 batches  then annealed to 0.001


class LearningRateDecayCallback(pl.Callback):
    def __init__(
        self,
        config,
        lr_decay=True,
    ):
        super().__init__()
        self.lr_warmup_end = config.lr_warmup_end
        self.lr_warmup_start = config.lr_warmup_start
        self.learning_rate = config.learning_rate
        self.warmup_batch = config.warmup_batch
        self.final_batch = config.final_batch

        self.lr_decay = lr_decay


    def on_train_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        """

        :param trainer:
        :type trainer:
        :param pl_module:
        :type pl_module:
        :param batch:
        :type batch:
        :param batch_idx:
        :type batch_idx:
        :param dataloader_idx:
        :type dataloader_idx:
        """
        optimizer = trainer.optimizers[0]

        if self.lr_decay:
            if batch_idx < self.warmup_batch:
                # linear warmup, in paper: start from 0.1 to 1 over 10000 batches
                lr_mult = float(batch_idx) / float(max(1, self.warmup_batch))
                lr = self.lr_warmup_start + lr_mult * (
                    self.lr_warmup_end - self.lr_warmup_start
                )

            else:
                # Cosine learning rate decay
                progress = float(batch_idx - self.warmup_batch) / float(
                    max(1, self.final_batch - self.warmup_batch)
                )

                lr = max(
                    self.learning_rate
                    + 0.5
                    * (1.0 + math.cos(math.pi * progress))
                    * (self.lr_warmup_end - self.learning_rate),
                    self.learning_rate,
                )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr


def _parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    #parser.add_argument("--gpus", type = int, default = 1)
    #parser.add_argument("--precision", type = int, default = 16)
    parser.add_argument("--progress_bar_refresh_rate", type = int, default = 1)
    parser.add_argument("--row_log_interval", type = int, default = 1)

    args = parser.parse_args()

    return args


def main(**kwargs):
    set_seed(1)
    train_config = ConveRTTrainConfig()
    model_config = ConveRTModelConfig()
    tokenizer = SentencePieceProcessor()
    args = _parse_args()
    tokenizer.Load(train_config.sp_model_path)
    train_instances = load_instances_from_reddit_json(train_config.dataset_path)
    RD = RedditData(train_instances, tokenizer, 60)
    dm = DataModule()
    train_loader = dm.train_dataloader(RD)
    model = SingleContextConvert(model_config, train_config)
    lr_decay = LearningRateDecayCallback(train_config)
    model.register_subword_params()

    trainer = (
        pl.Trainer.from_argparse_args(args, callbacks = [lr_decay],**kwargs)
    )  # ,checkpoint_callback = checkpoint_callback)  # ,resume_from_checkpoint=)
    trainer.fit(model, train_dataloader = train_loader, val_dataloaders = train_loader)


if __name__ == "__main__":
    main(fast_dev_run=True)