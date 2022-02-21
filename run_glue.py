import os
import json
import hydra
import wandb
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.lite import LightningLite

from transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers import BatchEncoding, DataCollatorForLanguageModeling

from src.data import prepare_dataloader
from src.model import prepare_model
from src.training import prepare_metric, train_epoch, predict_epoch

GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

class Lite(LightningLite):
    def run(self, config):
        if self.is_global_zero:
            print(OmegaConf.to_yaml(config))

        tokenizer, train_loader, valid_loader, test_loader = prepare_dataloader(config)
        if self.is_global_zero:
            batch = next(iter(train_loader))
            for k, v in batch.items():
                print(k, v.size())

        num_training_steps = len(train_loader) * config.num_epochs
        model, optimizer, scheduler = prepare_model(config, num_training_steps)
        _ = self.setup(model, optimizer)
        metric, metric_key = prepare_metric(config.task)

        if self.is_global_zero and not config.debug:
            wandb.init(project=f'glue-{config.task}', config=OmegaConf.to_container(config))
            # wandb.watch(model, log='gradients', log_freq=10)

        logs, best_score, best_log = [], 0., None
        for ep in range(config.num_epochs):
            loss = train_epoch(model, optimizer, scheduler, train_loader, show_pbar=self.is_global_zero)
            logits, preds, labels = predict_epoch(model, valid_loader)

            if config.task in ['stsb']:
                # regression
                res = metric.compute(predictions=logits.squeeze().numpy(), references=labels.numpy())
            else:
                # classification
                res = metric.compute(predictions=preds.numpy(), references=labels.numpy())
            log = {'ep': ep, 'loss': loss, **res}
            logs.append(log)
                
            score = log[metric_key]
            if self.is_global_zero and score > best_score:
                model.save_pretrained(os.path.join(config.save_dir, 'transformers'))
                print(f'best score {best_score:.3f} -> {score:.3f}')
                best_score, best_log = score, log

            if config.debug:
                print(log)
                break

            if self.is_global_zero:
                wandb.log(log)
                print(log)

        if self.is_global_zero:
            for k, v in best_log.items():
                wandb.summary[f'best_{k}'] = v
            with open(os.path.join(config.save_dir, 'logs.json'), 'w') as f:
                json.dump(logs, f)
            with open(os.path.join(config.save_dir, 'best.json'), 'w') as f:
                json.dump(best_log, f)
            if not config.debug:
                wandb.finish()


@hydra.main(config_path='.', config_name='config')
def main(config: DictConfig):
    Lite(**config.lite).run(config)


if __name__ == '__main__':
    main()