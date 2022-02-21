from tqdm import tqdm

import torch
from datasets import load_metric
from transformers import BatchEncoding

def get_metric_key(task):
    if task == 'cola':
        return 'matthews_correlation'
    elif task == 'stsb':
        return 'spearmanr'
    elif task in ['qqp', 'mrpc']:
        return 'f1'
    else:
        return 'accuracy'

def prepare_metric(task):
    metric_name = task if task != 'mnli-mm' else 'mnli'
    metric = load_metric('glue', metric_name)
    metric_key = get_metric_key(task)
    return metric, metric_key

def train_epoch(model, optimizer, scheduler, train_loader, show_pbar):
    _ = model.train()
    
    losses = 0.
    pbar = tqdm(train_loader, disable= not show_pbar)
    for batch in pbar:
        batch = BatchEncoding(batch).to(model.device)
        out = model(**batch)

        loss = out.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        pbar.set_postfix({'loss': loss.item()})
        losses += loss.item()
        
    return losses
    
        
def predict_epoch(model, dataloader):
    _ = model.eval()
    logits, labels = [], []
    for batch in dataloader:
        batch = BatchEncoding(batch).to(model.device)
        with torch.no_grad():
            out = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
        logits.append(out.logits.cpu())
        labels.append(batch.labels.cpu())
        
    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0)
    preds = logits.argmax(dim=-1)
    return logits, preds, labels