import os
import torch
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler

def get_num_labels(task):
    if task.startswith('mnli'):
        return 3
    elif task == 'stsb':
        return 1
    else:
        return 2


def get_param_groups(model, weight_decay):
    no_decay = ["bias", "bn", "ln", "norm"]
    param_groups = [
        {
            # apply weight decay
            "params": [p for n, p in model.named_parameters() if not any(nd in n.lower() for nd in no_decay)],
            "weight_decay": weight_decay
        },
        {
            # not apply weight decay
            "params": [p for n, p in model.named_parameters() if any(nd in n.lower() for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return param_groups


def prepare_model(config, num_training_steps):
    num_labels = get_num_labels(config.task)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(config.model_name_or_path, num_labels=num_labels)
    except:
        model = AutoModelForSequenceClassification.from_pretrained(os.path.join(config.working_dir, 'models', config.model_name_or_path), num_labels=num_labels)
    
    weight_decay = config.get('weight_decay', 0.)
    params = get_param_groups(model, weight_decay)
    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)

    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    if config.scheduler:
        scheduler = get_scheduler(config.scheduler, optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    else:
        scheduler = None
    return model, optimizer, scheduler