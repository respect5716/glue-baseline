import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer


task2column = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"), # matched
    "mnli-mm": ("premise", "hypothesis"), # mismatched
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def prepare_dataset(task):
    dataset_name = task if task != 'mnli-mm' else 'mnli'
    dataset = load_dataset("glue", dataset_name)
    
    train_dataset = dataset['train']
    if task == 'mnli':
        valid_dataset = dataset['validation_matched']
        test_dataset = dataset['test_matched']
    elif task == 'mnli-mm':
        valid_dataset = dataset['validation_mismatched']
        test_dataset = dataset['test_mismatched']
    else:
        valid_dataset = dataset['validation']
        test_dataset = dataset['test']
    return train_dataset, valid_dataset, test_dataset


def transform(batch, config, tokenizer):
    batch['labels'] = batch['label']
    col1, col2 = task2column[config.task]
    if col2 is None:
        return tokenizer(batch[col1], max_length=config.max_length, padding='max_length', truncation=True)
    else:
        return tokenizer(batch[col1], batch[col2], max_length=config.max_length, padding='max_length', truncation=True)


def prepare_dataloader(config):
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    except:
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(config.working_dir, 'models', config.model_name_or_path))
    
    train_dataset, valid_dataset, test_dataset = prepare_dataset(config.task)
    train_dataset = train_dataset.map(lambda x: transform(x, config, tokenizer), batched=True)
    valid_dataset = valid_dataset.map(lambda x: transform(x, config, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: transform(x, config, tokenizer), batched=True)

    train_dataset.set_format('torch', columns = ['input_ids', 'attention_mask', 'labels'])
    valid_dataset.set_format('torch', columns = ['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format('torch', columns = ['input_ids', 'attention_mask', 'labels'])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return tokenizer, train_loader, valid_loader, test_loader