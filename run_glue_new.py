

import hydra
from omegaconf import OmegaConf

import numpy as np

from datasets import load_dataset, load_metric

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PretrainedConfig,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed
)


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


@hydra.main(config_path='.', config_name='config')
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.trainer.seed)

    task = cfg.data.task_name
    raw_datasets = load_dataset('glue', task)

    is_regression = task == 'stsb'
    if is_regression:
        num_labels = 1
    else:
        label_list = raw_datasets['train'].features['label'].names
        num_labels = len(label_list)

    print(task, num_labels)

    config = AutoConfig.from_pretrained(
        cfg.model_name_or_path,
        num_labels = num_labels,
        finetuning_task = task
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        use_fast_tokenizer = True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name_or_path,
        config = config
    )

    sent1_key, sent2_key = task_to_keys[task]
    padding = 'max_length' if cfg.data.pad_to_max_length else False

    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and not is_regression
    ):
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            print(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    
    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}


    max_seq_length = min(cfg.data.max_seq_length, tokenizer.model_max_length)


    def preprocess_fn(examples):
        args = (
            (examples[sent1_key],) if sent2_key is None else (examples[sent1_key], examples[sent2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)
        return result

    with training_args.main_process_first(desc='dataset map pre-processing'):
        raw_datasets = raw_datasets.map(
            preprocess_fn,
            batched = True,
            desc = 'Running tokenizer on dataset'
        )

    train_dataset = raw_datasets['train']
    eval_dataset = raw_datasets["validation_matched" if task == "mnli" else "validation"]
    predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]

    metric = load_metric('glue', task)
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result
       

    if cfg.data.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset if training_args.do_train else None,
        eval_dataset = eval_dataset if training_args.do_eval else None,
        compute_metrics = compute_metrics,
        tokenizer = tokenizer,
        data_collator = data_collator
    )

    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics['train_samples'] = len(train_dataset)
        trainer.save_model()
        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()

if __name__ == '__main__':
    main()
