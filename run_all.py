import os
import json

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--tasks', type=str, default='all')
parser.add_argument('--output_dir', type=str, default='glue_output')
args = parser.parse_args()


def make_command(model_name_or_path, task_name, output_dir):
    output_dir = os.path.join(output_dir, model_name_or_path, task_name)

    cmd = 'python run_glue.py'
    cmd += f' --model_name_or_path {model_name_or_path}'
    cmd += f' --task_name {task_name}'
    cmd += f' --do_train'
    cmd += f' --do_eval'
    cmd += f' --max_seq_length 128'
    cmd += f' --per_device_train_batch_size 32'
    cmd += f' --learning_rate 2e-5'
    cmd += f' --num_train_epochs 3'
    cmd += f' --save_strategy no'
    cmd += f' --warmup_ratio 0.1'
    cmd += f' --weight_decay 0.01'
    cmd += f' --output_dir {output_dir}'
    
    return cmd


def main(args):
    if args.tasks.lower() == 'all':
        tasks = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']
    
    elif args.tasks.lower() == 'fast':
        tasks = ['cola', 'mrpc', 'rte', 'stsb', 'wnli']
    
    else:
        tasks = [t.lower() for t in args.tasks.split(',')]

    for task in tasks:
        command = make_command(args.model, task, args.output_dir)
        os.system(command)

if __name__ == '__main__':
    main(args)