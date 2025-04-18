import os
import sys
from typing import List
import argparse
import json
def arg_para():
    parser = argparse.ArgumentParser(description='tallrec')
    parser.add_argument('--data_name', choices=['amazon_book', 'ml-1m'], type=str, default='ml-1m')
    parser.add_argument('--train_start_num', type=int, default=0)
    parser.add_argument('--train_total_num', type=int, default=8)   
    parser.add_argument('--valid_num',type=int,default=7)
    parser.add_argument('--tune_num',type=int,default=7)
    parser.add_argument('--test_num',type=int,default=8)
    parser.add_argument('--cuda',type = str,default = '0')
    parser.add_argument('--sample', type=int, default=-1)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--mode', choices=["tune", "full"], type=str, default="full")
    parser.add_argument('--is_resume', type=bool, default=False)
    parser.add_argument('--eval_step', type=int, default=6)
    return parser.parse_args()
args = arg_para()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
import numpy as np 
import fire
import torch
import transformers
from datasets import load_dataset
from transformers import EarlyStoppingCallback
from transformers.utils import logging
from utils import *
from peft import  *
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: F402
from config import llm_path, data_dir

logger = logging.get_logger(__name__)


def train(
    # model/data params
    base_model: str = llm_path,  # the only required argument
    val_data_path: str = f'{data_dir}/{args.data_name}/val{args.valid_num}.json',
    test_data_path: str = f'{data_dir}/{args.data_name}/test{args.test_num}.json',
    output_dir: str = args.output_dir,
    sample: int = args.sample,
    seed: int = args.seed,
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 32,
    num_epochs: int = args.num_epochs,
    learning_rate: float = 1e-3,
    cutoff_len: int = 512,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter

):
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"val_data_path: {val_data_path}\n"
        f"sample: {sample}\n"
        f"seed: {seed}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    base_model = args.base_model
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = False # world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    seed=args.seed
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    create_parent_folders(args.output_dir)
    create_parent_folders(args.checkpoint_dir)
    try:
        model = prepare_model_for_int8_training(model)
    except:
        model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
     
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference



    if args.mode == 'full':
        full_train_list = list(range(args.train_start_num, args.train_start_num + args.train_total_num))
        train_data_paths = [f'{data_dir}/{args.data_name}/train{num}.json' for num in full_train_list]
        val_data_paths = [f'{data_dir}/{args.data_name}/val{num}.json' for num in full_train_list if num != full_train_list[-1]]
        train_data_paths.extend(val_data_paths)
        train_data = load_dataset("json", data_files=train_data_paths)

    elif args.mode == 'tune':
        try:
            train_data_paths = [
                f'{data_dir}/{args.data_name}/train{args.tune_num}.json',
                f'{data_dir}/{args.data_name}/val{args.tune_num-1}.json'
            ]
        except:
            train_data_paths = [f'{data_dir}/{args.data_name}/train{args.tune_num}.json']
        
        train_data = load_dataset("json", data_files=train_data_paths)
        val_data_path = f'{data_dir}/{args.data_name}/val{args.tune_num}.json'

    else:
        raise Exception("mode not in tune or full")


    if val_data_path.endswith(".json"): 
        val_data = load_dataset("json", data_files=val_data_path)
    else:
        val_data = load_dataset(val_data_path)

    if test_data_path.endswith(".json"):  
        test_data = load_dataset("json", data_files=test_data_path)
    else:
        test_data = load_dataset(test_data_path)
        
    if args.mode == "tune":
        if args.checkpoint_dir:
            best_model_dir = find_min_checkpoint_dir(args.checkpoint_dir)
        else:
            raise Exception('the checkpoint_dir to load is none')

        if best_model_dir and args.is_resume == False:
            # Check the available weights and load them
            checkpoint_name = os.path.join(
                best_model_dir, "pytorch_model.bin"
            )  # Full checkpoint
            if not os.path.exists(checkpoint_name):
                checkpoint_name = os.path.join(
                    best_model_dir, "adapter_model.bin"
                )  # only LoRA model - LoRA config above has to fit
                resume_from_checkpoint = (
                    False  # So the trainer won't try loading its state
                )
            if os.path.exists(checkpoint_name):
                print(f"Restarting from {checkpoint_name}")
                adapters_weights = torch.load(checkpoint_name)
                set_peft_model_state_dict(model, adapters_weights)
            else:
                print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt
    train_data["train"] = train_data["train"].shuffle(seed=seed).select(range(sample)) if sample > -1 else train_data["train"].shuffle(seed=seed)
    train_data["train"] = train_data["train"].shuffle(seed=seed)
    train_data = (train_data["train"].map(generate_and_tokenize_prompt))
    val_data = (val_data["train"].map(generate_and_tokenize_prompt))
    test_data = (test_data["train"].map(generate_and_tokenize_prompt))

    if args.data_name != 'ml-1m':
        train_data = select_data(train_data)
        val_data = select_data(val_data)
        test_data = select_data(test_data)
        
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = False # True
        model.model_parallel = False # True
    model.config.use_cache = False



    os.environ["WANDB_DISABLED"] = "true"

    if args.mode == 'full':
        eval_step = 23 
    else:
        try:
            eval_step = args.eval_step
        except:
            eval_step = 6

    begin_val_auc = 0
    if args.mode == 'tune':
        trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy='steps', # "steps",
            save_strategy="steps",
            eval_steps=eval_step,
            save_steps=eval_step,
            output_dir=output_dir,
            save_total_limit=5,
            load_best_model_at_end=True,
            metric_for_best_model="eval_auc",
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=args.patience)]
        )

        val_save_dir = os.path.join(output_dir, 'ckpt-old')
        val_results = trainer.evaluate(val_data)
        if not os.path.exists(val_save_dir):
            os.makedirs(val_save_dir)
        files = os.listdir(best_model_dir)
        for file_name in files:
            source_file = os.path.join(best_model_dir, file_name)
            target_file = os.path.join(val_save_dir, file_name)
            with open(source_file, 'rb') as fsrc, open(target_file, 'wb') as fdst:
                fdst.write(fsrc.read())
            print(f"File {source_file} copied to {target_file}")
        test_result_path = os.path.join(val_save_dir, 'val_result.json')
        with open(test_result_path, 'w') as json_file:
            json.dump(val_results, json_file, indent=4)
        test_result_path = os.path.join(output_dir, 'val_result.json')
        with open(test_result_path, 'w') as json_file:
            json.dump(val_results, json_file, indent=4)
        begin_val_auc = val_results['eval_auc']
        del trainer


    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy='steps', # "steps",
            save_strategy="steps",
            eval_steps=eval_step,
            save_steps=eval_step,
            # eval_delay=,
            output_dir=output_dir,
            save_total_limit=5,
            load_best_model_at_end=True,
            metric_for_best_model="eval_auc",
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to=None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=args.patience)]
    )

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    if args.is_resume:
        resume_from_checkpoint = find_max_checkpoint_dir(args.output_dir)
    else:
        resume_from_checkpoint = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
    model.save_pretrained(output_dir)
    best_model_dir = trainer.state.best_model_checkpoint
    
    # empty cache for test
    if args.mode == 'tune':
        best_metric = trainer.state.best_metric
        if best_metric >= begin_val_auc:
            folder_path = val_save_dir
            if os.path.exists(folder_path):
                for root, dirs, files in os.walk(folder_path, topdown=False):
                    for name in files:
                        file_path = os.path.join(root, name)
                        os.remove(file_path)  
                    for name in dirs:
                        dir_path = os.path.join(root, name)
                        os.rmdir(dir_path) 
                os.rmdir(folder_path) 
                print(f"Folder {folder_path} and its contents have been successfully deleted.")
            else:
                print(f"Folder {folder_path} does not exist.")
        else:
            print('ckpt-old has been saved')



if __name__ == "__main__":
    fire.Fire(train)
