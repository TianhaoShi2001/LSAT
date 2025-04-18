import os
import sys
from typing import List
import argparse
import re
import pandas as pd
import json

def arg_para():
    parser = argparse.ArgumentParser(description='tallrec')
    parser.add_argument('--data_name', choices=['amazon_book', 'ml-1m'], type=str, default='ml-1m')
    parser.add_argument('--train_start_num', type=int, default=0)
    parser.add_argument('--n', type=int, default=8)       # long-term lora
    parser.add_argument('--m', type=int, default=8)       # short-term lora
    parser.add_argument('--valid_num',type=int,default=7)
    parser.add_argument('--tune_num',type=int,default=7)
    parser.add_argument('--test_num',type=int,default=8)
    parser.add_argument('--cuda',type = str,default = '0')
    parser.add_argument('--sample', type=int, default=-1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--save_name', type=str, default=None)
    parser.add_argument('--mode', choices=["tune", "full"], type=str, default="full")
    return parser.parse_args()
args = arg_para()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
import numpy as np 
import fire
import torch
import transformers
from datasets import load_dataset, Dataset
from transformers import EarlyStoppingCallback
from utils import *
from config import llm_path, data_dir
from peft import *
from transformers import LlamaForCausalLM, LlamaTokenizer 


def train(
    # model/data params
    base_model: str = llm_path,  # the only required argument
    val_data_path: str = f'{data_dir}/{args.data_name}/val{args.valid_num}.json',
    test_data_path: str = f'{data_dir}/{args.data_name}/test{args.test_num}.json',
    sample: int = args.sample,
    seed: int = args.seed,
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 32,
    num_epochs: int = args.num_epochs,
    learning_rate: float = 1e-4,
    cutoff_len: int = 1024,
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
        # f"train_data_path: {train_data_path}\n"
        f"val_data_path: {val_data_path}\n"
        f"sample: {sample}\n"
        f"seed: {seed}\n"
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
    base_model = "decapoda-research/llama-7b-hf"
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
        # load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )


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

    n_adapter_dir = f'workspace/ckpt/{args.data_name}/retrain_trstart0_trtotal_{args.n}_val{args.n-1}_tes{args.n}'
    m_adapter_dir = f'workspace/ckpt/{args.data_name}/retrain_trstart{args.m}_trtotal_1_val{args.m}_tes{args.m+1}'
    n_adapter = torch.load(os.path.join(find_min_checkpoint_dir(n_adapter_dir), "adapter_model.bin"))
    m_adapter = torch.load(os.path.join(find_min_checkpoint_dir(m_adapter_dir), "adapter_model.bin"))

    for weight1 in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        ensemble_adapter = {}
        weight2 = 1 - weight1
        for key in n_adapter:
            ensemble_adapter[key] = weight1 * n_adapter[key] + weight2 * m_adapter[key]
        set_peft_model_state_dict(model,ensemble_adapter)

    model.print_trainable_parameters()  
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = False # True
        model.model_parallel = False # True
    model.config.use_cache = False
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)


    os.environ["WANDB_DISABLED"] = "true"
    eval_step = 23

    create_parent_folders(args.save_dir)
    trainer = transformers.Trainer(
        model=model,
        train_dataset=None,
        eval_dataset=None,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size = 32,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
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
            output_dir=args.save_dir,
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

    
    if args.data_name == 'ml-1m':
        # 修改文件路径
        file_path = f'{data_dir}/ml-1m/movies.dat'
        with open(file_path, 'r', encoding='ISO-8859-1') as f:
            movies = f.readlines()
        # 处理数据字段
        movie_names = [_.split('::')[1] for _ in movies]  # Movie names
        movie_ids = [_.split('::')[0] for _ in movies]     # Movie IDs
        movie_id_to_name = {}
        for movie_id, movie_name in zip(movie_ids, movie_names):
            movie_id_to_name[int(movie_id)] = movie_name

    def csv_to_json(data):
        part_data = data.copy()
        json_list = []
        # print(part_data.max())
        part_data.sort_values(by = ['user_id', 'timestamp'], ascending= True,inplace=True)
        for index, row in part_data.iterrows():
            row['history_movie_id'] = eval(row['history_movie_id'])
            row['history_rating'] = eval(row['history_rating'])
            if 'amazon_book' in args.data_name:
                row['history_movie_title'] = eval(row['history_movie_title'])
            L = len(row['history_movie_id'])
            preference = []
            unpreference = []
            for i in range(L):
                if int(row['history_rating'][i]) == 1:
                    if args.data_name =='ml-1m':
                        preference.append(movie_id_to_name[int(row['history_movie_id'][i])])
                    elif 'amazon_book' in args.data_name:
                        preference.append(row['history_movie_title'][i])
                else:
                    if args.data_name =='ml-1m':
                        unpreference.append(movie_id_to_name[int(row['history_movie_id'][i])])
                    elif 'amazon_book' in args.data_name:
                        unpreference.append(row['history_movie_title'][i])
            if args.data_name =='ml-1m':
                target_movie = movie_id_to_name[int(row['movie_id'])]
            # elif args.data_name == 'amazon_book':
            elif 'amazon_book' in args.data_name:
                target_movie = row['movie_title']
            preference_str = ""
            unpreference_str = ""
            for i in range(len(preference)):
                if i == 0:
                    preference_str += "\"" + preference[i] + "\""
                else:
                    preference_str += ", \"" + preference[i] + "\""
            for i in range(len(unpreference)):
                if i == 0:
                    unpreference_str += "\"" + unpreference[i] + "\""
                else:
                    unpreference_str += ", \"" + unpreference[i] + "\""
            target_preference = int(row['rating'])
            try:
                target_movie_str = "\"" + target_movie + "\""
            except:
                continue
            target_preference_str = "Yes." if target_preference == 1 else "No."
            json_list.append({
                "instruction": "Given the user's preference and unpreference, identify whether the user will like the target movie by answering \"Yes.\" or \"No.\".",
                "input": f"User Preference: {preference_str}\nUser Unpreference: {unpreference_str}\nWhether the user will like the target movie {target_movie_str}?",
                "output": target_preference_str,
            })
        return json_list
    full_train_list = list(args.warm_envs)
    train_data_paths = [f'{data_dir}/{args.data_name}/train{num}.csv' for num in full_train_list]
    val_data_paths = [f'{data_dir}/{args.data_name}/val{num}.csv' for num in full_train_list if num != max(full_train_list)]
    train_data_paths.extend(val_data_paths)

    train_df = pd.DataFrame()
    for path in train_data_paths:
        df = pd.read_csv(path)
        train_df = pd.concat([train_df, df], ignore_index=True)
    test_df = pd.read_csv(f'{data_dir}/{args.data_name}/test{args.test_num}.csv')

    train_items = train_df['movie_id'].unique().tolist()
    test_items = test_df['movie_id'].unique().tolist()
    warm_items = [item for item in test_items if item in train_items]
    cold_items = [item for item in test_items if item not in train_items]
    warm_test_df = test_df.loc[test_df['movie_id'].isin(warm_items)]
    cold_test_df = test_df.loc[test_df['movie_id'].isin(cold_items)]
    warm_test_data = csv_to_json(warm_test_df)
    cold_test_data = csv_to_json(cold_test_df)
    warm_test_data = Dataset.from_list(warm_test_data)
    cold_test_data = Dataset.from_list(cold_test_data)
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
    warm_test_data = (warm_test_data.map(generate_and_tokenize_prompt))
    cold_test_data = (cold_test_data.map(generate_and_tokenize_prompt))
    if val_data_path.endswith(".json"):  
        val_data = load_dataset("json", data_files=val_data_path)
    else:
        val_data = load_dataset(val_data_path)
    val_data = (val_data["train"].map(generate_and_tokenize_prompt))
    val_data = select_data(val_data)
    if test_data_path.endswith(".json"):  
        test_data = load_dataset("json", data_files=test_data_path)
    else:
        test_data = load_dataset(test_data_path)
    warm_test_data = select_data(warm_test_data)
    cold_test_data = select_data(cold_test_data)

    best_val_result = 0
    for weight1 in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        # model.set_adapter(f'{weight1}')
        ensemble_adapter = {}
        weight2 = 1 - weight1
        for key in n_adapter:
            ensemble_adapter[key] = weight1 * n_adapter[key] + weight2 * m_adapter[key]
        # model.load_adapter(ense)
        set_peft_model_state_dict(model,ensemble_adapter)
        val_result = trainer.evaluate(val_data)['eval_auc']
        if val_result > best_val_result:
            best_weight1 = weight1
            best_val_result = val_result

    ensemble_adapter = {}
    weight2 = 1 - best_weight1
    for key in n_adapter:
        ensemble_adapter[key] = best_weight1 * n_adapter[key] + weight2 * m_adapter[key]
    set_peft_model_state_dict(model,ensemble_adapter)

    warm_results = trainer.predict(warm_test_data)
    cold_results = trainer.predict(cold_test_data)
    test_pred = np.concatenate(((warm_results.predictions)[0],(cold_results.predictions)[0]))
    test_labels = np.concatenate(((warm_results.predictions)[1],(cold_results.predictions)[1]))
    warm_results = compute_metrics_for_results((warm_results.predictions))
    cold_results = compute_metrics_for_results((cold_results.predictions))
    test_results = compute_metrics_for_results((test_pred, test_labels))
    test_result_path = os.path.join(args.save_dir, args.save_name + f'_all_result.json')
    with open(test_result_path, 'w') as json_file:
        json.dump(test_results, json_file, indent=4)




if __name__ == "__main__":
    fire.Fire(train)
