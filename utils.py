import os
import torch
import re
from sklearn.metrics import roc_auc_score,accuracy_score, log_loss
def create_parent_folders(filename):
    if filename == None:
        return
    file_abs_path = os.path.abspath(filename)
    
    parent_folder = os.path.dirname(file_abs_path)

    while not os.path.exists(parent_folder):
        os.makedirs(parent_folder)
        print(f"Created parent folder '{parent_folder}'")
        parent_folder = os.path.dirname(parent_folder)



def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""
    
def compute_metrics_for_results(eval_preds):
    pre = eval_preds
    auc = roc_auc_score(pre[1], pre[0])
    acc = accuracy_score(pre[1], (pre[0] > 0.5).astype(int))
    logloss = log_loss(pre[1], (pre[0] > 0.5).astype(int))
    return {'eval_auc': auc, 'eval_acc':acc, 'eval_loss':logloss}


def compute_metrics(eval_preds):
    pre, labels = eval_preds
    auc = roc_auc_score(pre[1], pre[0])
    acc = accuracy_score(pre[1], (pre[0] > 0.5).astype(int))
    return {'auc': auc, 'acc':acc}

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    labels_index = torch.argwhere(torch.bitwise_or(labels == 8241, labels == 3782))
    gold = torch.where(labels[labels_index[:, 0], labels_index[:, 1]] == 3782, 0, 1)
    labels_index[: , 1] = labels_index[: , 1] - 1
    logits = logits.softmax(dim=-1)
    logits = torch.softmax(logits[labels_index[:, 0], labels_index[:, 1]][:,[3782, 8241]], dim = -1)
    return logits[:, 1][2::3], gold[2::3]





def find_min_checkpoint_dir(dir_path):
    min_suffix = None
    min_dir = None
    old_checkpoint_dir = None

    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isdir(item_path) and re.match(r'checkpoint-(\d+)', item):
            match = re.search(r'checkpoint-(\d+)', item)
            if match:
                suffix = int(match.group(1))
                if min_suffix is None or suffix < min_suffix:
                    min_suffix = suffix
                    min_dir = item_path
        if item == 'ckpt-old' and os.path.isdir(item_path):
            old_checkpoint_dir = item_path
    if old_checkpoint_dir:
        return old_checkpoint_dir

    return min_dir

def find_max_checkpoint_dir(dir_path):
    max_suffix = None
    max_dir = None

    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isdir(item_path) and re.match(r'checkpoint-(\d+)', item):
            match = re.search(r'checkpoint-(\d+)', item)
            if match:
                suffix = int(match.group(1))
                if max_suffix is None or suffix > max_suffix:
                    max_suffix = suffix
                    max_dir = item_path
    return max_dir





def select_data(data):
    select_idx = [idx for idx, sequence in enumerate(data['input_ids']) if ((sequence.count(8241) + sequence.count(3782)) == 3) & (len(sequence) <= cutoff_len)]
    return data.select(select_idx)