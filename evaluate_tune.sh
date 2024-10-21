sample=-1
cuda='1'
seed=2
tune_mode="tune"
train_start_num=0
train_total_num=1
valid_num=$((train_total_num - 1 + train_start_num)) 
tune_num=3
is_testall=False
finetune_mode="seq"
data_name='amazon_book_new'
test_num=19

# 设置保存路径
save_dir=workspace/test_results/${data_name}/
save_name=tune
checkpoint_dir=workspace/ckpt/${data_name}/tune_${tune_num}_trstart${train_start_num}_trtotal_${train_total_num}/

# 执行 Python 脚本
python evaluate.py \
    --train_start_num ${train_start_num} \
    --train_total_num ${train_total_num} \
    --patience 8 \
    --cuda ${cuda} \
    --test_num ${test_num} \
    --valid_num ${valid_num} \
    --sample ${sample} \
    --seed ${seed} \
    --mode ${tune_mode} \
    --tune_num ${tune_num} \
    --save_name ${save_name} \
    --finetune_mode ${finetune_mode} \
    --save_dir ${save_dir} \
    --checkpoint_dir ${checkpoint_dir} \
    --data_name ${data_name} 
