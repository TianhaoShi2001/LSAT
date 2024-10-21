data_name='ml-1m'
sample=-1
train_start_num=0
train_total_num=1
test_num=19
cuda='1'
tune_mode="full"
seed=0

valid_num=$((train_total_num - 1 + train_start_num))  # 0
load_test_num=$((train_total_num + train_start_num))  # 3

save_dir=workspace/test_results/${data_name}/
checkpoint_dir=workspace/ckpt/${data_name}/retrain_trstart${train_start_num}_trtotal_${train_total_num}_val${valid_num}_tes${load_test_num}/
save_name=retrain

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
    --save_name ${save_name} \
    --save_dir ${save_dir} \
    --checkpoint_dir ${checkpoint_dir} \
    --data_name ${data_name} 
