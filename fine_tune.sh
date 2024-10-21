sample=-1
cuda='2'
seed=0
tune_mode="tune"
train_start_num=0
train_total_num=1
tune_num=1
finetune_mode="seq"
data_name='amazon_book_new'

for tune_num in 1 2 3 4 5
do
    output_dir=workspace/ckpt/${data_name}/tune_${tune_num}_trstart${train_start_num}_trtotal_${train_total_num}/
    # set valid_num and test_num for loading checkpoint
    valid_num=$((train_total_num - 1 + train_start_num))  # 2
    test_num=$((train_total_num + train_start_num))  
    
    if [ ${tune_num} -eq $((valid_num + 1)) ]; then
        checkpoint_dir=workspace/ckpt/${data_name}/retrain_trstart${train_start_num}_trtotal_${train_total_num}_val${valid_num}_tes${test_num}
    else
        checkpoint_dir=workspace/ckpt/${data_name}/tune_$((tune_num - 1))_trstart${train_start_num}_trtotal_${train_total_num}
    fi
    
    # reset valid_num and test_num for evaluating
    valid_num=$((tune_num))
    test_num=$((tune_num + 1))
    python train.py \
        --train_start_num ${train_start_num} \
        --train_total_num ${train_total_num} \
        --patience 5 \
        --cuda ${cuda} \
        --test_num ${test_num} \
        --valid_num ${valid_num} \
        --sample ${sample} \
        --seed ${seed} \
        --mode ${tune_mode} \
        --tune_num ${tune_num} \
        --finetune_mode ${finetune_mode} \
        --output_dir ${output_dir} \
        --checkpoint_dir ${checkpoint_dir} \
        --data_name ${data_name}
done