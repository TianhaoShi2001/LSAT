sample=-1
cuda='5'
seed=0
tune_mode="full"
train_start_num=0

for train_total_num in  1 2 3 4 5 6 7 8 9 10 11 12 13 
do
    valid_num=$((train_total_num - 1 + train_start_num))  
    test_num=$((train_total_num + train_start_num)) 
    data_name='amazon_book'
    output_dir=workspace/ckpt/${data_name}/retrain_trstart${train_start_num}_trtotal_${train_total_num}_val${valid_num}_tes${test_num}/
    python train.py \
        --train_start_num ${train_start_num} \
        --train_total_num ${train_total_num} \
        --test_num ${test_num} \
        --valid_num ${valid_num} \
        --patience 10 \
        --cuda ${cuda} \
        --sample ${sample} \
        --seed ${seed} \
        --mode ${tune_mode} \
        --output_dir ${output_dir} \
        --data_name ${data_name}  
    valid_num=$((train_total_num - 1 + train_start_num))  # 2
    test_num=$((train_total_num + train_start_num)) # 3
done

