data_name='ml-1m'
sample=-1
cuda='7'
seed=0
tune_mode="full"
train_start_num=0

for n in 10  # long term lora
do
for m in 16  # short term lora 
do
    valid_num=${m}
    test_num=$((m + 1))
    save_dir=workspace/soup_results/${data_name}/arithmetic_long-term${n}_short-term${m}_tes${test_num}
    save_name=arithmetic
    python arithmetic.py \
        --warm_envs 0 \
        --train_start_num ${train_start_num} \
        --n ${n} \
        --m ${m} \
        --patience 8 \
        --cuda ${cuda} \
        --test_num ${test_num} \
        --valid_num ${valid_num} \
        --sample ${sample} \
        --seed ${seed} \
        --mode ${tune_mode} \
        --save_name ${save_name} \
        --save_dir ${save_dir} \
        --data_name ${data_name}
done
done