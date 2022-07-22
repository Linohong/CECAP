MODEL='roberta-large'
DATADIR='dataset/wizard_kat'
OUT_PREFIX='ks_roberta_cecap_negsamp_74092'
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=0,1

OUTPUT=checkpoints/${OUT_PREFIX}
PREFIX=train

for seed in 42 52 62 72 82
do
python run_ks_lino.py \
    --model_name_or_path $MODEL \
    --data_dir $DATADIR \
    --cache_dir 'cached' \
    --task 'Wizard_LowResource' \
    --train_prefix $PREFIX \
    --eval_prefix 'test_seen' \
    --max_source_length 256 \
    --max_target_length 64 \
    --max_kno_length 64 \
    --max_num_kno 40 \
    --do_train \
    --num_train_epochs 3 \
    --save_steps 9000 \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --output_dir $OUTPUT_$seed \
    --overwrite_output_dir \
    --seed $seed \
    --cecap 'cecap_wikipedia' \
    --fp16 
done
