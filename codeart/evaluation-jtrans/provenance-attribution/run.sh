TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=7

MODEL_NAME_OR_PATH=../../save/jtrans/models/jTrans-pretrain/
# MODEL_NAME_OR_PATH=../../save/jtrans/models/jTrans-finetune/

python run.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name PurCL/binkit-jtrans-all \
    --use_auth_token \
    --dataloader_num_workers 2 \
    --max_train_samples 100 \
    --max_eval_samples 10000 \
    --max_predict_samples 10000 \
    --max_seq_length 512 \
    --labels O0,O1,O2,O3 \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --do_predict \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 5 \
    --learning_rate 5e-5 \
    --evaluation_strategy steps \
    --eval_steps 2 \
    --save_steps 100 \
    --logging_step 2 \
    --report_to tensorboard \
    --cache_dir ../save/.cache \
    --output_dir ../save/jtrans-pa \
    --overwrite_output_dir \
    --load_best_model_at_end=True \
    --metric_for_best_model eval_loss \
    --load_best_model_at_end
    # --overwrite_cache \
    # --warmup_steps 1000 \