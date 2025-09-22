accelerate launch train_qa_lora.py \
    --experiment_name 'LoRA_RoBERTa_QA' \
    --working_directory 'work_dir' \
    --checkpoint_weights_dir 'checkpoints' \
    --path_to_cache_dir "data/sqad_data" \
    --hf_model_name 'FacebookAI/roberta-base' \
    --hf_dataset 'squad' \
    --use_lora \
    --lora_rank 8 \
    --lora_alpha 8 \
    --lora_use_rslora \
    --lora_dropout 0.1 \
    --lora_bias 'lora_only' \
    --lora_target_modules 'word_embeddings,query,key,value,dense' \
    --lora_exclude_modules 'pooler,head' \
    --max_grad_norm 1.0 \
    --per_gpu_batch_size 64 \
    --warmup_steps 100 \
    --epochs 3 \
    --num_workers 0 \
    --learning_rate 3e-5 \
    --weight_decay 0.001 \
    --gradient_checkpointing \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --seed 42 \
    --max_no_of_checkpoints 3 \
    --pretrained_backbone "pretrained_huggingface" \
#    --wandb \
#    --loading_from_checkpoint
#    --train_head_only \


