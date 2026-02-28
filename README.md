export CUDA_VISIBLE_DEVICES=0,1

# stage1
accelerate launch --num_processes 2 --num_machines 1 --deepspeed_config_file ds_config.json scripts/train_from_scratch.py --run_name Evo1_metaworld_stage1 --action_head flowmatching --use_augmentation --lr 5e-5 --dropout 0.2 --weight_decay 1e-3 --batch_size 128 --image_size 448 --max_steps 5000 --log_interval 10 --ckpt_interval 10000 --warmup_steps 1000 --grad_clip_norm 1.0 --num_layers 8 --horizon 50 --finetune_action_head --disable_wandb --vlm_name OpenGVLab/InternVL3-1B --dataset_config_path dataset/config_metaworld.yaml --per_action_dim 24 --state_dim 24 --save_dir /mnt/data_ssd/zhoufang/code/Evo-1/Evo_1/checkpoints/stage1


# stage2
accelerate launch --num_processes 1 --num_machines 1 --deepspeed_config_file ds_config.json scripts/train_from_scratch.py --run_name Evo1_metaworld_stage2 --action_head flowmatching --use_augmentation --lr 1e-5 --dropout 0.2 --weight_decay 1e-3 --batch_size 4 --image_size 448 --max_steps 80000 --log_interval 10 --ckpt_interval 10000 --warmup_steps 1000 --grad_clip_norm 1.0 --num_layers 8 --horizon 50 --finetune_vlm --finetune_action_head --disable_wandb --vlm_name OpenGVLab/InternVL3-1B --dataset_config_path dataset/config_metaworld.yaml --per_action_dim 24 --state_dim 24 --save_dir /mnt/data_ssd/zhoufang/code/Evo-1/Evo_1/checkpoints/stage2 --resume --resume_pretrain --resume_path /mnt/data_ssd/zhoufang/code/Evo-1/Evo_1/checkpoints/stage2/step_80000


# real world
# stage 1
accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --deepspeed_config_file ds_config.json \
    scripts/train_from_scratch.py \
    --run_name Evo1_real_data_task1 \
    --action_head flowmatching \
    --lr 5e-5 \
    --batch_size 128 \
    --image_size 448 \
    --max_steps 10000 \
    --log_interval 10 \
    --ckpt_interval 1000 \
    --warmup_steps 500 \
    --num_layers 8 \
    --horizon 16 \
    --per_action_dim 7 \
    --state_dim 7 \
    --finetune_action_head \
    --disable_wandb \
    --vlm_name OpenGVLab/InternVL3-1B \
    --dataset_config_path dataset/config_metaworld.yaml \
    --data_paths /mnt/data_ssd/zhoufang/code/evo-fast/Evo_1/dataset/real_data/task_1 \
    --save_dir ./checkpoints/real_task_1

#stage 2
# 切换到工作目录
cd /mnt/data_ssd/zhoufang/code/evo-fast/Evo_1

# 启动训练
export CUDA_VISIBLE_DEVICES=2,3
accelerate launch \
    --num_processes 2 \
    --num_machines 1 \
    --deepspeed_config_file ds_config.json \
    scripts/train_from_scratch.py \
    --run_name Evo1_real_data_task1_stage2 \
    --action_head flowmatching \
    --lr 1e-5 \
    --batch_size 32 \
    --image_size 448 \
    --max_steps 160000 \
    --log_interval 10 \
    --ckpt_interval 1000 \
    --warmup_steps 500 \
    --num_layers 8 \
    --horizon 16 \
    --per_action_dim 7 \
    --state_dim 7 \
    --finetune_action_head \
    --finetune_vlm \
    --disable_wandb \
    --vlm_name OpenGVLab/InternVL3-1B \
    --dataset_config_path dataset/config_metaworld.yaml \
    --data_paths /mnt/data_ssd/zhoufang/code/evo-fast/Evo_1/dataset/real_data/task_1 \
    --save_dir ./checkpoints/real_task_1_stage2 \
    --resume \
    --resume_pretrain \
    --resume_path ./checkpoints/real_stage1/step_10000