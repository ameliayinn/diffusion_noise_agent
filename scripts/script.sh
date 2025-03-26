num_gpus=2
master_port=29500
learning_rate=2e-5

deepspeed --num_gpus $num_gpus --master_port $master_port main.py \
    --image_size 64 \
    --batch_size 256 \
    --num_epochs 3000 \
    --timesteps 1000 \
    --lr $learning_rate \
    --time_emb_dim 256 \
    --dataset_name "cats_vs_dogs" \
    --samples_dir "./samples" \
    --checkpoints_dir "./checkpoints" \
    --fp16 \
    --mode "train" \
    --model_path "" \
    --num_images 16 \