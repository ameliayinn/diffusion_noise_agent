num_gpus=2
master_port=29500

deedeepspeed --num_gpus $num_gpus --master_port $master_port main.py \
    --image_size 8 \
    --batch_size 256 \
    --num_epochs 1000 \
    --timesteps 1000 \
    --lr 2e-4 \
    --time_emb_dim 256 \
    --lambda1 1\
    --num1 8000\
    --lambda2 10\
    --num2 2000\
    --samples_dir "./samples" \
    --checkpoints_dir "./checkpoints" \
    --fp16 \
    --mode "train" \
    --model_path "" \
    --num_images 80000 \
    --simulation_distribution "poisson"\
    --use_different_noise \

deepspeed --num_gpus $num_gpus --master_port $master_port main.py \
    --image_size 8 \
    --batch_size 256 \
    --num_epochs 1000 \
    --timesteps 1000 \
    --lr 2e-4 \
    --time_emb_dim 256 \
    --lambda1 1\
    --num1 5000\
    --lambda2 5\
    --num2 5000\
    --samples_dir "./samples" \
    --checkpoints_dir "./checkpoints" \
    --fp16 \
    --mode "train" \
    --model_path "" \
    --num_images 80000 \
    --simulation_distribution "poisson"\
    --use_different_noise \

deepspeed --num_gpus $num_gpus --master_port $master_port main.py \
    --image_size 8 \
    --batch_size 256 \
    --num_epochs 1000 \
    --timesteps 1000 \
    --lr 2e-4 \
    --time_emb_dim 256 \
    --lambda1 1\
    --num1 8000\
    --lambda2 5\
    --num2 2000\
    --samples_dir "./samples" \
    --checkpoints_dir "./checkpoints" \
    --fp16 \
    --mode "train" \
    --model_path "" \
    --num_images 80000 \
    --simulation_distribution "poisson"\
    --use_different_noise \