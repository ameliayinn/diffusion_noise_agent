num_gpus=2
master_port=29500

deepspeed --num_gpus $num_gpus --master_port $master_port main.py \
    --image_size 8 \
    --batch_size 256 \
    --num_epochs 1000 \
    --timesteps 1000 \
    --lr 2e-4 \
    --time_emb_dim 256 \
    --mu1 4\
    --sigma1 1\
    --num1 9000\
    --mu2 10\
    --sigma2 4\
    --num2 1000\
    --samples_dir "./samples" \
    --checkpoints_dir "./checkpoints" \
    --fp16 \
    --mode "train" \
    --model_path "" \
    --num_images 10000 \
    --simulation_distribution "normal"\

deepspeed --num_gpus $num_gpus --master_port $master_port main.py \
    --image_size 8 \
    --batch_size 256 \
    --num_epochs 1000 \
    --timesteps 1000 \
    --lr 2e-4 \
    --time_emb_dim 256 \
    --mu1 2\
    --sigma1 2\
    --num1 9000\
    --mu2 5\
    --sigma2 4\
    --num2 1000\
    --samples_dir "./samples" \
    --checkpoints_dir "./checkpoints" \
    --fp16 \
    --mode "train" \
    --model_path "" \
    --num_images 10000 \
    --simulation_distribution "normal"\