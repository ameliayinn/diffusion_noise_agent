num_gpus=1
master_port=29500

deepspeed --num_gpus $num_gpus --master_port $master_port main.py \
    --image_size 16 \
    --batch_size 1 \
    --num_epochs 200 \
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
    --num_images 1000 \
    --simulation_distribution "normal"\
    --use_different_noise \
    --use_moe \
    --num_experts 4\
    --moe_hidden_dim 64\
    --moe_tau 0.1\