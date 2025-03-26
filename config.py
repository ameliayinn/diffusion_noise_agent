import argparse

def get_config():
    parser = argparse.ArgumentParser(description="Diffusion Model Training with DeepSpeed")
    
    # 训练参数
    parser.add_argument("--image_size", type=int, default=64, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=256, help="Global batch size")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--timesteps", type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--dataset_name", type=str, default="cats_vs_dogs", help="Dataset name")
    
    # 路径参数
    parser.add_argument("--samples_dir", type=str, default="./samples", help="Directory to save samples")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--logs_dir", type=str, default="./logs", help="Directory to save logs")
    
    # 精度设置
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training (default: True)")
    parser.add_argument("--no-fp16", action="store_false", dest="fp16", help="Disable mixed precision training")
    
    # 运行模式
    parser.add_argument("--mode", choices=["train", "generate"], default="train", help="Run mode")
    parser.add_argument("--model_path", type=str, help="Model path for generation mode")
    parser.add_argument("--num_images", type=int, default=16, help="Number of images to generate")
    
    # 时间嵌入维度
    parser.add_argument("--time_emb_dim", type=int, default=128, help="Dimension of time embeddings")
    
    # MoE 相关参数
    parser.add_argument("--use_moe", action="store_true", default=False,
                      help="Enable Mixture of Experts module (default: False)")
    parser.add_argument("--num_experts", type=int, default=4,
                      help="Number of experts in MoE (default: 4)")
    parser.add_argument("--moe_hidden_dim", type=int, default=64,
                      help="Hidden dimension for MoE experts (default: 64)")
    parser.add_argument("--moe_tau", type=float, default=0.1,
                      help="Temperature for Gumbel-Softmax in MoE (default: 0.1)")
    
    # simulation data
    parser.add_argument("--simulation_distribution", type=str, default="data", help="Distribution of the simulation data")
    parser.add_argument("--num1", type=int, default=2, help="Number of data of the first normal distribution")
    parser.add_argument("--num2", type=int, default=2, help="Number of data of the second normal distribution")
    
    # normal distribution
    parser.add_argument("--mu1", type=int, default=2, help="Mu value of the first normal distribution")
    parser.add_argument("--sigma1", type=int, default=2, help="Sigma value of the first normal distribution")
    parser.add_argument("--mu2", type=int, default=2, help="Mu value of the second normal distribution")
    parser.add_argument("--sigma2", type=int, default=2, help="Sigma value of the second normal distribution")
    
    # poisson distribution
    parser.add_argument("--lambda1", type=int, default=2, help="Lambda value of the first poisson distribution")
    parser.add_argument("--lambda2", type=int, default=2, help="Lambda value of the second poisson distribution")
    
    # deepspeed 自动添加的参数
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed from deepspeed")
    
    # use different noise
    parser.add_argument("--use_different_noise", action="store_true", default=False, help="Use different noise for diffusion or not")
    
    # parser.set_defaults(fp16=True)
    return parser.parse_args()