from config import get_config
from generate import generate_samples
from train import train_deepspeed

if __name__ == "__main__":
    cfg = get_config() 

    if cfg.mode == "train":
        train_deepspeed(cfg)
    elif cfg.mode == "generate":
        if not cfg.model_path:
            raise ValueError("Model path must be specified for generation mode")
        generate_samples(cfg, cfg.model_path, cfg.num_images)