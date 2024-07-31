import torch
from torch.optim import AdamW
from prodigyopt import Prodigy

from modules.config import Config

def init_optimizer(model, config: Config):
    if config.optimizer == "AdamW":
        optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "Prodigy":
        if config.learning_rate <= 1 and config.learning_rate >= 0.1:
            optimizer = Prodigy(model.parameters(), 
                                lr=config.learning_rate,
                                decouple=True,
                                d_coef=config.d_coef,
                                safeguard_warmup=True if config.warmup_steps > 0 else False,
                                use_bias_correction=True)
        else:
            raise ValueError("The learning rate for Prodigy must be between 0.1 and 1.")
    else:
        raise ValueError(f"The optimizer '{config.optimizer}' is not supported.")
    
    return optimizer

def load_optimizer(optimizer, checkpoint_path):
    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    optimizer.load_state_dict(checkpoint['optimizer'])
    del checkpoint
    
    return optimizer

def set_lr_optimizer(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer