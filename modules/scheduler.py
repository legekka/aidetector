import torch

def init_scheduler(optimizer, config, max_steps=None):
    if config.scheduler == "WarmupThenCosineAnnealingLR":
        if max_steps is None:
            raise ValueError("The max_steps is not provided.")
        warmup_steps = config.warmup_steps
        cosine_annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps - warmup_steps, eta_min=config.eta_min)
        scheduler = WarmupThenCosineAnnealingLR(optimizer, warmup_steps, cosine_annealing_scheduler)
    elif config.scheduler == "LinearWarmupLR":
        warmup_steps = config.warmup_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, float(step + 1) / warmup_steps))
    elif config.scheduler == "CosineAnnealingLR":
        if max_steps is None:
            raise ValueError("The max_steps is not provided.")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=config.eta_min)    
    elif config.scheduler == "CosineAnnealingEpochLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=config.eta_min)
    elif config.scheduler == "ConstantLR":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    elif config.scheduler == "WarmupThenConstantLR":
        if max_steps is None:
            raise ValueError("The max_steps is not provided.")
        warmup_steps = config.warmup_steps
        constant_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        scheduler = WarmupThenCosineAnnealingLR(optimizer, warmup_steps, constant_scheduler)
    else:
        raise ValueError(f"The scheduler '{config.scheduler}' is not supported.")
    
    return scheduler

def load_scheduler(scheduler, checkpoint_path):
    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    scheduler.load_state_dict(checkpoint['scheduler'])
    del checkpoint
    
    return scheduler

class WarmupThenCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, cosine_annealing_scheduler, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.cosine_annealing_scheduler = cosine_annealing_scheduler
        self.initial_lr = optimizer.param_groups[0]['lr']
        super(WarmupThenCosineAnnealingLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = float(self.last_epoch) / float(max(1, self.warmup_steps))
            return [self.initial_lr * warmup_factor for _ in self.optimizer.param_groups]
        else:
            # Cosine annealing
            self.cosine_annealing_scheduler.last_epoch = self.last_epoch - self.warmup_steps
            return self.cosine_annealing_scheduler.get_lr()

