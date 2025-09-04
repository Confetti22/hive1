from torch.optim.lr_scheduler import _LRScheduler
import math

class WarmupCosineLR(_LRScheduler):
    """
    Linearly warm-up the learning-rate for `warmup_epochs`,
    then cosine-anneal it to zero over the remaining epochs.

    Args
    ----
    optimizer : torch.optim.Optimizer
    warmup_epochs : int
        Number of warm-up epochs (≥1).
    max_epochs : int
        Total number of training epochs.
    last_epoch : int, default -1
        Start epoch.  Leave at -1 unless you are resuming training.
    """

    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # `self.last_epoch` is the *next* epoch index PyTorch will enter (0-based)
        epoch = self.last_epoch + 1
        if epoch <= self.warmup_epochs:                       # linear warm-up
            warmup_factor = epoch / float(self.warmup_epochs)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        # cosine decay (epoch > warm-up)
        progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return [base_lr * cosine_factor for base_lr in self.base_lrs]


def get_scheduler(optimizer, args):
    schedulers ={
        'cosine':  WarmupCosineLR(optimizer,args.warmup_epochs,args.max_epochs),
    }
    return schedulers.get(args.lr_scheduler, "Invalid Optimizer" ) 
