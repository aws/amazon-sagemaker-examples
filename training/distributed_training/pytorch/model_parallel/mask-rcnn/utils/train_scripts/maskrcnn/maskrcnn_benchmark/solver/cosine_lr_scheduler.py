import math
from torch.optim.lr_scheduler import _LRScheduler
# Credit: https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, warmup_lr=0.000133, eta_max=0.1, alpha=0.001, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        # self.base_eta_max = eta_max
        self.alpha = alpha
        self.eta_max = eta_max
        self.eta_min = alpha * eta_max
        self.warmup_lr = warmup_lr
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1: # warmup init lr
            return [self.warmup_lr for _ in self.base_lrs]
        elif self.T_cur < self.T_up: # linear warmup to eta_max
            return [(self.eta_max - self.warmup_lr)*self.T_cur / self.T_up + self.warmup_lr for _ in self.base_lrs]
        else:
            completed_fraction = (self.T_cur-self.T_up) / (self.T_i - self.T_up)
            cosine_decayed = (1 + math.cos(math.pi * completed_fraction)) / 2
            return [base_lr * ((1 - self.alpha) * cosine_decayed + self.alpha) for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

