import torch


class WarmupScheduler:

    def __init__(self, optimizer, warmup_iters: int, base_lr: float,
                 start_lr: float = 0.0, last_iter: int = -1):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.base_lr = base_lr
        self.start_lr = start_lr
        self.last_iter = last_iter

        if last_iter == -1:
            for group in self.optimizer.param_groups:
                group['lr'] = start_lr

    def step(self, current_iter: int):
        if current_iter < self.warmup_iters:

            alpha = current_iter / self.warmup_iters
            lr = self.start_lr + (self.base_lr - self.start_lr) * alpha
        else:

            lr = self.base_lr

        for group in self.optimizer.param_groups:
            group['lr'] = lr

        return lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


class StageWarmupScheduler:

    def __init__(self, optimizer, base_lr: float,
                 initial_warmup_iters: int = 2000,
                 stage_warmup_iters: int = 500):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.initial_warmup_iters = initial_warmup_iters
        self.stage_warmup_iters = stage_warmup_iters

        self.current_stage = None
        self.stage_start_iter = 0
        self.global_start_iter = 0


        for group in self.optimizer.param_groups:
            group['lr'] = 0.0

    def set_stage(self, stage_name: str, current_iter: int):
        if self.current_stage != stage_name:
            print(f"\n [LR Scheduler] Stage transition: {self.current_stage} → {stage_name}")
            print(f"   Starting stage warmup ({self.stage_warmup_iters} iters) at iter {current_iter}")
            self.current_stage = stage_name
            self.stage_start_iter = current_iter

    def step(self, global_iter: int) -> float:

        if global_iter < self.initial_warmup_iters:
            alpha = global_iter / self.initial_warmup_iters
            lr = self.base_lr * alpha


        elif (global_iter - self.stage_start_iter) < self.stage_warmup_iters:

            iters_since_stage = global_iter - self.stage_start_iter
            alpha = iters_since_stage / self.stage_warmup_iters

            lr = self.base_lr * (0.5 + 0.5 * alpha)


        else:
            lr = self.base_lr

        for group in self.optimizer.param_groups:
            group['lr'] = lr

        return lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


class CosineAnnealingWarmup:

    def __init__(self, optimizer, warmup_iters: int, max_iters: int,
                 base_lr: float, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.base_lr = base_lr
        self.min_lr = min_lr

        for group in self.optimizer.param_groups:
            group['lr'] = 0.0

    def step(self, current_iter: int) -> float:
        import math

        if current_iter < self.warmup_iters:

            alpha = current_iter / self.warmup_iters
            lr = self.base_lr * alpha
        else:

            progress = (current_iter - self.warmup_iters) / (self.max_iters - self.warmup_iters)
            progress = min(progress, 1.0)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        for group in self.optimizer.param_groups:
            group['lr'] = lr

        return lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']
