"""Custom AdamW optimizer (Adam + decoupled weight decay)."""

import torch


class AdamW:
    """
    AdamW: Adam with decoupled weight decay.
    Unlike Adam: weight decay applied directly to params, not through gradient.
    """

    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.01):
        self.params = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        self.t += 1
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue

                g = p.grad

                # Decoupled weight decay (before update)
                if self.weight_decay > 0:
                    p.mul_(1.0 - self.lr * self.weight_decay)

                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)

                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def state_dict(self):
        return {"lr": self.lr, "t": self.t}

    def load_state_dict(self, state_dict):
        self.lr = state_dict.get("lr", self.lr)
        self.t = state_dict.get("t", self.t)


class CosineScheduler:
    """Cosine annealing LR scheduler."""

    def __init__(self, optimizer, total_steps, warmup_steps=0, min_lr=1e-6):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        import math

        if self.current_step <= self.warmup_steps:
            lr = self.base_lr * self.current_step / max(1, self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        self.optimizer.lr = lr


def get_optimizer(model, lr=0.001, weight_decay=0.01):
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
