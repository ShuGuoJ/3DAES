"""
监控参数梯度的变化
"""
import torch


class GradMonitor(object):
    def __call__(self, parameters, ord=1):
        grad_norm = []
        for p in parameters:
            if p.requires_grad:
                norm = p.grad.norm(ord)
                grad_norm.append(norm)
            else:
                continue
        grad_norm = torch.tensor(grad_norm, dtype=torch.float)
        return float(grad_norm.norm(ord))

