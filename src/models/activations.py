"""Activations for ViT."""
import torch
import math

def relu(x):
    return torch.clamp(x, min=0)

def gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))

def softmax(x, dim=-1):
    e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
    return e_x / torch.sum(e_x, dim=dim, keepdim=True)
