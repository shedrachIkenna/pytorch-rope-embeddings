import torch 
import torch.nn as nn 
import math
from typing import Tuple

class RotaryPositionalEmbedding(nn.Module):
    """
    Implements Rotary Positional Embedding (RoPe) using PyTorch 
    """
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super.__init__()

        # Base frequency calculation: inv_freq = 1 / theta^(2i/d)
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))

        # Positional angle calculation: theta_t,i = t * inv_freq
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq) # shape [max_seq_len, head_dim / 2]

        # Complex Rotation: e^(i*theta) = cos(theta) * i*sin(theta) (Euler's formula)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        
