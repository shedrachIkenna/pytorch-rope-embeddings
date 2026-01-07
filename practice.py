import torch 
import torch.nn as nn 
import math 
from typing import Tuple 

class RotaryPositionalEmbedding(nn.Module):
    """
    Implements Rotary Positional Embedding (RoPe) using pytorch 

    """
    def __init__(self, dim_head: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()

        assert dim_head % 2 == 0, f"dim_head must be even, got {dim_head}"

        # Base Frequency Calculation: inv_freq = 1 / theta^(2i/d)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim_head, 2).float() / dim_head))

        # Positional Angle: theta_i, t = t * inv_freq 
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)

        # Complex Rotation: e^(theta*2i) == cos(theta) + i*sin(theta) // Euler's formula 
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        # Register freqs_cis as a Buffer so that it is moved with the model and not trained as a parameter 
        self.register_buffer("freqs_cis", freqs_cis)