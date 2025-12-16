import torch 
import torch.nn as nn 
import math
from typing import Tuple

class RotaryPositionalEmbedding(nn.Module):
    """
    Implements Rotary Positional Embedding (RoPe) using PyTorch 
    """
