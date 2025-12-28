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

        # Register freqs_cis as a Buffer so that it's moved with the model but not trained 
        self.register_buffer("freqs_cis", freqs_cis)

    def _reshape_for_broadcast(self, freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor: 
        """
        Reshapes freqs_cis to be compactible for broadcasting across batch and head dimensions

        freqs_cis shape is [Seq, D/2]. We need to broadcast it to [1, Seq, 1, D/2] for matrix multiplication with Q and K which are [B, Seq, H, D/2]

        Seq - number of words/tokens in a sentence/sequence 
        D/2 - embedding dimensions/2 = feature pairs 
        B - Batch (how many sentences/sequences the transformer sees at once)
        H - Head (How many attention heads)
        """

        # Get the number of dimensions for x 
        ndim = x.ndim

        # raise assertion error if condition is false 
        assert ndim >= 4, "Input tensor must have at least 4 dimensions (B, S, H, D/2)"

        # looks at x tensor, loops through each dimension
        # replace all the dimensions with 1s except dimensions at index 2 and the last index (4)
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    
        # view([1, 10, 1, 5]) becomes view(1, 10, 1, 5)
        return freqs_cis.view(*shape)

