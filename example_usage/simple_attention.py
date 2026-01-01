from rope_embedding.rope import RotaryPositionalEmbedding 
import torch.nn.functional as F 
import torch.nn as nn 

class SimpleAttention(nn.Module):
    def __init__(self, dim, n_heads, max_seq_len):
        super().__init__()
        head_dim = dim // n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim

        # Initialize RoPe 
        self.rope = RotaryPositionalEmbedding(head_dim, max_seq_len)

        # Q, K, V projection layers (simplified example)
        self.Wq = nn.Linear(dim, dim, bias=False)
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.Wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x, seqlen):
        B, S, D = x.shape # Batch, Sequence length and Dimension 

        # Project and reshape Q, K, V 
        Q = self.Wq(x).view(B, S, self.n_heads, self.head_dim)
        K = self.Wk(x).view(B, S, self.n_heads, self.head_dim)
        V = self.Wv(x).view(B, S, self.n_heads, self.head_dim)

        # Apply RoPe 
        Q_rotated, K_rotated = self.rope(Q, K, seqlen)

        # Transpose for scaled dot-product attention (Batch, Head, Seq, Dim)
        Q_r = Q_rotated.transpose(1, 2)
        K_r = K_rotated.transpose(1, 2)
        V_t = V.transpose(1, 2)

        # Attention Calculation 
        attn_output = F.scaled_dot_product_attention(Q_r, K_r, V_t)

        # Final reshape and output 
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)
        return self.Wo(attn_output)
