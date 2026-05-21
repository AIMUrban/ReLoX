import math

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class CNNEncoder(nn.Module):
    """
    64x64 -> 32x32 -> 16x16 -> 8x8 (tokens=64)
    """

    def __init__(self, out_dim=256, c=64):
        super().__init__()

        self.body = nn.Sequential(
            ConvBlock(3, c, stride=1),
            ConvBlock(c, c, stride=2),
            ConvBlock(c, 2 * c, stride=2),
            ConvBlock(2 * c, 4 * c, stride=2),
            ConvBlock(4 * c, 8 * c, stride=2),
        )
        self.proj = nn.Conv2d(8 * c, out_dim, 1, bias=False)
        self.ln = nn.LayerNorm(out_dim)

    def _fm_to_tokens(self, fm):
        _, _, h, w = fm.shape
        tokens = fm.flatten(2).transpose(1, 2)  # [B, N, D]
        tokens = self.ln(tokens)
        tokens_pos = add_2d_pos(tokens, h, w)
        return tokens, tokens_pos

    def forward(self, x):
        fm = self.proj(self.body(x))
        t, t_pos = self._fm_to_tokens(fm)
        return {"tokens": t, "tokens_pos": t_pos}


def add_2d_pos(tokens, h=None, w=None, base=10000.0):
    """
    2D sinusoidal positional encoding.
    tokens: [B, N, D]
    return: [B, N, D]
    """
    _, N, D = tokens.shape
    device = tokens.device

    if h is None or w is None:
        s = int(math.sqrt(N))
        assert s * s == N, f"N={N} is not a perfect square; please pass h and w explicitly."
        h = w = s
    assert h * w == N

    # D must be divisible by 4 so each axis can use paired sin/cos channels.
    assert D % 4 == 0, f"out_dim={D} must be divisible by 4 for 2D sin-cos encoding."

    # Normalize to [-1, 1] for better scale robustness.
    ys = torch.linspace(-1.0, 1.0, steps=h, device=device)
    xs = torch.linspace(-1.0, 1.0, steps=w, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [h,w]
    grid_x = grid_x.reshape(-1, 1)  # [N,1]
    grid_y = grid_y.reshape(-1, 1)  # [N,1]

    dim_each = D // 2  # half for x, half for y
    dim_half = dim_each // 2  # per axis: half sin, half cos

    i = torch.arange(dim_half, device=device).float()  # [dim_half]
    inv_freq = base ** (-i / dim_half)  # [dim_half]

    sinus_x = grid_x * inv_freq
    sinus_y = grid_y * inv_freq

    pos_x = torch.cat([torch.sin(sinus_x), torch.cos(sinus_x)], dim=1)
    pos_y = torch.cat([torch.sin(sinus_y), torch.cos(sinus_y)], dim=1)

    pos = torch.cat([pos_x, pos_y], dim=1).unsqueeze(0)  # [1,N,D]
    return tokens + pos
