"""
UNet in DDPM.
Modified from https://github.com/w86763777/pytorch-ddpm/blob/f804ccbd58a758b07f79a3b9ecdfb1beb67258f6/model.py

Move the TimeEmbedding part out of UNet. UNet takes the embedding as input.
Change the input of TimeEmbedding from discrete to conitnuous values.
The input range of TimeEmbedding is changed from [0, T-1] to [0, 1].
Rename TimeEmbedding to ScalarEmbedding.
Add input channels to the arguments of UNet.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from ..sinusoidal_embedding import get_sinusoidal_embedding


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ScalarEmbedding(nn.Module):
    """Embed scalars to vectors."""

    def __init__(self, d_model, dim):
        """
        Args:
            d_model (int): dimension of the sinusoidal embedding
            dim (int): dimension of the output embedding after MLP
        """
        assert d_model % 2 == 0
        super().__init__()
        self.d_model = d_model

        self.embedding_mlp = nn.Sequential(
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        """
        Args:
            t: (b,)

        Returns:
            (b, dim)
        """
        emb = get_sinusoidal_embedding(t, self.d_model)
        emb = self.embedding_mlp(emb)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.main(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def _forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h

    def forward(self, x, temb):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, temb, use_reentrant=False
            )
        else:
            return self._forward(x, temb)


class UNet(nn.Module):
    def __init__(
        self,
        ch,
        ch_mult,
        attn,
        num_res_blocks,
        dropout,
        edim,
        in_ch,
        use_checkpoint=False,
    ):
        """
        Args:
            edim (int): embedding dimension
            in_ch (int): number of channels in the input (and output) of the network
        """
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), "attn index out of bound"

        self.head = nn.Conv2d(in_ch, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(
                    ResBlock(
                        in_ch=now_ch,
                        out_ch=out_ch,
                        tdim=edim,
                        dropout=dropout,
                        attn=(i in attn),
                        use_checkpoint=use_checkpoint,
                    )
                )
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList(
            [
                ResBlock(
                    now_ch,
                    now_ch,
                    edim,
                    dropout,
                    attn=True,
                    use_checkpoint=use_checkpoint,
                ),
                ResBlock(
                    now_ch,
                    now_ch,
                    edim,
                    dropout,
                    attn=False,
                    use_checkpoint=use_checkpoint,
                ),
            ]
        )

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(
                    ResBlock(
                        in_ch=chs.pop() + now_ch,
                        out_ch=out_ch,
                        tdim=edim,
                        dropout=dropout,
                        attn=(i in attn),
                        use_checkpoint=use_checkpoint,
                    )
                )
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, in_ch, 3, stride=1, padding=1),
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, emb):
        """
        Args:
            x: (b, *data)
            emb: (b, edim)
        """
        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, emb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, emb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, emb)
        h = self.tail(h)

        assert len(hs) == 0
        return h


class UNetTime(nn.Module):
    """Time conditioned UNet."""

    def __init__(
        self, ch, ch_mult, attn, num_res_blocks, dropout, in_ch, use_checkpoint=False
    ):
        super().__init__()
        edim = ch * 4  # embedding dimension is 4x feature channel number
        self.time_embedding = ScalarEmbedding(ch, edim)

        self.unet = UNet(
            ch,
            ch_mult,
            attn,
            num_res_blocks,
            dropout,
            edim,
            in_ch,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, x, t):
        """
        Args:
            x: (b, *data)
            t: (b,)
        """
        # Time embedding
        emb = self.time_embedding(t)

        # UNet
        return self.unet(x, emb)


if __name__ == "__main__":
    batch_size = 8
    model = UNetTime(
        ch=128, ch_mult=[1, 2, 2, 2], attn=[1], num_res_blocks=2, dropout=0.1, in_ch=3
    )
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.rand(batch_size)
    y = model(x, t)
