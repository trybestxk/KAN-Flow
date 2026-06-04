import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from flow_matching.utils import ModelWrapper


class RadialBasisFunction(nn.Module):
    def __init__(self, grid_min: float = -2., grid_max: float = 2., num_grids: int = 8, denominator: float = None):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = nn.Parameter(grid, requires_grad=True)

        initial_denom = denominator or (grid_max - grid_min) / (num_grids - 1)
        self.log_denominator = nn.Parameter(
            torch.log(torch.tensor(initial_denom)),
            requires_grad=True
        )

    def forward(self, x):
        denominator = torch.exp(self.log_denominator).clamp(min=1e-3, max=10.0)
        return torch.exp(-((x[..., None] - self.grid) / denominator) ** 2)


class SplineConv1D(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, init_scale=0.1, padding_mode="zeros"):
        self.init_scale = init_scale
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias, padding_mode)

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class FastKANConv1DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, grid_min=-2., grid_max=2.,
                 num_grids=8, use_base_update=True, base_activation=F.silu,
                 spline_weight_init_scale=0.01, padding_mode="zeros"):
        super().__init__()

        self.num_grids = num_grids
        self.basis_function = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.input_norm = nn.LayerNorm(in_channels)

        self.spline_conv = SplineConv1D(
            in_channels * num_grids, out_channels, kernel_size, stride, padding,
            dilation, groups, bias, spline_weight_init_scale, padding_mode
        )

        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_conv = nn.Conv1d(
                in_channels, out_channels, kernel_size, stride, padding,
                dilation, groups, bias, padding_mode
            )

    def forward(self, x):
        batch_size, channels, length = x.shape
        x_transposed = x.permute(0, 2, 1).contiguous()
        x_transposed = self.input_norm(x_transposed)
        x_basis = self.basis_function(x_transposed)
        x_basis = x_basis.permute(0, 2, 3, 1).contiguous()
        x_basis = x_basis.view(batch_size, channels * self.num_grids, length)

        ret = self.spline_conv(x_basis)

        if self.use_base_update:
            base = self.base_conv(self.base_activation(x))
            ret = ret + base

        return ret


class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)


class Swish(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x) * x


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, ffn_dim=None):
        super().__init__()
        ffn_dim = ffn_dim or embed_dim * 4

        self.binder_to_target = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.target_to_binder = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.binder_norm1 = nn.LayerNorm(embed_dim)
        self.target_norm1 = nn.LayerNorm(embed_dim)

        self.binder_ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.target_ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )

        self.binder_norm2 = nn.LayerNorm(embed_dim)
        self.target_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, binder, target, binder_mask=None, target_mask=None):
        binder_cross, _ = self.binder_to_target(
            query=binder,
            key=target,
            value=target,
            key_padding_mask=target_mask
        )
        binder = self.binder_norm1(binder + binder_cross)

        target_cross, _ = self.target_to_binder(
            query=target,
            key=binder,
            value=binder,
            key_padding_mask=binder_mask
        )
        target = self.target_norm1(target + target_cross)

        binder = self.binder_norm2(binder + self.binder_ffn(binder))
        target = self.target_norm2(target + self.target_ffn(target))

        return binder, target


class TargetEmbeddingEncoder(nn.Module):
    def __init__(self, esm_hidden_dim, embed_dim=256):
        super().__init__()
        self.esm_hidden_dim = esm_hidden_dim
        self.embed_dim = embed_dim

        self.layer_norm_input = nn.LayerNorm(esm_hidden_dim)
        self.proj = nn.Linear(esm_hidden_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, target_embeddings, target_mask):
        target_embeddings = self.layer_norm_input(target_embeddings)
        encoded = self.proj(target_embeddings)
        encoded = self.layer_norm(encoded)
        return encoded


class ConditionalCNNModel(nn.Module):
    def __init__(self, vocab_size, esm_embed_dim, esm_hidden_dim, embed_dim=256,
                 hidden_dim=256, cross_attn_heads=8, cross_attn_layers=3, max_seq_len=512,
                 kan_num_grids=6, kan_grid_range=(-3., 3.),
                 esm_embedding_weights=None, finetune_embedding=True):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.esm_hidden_dim = esm_hidden_dim

        self.binder_embedding = nn.Embedding(vocab_size, esm_embed_dim)
        if esm_embedding_weights is not None:
            with torch.no_grad():
                self.binder_embedding.weight.copy_(esm_embedding_weights)
            print(f"Initialized binder embedding from ESM weights")
        self.binder_embedding.weight.requires_grad = finetune_embedding

        self.binder_proj = nn.Sequential(
            nn.Linear(esm_embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)

        self.target_encoder = TargetEmbeddingEncoder(
            esm_hidden_dim=esm_hidden_dim,
            embed_dim=embed_dim
        )

        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, cross_attn_heads, dropout=0.1)
            for _ in range(cross_attn_layers)
        ])
        self.cross_attn_norm = nn.LayerNorm(embed_dim)

        self.target_global_proj = nn.Linear(embed_dim, embed_dim)

        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        self.swish = Swish()

        self.embed_to_cnn = nn.Linear(embed_dim, hidden_dim)

        self.kan_blocks = nn.ModuleList([
            FastKANConv1DLayer(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1,
                               num_grids=kan_num_grids, grid_min=kan_grid_range[0], grid_max=kan_grid_range[1]),
            FastKANConv1DLayer(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2,
                               num_grids=kan_num_grids, grid_min=kan_grid_range[0], grid_max=kan_grid_range[1]),
            FastKANConv1DLayer(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4,
                               num_grids=kan_num_grids, grid_min=kan_grid_range[0], grid_max=kan_grid_range[1]),
            FastKANConv1DLayer(hidden_dim, hidden_dim, kernel_size=5, padding=6, dilation=3,
                               num_grids=kan_num_grids, grid_min=kan_grid_range[0], grid_max=kan_grid_range[1]),
            FastKANConv1DLayer(hidden_dim, hidden_dim, kernel_size=3, padding=8, dilation=8,
                               num_grids=kan_num_grids, grid_min=kan_grid_range[0], grid_max=kan_grid_range[1]),
            FastKANConv1DLayer(hidden_dim, hidden_dim, kernel_size=3, padding=9, dilation=9,
                               num_grids=kan_num_grids, grid_min=kan_grid_range[0], grid_max=kan_grid_range[1]),
        ])

        self.time_denses = nn.ModuleList([Dense(embed_dim, hidden_dim) for _ in range(6)])
        self.target_denses = nn.ModuleList([Dense(embed_dim, hidden_dim) for _ in range(6)])
        self.norms = nn.ModuleList([nn.GroupNorm(1, hidden_dim) for _ in range(6)])

        self.final = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, vocab_size, kernel_size=1)
        )

    def get_binder_embeddings(self, binder_ids):
        x_embed = self.binder_embedding(binder_ids)
        x_embed = self.binder_proj(x_embed)
        seq_len = x_embed.size(1)
        x_embed = x_embed + self.pos_encoding[:, :seq_len, :]
        return x_embed

    def forward(self, x, t, target_embeddings=None, target_mask=None):
        batch_size, seq_len = x.shape

        x_embed = self.get_binder_embeddings(x)

        if target_embeddings is not None and target_mask is not None:
            target_encoded = self.target_encoder(target_embeddings, target_mask)

            target_key_padding_mask = ~target_mask.bool()

            for cross_attn_block in self.cross_attn_layers:
                x_embed, target_encoded = cross_attn_block(
                    binder=x_embed,
                    target=target_encoded,
                    target_mask=target_key_padding_mask
                )

            x_embed = self.cross_attn_norm(x_embed)

            mask_expanded = target_mask.unsqueeze(-1).float()
            target_global = (target_encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            target_global = self.target_global_proj(target_global)
        else:
            target_global = torch.zeros(batch_size, self.embed_dim, device=x.device)

        time_embed = self.swish(self.time_embed(t))

        out = self.embed_to_cnn(x_embed)
        out = out.permute(0, 2, 1)
        out = self.swish(out)

        for kan_block, time_dense, target_dense, norm in zip(
                self.kan_blocks, self.time_denses, self.target_denses, self.norms
        ):
            time_cond = time_dense(time_embed)[:, :, None]
            target_cond = target_dense(target_global)[:, :, None]

            conditioned = out + time_cond + target_cond
            h = self.swish(kan_block(norm(conditioned)))
            out = h + out

        out = self.final(out)
        out = out.permute(0, 2, 1)
        out = out - out.mean(dim=-1, keepdim=True)

        return out


class ConditionalWrappedModel(ModelWrapper):
    def forward(self, x, t, **extras):
        target_embeddings = extras.get('target_embeddings', None)
        target_mask = extras.get('target_mask', None)

        logits = self.model(x, t, target_embeddings, target_mask)
        return torch.softmax(logits, dim=-1)


class ConditionalConstrainedWrapper(ModelWrapper):
    def __init__(self, model, target_lengths, alphabet_config):
        super().__init__(model)
        self.target_lengths = target_lengths

        self.CLS_TOKEN = alphabet_config['cls_idx']
        self.PAD_TOKEN = alphabet_config['pad_idx']
        self.EOS_TOKEN = alphabet_config['eos_idx']
        self.AA_START = alphabet_config['aa_start_idx']
        self.AA_END = alphabet_config['aa_end_idx']

    def forward(self, x, t, **extras):
        target_embeddings = extras.get('target_embeddings', None)
        target_mask = extras.get('target_mask', None)

        logits = self.model(x, t, target_embeddings, target_mask)

        if self.target_lengths is not None:
            logits = self.apply_position_constraints(logits, x)

        return torch.softmax(logits, dim=-1)

    def apply_position_constraints(self, logits, current_x):
        batch_size, seq_len, vocab_size = logits.shape

        LARGE_NEGATIVE = -1e10
        LARGE_POSITIVE = 1e10

        for i in range(min(batch_size, len(self.target_lengths))):
            target_length = self.target_lengths[i]
            if torch.is_tensor(target_length):
                target_length = target_length.item()

            target_length = min(target_length, seq_len)

            if target_length <= 0:
                continue

            if seq_len > 0:
                logits[i, 0, :] = LARGE_NEGATIVE
                logits[i, 0, self.CLS_TOKEN] = LARGE_POSITIVE

            if target_length > 1:
                eos_pos = target_length - 1
                logits[i, eos_pos, :] = LARGE_NEGATIVE
                logits[i, eos_pos, self.EOS_TOKEN] = LARGE_POSITIVE

            if target_length > 2:
                logits[i, 1:target_length - 1, self.CLS_TOKEN] = LARGE_NEGATIVE
                logits[i, 1:target_length - 1, self.PAD_TOKEN] = LARGE_NEGATIVE
                logits[i, 1:target_length - 1, self.EOS_TOKEN] = LARGE_NEGATIVE

            if target_length < seq_len:
                logits[i, target_length:, :] = LARGE_NEGATIVE
                logits[i, target_length:, self.PAD_TOKEN] = LARGE_POSITIVE

        return logits


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print("=" * 60)
    print("Model Parameter Statistics")
    print("=" * 60)
    print(f"Total parameters:      {total_params:,}")
    print(f"Trainable parameters:  {trainable_params:,}")
    print(f"Frozen parameters:     {frozen_params:,}")
    print(f"Trainable percentage:  {100 * trainable_params / total_params:.2f}%")
    print("=" * 60)

    return total_params, trainable_params