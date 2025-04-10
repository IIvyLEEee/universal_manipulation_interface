import torch
import torch.nn as nn

import torch.nn.functional as F
from diffusion_policy.module.quant_linear import Linear

from diffusion_policy.module.softmax import SoftmaxFunc
from torch.autograd import Function
from torch.nn.modules.activation import MultiheadAttention


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def int8_quantizer(input: torch.Tensor, input_delta: torch.Tensor):
    x_int = round_ste(input / input_delta)
    x_quant = torch.clamp(x_int, -127, 127)
    return x_quant


def mint8_quantizer(input: torch.Tensor, input_delta: torch.Tensor):
    x_int = round_ste(input * input_delta)
    x_quant = torch.clamp(x_int, -127, 127)
    return x_quant


class MultiHeadAttentionFunc1(Function):
    @staticmethod
    def forward(ctx, queries, keys, num_heads, atten_mask, q_delta, k_delta):
        _, key_len, _ = keys.shape
        N, query_len, embed_size = queries.shape

        head_dim = embed_size // num_heads
        keys = keys.reshape(N, key_len, num_heads, head_dim)
        queries = queries.reshape(N, query_len, num_heads, head_dim) / (
            head_dim ** (1 / 2)
        )
        queries = int8_quantizer(queries, q_delta)
        # queries = queries / q_delta
        keys = int8_quantizer(keys, k_delta)
        # keys = keys / k_delta
        energy = torch.einsum("nqhd,nlhd->nhql", [queries, keys])
        atten_mask = atten_mask.unsqueeze(0).unsqueeze(0)
        if atten_mask is not None:
            energy = energy + atten_mask
        energy = energy * q_delta * k_delta
        ctx.save_for_backward(queries, keys, atten_mask)
        ctx.embed_size = embed_size
        return energy

    @staticmethod
    def backward(ctx, grad_output):
        # print(grad_output)
        queries, keys, _ = ctx.saved_tensors
        embed_size = ctx.embed_size
        N, num_heads, query_len, keys_len = grad_output.shape
        head_dim = embed_size // num_heads
        grad_attention = grad_output / (head_dim ** (1 / 2))
        grad_keys = torch.einsum("nhql,nqhd->nlhd", [grad_attention, queries])
        grad_keys = grad_keys.reshape(N, keys_len, embed_size)

        grad_queries = torch.einsum("nhql,nlhd->nqhd", [grad_attention, keys])
        grad_queries = grad_queries.reshape(N, query_len, embed_size)
        return grad_queries, grad_keys, None, None, None, None


class MultiHeadAttentionFunc2(Function):
    @staticmethod
    def forward(ctx, qk, values, num_heads, v_delta, qk_delta, qkv_delta):
        N, _, query_len, _ = qk.shape
        _, value_len, embed_size = values.shape

        qk = int8_quantizer(qk, qk_delta)
        # qk = qk / qk_delta
        values = int8_quantizer(values, v_delta)
        # values = values / v_delta
        head_dim = embed_size // num_heads
        values = values.reshape(N, value_len, num_heads, head_dim)
        quant_out = torch.einsum("nhql,nlhd->nqhd", [qk, values]).reshape(
            N, query_len, embed_size
        )
        quant_out = quant_out * qk_delta * v_delta
        out = int8_quantizer(quant_out, qkv_delta)
        # out = quant_out / qkv_delta
        dequant_out = out * qkv_delta
        ctx.save_for_backward(qk, values)
        ctx.head_dim = head_dim
        ctx.num_heads = num_heads
        return dequant_out

    @staticmethod
    def backward(ctx, grad_output):
        qk, values = ctx.saved_tensors
        N, query_len, embed_size = grad_output.shape
        head_dim = ctx.head_dim
        num_heads = ctx.num_heads
        grad_output = grad_output.reshape(N, query_len, num_heads, head_dim)
        grad_qk = torch.einsum("nqhd,nlhd->nhql", [grad_output, values])
        grad_values = torch.einsum("nqhd,nhql->nlhd", [grad_output, qk])
        grad_values = grad_values.reshape(N, -1, embed_size)

        return grad_qk, grad_values, None, None, None, None


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.0, batch_first=True):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.dropout = dropout
        self.batch_first = batch_first

        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = Linear(embed_size, embed_size, bias=False)
        self.keys = Linear(embed_size, embed_size, bias=False)
        self.queries = Linear(embed_size, embed_size, bias=False)
        self.fc_out = Linear(embed_size, embed_size, bias=False)
        self.softmax = nn.Softmax(dim=3)
        self.dropout = nn.Dropout(self.dropout)
        self.input_n_bits = 8
        self.input_n_levels = 2 ** (self.input_n_bits - 1) - 1
        self.q_delta = None
        self.k_delta = None
        self.v_delta = None
        self.qk_delta = None
        self.qkv_delta = None
        self.init = True

    def int8_init_scale(self, x: torch.Tensor):
        delta = None

        x_min = min(x.data.min().item(), 0)
        x_max = max(x.data.max().item(), 0)
        # x_min = x_min * (self.input_n_bits + 2) / 8
        # x_max = x_max * (self.input_n_bits + 2) / 8

        x_absmax = max(abs(x_min), x_max)
        delta = x_absmax / self.input_n_levels

        delta = torch.tensor(delta).type_as(x)
        return delta

    def forward(self, queries, keys, values, attn_mask=None):
        values = self.values(values)
        # print("input:", keys.max(), keys.min())
        keys = self.keys(keys)
        # print("output:", keys.max(), keys.min())
        queries = self.queries(queries)
        if self.init is True:
            q_ = queries.clone().detach()
            k_ = keys.clone().detach()
            v_ = values.clone().detach()
            _, key_len, _ = k_.shape
            N, query_len, embed_size = q_.shape
            head_dim = embed_size // self.num_heads
            k_ = k_.reshape(N, key_len, self.num_heads, head_dim)
            q_ = q_.reshape(N, query_len, self.num_heads, head_dim) / (
                head_dim ** (1 / 2)
            )
            qk_ = torch.einsum("nqhd,nlhd->nhql", [q_, k_])
            attn_mask_ = attn_mask.unsqueeze(0).unsqueeze(0)
            if attn_mask_ is not None:
                qk_ = qk_ + attn_mask_

            qk_softmax_ = F.softmax(qk_, 3, _stacklevel=5)
            qk_dropout_ = self.dropout(qk_softmax_)

            _, value_len, _ = v_.shape
            v_ = v_.reshape(N, value_len, self.num_heads, head_dim)
            qkv_ = torch.einsum("nhql,nlhd->nqhd", [qk_dropout_, v_]).reshape(
                N, query_len, embed_size
            )
            self.q_delta = self.int8_init_scale(q_) * 2
            self.k_delta = self.int8_init_scale(k_) * 2
            self.v_delta = self.int8_init_scale(v_)  * 2
            self.qk_delta = self.int8_init_scale(qk_dropout_) * 2
            self.qkv_delta = self.int8_init_scale(qkv_) * 2
            self.init = False

        qk = MultiHeadAttentionFunc1.apply(
            queries, keys, self.num_heads, attn_mask, self.q_delta, self.k_delta
        )
        qk_softmax = self.softmax(qk)
        qk_dropout = self.dropout(qk_softmax)
        qkv = MultiHeadAttentionFunc2.apply(
            qk_dropout,
            values,
            self.num_heads,
            self.v_delta,
            self.qk_delta,
            self.qkv_delta,
        )
        out = self.fc_out(qkv)
        return out


if __name__ == "__main__":
    embed_size = 256
    num_heads = 4
    dropout_p = 0.0
    batch_size = 56
    sequence_length = 10

    x1 = torch.randn(batch_size, sequence_length, embed_size, requires_grad=True)
    y1 = x1
    x2 = torch.randn(batch_size, sequence_length, embed_size, requires_grad=True)
    y2 = x2
    x3 = torch.randn(batch_size, sequence_length, embed_size, requires_grad=True)
    y3 = x3

    in_proj_weight = nn.Parameter(
        torch.randn(3 * embed_size, embed_size, requires_grad=True)
    )
    out_proj_weight = nn.Parameter(
        torch.randn(embed_size, embed_size, requires_grad=True)
    )

    atten_mask = torch.randn(sequence_length, sequence_length)

    true_layer = MultiheadAttention(embed_size, num_heads, dropout_p, False)
    true_layer.in_proj_weight = nn.Parameter(in_proj_weight)
    true_layer.out_proj.weight = nn.Parameter(out_proj_weight)
    out = true_layer(x1, x2, x3, atten_mask)[0]
    dout = torch.randn_like(out)
    loss = (out * dout).sum()
    loss.backward()

    attn_layer = MultiHeadAttention(
        embed_size=embed_size, num_heads=num_heads, dropout_p=dropout_p
    )
    attn_layer.queries.weight = nn.Parameter(in_proj_weight[0:embed_size, :])
    attn_layer.keys.weight = nn.Parameter(
        in_proj_weight[embed_size : 2 * embed_size, :]
    )
    attn_layer.values.weight = nn.Parameter(
        in_proj_weight[2 * embed_size : 3 * embed_size, :]
    )
    attn_layer.fc_out.weight = nn.Parameter(out_proj_weight)
    attn_layer.queries.bias = None
    attn_layer.keys.bias = None
    attn_layer.values.bias = None
    attn_layer.fc_out.bias = None
    fake_out = attn_layer(y1, y2, y3, atten_mask)

    # print("out:\n", out)
    # print("fake_out:\n", fake_out)
    print(out - fake_out)