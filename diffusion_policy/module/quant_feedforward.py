

import torch
import math

"""
Layer:  Linear
Author: cxz21
Data:   2024/10/30
"""


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def fp4_quantizer(weight: torch.Tensor, weight_delta: torch.Tensor):
    w_max = (2 - 2 ** (-1)) * 2 ** (2**2 - 1 - weight_delta)
    w_min = -w_max
    x_R = torch.min(torch.max(weight, w_min), w_max)
    w_log_scales = torch.clamp(
        (torch.floor(torch.log2(torch.abs(x_R) + 1e-5) + weight_delta)).detach(),
        1.0,
    )
    x_scales = 2.0 ** (w_log_scales - 1 - weight_delta)
    x_quant_m = (weight / x_scales).round_()
    x_dequant = x_quant_m.mul_(x_scales)
    return x_dequant / 2.0 ** (-weight_delta) / 2
    # return x_dequant


def int8_quantizer(input: torch.Tensor, input_delta: torch.Tensor):
    x_int = round_ste(input / input_delta)
    x_quant = torch.clamp(x_int, -127, 127)
    return x_quant


def mint8_quantizer(input: torch.Tensor, input_delta: torch.Tensor):
    x_int = round_ste(input * input_delta)
    x_quant = torch.clamp(x_int, -127, 127)
    return x_quant


class LinearFunc(torch.autograd.Function):
    """
    Custom quantizated autograd function for Linear in int8 precision.
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_delta: torch.Tensor,
        weight_delta: torch.Tensor,
        output_delta: torch.Tensor,
        bias=None,
    ):
        quant_input = int8_quantizer(input, input_delta)
        quant_weight = fp4_quantizer(weight, weight_delta)
        # quant_input = input/input_delta
        # quant_weight = weight / 2.0 ** (-weight_delta) / 2
        output = quant_input @ quant_weight.transpose(0, 1)

        scaling_factor = input_delta * (2.0 ** (-weight_delta + 1)) / output_delta
        scaling_factor = scaling_factor.transpose(0, 1)
        quant_output = mint8_quantizer(output, scaling_factor)
        # quant_output = output * scaling_factor
        dequant_output = quant_output * output_delta
        ctx.save_for_backward(
            input, weight, input_delta, weight_delta, output_delta, bias
        )
        return dequant_output

    @staticmethod
    def backward(ctx, dequant_grad_output: torch.Tensor):
        input, weight, input_delta, weight_delta, output_delta, _ = (
            ctx.saved_tensors
        )
        dequant_grad_input = dequant_grad_weight = None
        dequant_grad_input = dequant_grad_output @ weight
        dequant_grad_weight = dequant_grad_output.transpose(-2, -1) @ input
        return dequant_grad_input, dequant_grad_weight, None, None, None, None


class Linear(torch.nn.Module):
    def __init__(self, in_feature: int, out_feature: int, bias: bool = False):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.weight = torch.nn.Parameter(torch.empty((out_feature, in_feature)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_feature))
        else:
            self.register_parameter("bias", None)

        self.weight_n_bits = 4
        self.weight_n_levels = 2 ** (self.weight_n_bits - 1) - 1
        self.weight_delta = None
        self.input_n_bits = 8
        self.input_n_levels = 2 ** (self.input_n_bits - 1) - 1
        self.input_delta = None
        self.output_delta = None
        self.calibration = False
        self.cali_p = 0.05
        self.init = True

    def fp4_init_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta = None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            elif len(x.shape) == 3:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            delta = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                delta[c] = self.fp4_init_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
            elif len(x.shape) == 3:
                delta = delta.view(-1, 1, 1)
            else:
                delta = delta.view(-1, 1)
        else:
            w_max = x.abs().max()
            delta = 2**2 - torch.log2(w_max) + math.log2(2 - 2 ** (-1)) + 0.55
            delta = delta.clone().detach().type_as(x)

        return delta

    def int8_init_scale(self, x: torch.Tensor):
        delta = None

        x_min = min(x.data.min().item(), 0)
        x_max = max(x.data.max().item(), 0)

        x_absmax = max(abs(x_min), x_max)
        delta = x_absmax / self.input_n_levels

        delta = torch.tensor(delta).type_as(x)
        return delta

    def forward(self, input: torch.Tensor, init=False, number=0):
        if self.init is True and self.weight_delta is None:
            weight_ = self.weight.clone().detach()
            input_ = input.clone().detach()
            self.weight_delta = self.fp4_init_scale(weight_, True).requires_grad_(False) 
            self.input_delta = self.int8_init_scale(input_).requires_grad_(False) * 2
            output_ = input_ @ weight_.transpose(0, 1)
            self.output_delta = self.int8_init_scale(output_).requires_grad_(False) * 1.5
            self.init = False
        
        return LinearFunc.apply(
            input,
            self.weight,
            self.input_delta,
            self.weight_delta,
            self.output_delta,
            self.bias,
        )


if __name__ == "__main__":
    in_features = 5
    batch_size = 10
    out_features = 5
    x = torch.randn(batch_size, in_features, requires_grad=True)  # .type(torch.int8)
    y = x.clone().detach().requires_grad_(True)
    w = torch.randn(out_features, in_features, requires_grad=True)
    b = torch.randn(out_features, requires_grad=True).to(torch.bfloat16)

    layer = Linear(in_features, out_features, True)
    layer.weight = torch.nn.Parameter(w)
    # layer.bias = torch.nn.Parameter(b).to(torch.bfloat16)
    out = layer(x)
    outy = y.matmul(w.transpose(0, 1))  # + b  # .type(torch.float32)

    # print("outy: ", outy)

    dout = torch.randn(batch_size, out_features) / 10  # .type(torch.float32)

    fakeloss = (out * dout).sum()
    fakeloss.backward()

    loss = (outy * dout).sum()
    loss.backward()
    # print("out: ", out)
    print("dx: ", x.grad)
    print("dy: ", y.grad)
    print(torch.min(x.grad / y.grad))
    print(torch.max(x.grad / y.grad))
