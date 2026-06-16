# RMSNorm 知识点整理：公式、AMP 陷阱与标准实现

RMSNorm, 全称 Root Mean Square Normalization, 是 LLaMA、Gemma 等大语言模型里常见的归一化层。它可以看作是 LayerNorm 的一个工程友好版本：不再显式减去均值，只用均方根来约束激活的尺度，再乘上一个可学习的缩放参数。

这篇文章整理 RMSNorm 在实现和面试中最容易被问到的几个点：

- RMSNorm 解决了 LayerNorm 的什么问题
- RMSNorm 的计算公式和张量维度
- AMP / FP16 下为什么要先转 `float32`
- PyTorch 标准实现应该怎么写
- 为什么实现里用 `rsqrt` 而不是 `1 / sqrt`
- 为什么归一化后的乘法可以放在低精度下做

## 1. RMSNorm 解决的问题

标准 LayerNorm 对一个 token 的 hidden dimension 做归一化。给定向量 `x in R^d`，LayerNorm 通常会做两件事：

1. 减去均值，让特征重新居中。
2. 除以标准差，让特征尺度稳定。

公式是：

$$
\mu = \frac{1}{d}\sum_{i=1}^{d}x_i
$$

$$
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2
$$

$$
y_i = \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}\cdot \gamma_i + \beta_i
$$

LayerNorm 的好处是训练稳定，但它也有代价：

- 需要计算均值 `mean(x)`。
- 需要做居中操作 `x - mean(x)`。
- 需要再计算方差。
- 通常还带有 `gamma` 和 `beta` 两组参数。
- 在 GPU 上，这些 reduction 和逐元素操作会带来额外同步、访存和 kernel 开销。

RMSNorm 的核心观察是：在大模型的深层网络中，激活的均值通常已经比较接近 0。既然如此，可以不强制做 re-centering, 只做 re-scaling。也就是说，RMSNorm 保留“控制激活尺度”的能力，但去掉“减均值”的步骤。

RMSNorm 简化后的特点是：

- 不计算 `mean(x)` 作为中心化项。
- 不做 `x - mean(x)`。
- 用 `mean(x^2)` 直接衡量向量尺度。
- 通常只有一个可学习参数 `weight / gamma`，没有 bias。
- 前向和反向都更简单，更适合大模型中的高频调用。

一句话总结：RMSNorm 不是为了改变模型表达能力的上限，而是为了用更少的计算和访存，获得接近 LayerNorm 的训练稳定性。

## 2. RMSNorm 计算公式

给定输入向量：

$$
x \in \mathbb{R}^{d}
$$

RMSNorm 先计算均方根：

$$
\operatorname{RMS}(x)=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}
$$

其中 `epsilon` 是一个很小的常数，用于防止除零，常见取值如 `1e-6`。

然后做归一化和缩放：

$$
y_i = \frac{x_i}{\operatorname{RMS}(x)}\cdot \gamma_i
$$

也可以写成更接近代码的形式：

$$
\operatorname{variance} = \frac{1}{d}\sum_{i=1}^{d}x_i^2
$$

$$
\operatorname{inv\_rms} = \frac{1}{\sqrt{\operatorname{variance}+\epsilon}}
$$

$$
y = x \cdot \operatorname{inv\_rms} \odot \gamma
$$

其中：

- `x` 的形状通常是 `[batch_size, seq_len, hidden_size]`。
- RMSNorm 沿最后一维 `hidden_size` 计算。
- `variance` 的形状是 `[batch_size, seq_len, 1]`，需要 `keepdim=True` 方便广播。
- `gamma / weight` 的形状是 `[hidden_size]`。
- RMSNorm 通常没有 bias。

因此，PyTorch 里的关键代码就是：

```python
variance = x.pow(2).mean(dim=-1, keepdim=True)
x_norm = x * torch.rsqrt(variance + eps)
y = x_norm * weight
```

不过，这段代码还不够安全。真正的大模型实现里必须处理混合精度问题。

## 3. AMP 下的数值问题

现代大模型训练和推理通常会使用 AMP, FP16 或 BF16，以降低显存占用并提高吞吐。但 RMSNorm 有一个典型陷阱：第一步要计算 `x^2`。

FP16 的最大有限值大约是：

```text
65504
```

如果某个输入值大于 `256`，那么平方后就可能溢出：

$$
256^2 = 65536 > 65504
$$

这意味着在 FP16 中：

```python
x_fp16.pow(2)
```

可能直接变成 `inf`。接下来 `mean` 会得到 `inf`，`rsqrt(inf)` 会得到接近 0 的值，严重时后续计算会传播出 `NaN`，导致训练不稳定甚至崩溃。

所以 RMSNorm 的标准工程写法是：在平方、求均值和求倒数平方根之前，先把输入 upcast 到 `float32`。

```python
x_fp32 = x.float()
variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
x_norm = x_fp32 * torch.rsqrt(variance + eps)
```

这样做的意义是：

- `pow(2)` 在 FP32 下不容易溢出。
- reduction 的累加误差更小。
- `rsqrt` 的输入更稳定。
- 最终结果可以再转回输入 dtype，以保持和模型其他部分一致。

BF16 的情况略有不同。BF16 的指数位和 FP32 一样，动态范围比 FP16 大很多，所以溢出风险小得多。但 BF16 的尾数精度低，reduction 仍然更适合用 FP32 做。因此，在通用 RMSNorm 实现里，不管输入是 FP16 还是 BF16，核心统计量一般都用 FP32 计算。

## 4. 标准 PyTorch 实现

下面是一个接近 HuggingFace LLaMA 风格的 RMSNorm 实现。重点是：统计量用 FP32，输出 dtype 与输入保持一致。

```python
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x if x.dtype == torch.float32 else x.float()
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        return x_fp32 * torch.rsqrt(variance + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x) * self.weight.float()
        return output.to(dtype=x.dtype)
```

也可以写成更贴近 HuggingFace 常见风格的版本：

```python
class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype

        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)
```

这个实现里有几个细节值得强调。

第一，`weight` 初始化为全 1。这样模型刚开始训练时，RMSNorm 的缩放不会破坏归一化后的尺度。

第二，`mean(dim=-1, keepdim=True)` 必须保留最后一维。假设输入是 `[B, S, H]`，那么 `variance` 应该是 `[B, S, 1]`，这样才能和原始输入按 hidden dimension 广播相乘。

第三，`_norm` 中间结果保持 FP32，不要过早转回 FP16。因为归一化后还要乘 `weight`，过早降精度会引入额外舍入误差。

第四，最后输出转回 `input_dtype`。这能让 RMSNorm 和模型后续层保持一致，避免无意中把整个计算图推到 FP32，造成显存和带宽开销上升。

## 5. 为什么用 `rsqrt` 更快

RMSNorm 需要的是：

$$
\frac{1}{\sqrt{x}}
$$

直接写成代码可以是：

```python
inv_rms = 1.0 / torch.sqrt(variance + eps)
```

但更推荐：

```python
inv_rms = torch.rsqrt(variance + eps)
```

原因有三点。

第一，`rsqrt` 表达的就是 reciprocal square root, 也就是“倒数平方根”。它直接对应目标数学含义，不需要先 `sqrt` 再 `div`。

第二，`sqrt + reciprocal/div` 通常会拆成两个操作：先计算平方根，再做除法。`rsqrt` 可以让底层编译器或 GPU 后端直接选择倒数平方根相关指令或更短的计算路径。

第三，RMSNorm 是典型的高频小算子。单次节省可能很小，但大模型每层、每个 token、前向反向都要做归一化，累计收益就明显了。尤其是在 fused RMSNorm kernel 中，`rsqrt` 能更自然地和后续乘法组合：

```python
y = x * torch.rsqrt(variance + eps) * weight
```

从计算图角度看，`rsqrt` 也给编译器更清晰的优化机会。它说明我们真正需要的是倒数平方根，而不是需要单独保留平方根结果。

## 6. 为什么最后的乘法可以低精度做

RMSNorm 最危险的地方是 `x^2`，不是归一化后的 `x_norm * weight`。

原因是 `_norm(x)` 之后，向量的 RMS 被压到了接近 1：

$$
\operatorname{RMS}(x_{\text{norm}}) \approx 1
$$

这意味着归一化后的大多数元素会落在一个很小的范围内。粗略理解，可以认为绝大多数值通常在 `[-3, 3]` 附近。虽然这不是严格边界，但它足以说明尺度已经被控制住了。

然后再乘以 `weight`。RMSNorm 的 `weight` 初始化为 1，训练过程中通常也不会变成非常大的数，很多情况下会在 1 附近波动，例如 `[0.5, 2.0]` 这样的量级。

于是最终乘法的数值范围通常类似：

```text
[-3, 3] * [0.5, 2.0] -> 大致 [-6, 6]
```

这和 FP16 的最大有限值 `65504` 相比差得非常远。因此，归一化后的逐元素乘法即使用 FP16 / BF16 做，溢出风险也很低。

换句话说：

- `x^2` 前的 `x` 可能还没有被尺度约束，平方会放大异常值，所以必须谨慎。
- `x_norm` 已经被 RMSNorm 压到 RMS 约等于 1，再乘 `weight` 通常很安全。

这也是为什么很多工业实现会采用这样的精度策略：

1. 输入是 FP16 / BF16。
2. 统计量计算转 FP32。
3. 归一化中间结果可以保持 FP32。
4. 最终输出转回 FP16 / BF16。
5. 在融合 kernel 中，最后写回低精度输出，节省带宽和显存。

需要注意的是，“可以低精度做”不代表所有中间计算都应该低精度做。核心原则是：涉及平方和 reduction 的统计量用高精度，已经被归一化约束住的逐元素乘法可以回到低精度。

## 7. RMSNorm 实现检查清单

写 RMSNorm 时，可以用下面这个清单快速检查：

- `weight` 形状是否为 `[hidden_size]`。
- `weight` 是否初始化为全 1。
- 是否沿最后一维 `dim=-1` 计算均方。
- `mean(..., keepdim=True)` 是否保留维度，方便广播。
- `pow(2).mean(...)` 前是否把输入转成 FP32。
- 是否使用 `torch.rsqrt(variance + eps)`。
- 是否没有额外 bias。
- 输出 dtype 是否和输入 dtype 一致。
- 是否避免过早把 `_norm` 的中间结果转回低精度。

一个正确的最小版本如下：

```python
class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype

        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)

        return (x * self.weight.float()).to(input_dtype)
```

## 8. 总结

RMSNorm 的本质是：去掉 LayerNorm 中的均值中心化，只保留基于均方根的尺度归一化。

它解决的主要问题不是“LayerNorm 不稳定”，而是 LayerNorm 在大模型中调用频率极高，计算均值、中心化、方差和额外参数都会带来开销。RMSNorm 用更少的 reduction、更少的逐元素操作和更简单的参数结构，在大多数 Transformer 场景下获得了足够好的稳定性。

工程实现时，最重要的细节是混合精度。RMSNorm 里的 `x^2` 很容易在 FP16 下溢出，因此统计量必须 upcast 到 FP32。归一化完成后，数值尺度已经被控制到 RMS 约等于 1，最后乘 `weight` 并转回低精度通常是安全的。

最后，`torch.rsqrt` 是 RMSNorm 里更自然也更高效的写法。因为我们真正需要的是 `1 / sqrt(x)`，直接使用 `rsqrt` 可以减少操作表达，给底层编译器和 GPU 指令选择更好的优化空间。
