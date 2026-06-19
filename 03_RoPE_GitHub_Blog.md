# RoPE 知识点整理：旋转位置编码、复数实现与上下文外推

RoPE, Rotary Position Embedding, 中文通常叫旋转位置编码，是 LLaMA、Qwen、DeepSeek、Mistral 等主流大语言模型中最常见的位置编码方式之一。它不像原始 Transformer 那样把位置编码直接加到 token embedding 上，而是在 Attention 计算前，对 Query 和 Key 做和位置相关的旋转。

这篇文章整理 RoPE 在实现和面试中最容易被问到的几个点：

- 为什么需要 RoPE
- RoPE 和绝对位置编码有什么区别
- 为什么旋转可以表达相对位置信息
- RoPE 的二维旋转矩阵公式是什么
- 如何用复数乘法高效实现 RoPE
- `precompute_freqs_cis` 和 `apply_rotary_emb` 应该怎么写
- 为什么实现中要把输入临时转成 FP32
- RoPE Scaling 如何支持更长上下文
- 实现时有哪些常见错误

## 1. 为什么需要位置编码

Transformer 的 Attention 本身是 permutation-invariant 的。也就是说，如果只看自注意力公式：

$$
\mathrm{Attention}(Q,K,V)=\mathrm{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

模型可以看到 token 之间的内容相似度，但不知道 token 出现在第几个位置。

例如下面两个序列：

```text
我 喜欢 你
你 喜欢 我
```

如果没有位置信息，模型很难区分“我”和“你”的顺序变化。对于语言模型来说，顺序就是语义的一部分，所以必须把位置信息注入模型。

原始 Transformer 使用的是绝对位置编码。它会为每个位置生成一个位置向量，然后和 token embedding 相加：

```text
hidden_states = token_embedding + position_embedding
```

这种方法简单，但有一个问题：模型学到的更像是“第 10 个位置是什么样”，而不是“两个 token 相隔多远”。当推理长度超过训练长度时，绝对位置编码往往泛化较弱。

RoPE 的目标是：让 Attention 在计算 Query 和 Key 的内积时，自然感知相对距离。

## 2. RoPE 的核心思想

RoPE 的核心思想可以用一句话概括：

> 对不同位置的 Query 和 Key 施加不同角度的旋转，使它们的点积天然依赖相对位置差。

设 token 位于位置 `m`，另一个 token 位于位置 `n`。RoPE 不直接给它们加一个位置向量，而是分别对它们做旋转：

$$
q_m' = R_m q
$$

$$
k_n' = R_n k
$$

其中 $R_m$ 和 $R_n$ 是由位置决定的旋转矩阵。Attention score 计算的是：

$$
(q_m')^T k_n'
$$

代入旋转矩阵后：

$$
(R_m q)^T(R_n k)=q^T R_m^T R_n k
$$

二维旋转矩阵有一个性质：

$$
R_m^T R_n = R_{n-m}
$$

所以：

$$
(q_m')^T k_n'=q^T R_{n-m}k
$$

这说明旋转后的点积不再只依赖内容 `q` 和 `k`，还依赖位置差 `n - m`。这就是 RoPE 能表达相对位置的关键。

一句话总结：

```text
绝对位置编码：把“当前位置是多少”加到 hidden states 里。
RoPE：让 Q 和 K 的内积天然包含“两个位置相差多少”。
```

## 3. 二维旋转矩阵

先看二维向量的旋转。给定向量：

$$
x = \begin{bmatrix}x_1 \\ x_2\end{bmatrix}
$$

把它旋转角度 $\theta$，可以使用旋转矩阵：

$$
R_\theta =
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
$$

旋转后的结果是：

$$
R_\theta x =
\begin{bmatrix}
x_1\cos\theta - x_2\sin\theta \\
x_1\sin\theta + x_2\cos\theta
\end{bmatrix}
$$

RoPE 会把 head dimension 中的相邻两个元素看成一组二维向量：

```text
[x0, x1, x2, x3, x4, x5, ...]
 -> [(x0, x1), (x2, x3), (x4, x5), ...]
```

每一组二维向量使用不同频率的角度旋转。位置越大，旋转角越大；维度频率不同，旋转速度也不同。

## 4. RoPE 的频率设计

设 head dimension 为 `dim`，RoPE 会对最后一维两两配对，所以一共有：

$$
\frac{dim}{2}
$$

组二维向量。

每组向量使用一个频率。常见公式是：

$$
\theta_i = 10000^{-\frac{2i}{dim}}
$$

其中：

- `i` 是配对后的维度索引。
- `dim` 是每个 attention head 的维度，也就是 head_dim。
- `10000` 是原始 Transformer 位置编码中沿用下来的基频参数。

在代码中常写成：

```python
freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
```

这里 `torch.arange(0, dim, 2)` 生成的是：

```text
0, 2, 4, ..., dim - 2
```

把它除以 `dim` 后，正好对应公式里的：

$$
\frac{2i}{dim}
$$

对于每个位置 `m`，旋转角度是：

$$
m\theta_i
$$

所以所有位置和所有频率组成一个角度矩阵：

$$
\mathrm{freqs}_{m,i}=m\theta_i
$$

对应代码是：

```python
t = torch.arange(end, dtype=torch.float32)
freqs = torch.outer(t, freqs)
```

如果 `end = seq_len`，`freqs` 的形状就是：

```text
[seq_len, dim // 2]
```

## 5. 为什么可以用复数实现

二维旋转和复数乘法是等价的。

把二维向量：

$$
(x_1, x_2)
$$

看成一个复数：

$$
z = x_1 + i x_2
$$

再构造单位复数：

$$
e^{i\theta}=\cos\theta+i\sin\theta
$$

两者相乘：

$$
z' = z \cdot e^{i\theta}
$$

展开可得：

$$
(x_1+i x_2)(\cos\theta+i\sin\theta)
$$

$$
= (x_1\cos\theta - x_2\sin\theta)
+ i(x_1\sin\theta + x_2\cos\theta)
$$

这正好等价于二维旋转矩阵：

$$
\begin{bmatrix}
x_1\cos\theta - x_2\sin\theta \\
x_1\sin\theta + x_2\cos\theta
\end{bmatrix}
$$

因此，RoPE 可以不显式构造旋转矩阵，而是：

1. 把实数张量最后一维两两组合成复数。
2. 预计算复数旋转因子 $e^{im\theta}$。
3. 用一次复数乘法完成旋转。
4. 再把复数结果恢复成实数张量。

这就是 notebook 中 `torch.view_as_complex` 和 `torch.view_as_real` 的核心意义。

## 6. 预计算复数频率

RoPE 中与位置有关的复数旋转因子通常会预先计算好，避免每次 forward 都重复生成。

标准实现如下：

```python
import torch


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    计算 RoPE 的复数旋转因子。

    Args:
        dim: head_dim，必须是偶数。
        end: 最大序列长度。
        theta: RoPE 基频，常见默认值是 10000。

    Returns:
        freqs_cis: shape 为 [end, dim // 2] 的 complex64 张量。
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis
```

这里的关键点是 `torch.polar`。

`torch.polar(abs, angle)` 会根据极坐标生成复数：

$$
abs \cdot (\cos(angle)+i\sin(angle))
$$

RoPE 只需要旋转，不需要改变向量模长，所以 `abs` 使用全 1：

```python
torch.ones_like(freqs)
```

于是：

```python
freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
```

生成的就是：

$$
e^{im\theta_i} = \cos(m\theta_i) + i\sin(m\theta_i)
$$

其中 `cis` 通常表示：

```text
cos + i * sin
```

## 7. 广播形状设计

Attention 中 Query 和 Key 的常见形状是：

```text
[batch_size, seq_len, num_heads, head_dim]
```

当把最后一维两两组成复数后，形状变为：

```text
[batch_size, seq_len, num_heads, head_dim // 2]
```

而预计算的 `freqs_cis` 形状是：

```text
[seq_len, head_dim // 2]
```

为了让它能和 Query / Key 广播相乘，需要把它 reshape 成：

```text
[1, seq_len, 1, head_dim // 2]
```

也就是在 batch 和 num_heads 维度上广播。

对应函数可以写成：

```python
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)
```

如果 `x` 是复数形式的 Query，形状为：

```text
[B, S, H, D // 2]
```

那么这个函数会把 `freqs_cis` 变成：

```text
[1, S, 1, D // 2]
```

这样：

```python
x * freqs_cis
```

就会自动在 batch 和 head 维度上广播。

## 8. 应用 RoPE 到 Query 和 Key

完整的复数版 RoPE 实现如下：

```python
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将 RoPE 应用到 Query 和 Key 上。

    xq, xk shape: [batch_size, seq_len, num_heads, head_dim]
    freqs_cis shape: [seq_len, head_dim // 2]
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)
```

这段代码可以拆成四步理解。

第一步，把输入临时转成 FP32：

```python
xq.float()
xk.float()
```

这是为了让复数乘法更稳定，尤其是输入为 FP16 / BF16 时。

第二步，把最后一维拆成复数对：

```python
xq.float().reshape(*xq.shape[:-1], -1, 2)
```

如果原始 shape 是：

```text
[B, S, H, D]
```

reshape 后变为：

```text
[B, S, H, D // 2, 2]
```

最后那个大小为 2 的维度分别表示实部和虚部。

第三步，用 `torch.view_as_complex` 把实数张量解释成复数张量：

```python
xq_ = torch.view_as_complex(...)
```

形状变为：

```text
[B, S, H, D // 2]
```

第四步，乘以旋转因子，然后恢复为实数：

```python
xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
```

`torch.view_as_real` 会把复数张量重新变成：

```text
[B, S, H, D // 2, 2]
```

最后 `flatten(3)` 把最后两个维度合并回：

```text
[B, S, H, D]
```

## 9. 为什么 RoPE 不改变向量模长

RoPE 是旋转操作，而旋转不会改变向量长度。

二维情况下：

$$
x' = R_\theta x
$$

旋转矩阵是正交矩阵，因此：

$$
\|x'\|_2 = \|x\|_2
$$

复数角度看也一样。RoPE 使用的是单位复数：

$$
e^{i\theta}
$$

它的模长是：

$$
|e^{i\theta}|=1
$$

所以：

$$
|z \cdot e^{i\theta}| = |z| \cdot |e^{i\theta}| = |z|
$$

这也是测试代码中会检查：

```python
norm_before = torch.norm(xq, dim=-1)
norm_after = torch.norm(xq_out, dim=-1)
assert torch.allclose(norm_before, norm_after)
```

如果 RoPE 后向量模长发生明显变化，通常说明旋转实现、维度配对或 dtype 处理有问题。

## 10. 为什么只作用在 Query 和 Key 上

RoPE 通常只应用在 Query 和 Key 上，而不是 Value 上。

原因是位置信息主要影响 Attention score 的计算：

$$
QK^T
$$

Attention score 决定每个 token 应该关注哪些位置。只要 Query 和 Key 的点积包含相对位置信息，注意力权重就能感知 token 之间的距离。

Value 负责提供被加权汇聚的内容。如果把 RoPE 也作用到 Value 上，会改变实际被汇聚的内容表示，通常不是必要的，也不符合主流 LLaMA 风格实现。

所以常见流程是：

```text
hidden_states
 -> q_proj, k_proj, v_proj
 -> apply_rope(q, k)
 -> attention(q_rot, k_rot, v)
```

## 11. FP16 / BF16 下的精度细节

notebook 中强调了一个工程细节：在执行 RoPE 的复数转换和复数乘法前，最好先把输入转为 FP32。

代码是：

```python
xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
```

这样做有几个原因：

- 复数乘法内部包含多次乘法和加减法，低精度下更容易累积误差。
- FP16 的数值范围和精度都有限，长序列、高频旋转时更容易出现不稳定。
- 预计算的 `freqs_cis` 通常是 `complex64`，和 FP32 实部/虚部更匹配。
- RoPE 后还会进入 Attention score 计算，前面的误差会继续影响 softmax。

最终输出再转回输入 dtype：

```python
return xq_out.type_as(xq), xk_out.type_as(xk)
```

这样既保证了旋转计算的稳定性，又不会让后续模型无意中全部变成 FP32，避免额外显存和带宽开销。

一句话总结：

```text
RoPE 旋转计算临时用 FP32，输出恢复到原始 dtype。
```

## 12. 快速测试

可以用下面的代码检查 RoPE 实现是否正确。

```python
def test_rope():
    batch_size, seq_len, num_heads, head_dim = 2, 16, 4, 64

    xq = torch.randn(batch_size, seq_len, num_heads, head_dim)
    xk = torch.randn(batch_size, seq_len, num_heads, head_dim)

    freqs_cis = precompute_freqs_cis(head_dim, seq_len)
    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)

    assert freqs_cis.shape == (seq_len, head_dim // 2)
    assert xq_out.shape == xq.shape
    assert xk_out.shape == xk.shape

    assert not torch.allclose(xq, xq_out, atol=1e-5)

    norm_before = torch.norm(xq, dim=-1)
    norm_after = torch.norm(xq_out, dim=-1)
    assert torch.allclose(norm_before, norm_after, rtol=1e-4, atol=1e-5)

    assert not torch.isnan(xq_out).any()
    assert not torch.isinf(xq_out).any()

    xq_fp16 = torch.randn(1, 8, 2, head_dim, dtype=torch.float16)
    xk_fp16 = torch.randn(1, 8, 2, head_dim, dtype=torch.float16)
    freqs_fp16 = precompute_freqs_cis(head_dim, 8)

    xq_out_fp16, xk_out_fp16 = apply_rotary_emb(xq_fp16, xk_fp16, freqs_fp16)

    assert xq_out_fp16.dtype == torch.float16
    assert not torch.isnan(xq_out_fp16).any()

    print("RoPE tests passed.")
```

这个测试覆盖了几个关键点：

- 输出 shape 是否和输入一致。
- 预计算频率 shape 是否正确。
- RoPE 是否真的改变了输入表示。
- 旋转是否保持向量模长。
- 输出是否没有 NaN / Inf。
- FP16 输入是否能正确恢复 FP16 输出。

## 13. 常见实现错误

### 13.1 head_dim 不是偶数

RoPE 会把最后一维两两配对成复数：

```text
[x0, x1], [x2, x3], ...
```

因此 `head_dim` 必须是偶数。如果 `head_dim` 是奇数，最后一个元素无法配对，`reshape(..., -1, 2)` 会失败。

### 13.2 忘记把最后一维拆成 `[dim // 2, 2]`

`torch.view_as_complex` 要求输入最后一维大小为 2，分别表示实部和虚部。

正确写法：

```python
xq.float().reshape(*xq.shape[:-1], -1, 2)
```

错误写法通常是直接：

```python
torch.view_as_complex(xq)
```

如果 `xq` 的最后一维不是 2，就会报错或语义错误。

### 13.3 freqs_cis 广播形状不对

`freqs_cis` 原始形状是：

```text
[seq_len, head_dim // 2]
```

用于 `[B, S, H, D // 2]` 的复数 Query / Key 时，应该 reshape 为：

```text
[1, S, 1, D // 2]
```

如果错误 reshape 成 `[S, 1, D // 2]` 或其他形状，可能会广播失败，也可能静默广播到错误维度。

### 13.4 忘记 `flatten(3)`

`torch.view_as_real` 后的形状是：

```text
[B, S, H, D // 2, 2]
```

必须把最后两个维度合并回：

```text
[B, S, H, D]
```

对应代码：

```python
torch.view_as_real(xq_ * freqs_cis).flatten(3)
```

如果忘记 `flatten(3)`，后续 Attention 会收到五维张量。

### 13.5 对 Value 也应用 RoPE

LLaMA 风格 RoPE 只应用到 Query 和 Key。Value 通常不旋转。把 RoPE 加到 Value 上，会改变被 attention 权重汇聚的内容表示，不是常见实现。

### 13.6 全程使用 FP16 做复数乘法

低精度复数乘法可能造成数值误差放大。更稳妥的写法是：

```python
x.float()
```

完成旋转后再：

```python
type_as(x)
```

这也是很多开源模型实现中常见的工程处理方式。

## 14. RoPE 与上下文外推

RoPE 还有一个重要话题：上下文扩展，也就是 context extension。

假设模型训练时只见过 4K 长度，但推理时希望支持 16K、32K 甚至 128K。直接把位置 `m` 扩展到训练外长度会导致旋转角进入模型没见过的范围，Attention 行为可能明显退化。

为了解决这个问题，工业界提出了多种 RoPE Scaling 方法。

### 14.1 Linear Scaling

线性插值的思想是压缩位置索引。例如把原始位置：

```text
m
```

替换成：

```text
m / scale
```

这样更长的上下文会被压缩回模型训练时更熟悉的位置范围。

直观理解：

```text
原本 0 到 32768 的位置
压缩成 0 到 4096 附近
```

优点是简单，缺点是所有频率都被统一缩放，可能损失局部位置分辨率。

### 14.2 NTK-aware Scaling

NTK-aware Scaling 的思想是调整 RoPE 的基频 `theta`。例如从：

```text
theta = 10000
```

增大到：

```text
theta = 100000
```

增大 `theta` 会降低部分维度的旋转速度，使长距离位置不至于旋转得过快。它比简单线性缩放更关注 RoPE 频率分布对长上下文的影响。

### 14.3 YaRN

YaRN 是更复杂的 RoPE Scaling 方法。它的直觉是：不同频率维度承担的功能不同，不应该全部用同一个缩放策略。

可以粗略理解为：

- 低频维度更适合做长距离外推。
- 高频维度更适合保留局部位置信息。
- 不同频段采用不同缩放或插值策略。

这类方法使得 RoPE 在长上下文模型中依然非常常见。

## 15. 面试速记

如果面试中被问到 RoPE，可以按下面顺序回答。

第一，RoPE 是旋转位置编码。它不是把位置 embedding 加到 hidden states 上，而是在 Attention 前对 Query 和 Key 做位置相关的旋转。

第二，RoPE 的核心优势是点积中自然包含相对位置信息。因为：

$$
(R_m q)^T(R_n k)=q^T R_{n-m}k
$$

所以 Attention score 会依赖位置差 `n - m`。

第三，RoPE 会把 head dimension 两两配对成二维向量，并对每一对使用不同频率的旋转角：

$$
\theta_i = 10000^{-2i/d}
$$

第四，工程实现中常用复数乘法。把 `[x0, x1]` 看成 `x0 + i*x1`，再乘以：

$$
e^{im\theta_i}
$$

就等价于二维旋转。

第五，实现时要注意 shape。输入一般是 `[B, S, H, D]`，转复数后是 `[B, S, H, D // 2]`，`freqs_cis` 是 `[S, D // 2]`，需要 reshape 成 `[1, S, 1, D // 2]` 广播。

第六，低精度下最好临时转 FP32 做复数旋转，最后再转回原始 dtype。

第七，长上下文场景通常需要 RoPE Scaling，例如 Linear Scaling、NTK-aware Scaling 或 YaRN。

## 16. 实现检查清单

写完 RoPE 后，可以用下面的清单快速检查：

- `head_dim` 是否为偶数。
- `freqs_cis` 的 shape 是否是 `[seq_len, head_dim // 2]`。
- 频率公式是否使用 `theta ** (torch.arange(0, dim, 2) / dim)`。
- 是否使用 `torch.outer(position_ids, inv_freq)` 生成角度矩阵。
- 是否用 `torch.polar(torch.ones_like(freqs), freqs)` 生成单位复数。
- Query 和 Key 是否在最后一维 reshape 成 `[head_dim // 2, 2]`。
- 是否使用 `torch.view_as_complex` 转成复数。
- `freqs_cis` 是否 reshape 成 `[1, seq_len, 1, head_dim // 2]` 进行广播。
- 是否只对 Query 和 Key 应用 RoPE，而不是 Value。
- 是否用 `torch.view_as_real(...).flatten(3)` 恢复原始 shape。
- 输出 shape 是否和输入 shape 一致。
- 旋转前后向量 norm 是否基本不变。
- FP16 / BF16 输入是否临时转 FP32 做旋转，最后再恢复 dtype。

一个正确的最小复数版实现如下：

```python
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)
```

## 17. 总结

RoPE 的本质是把位置编码变成 Query 和 Key 上的位置相关旋转。它不直接把位置向量加到 hidden states 上，而是通过旋转矩阵让 Attention score 自然包含相对位置差。

数学上，RoPE 依赖旋转矩阵的性质：

$$
(R_m q)^T(R_n k)=q^T R_{n-m}k
$$

工程上，RoPE 可以利用复数乘法优雅实现。把 head dimension 两两配对成复数，乘以预计算的单位复数 $e^{im\theta}$，再转回实数张量，就完成了旋转位置编码。

实现时最重要的是三件事：第一，维度配对和广播 shape 必须正确；第二，只对 Query 和 Key 应用 RoPE；第三，低精度输入最好临时提升到 FP32 做复数旋转，最后再恢复原始 dtype。

RoPE 之所以成为主流大模型的标准位置编码方案，不只是因为它能表达相对位置，还因为它能通过 Linear Scaling、NTK-aware Scaling、YaRN 等方法扩展到更长上下文。这让它在现代 LLM 架构中非常实用。

相关阅读：如果想继续理解如何用 Triton 融合 RoPE 并降低访存开销，可以看 `../03_CUDA_and_Triton_Kernels/07_Triton_Fused_RoPE.ipynb`。
