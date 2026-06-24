# LLaMA3 Block 知识点整理：Pre-Norm、RMSNorm、SwiGLU、RoPE 与 GQA 的工程组装

LLaMA3 Block 是现代 Decoder-only 大语言模型中最典型的基础结构之一。单独看 RMSNorm、RoPE、GQA、SwiGLU 时，它们都是局部模块；真正搭建模型时，需要把这些模块按正确顺序拼接成一个稳定、可训练、可高效推理的 Transformer Decoder Layer。

这篇文章整理 LLaMA3 Block 在实现和面试中最常被问到的知识点：

- LLaMA Block 和传统 Transformer Block 有什么区别
- 为什么 LLaMA 使用 Pre-Norm 而不是 Post-Norm
- RMSNorm 在 Block 中放在哪里
- Attention 子层和 MLP 子层的残差路径如何连接
- SwiGLU MLP 为什么需要 `gate_proj`、`up_proj`、`down_proj` 三个线性层
- RoPE 和 GQA 在 Attention 内部承担什么角色
- LLaMA Decoder Layer 的 PyTorch 实现应该怎么写
- 实现时有哪些常见 shape、残差和梯度错误

## 1. LLaMA Block 在做什么

Transformer Decoder Layer 的目标是：给定输入 hidden states，让每个 token 一边从上下文中聚合信息，一边通过前馈网络进行非线性特征变换。

在 LLaMA 系列模型中，一个 Decoder Layer 可以概括为两段：

```text
Attention 子层:
hidden_states -> RMSNorm -> Self-Attention(RoPE + GQA) -> Residual Add

MLP 子层:
hidden_states -> RMSNorm -> SwiGLU MLP -> Residual Add
```

对应公式是：

$$
h = x + \mathrm{Attention}(\mathrm{RMSNorm}(x))
$$

$$
\mathrm{out} = h + \mathrm{MLP}(\mathrm{RMSNorm}(h))
$$

这里有两个关键点：

- 归一化发生在子层计算之前，所以叫 Pre-Norm。
- Attention 和 MLP 各自都有独立的残差连接。

一句话总结：

```text
LLaMA Block = Pre-RMSNorm + RoPE/GQA Attention + SwiGLU MLP + 双残差连接。
```

## 2. LLaMA 相比传统 Transformer 的核心变化

早期 Transformer 或 GPT-2 风格 Block 常见结构是 LayerNorm、标准 MHA、GELU MLP、绝对或学习式位置编码。LLaMA 系列在这些位置做了更适合大模型训练和推理的替换。

| 模块 | 传统 Transformer / GPT 风格 | LLaMA 风格 | 主要收益 |
|---|---|---|---|
| 归一化位置 | Post-Norm 或混合结构 | Pre-Norm | 深层训练更稳定 |
| 归一化算法 | LayerNorm | RMSNorm | 计算更简单，省去均值中心化 |
| MLP 激活 | ReLU / GELU | SwiGLU | 门控机制提升表达能力 |
| 位置编码 | 绝对位置编码 | RoPE | Attention 分数天然感知相对位置 |
| 注意力结构 | MHA | GQA | 降低 KV Cache 显存和带宽开销 |
| 线性层 bias | 常见有 bias | 通常无 bias | 减少参数和实现复杂度 |

这些改动不是互相独立的装饰，而是共同服务于两个目标：

- 训练时让深层 Decoder-only 模型更稳定。
- 推理时减少长上下文和自回归生成的显存压力。

## 3. Pre-Norm 与 Post-Norm

Post-Norm 的典型形式是：

$$
h = \mathrm{Norm}(x + \mathrm{Sublayer}(x))
$$

Pre-Norm 的典型形式是：

$$
h = x + \mathrm{Sublayer}(\mathrm{Norm}(x))
$$

LLaMA 使用的是 Pre-Norm，也就是先归一化，再进入 Attention 或 MLP，最后再和原始输入做残差相加。

### 3.1 为什么 Pre-Norm 更适合深层模型

在深层 Transformer 中，梯度需要穿过很多层。如果每一层都把残差结果再做归一化，梯度路径会更复杂，训练更容易不稳定。

Pre-Norm 的优势在于：

- 子层输入分布更稳定，因为进入 Attention/MLP 之前已经归一化。
- 残差路径更直接，梯度可以更顺畅地从深层传回浅层。
- 深层 Decoder 堆叠时更容易收敛。

可以把 Pre-Norm 的残差主干理解成：

```text
x -> x + f(norm(x)) -> h + g(norm(h)) -> ...
```

其中 `x` 到后续层始终保留一条较直接的加法路径。

## 4. RMSNorm 在 Block 中的位置

LLaMA Block 中有两个 RMSNorm：

```python
self.input_layernorm = RMSNorm(hidden_size)
self.post_attention_layernorm = RMSNorm(hidden_size)
```

它们分别放在：

- `input_layernorm`: Attention 子层之前。
- `post_attention_layernorm`: MLP 子层之前。

完整流程是：

```text
输入 hidden_states
  |
  |-- residual = hidden_states
  |-- hidden_states = input_layernorm(hidden_states)
  |-- hidden_states = self_attn(hidden_states)
  |-- hidden_states = residual + hidden_states
  |
  |-- residual = hidden_states
  |-- hidden_states = post_attention_layernorm(hidden_states)
  |-- hidden_states = mlp(hidden_states)
  |-- hidden_states = residual + hidden_states
  |
输出 hidden_states
```

注意，第二个 RMSNorm 的输入不是原始 `x`，而是 Attention 残差相加后的 `hidden_states`。这是很多初学者容易写错的地方。

## 5. Attention 子层：RoPE 与 GQA

在 LLaMA3 Block 里，Attention 子层通常不是普通 MHA，而是带 RoPE 和 GQA 的 Causal Self-Attention。

它内部一般完成这些步骤：

```text
hidden_states
  -> q_proj, k_proj, v_proj
  -> reshape 成多头形式
  -> 对 Q、K 应用 RoPE
  -> 读取或更新 KV Cache
  -> 如果是 GQA，将 K/V head repeat 到 Q head 数量
  -> causal attention
  -> o_proj
```

### 5.1 RoPE 的作用

RoPE, Rotary Position Embedding, 不把位置向量直接加到 hidden states 上，而是在 Attention 计算前旋转 Query 和 Key。

核心思想是：

$$
q_m' = R_m q,\quad k_n' = R_n k
$$

Attention score 使用旋转后的向量：

$$
(q_m')^T k_n' = q^T R_{n-m} k
$$

因此 Q 和 K 的点积天然依赖位置差 `n - m`，模型可以感知相对位置。

### 5.2 GQA 的作用

GQA, Grouped-Query Attention, 让多个 Query heads 共享一组 Key/Value heads。

假设：

```text
num_query_heads = 32
num_kv_heads    = 8
```

那么每 4 个 Query heads 共享 1 个 KV head。

这样做的主要收益是降低 KV Cache：

$$
\mathrm{KVCacheSize} \propto 2 \times L \times B \times S \times H_{kv} \times D
$$

其中：

- `2`: K 和 V 两份缓存。
- `L`: 层数。
- `B`: batch size。
- `S`: sequence length。
- `H_kv`: KV head 数量。
- `D`: head dimension。

在自回归推理中，KV Cache 经常是显存和带宽瓶颈。减少 `H_kv` 可以显著降低长上下文推理成本。

## 6. MLP 子层：SwiGLU

传统 Transformer MLP 通常是：

$$
\mathrm{MLP}(x)=W_{down}\,\sigma(W_{up}x)
$$

LLaMA 使用 SwiGLU，它引入了门控分支：

$$
\mathrm{SwiGLU}(x) =
\left(\mathrm{SiLU}(xW_{gate}) \odot xW_{up}\right)W_{down}
$$

其中：

$$
\mathrm{SiLU}(z)=z\cdot\sigma(z)
$$

在 PyTorch 中对应 `F.silu`。

### 6.1 为什么需要三个线性层

SwiGLU MLP 通常包含三个无 bias 线性层：

```python
self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
```

三者作用不同：

| 线性层 | 形状 | 作用 |
|---|---|---|
| `gate_proj` | `hidden_size -> intermediate_size` | 生成门控信号 |
| `up_proj` | `hidden_size -> intermediate_size` | 生成被门控的特征 |
| `down_proj` | `intermediate_size -> hidden_size` | 映射回模型主干维度 |

前向传播是：

```python
return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

这里的 `*` 是逐元素乘法，也就是 Hadamard product。

### 6.2 intermediate_size 为什么不是简单的 4 倍

传统 MLP 常使用：

```text
hidden_size -> 4 * hidden_size -> hidden_size
```

SwiGLU 有两条上投影分支：`gate_proj` 和 `up_proj`。如果仍然使用 `4 * hidden_size`，参数量会明显增加。

因此 LLaMA 风格模型通常会把中间维度设成接近：

$$
\frac{8}{3} \times hidden\_size
$$

再根据硬件友好的倍数做向上取整。这样可以在引入门控机制后，让参数量大致接近传统 `4x` MLP。

以简化计算看：

传统 MLP 参数量约为：

$$
hidden \times 4hidden + 4hidden \times hidden = 8hidden^2
$$

SwiGLU 参数量约为：

$$
hidden \times intermediate \times 2 + intermediate \times hidden
= 3hidden \times intermediate
$$

令二者相近：

$$
3hidden \times intermediate \approx 8hidden^2
$$

得到：

$$
intermediate \approx \frac{8}{3}hidden
$$

## 7. Shape Tracking

假设输入：

```text
B = batch size
S = sequence length
H = hidden_size
I = intermediate_size
```

LLaMA Block 的主要 shape 如下：

| 步骤 | 张量 | 形状 |
|---|---|---|
| 输入 | `hidden_states` | `[B, S, H]` |
| Attention 前 RMSNorm | `input_layernorm(hidden_states)` | `[B, S, H]` |
| Attention 输出 | `self_attn(...)` | `[B, S, H]` |
| Attention 残差后 | `residual + attn_out` | `[B, S, H]` |
| MLP 前 RMSNorm | `post_attention_layernorm(hidden_states)` | `[B, S, H]` |
| `gate_proj` | `gate_proj(x)` | `[B, S, I]` |
| `up_proj` | `up_proj(x)` | `[B, S, I]` |
| 门控乘法 | `silu(gate) * up` | `[B, S, I]` |
| `down_proj` | `down_proj(...)` | `[B, S, H]` |
| MLP 残差后 | `residual + mlp_out` | `[B, S, H]` |

最重要的约束是：

```text
Attention 输出必须回到 hidden_size。
MLP 输出也必须回到 hidden_size。
否则无法和 residual 相加。
```

## 8. PyTorch 参考实现

下面代码用简化版 `DummyRMSNorm` 和 `DummyAttention` 占位，重点展示 LLaMA Decoder Layer 的组装方式。真实模型中，`DummyAttention` 应替换为支持 RoPE、GQA、Causal Mask、KV Cache 的 Attention 模块。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DummyRMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.w


class DummyAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        self.input_layernorm = DummyRMSNorm(hidden_size)
        self.self_attn = DummyAttention(hidden_size)

        self.post_attention_layernorm = DummyRMSNorm(hidden_size)
        self.mlp = LlamaMLP(hidden_size, intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
```

## 9. 最小测试

一个正确的 Block 至少应该满足两个条件：

- 输出形状和输入形状一致。
- 所有参数都能收到梯度。

```python
def test_llama_block():
    batch_size, seq_len, hidden_size = 2, 16, 512
    intermediate_size = 1376

    layer = LlamaDecoderLayer(hidden_size, intermediate_size)
    x = torch.randn(batch_size, seq_len, hidden_size)

    out = layer(x)
    assert out.shape == (batch_size, seq_len, hidden_size)

    out.sum().backward()
    for name, param in layer.named_parameters():
        assert param.grad is not None, f"{name} did not receive gradient"

    print("All tests passed.")


test_llama_block()
```

这个测试虽然简单，但能抓住几个常见实现错误：

- `down_proj` 输出维度写错，导致不能残差相加。
- `gate_proj` 或 `up_proj` 没参与前向传播，导致没有梯度。
- 忘记第二段 MLP 残差连接。
- `forward` 没有返回最终 `hidden_states`。

## 10. 常见实现错误

### 10.1 把 Pre-Norm 写成 Post-Norm

错误写法：

```python
hidden_states = self.self_attn(hidden_states)
hidden_states = self.input_layernorm(residual + hidden_states)
```

这变成了 Post-Norm，不是 LLaMA Block 的结构。

正确写法：

```python
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)
hidden_states = self.self_attn(hidden_states)
hidden_states = residual + hidden_states
```

### 10.2 第二个残差保存位置错误

错误写法：

```python
residual = original_x
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)
hidden_states = residual + hidden_states
```

MLP 子层的 residual 应该是 Attention 子层之后的 `hidden_states`，不是最初输入。

正确写法：

```python
residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)
hidden_states = residual + hidden_states
```

### 10.3 SwiGLU 写成普通 FFN

错误写法：

```python
return self.down_proj(F.silu(self.up_proj(x)))
```

这只是普通两层 MLP，没有门控分支。

正确写法：

```python
return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

### 10.4 线性层维度反了

`down_proj` 的输入必须是 `intermediate_size`，输出必须是 `hidden_size`：

```python
self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
```

如果写成：

```python
self.down_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
```

MLP 输出就会变成 `[B, S, I]`，无法和 `[B, S, H]` 的 residual 相加。

### 10.5 忘记无 bias 配置

LLaMA 风格线性层通常使用 `bias=False`：

```python
nn.Linear(hidden_size, intermediate_size, bias=False)
```

这不是影响 shape 的错误，但如果目标是复现 LLaMA 结构，应保持这个配置。

## 11. 面试高频问题

### 11.1 LLaMA Decoder Layer 的执行顺序是什么

可以直接回答：

```text
先对输入做 RMSNorm，再进入 Self-Attention，做残差相加；
然后对 Attention 后的 hidden states 再做 RMSNorm，进入 SwiGLU MLP，再做残差相加。
```

对应公式：

$$
h = x + \mathrm{Attention}(\mathrm{RMSNorm}(x))
$$

$$
out = h + \mathrm{MLP}(\mathrm{RMSNorm}(h))
$$

### 11.2 为什么 LLaMA 使用 RMSNorm

RMSNorm 只根据均方根进行归一化，不做均值中心化：

$$
\mathrm{RMSNorm}(x)=\frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2+\epsilon}}\odot w
$$

相比 LayerNorm，它少了减均值和 bias 相关计算，实现更简单，在大模型中通常能保持效果并提升效率。

### 11.3 为什么 LLaMA 使用 SwiGLU

SwiGLU 给 MLP 增加了门控分支：

```text
gate = SiLU(gate_proj(x))
up   = up_proj(x)
out  = down_proj(gate * up)
```

门控机制可以动态选择哪些中间特征通过，比普通 `GELU(Wx)` 表达能力更强。

### 11.4 RoPE 和 GQA 分别解决什么问题

RoPE 解决位置信息注入问题：

```text
让 Q/K 的点积天然包含相对位置信息。
```

GQA 解决推理效率问题：

```text
减少 KV head 数量，从而降低 KV Cache 的显存和带宽开销。
```

### 11.5 为什么 Attention 和 MLP 都要残差连接

残差连接有两个作用：

- 保留输入信息，避免每个子层必须完全重写表示。
- 提供更直接的梯度传播路径，缓解深层网络训练困难。

在 LLaMA Block 中，Attention 和 MLP 是两个独立子层，所以每个子层都需要自己的 residual。

## 12. 工程实现 Checklist

实现一个 LLaMA Decoder Layer 时，可以按下面清单检查：

- `input_layernorm` 放在 Attention 前面。
- `post_attention_layernorm` 放在 MLP 前面。
- Attention 输出 shape 是 `[B, S, hidden_size]`。
- MLP 输出 shape 是 `[B, S, hidden_size]`。
- Attention 和 MLP 各有一次 residual add。
- `gate_proj` 和 `up_proj` 都从 `hidden_size` 投影到 `intermediate_size`。
- `down_proj` 从 `intermediate_size` 投影回 `hidden_size`。
- SwiGLU 使用 `F.silu(gate_proj(x)) * up_proj(x)`。
- LLaMA 风格线性层通常设置 `bias=False`。
- 如果接入真实 Attention，需要保证 RoPE 只作用在 Q/K 上，不作用在 V 上。
- 如果接入 KV Cache，需要保证 cache 的 sequence 维度拼接正确。

## 13. 总结

LLaMA3 Block 的核心不是某个单独技巧，而是多个现代大模型组件的组合：

```text
Pre-Norm 负责训练稳定性；
RMSNorm 负责高效归一化；
RoPE 负责相对位置信息；
GQA 负责降低 KV Cache 成本；
SwiGLU 负责增强 MLP 表达能力；
Residual Connection 负责信息和梯度流动。
```

最终结构可以记成一行：

```text
x -> x + Attention(RMSNorm(x)) -> h + MLP(RMSNorm(h))
```

只要掌握这条主线，就能清楚解释 LLaMA Decoder Layer 的代码实现、shape 流程、训练稳定性原因，以及推理阶段的效率设计。
