# Attention 知识点整理：MHA、MQA、GQA 与 KV Cache

Attention 是 Transformer 和现代大语言模型中最核心的模块。它决定了模型在处理当前 token 时，如何从历史上下文中选择、聚合和利用信息。

这篇文章围绕工程实现中最常见的几个问题展开：

- Self-Attention 的计算公式是什么
- Query、Key、Value 分别承担什么角色
- 多头注意力 MHA 为什么需要拆分 head
- 自回归推理中为什么需要 KV Cache
- KV Cache 为什么会成为显存和带宽瓶颈
- MHA、MQA、GQA 的区别是什么
- GQA 为什么能显著降低 KV Cache 显存占用
- `repeat_kv` 为什么要延迟执行
- PyTorch 中如何实现支持 KV Cache 的 GQA
- 实现时有哪些常见 shape 错误和工程陷阱

## 1. Attention 在做什么

Attention 可以理解为一种“按相关性检索上下文”的机制。

给定当前 token 的隐藏状态，模型会构造三个向量：

- `Query`: 当前 token 想要查找什么信息。
- `Key`: 每个历史 token 提供的可检索标签。
- `Value`: 每个历史 token 真正被聚合的内容。

Attention 的标准公式是：

$$
\mathrm{Attention}(Q,K,V)=
\mathrm{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中：

- $QK^T$ 计算 Query 和 Key 的相似度。
- $\sqrt{d_k}$ 是缩放因子，避免点积数值过大导致 softmax 饱和。
- `Softmax` 把相似度变成概率分布。
- 最后乘以 $V$，得到根据注意力权重加权后的上下文表示。

一句话总结：

```text
Q 决定“我想找什么”，K 决定“我有什么标签”，V 决定“真正取回什么内容”。
```

## 2. 为什么要做多头注意力

如果只有一个 attention head，模型只能在一个表示子空间里计算 token 之间的关系。多头注意力 MHA, Multi-Head Attention, 会把 hidden dimension 拆成多个 head，让不同 head 学习不同类型的关系。

例如：

- 某些 head 关注局部语法关系。
- 某些 head 关注长距离依赖。
- 某些 head 关注实体指代。
- 某些 head 关注格式、标点或结构信息。

假设：

- `B`: batch size
- `S`: sequence length
- `H`: number of query heads
- `D`: head dimension
- `hidden_dim = H * D`

线性投影后，`Q` 的形状通常是：

```text
[B, S, H * D]
```

为了并行计算多头注意力，需要把它 reshape 成：

```text
[B, S, H, D]
```

然后转置成：

```text
[B, H, S, D]
```

这样做的好处是：`B` 和 `H` 都可以作为批量维度，矩阵乘法只发生在最后两个维度上。

## 3. Shape Tracking

Attention 实现中最容易出错的地方就是维度。下面是标准 MHA 的 shape 流程：

| 步骤 | 张量 | 形状 |
|---|---|---|
| 输入 hidden states | `x` | `[B, S, hidden_dim]` |
| Q 投影 | `xq` | `[B, S, H * D]` |
| K 投影 | `xk` | `[B, S, H * D]` |
| V 投影 | `xv` | `[B, S, H * D]` |
| 切分 Q 头 | `xq` | `[B, H, S, D]` |
| 切分 K 头 | `xk` | `[B, H, S, D]` |
| 切分 V 头 | `xv` | `[B, H, S, D]` |
| 注意力分数 | `scores = Q @ K^T` | `[B, H, S, S]` |
| softmax 后 | `probs` | `[B, H, S, S]` |
| 加权求和 | `output = probs @ V` | `[B, H, S, D]` |
| 合并 head | `output` | `[B, S, H * D]` |
| 输出投影 | `out` | `[B, S, hidden_dim]` |

核心矩阵乘法是：

```text
Q:        [B, H, S, D]
K^T:      [B, H, D, S]
Q @ K^T:  [B, H, S, S]

probs:    [B, H, S, S]
V:        [B, H, S, D]
probs@V:  [B, H, S, D]
```

## 4. 自回归生成与 KV Cache

大语言模型推理通常是自回归生成：

```text
输入 prompt -> 生成第 1 个 token -> 生成第 2 个 token -> ... -> 生成第 N 个 token
```

生成第 $t$ 个 token 时，当前 token 需要 attend 到前面所有 token。

如果不使用 KV Cache，每一步都要重新计算整个前缀的 Key 和 Value：

```text
step 1: 计算 token 1 的 K,V
step 2: 重新计算 token 1,2 的 K,V
step 3: 重新计算 token 1,2,3 的 K,V
...
```

这会产生大量重复计算。

KV Cache 的做法是：第一次计算过的 Key 和 Value 直接缓存起来，后续 decode step 只计算新 token 的 Key 和 Value，然后和历史 cache 拼接。

```text
历史 cache:
k_cache: [B, H, old_len, D]
v_cache: [B, H, old_len, D]

当前 token:
xk:      [B, H, 1, D]
xv:      [B, H, 1, D]

拼接后:
new_k:   [B, H, old_len + 1, D]
new_v:   [B, H, old_len + 1, D]
```

对应 PyTorch 代码：

```python
xk = torch.cat([k_cache, xk], dim=2)
xv = torch.cat([v_cache, xv], dim=2)
new_kv_cache = (xk, xv)
```

这里 `dim=2` 是 sequence length 维度，因为张量布局是：

```text
[B, H, S, D]
```

## 5. KV Cache 为什么是性能瓶颈

KV Cache 解决了重复计算问题，但引入了显存和带宽问题。

在 decode 阶段，每生成一个 token，都要读取历史所有 token 的 K 和 V。序列越长，KV Cache 越大。

单层 KV Cache 的元素数量大致是：

$$
2 \times B \times S \times H_{kv} \times D
$$

其中：

- `2` 表示 K 和 V 两份缓存。
- `B` 是 batch size。
- `S` 是当前上下文长度。
- `H_kv` 是 KV head 数量。
- `D` 是 head dimension。

如果模型有 `L` 层，总 KV Cache 元素数量是：

$$
2 \times L \times B \times S \times H_{kv} \times D
$$

这说明 KV Cache 占用与层数、batch size、上下文长度、KV head 数量都线性相关。

对于长上下文推理，Attention decode 往往不是算力不够，而是显存带宽不够。GPU 需要不断从显存读取巨大的 KV Cache，这就是常说的 memory-bound。

## 6. MHA、MQA、GQA 的区别

MHA、MQA、GQA 的核心区别是：Query head 和 Key/Value head 是否一一对应。

| 结构 | Query heads | KV heads | 关系 | KV Cache 占用 | 表达能力 |
|---|---:|---:|---|---:|---|
| MHA | 多个 | 多个 | 每个 Q head 对应独立 KV head | 最高 | 最强 |
| MQA | 多个 | 1 个 | 所有 Q head 共享同一组 KV | 最低 | 较弱 |
| GQA | 多个 | 少数几个 | 一组 Q heads 共享一个 KV head | 折中 | 接近 MHA |

### 6.1 MHA

MHA, Multi-Head Attention, 是标准多头注意力。

如果 `num_heads = 8`，那么：

```text
Q heads:  8
K heads:  8
V heads:  8
```

每个 Query head 都有自己的 Key 和 Value head。表达能力强，但 KV Cache 最大。

### 6.2 MQA

MQA, Multi-Query Attention, 让所有 Query heads 共享同一个 Key head 和 Value head。

如果 `num_heads = 8`，那么：

```text
Q heads:  8
K heads:  1
V heads:  1
```

KV Cache 直接减少到 MHA 的 `1/8`。缺点是所有 Query heads 只能看同一组 K/V，表达能力可能下降。

### 6.3 GQA

GQA, Grouped-Query Attention, 是 MHA 和 MQA 的折中方案。

如果 `num_heads = 8`，`num_kv_heads = 2`，那么：

```text
Q heads:  8
K heads:  2
V heads:  2
```

每 4 个 Query heads 共享一个 KV head：

```text
Q0 Q1 Q2 Q3 -> KV0
Q4 Q5 Q6 Q7 -> KV1
```

此时 KV Cache 占用是 MHA 的：

$$
\frac{H_{kv}}{H}=\frac{2}{8}=\frac{1}{4}
$$

GQA 在显存占用和模型表达能力之间取得了更好的工程平衡，因此被 LLaMA、Qwen、Mistral、DeepSeek 等模型广泛采用。

## 7. GQA 的投影矩阵

在标准 MHA 中，Q/K/V 的输出维度通常都是 `hidden_dim`：

```python
self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
self.k_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
self.v_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
```

在 GQA 中，Q 仍然有 `num_heads` 个 head，但 K/V 只有 `num_kv_heads` 个 head：

```python
self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
self.k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
self.v_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
```

例如：

```text
hidden_dim = 4096
num_heads = 32
head_dim = 128
num_kv_heads = 8
```

则：

```text
q_proj 输出维度 = 32 * 128 = 4096
k_proj 输出维度 =  8 * 128 = 1024
v_proj 输出维度 =  8 * 128 = 1024
```

这就是 GQA 能减少 KV Cache 的根本原因：缓存的 K/V head 数量变少了。

## 8. repeat_kv 的作用

虽然 GQA 中 K/V head 数量比 Q head 少，但做 attention 矩阵乘法时，Q 和 K 的 head 数必须对齐。

例如：

```text
Q: [B, 8, S, D]
K: [B, 2, S, D]
V: [B, 2, S, D]
```

不能直接计算：

```text
Q @ K^T
```

因为 head 维度 `8` 和 `2` 不一致。

所以需要把 K/V 在 head 维度上复制，让它们变成：

```text
K: [B, 8, S, D]
V: [B, 8, S, D]
```

这就是 `repeat_kv` 的作用。

```python
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
```

假设：

```text
num_heads = 8
num_kv_heads = 2
n_rep = num_heads // num_kv_heads = 4
```

那么：

```text
[B, 2, S, D]
 -> [B, 2, 1, S, D]
 -> [B, 2, 4, S, D]
 -> [B, 8, S, D]
```

这里使用 `expand` 而不是 `repeat`，可以避免立即复制底层数据。后续 `reshape` 会得到 attention 计算需要的布局。

## 9. 为什么 repeat_kv 要延迟执行

GQA 的关键工程原则是：

```text
只缓存原始 num_kv_heads 个 K/V head，不缓存 repeat 后的 K/V。
```

正确流程：

```text
1. 计算当前 token 的 xk, xv: [B, H_kv, S, D]
2. 和历史 KV Cache 拼接: [B, H_kv, old_len + S, D]
3. 保存 new_kv_cache，仍然是 H_kv 个 head
4. attention 计算前临时 repeat_kv: [B, H, old_len + S, D]
```

错误流程：

```text
1. 先 repeat_kv，把 K/V 扩展到 num_heads
2. 再写入 KV Cache
```

这样会让 KV Cache 的大小退化回 MHA，GQA 的显存优势完全消失。

因此代码中一定要注意顺序：

```python
if kv_cache is not None:
    k_cache, v_cache = kv_cache
    xk = torch.cat([k_cache, xk], dim=2)
    xv = torch.cat([v_cache, xv], dim=2)

new_kv_cache = (xk, xv)

xk = repeat_kv(xk, self.num_queries_per_kv)
xv = repeat_kv(xv, self.num_queries_per_kv)
```

`new_kv_cache` 必须在 `repeat_kv` 之前生成。

## 10. 完整 PyTorch 实现

下面是一个支持 MHA、MQA、GQA 和 KV Cache 的简化实现。

```python
import math

import torch
import torch.nn as nn


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads

        assert hidden_dim % num_heads == 0
        assert num_heads % self.num_kv_heads == 0

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            hidden_dim, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_dim, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        batch_size, seq_len, _ = x.shape

        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        xq = xq.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        xk = xk.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        xv = xv.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            xk = torch.cat([k_cache, xk], dim=2)
            xv = torch.cat([v_cache, xv], dim=2)

        new_kv_cache = (xk, xv)

        xk = repeat_kv(xk, self.num_queries_per_kv)
        xv = repeat_kv(xv, self.num_queries_per_kv)

        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores = scores + attention_mask

        probs = torch.nn.functional.softmax(scores, dim=-1)
        output = torch.matmul(probs, xv)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.o_proj(output), new_kv_cache
```

## 11. 代码逐段解析

### 11.1 初始化参数

```python
self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
self.num_queries_per_kv = self.num_heads // self.num_kv_heads
self.head_dim = hidden_dim // num_heads
```

这里有三个关键变量：

- `num_heads`: Query head 数量。
- `num_kv_heads`: Key/Value head 数量。
- `num_queries_per_kv`: 每个 KV head 对应多少个 Query heads。

三种结构可以统一表示：

| 结构 | `num_heads` | `num_kv_heads` | `num_queries_per_kv` |
|---|---:|---:|---:|
| MHA | 8 | 8 | 1 |
| GQA | 8 | 2 | 4 |
| MQA | 8 | 1 | 8 |

### 11.2 Q/K/V 投影

```python
xq = self.q_proj(x)
xk = self.k_proj(x)
xv = self.v_proj(x)
```

投影后：

```text
xq: [B, S, H * D]
xk: [B, S, H_kv * D]
xv: [B, S, H_kv * D]
```

GQA 的特殊点是：`xk` 和 `xv` 的最后一维比 `xq` 小。

### 11.3 多头切分

```python
xq = xq.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
xk = xk.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
xv = xv.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
```

切分后：

```text
xq: [B, H,    S, D]
xk: [B, H_kv, S, D]
xv: [B, H_kv, S, D]
```

注意：K/V 使用的是 `num_kv_heads`，不是 `num_heads`。

### 11.4 KV Cache 拼接

```python
if kv_cache is not None:
    k_cache, v_cache = kv_cache
    xk = torch.cat([k_cache, xk], dim=2)
    xv = torch.cat([v_cache, xv], dim=2)
```

decode 阶段通常 `seq_len = 1`：

```text
k_cache: [B, H_kv, old_len, D]
xk:      [B, H_kv, 1,       D]
new_k:   [B, H_kv, old_len + 1, D]
```

### 11.5 GQA 扩展 KV heads

```python
xk = repeat_kv(xk, self.num_queries_per_kv)
xv = repeat_kv(xv, self.num_queries_per_kv)
```

扩展后：

```text
xq: [B, H, total_len, D] 或 [B, H, query_len, D]
xk: [B, H, total_len, D]
xv: [B, H, total_len, D]
```

这样才能计算 `xq @ xk.transpose(2, 3)`。

### 11.6 Scaled Dot-Product Attention

```python
scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
```

对应 shape：

```text
xq:                [B, H, query_len, D]
xk.transpose(2,3): [B, H, D, total_len]
scores:            [B, H, query_len, total_len]
```

如果是训练或 prefill 阶段，`query_len = total_len = S`。

如果是 decode 阶段，`query_len = 1`，`total_len = old_len + 1`。

### 11.7 Attention Mask

```python
if attention_mask is not None:
    scores = scores + attention_mask
```

mask 通常使用很大的负数或 `-inf` 屏蔽不可见位置。

训练 decoder-only 模型时，常用 causal mask：

```text
token 0: 只能看 0
token 1: 能看 0,1
token 2: 能看 0,1,2
```

mask 的 shape 通常可以广播到：

```text
[B, H, query_len, key_len]
```

常见写法是：

```text
[1, 1, S, S]
```

### 11.8 Softmax 与加权求和

```python
probs = torch.nn.functional.softmax(scores, dim=-1)
output = torch.matmul(probs, xv)
```

这里 softmax 必须在最后一维做，也就是对所有 key positions 归一化：

```text
scores: [B, H, query_len, key_len]
dim=-1: key_len 维度
```

加权求和后：

```text
output: [B, H, query_len, D]
```

### 11.9 合并多头

```python
output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
```

shape 变化：

```text
[B, H, S, D]
 -> transpose(1, 2)
[B, S, H, D]
 -> view
[B, S, H * D]
```

这里的 `.contiguous()` 很重要。`transpose` 后张量的内存布局通常不是连续的，直接 `view` 可能报错。

## 12. 测试代码

可以用下面的测试同时验证 MHA、GQA 和 KV Cache。

```python
def test_mha_mqa_gqa():
    batch_size, seq_len, hidden_dim, num_heads = 2, 16, 128, 4

    print("Testing MHA...")
    mha = GroupedQueryAttention(hidden_dim, num_heads, num_kv_heads=num_heads)
    x = torch.randn(batch_size, seq_len, hidden_dim)
    out, _ = mha(x)
    assert out.shape == (batch_size, seq_len, hidden_dim)

    print("Testing GQA...")
    gqa = GroupedQueryAttention(hidden_dim, num_heads, num_kv_heads=2)
    out, _ = gqa(x)
    assert out.shape == (batch_size, seq_len, hidden_dim)

    print("Testing MQA...")
    mqa = GroupedQueryAttention(hidden_dim, num_heads, num_kv_heads=1)
    out, _ = mqa(x)
    assert out.shape == (batch_size, seq_len, hidden_dim)

    print("Testing KV Cache...")
    prefill_len = 5
    x_prefill = torch.randn(batch_size, prefill_len, hidden_dim)
    _, kv_cache = mha(x_prefill)

    x_decode = torch.randn(batch_size, 1, hidden_dim)
    out_decode, new_kv_cache = mha(x_decode, kv_cache=kv_cache)

    assert out_decode.shape == (batch_size, 1, hidden_dim)
    assert new_kv_cache[0].shape == (
        batch_size,
        num_heads,
        prefill_len + 1,
        hidden_dim // num_heads,
    )

    print("All tests passed.")
```

## 13. Prefill 和 Decode 的区别

LLM 推理通常分成两个阶段：

| 阶段 | 输入长度 | 是否使用已有 KV Cache | 主要操作 |
|---|---:|---|---|
| Prefill | prompt 长度，通常大于 1 | 否 | 一次性计算 prompt 的 KV Cache |
| Decode | 每次通常 1 个 token | 是 | 读取历史 KV Cache 并追加新 token |

### 13.1 Prefill

Prefill 阶段输入整个 prompt：

```text
x: [B, prompt_len, hidden_dim]
```

生成：

```text
kv_cache:
K: [B, H_kv, prompt_len, D]
V: [B, H_kv, prompt_len, D]
```

### 13.2 Decode

Decode 阶段每次输入一个新 token：

```text
x: [B, 1, hidden_dim]
```

读取历史 cache：

```text
K_cache: [B, H_kv, old_len, D]
V_cache: [B, H_kv, old_len, D]
```

追加后：

```text
K_new: [B, H_kv, old_len + 1, D]
V_new: [B, H_kv, old_len + 1, D]
```

decode 阶段的 attention score 形状是：

```text
[B, H, 1, old_len + 1]
```

这意味着当前 token 会 attend 到所有历史 token 和自己。

## 14. 显存占用对比

假设：

```text
num_heads = 32
head_dim = 128
seq_len = S
batch_size = B
layers = L
```

不同结构的单层 KV Cache 元素数量：

| 结构 | `num_kv_heads` | 单层 KV Cache 元素数量 |
|---|---:|---|
| MHA | 32 | `2 * B * S * 32 * 128` |
| GQA | 8 | `2 * B * S * 8 * 128` |
| MQA | 1 | `2 * B * S * 1 * 128` |

相对 MHA 的占用：

| 结构 | 相对 KV Cache 占用 |
|---|---:|
| MHA | 100% |
| GQA, `num_kv_heads=8` | 25% |
| MQA, `num_kv_heads=1` | 3.125% |

如果使用 FP16 或 BF16，每个元素 2 bytes。完整模型的 KV Cache 字节数约为：

$$
2 \times L \times B \times S \times H_{kv} \times D \times 2
$$

第一个 `2` 是 K/V 两份缓存，最后一个 `2` 是 FP16/BF16 的字节数。

## 15. 常见错误

### 15.1 忘记 transpose

错误：

```python
xq = xq.view(batch_size, seq_len, self.num_heads, self.head_dim)
scores = torch.matmul(xq, xk.transpose(2, 3))
```

此时 shape 是 `[B, S, H, D]`，矩阵乘法的语义不对。

正确：

```python
xq = xq.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
```

### 15.2 K/V 用错 head 数

错误：

```python
xk = xk.view(batch_size, seq_len, self.num_heads, self.head_dim)
```

GQA 中 K/V 应该使用 `num_kv_heads`：

```python
xk = xk.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
```

### 15.3 先 repeat_kv 再写 cache

错误：

```python
xk = repeat_kv(xk, self.num_queries_per_kv)
xv = repeat_kv(xv, self.num_queries_per_kv)
new_kv_cache = (xk, xv)
```

这样会缓存扩展后的 K/V，显存占用退化为 MHA。

正确：

```python
new_kv_cache = (xk, xv)
xk = repeat_kv(xk, self.num_queries_per_kv)
xv = repeat_kv(xv, self.num_queries_per_kv)
```

### 15.4 拼接 cache 的维度错了

错误：

```python
xk = torch.cat([k_cache, xk], dim=1)
```

`dim=1` 是 head 维度。拼接历史 token 应该沿 sequence 维度：

```python
xk = torch.cat([k_cache, xk], dim=2)
```

### 15.5 忘记 contiguous

错误：

```python
output = output.transpose(1, 2).view(batch_size, seq_len, -1)
```

`transpose` 后内存布局可能不连续。更稳妥的写法：

```python
output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
```

### 15.6 softmax 维度错了

错误：

```python
probs = torch.nn.functional.softmax(scores, dim=2)
```

应该沿 key length 维度归一化，也就是最后一维：

```python
probs = torch.nn.functional.softmax(scores, dim=-1)
```

## 16. 面试常见问题

### Q1: 为什么 attention score 要除以 sqrt(head_dim)

如果 $Q$ 和 $K$ 的元素方差近似为 1，那么点积 $QK^T$ 的方差会随着维度 $d_k$ 增大而增大。分数过大时，softmax 会变得非常尖锐，梯度容易变小。

除以 $\sqrt{d_k}$ 可以稳定数值范围，让训练更稳定。

### Q2: KV Cache 缓存的是什么

KV Cache 缓存的是每一层 attention 中，历史 token 经过 `k_proj` 和 `v_proj` 后的 Key 和 Value。

它不是缓存 hidden states，也不是缓存 attention score。

缓存 K/V 的原因是：后续 token 仍然需要和历史 token 的 K 做相关性计算，并用历史 token 的 V 做加权求和。

### Q3: 为什么不缓存 Query

Query 只用于当前 step 主动查询历史信息。历史 token 的 Query 后续不会再被新 token 使用。

decode 阶段只需要当前 token 的 Query，以及所有历史 token 的 Key/Value。

### Q4: GQA 为什么能减少推理显存

KV Cache 的大小正比于 `num_kv_heads`。GQA 让多个 Query heads 共享较少的 K/V heads，因此缓存的 K/V head 数量变少。

例如 `num_heads=32`，`num_kv_heads=8`，KV Cache 只有 MHA 的 `8/32 = 25%`。

### Q5: MQA 和 GQA 有什么区别

MQA 是极端版本的 GQA：

```text
MQA: num_kv_heads = 1
GQA: 1 < num_kv_heads < num_heads
MHA: num_kv_heads = num_heads
```

MQA 最省显存，但表达能力下降更明显。GQA 在效果和推理效率之间更平衡。

### Q6: 为什么 GQA 需要 repeat_kv

因为 attention 计算时 Q 和 K 的 head 维度需要一致。

GQA 中：

```text
Q: [B, H, S, D]
K: [B, H_kv, S, D]
```

需要把 K/V 从 `H_kv` 个 head 临时扩展到 `H` 个 head，才能逐 head 做矩阵乘法。

### Q7: repeat_kv 会不会抵消 GQA 的显存优势

不会，前提是只在 attention 计算前临时 repeat，不把 repeat 后的 K/V 写入 cache。

GQA 的核心节省来自 KV Cache。只要 cache 中保存的仍然是 `num_kv_heads` 个 K/V heads，显存优势就保留了。

### Q8: Prefill 和 Decode 哪个更吃算力

Prefill 通常更像大矩阵并行计算，GPU 利用率较高。

Decode 每步只处理一个 token，但要读取越来越长的 KV Cache，常常受显存带宽限制，吞吐不容易提升。

## 17. 工业界源码映射

在 HuggingFace Transformers 中，LLaMA 系列 attention 的核心逻辑可以在下面位置找到：

```text
transformers/models/llama/modeling_llama.py
```

重点关注：

- `LlamaAttention`
- `num_heads`
- `num_key_value_heads`
- `num_key_value_groups`
- `repeat_kv`
- `past_key_value`
- causal mask 的构造和传递

在推理框架中，例如 vLLM，核心问题会进一步扩展为：

- KV Cache 如何分块管理
- 如何避免显存碎片
- 多请求 batch 时如何调度不同长度的序列
- PagedAttention 如何把 KV Cache 当成分页内存管理

本篇实现的是 attention 的基础逻辑；vLLM 的 PagedAttention 是在这个基础上解决大规模推理服务中的内存管理和调度问题。

## 18. 复习速记

### 核心公式

```text
Attention(Q, K, V) = Softmax(QK^T / sqrt(d_k))V
```

### MHA/MQA/GQA

```text
MHA: num_kv_heads = num_heads
MQA: num_kv_heads = 1
GQA: 1 < num_kv_heads < num_heads
```

### GQA 关键变量

```text
num_queries_per_kv = num_heads // num_kv_heads
```

### 标准 shape

```text
x:      [B, S, hidden_dim]
Q:      [B, H, S, D]
K/V:    [B, H_kv, S, D]
cache:  [B, H_kv, total_len, D]
repeat: [B, H, total_len, D]
score:  [B, H, query_len, total_len]
out:    [B, S, hidden_dim]
```

### KV Cache 正确顺序

```text
投影 -> reshape -> 拼接 cache -> 保存 new cache -> repeat_kv -> attention
```

### 最容易错的点

- K/V reshape 时用了 `num_heads` 而不是 `num_kv_heads`。
- cache 拼接维度写成了 `dim=1`，正确是 `dim=2`。
- 在写 cache 前做了 `repeat_kv`。
- `transpose` 后直接 `view`，忘记 `.contiguous()`。
- softmax 没有沿最后一维做。
- attention mask shape 不能广播到 `[B, H, query_len, key_len]`。

## 19. 最小心智模型

可以用下面这句话记住整套逻辑：

```text
Q 负责发起查询，K/V 负责提供历史记忆；
MHA 每个 Q head 有独立记忆，MQA 所有 Q head 共享一份记忆；
GQA 让一组 Q heads 共享一份记忆；
KV Cache 缓存的是历史 K/V；
GQA 的工程关键是只缓存少量 KV heads，在计算 attention 前再临时扩展。
```

