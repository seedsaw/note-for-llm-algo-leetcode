# MoE Router 知识点整理：稀疏激活、Top-K Routing 与专家加权聚合

MoE, Mixture of Experts, 中文通常叫混合专家模型，是 Mixtral、Grok、DeepSeek 等大模型中非常重要的架构方向。它的核心思想不是让每个 token 都经过所有参数，而是通过 Router 为每个 token 选择少数几个专家，从而实现“大参数容量、低实际计算量”。

这篇文章整理 MoE Router 在实现和面试中最常被问到的知识点：

- Dense 模型的计算瓶颈是什么
- MoE 为什么能做到稀疏激活
- Router / Gate 在 MoE 中负责什么
- Top-K Routing 的完整流程是什么
- 为什么必须先对所有专家做 softmax，再取 top-k
- 为什么 top-k 权重还需要重新归一化
- MoE 输出如何按专家权重加权聚合
- PyTorch 中如何实现 `TopKRouter` 和简化版 `SparseMoEBlock`
- 实现时有哪些常见 shape、dtype 和路由错误

## 1. Dense 模型的痛点

标准 Transformer 是 Dense 模型。也就是说，每个 token 都会经过每一层的全部参数。

如果一个模型有 70B 参数，那么每个 token 在前向计算时都要激活大量参数。随着模型规模变大，训练和推理的计算量也会线性增加。

Dense 模型的问题可以概括为：

```text
模型参数越多 -> 知识容量越大
模型参数越多 -> 每个 token 的计算成本也越高
```

这导致一个矛盾：

- 想提升模型能力，需要增加参数量。
- 但参数量越大，训练和推理越贵。

MoE 的目标就是缓解这个矛盾。

## 2. MoE 的核心思想

MoE 的核心是稀疏激活，Sparse Activation。

它通常会把 Transformer 中的 MLP 层替换成多个并列的 Expert。每个 Expert 本质上可以是一个小型前馈网络，例如 SwiGLU MLP。

对于每个 token，MoE 不会调用所有专家，而是通过一个轻量 Router 选择其中 `K` 个专家。

例如：

```text
专家总数 num_experts = 8
每个 token 只选择 top_k = 2 个专家
```

那么每个 token 只会经过 2 个专家，而不是 8 个专家。

这就是 MoE 的关键收益：

```text
总参数量很大，但每个 token 实际激活的参数很少。
```

以直观例子理解：

```text
Dense 70B:
每个 token 激活约 70B 参数。

MoE 8x7B, top_k=2:
总专家参数约 56B，但每个 token 只激活约 14B 专家参数。
```

因此 MoE 可以在保持较低计算量的同时，显著增加模型的知识容量。

## 3. Router 在 MoE 中做什么

Router 也叫 Gate，它是 MoE 中的路由网络。

给定一个 token 的 hidden state：

$$
x \in \mathbb{R}^{d}
$$

Router 使用一个线性层，把它映射成每个专家的打分：

$$
h = xW_{gate}
$$

其中：

- `d` 是 hidden size。
- `E` 是专家数量 `num_experts`。
- $W_{gate} \in \mathbb{R}^{d \times E}$。
- $h \in \mathbb{R}^{E}$，表示这个 token 对每个专家的原始偏好分数。

在代码中就是：

```python
self.gate = nn.Linear(hidden_size, num_experts, bias=False)
router_logits = self.gate(hidden_states)
```

如果输入是 `[B, S, H]`，通常会先展平成 token 维度：

```text
[batch_size, seq_len, hidden_size]
  -> [batch_size * seq_len, hidden_size]
```

这样每一行就是一个 token。

## 4. Top-K Routing 的完整流程

Top-K Routing 可以分成四步：

```text
1. 用 gate 计算每个 token 对所有专家的 logits。
2. 对所有专家维度做 softmax，得到全局专家概率。
3. 从概率中取 top_k 个最大值及其专家索引。
4. 对 top_k 权重重新归一化，然后按权重聚合专家输出。
```

对应数学过程如下。

### 4.1 计算 Router Logits

给定 token 表示 $x$：

$$
h = xW_{gate}
$$

其中：

$$
h \in \mathbb{R}^{E}
$$

`h[i]` 表示当前 token 分配给第 `i` 个专家的原始打分。

### 4.2 全局 Softmax

先在所有专家维度上做 softmax：

$$
p = \mathrm{Softmax}(h)
$$

其中：

$$
p \in \mathbb{R}^{E},\quad \sum_{i=1}^{E}p_i=1
$$

代码是：

```python
routing_probs = F.softmax(router_logits.float(), dim=-1)
```

注意这里使用 `.float()`，通常是为了在 FP16/BF16 训练中把 softmax 临时提升到 FP32，降低数值溢出或概率塌缩的风险。

### 4.3 Top-K 选择

从全局概率中选出最大的 `K` 个：

$$
p_{topk}, idx_{topk} = \mathrm{TopK}(p, K)
$$

代码是：

```python
routing_weights, selected_experts = torch.topk(
    routing_probs,
    self.top_k,
    dim=-1,
)
```

返回值含义：

- `routing_weights`: 每个 token 选中的 `K` 个专家概率。
- `selected_experts`: 每个 token 选中的 `K` 个专家索引。

如果 `num_tokens = batch_size * seq_len`，那么：

```text
routing_weights:  [num_tokens, top_k]
selected_experts: [num_tokens, top_k]
```

### 4.4 Top-K 权重重归一化

取出 top-k 后，被保留下来的概率和通常小于 1：

$$
\sum_{i \in TopK}p_i \le 1
$$

为了让专家输出的加权和保持稳定尺度，需要重新归一化：

$$
w_i = \frac{p_i}{\sum_{j \in TopK}p_j}
$$

代码是：

```python
routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
```

归一化后，每个 token 的 top-k 权重之和重新变成 1：

```text
routing_weights.sum(dim=-1) == 1
```

## 5. Softmax Trap：为什么不能先 Top-K 再 Softmax

MoE Router 里最容易犯的错误是：

```text
先从 logits 中取 top-k
再只对这 k 个 logits 做 softmax
```

错误写法类似：

```python
topk_logits, selected_experts = torch.topk(router_logits, top_k, dim=-1)
routing_weights = F.softmax(topk_logits, dim=-1)
```

这种写法的问题是：它只在局部 top-k logits 内部做归一化，丢掉了这些专家相对于所有专家的全局置信度。

正确做法是：

```python
routing_probs = F.softmax(router_logits.float(), dim=-1)
routing_weights, selected_experts = torch.topk(routing_probs, top_k, dim=-1)
routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
```

二者的差别可以这样理解：

```text
错误做法:
只关心 top-k 内部谁更大。

正确做法:
先判断每个专家在全局专家集合中的概率，再保留概率最高的 top-k。
```

在 Mixtral 风格 MoE 中，常见实现就是先对所有专家做 softmax，再取 top-k，并对 top-k 权重重归一化。

## 6. 输出融合：按专家权重加权求和

Router 只决定 token 去哪些专家，以及每个专家输出占多大权重。

真正的 token 输出是专家输出的加权和：

$$
y = \sum_{i \in TopK} w_i \cdot \mathrm{Expert}_{idx_i}(x)
$$

其中：

- `idx_i` 是第 `i` 个被选中的专家编号。
- `w_i` 是该专家对应的归一化路由权重。
- `Expert_{idx_i}(x)` 是该专家对 token 的输出。

如果 `top_k = 2`，可以直观写成：

```text
y = w_1 * Expert_a(x) + w_2 * Expert_b(x)
```

并且：

```text
w_1 + w_2 = 1
```

## 7. Shape Tracking

假设：

```text
B = batch size
S = sequence length
H = hidden_size
E = num_experts
K = top_k
T = B * S
```

Router 的主要 shape 如下：

| 步骤 | 张量 | 形状 |
|---|---|---|
| 输入 | `hidden_states` | `[B, S, H]` |
| 展平 token | `flat_hidden_states` | `[T, H]` |
| Router logits | `router_logits` | `[T, E]` |
| 全局 softmax | `routing_probs` | `[T, E]` |
| Top-K 权重 | `routing_weights` | `[T, K]` |
| Top-K 专家索引 | `selected_experts` | `[T, K]` |
| 单个专家输入 | `current_state` | `[num_tokens_for_expert, H]` |
| 单个专家输出 | `current_output` | `[num_tokens_for_expert, H]` |
| 聚合输出 | `final_hidden_states` | `[T, H]` |
| 恢复序列形状 | `output` | `[B, S, H]` |

最重要的约束是：

```text
专家输出维度必须是 hidden_size，否则无法聚合回原 hidden states。
```

## 8. PyTorch 参考实现：TopKRouter

下面是一个最小但完整的 Top-K Router 实现：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)

        router_logits = self.gate(hidden_states)

        routing_probs = F.softmax(router_logits.float(), dim=-1)
        routing_weights, selected_experts = torch.topk(
            routing_probs,
            self.top_k,
            dim=-1,
        )
        routing_weights = routing_weights / routing_weights.sum(
            dim=-1,
            keepdim=True,
        )

        routing_weights = routing_weights.to(hidden_states.dtype)
        return routing_weights, selected_experts
```

这里有三个细节值得注意：

- `router_logits.float()`：softmax 用 FP32 计算更稳定。
- `torch.topk(..., dim=-1)`：沿专家维度选择 top-k。
- `routing_weights.to(hidden_states.dtype)`：最后把权重转回输入 dtype，方便后续混合精度计算。

## 9. PyTorch 参考实现：SparseMoEBlock

下面是一个便于理解的简化 MoE 聚合层。真实模型中的 Expert 通常是 SwiGLU MLP，这里用 `nn.Linear` 代替。

```python
class SparseMoEBlock(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.router = TopKRouter(hidden_size, num_experts, top_k)
        self.experts = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, hidden_size = hidden_states.shape
        routing_weights, selected_experts = self.router(hidden_states)

        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        flat_hidden_states = hidden_states.view(-1, hidden_size)

        for expert_idx, expert in enumerate(self.experts):
            token_idx, kth_expert = torch.where(selected_experts == expert_idx)

            if token_idx.shape[0] > 0:
                current_state = flat_hidden_states[token_idx]
                current_output = expert(current_state)
                current_weight = routing_weights[token_idx, kth_expert].unsqueeze(-1)
                final_hidden_states[token_idx] += current_output * current_weight

        return final_hidden_states.view(batch_size, seq_len, hidden_size)
```

这个实现的核心逻辑是：

```text
遍历每个 expert
  -> 找出哪些 token 选择了这个 expert
  -> 把这些 token 批量送入该 expert
  -> 取出对应的 routing weight
  -> 加权累加到 final_hidden_states
```

## 10. `torch.where(selected_experts == expert_idx)` 如何理解

`selected_experts` 的形状是：

```text
[num_tokens, top_k]
```

例如：

```text
selected_experts =
[
  [1, 3],
  [0, 2],
  [3, 5],
]
```

表示：

```text
token 0 选择 expert 1 和 expert 3
token 1 选择 expert 0 和 expert 2
token 2 选择 expert 3 和 expert 5
```

当遍历到 `expert_idx = 3` 时：

```python
token_idx, kth_expert = torch.where(selected_experts == 3)
```

得到的含义是：

```text
token_idx:   哪些 token 选择了 expert 3
kth_expert:  expert 3 是该 token 的第几个 top-k 专家
```

对于上面的例子：

```text
token 0 的第 2 个专家是 3
token 2 的第 1 个专家是 3
```

所以可以用：

```python
routing_weights[token_idx, kth_expert]
```

取出每个 token 对当前 expert 的权重。

## 11. 最小测试

一个 Router 至少应该通过下面几个检查：

- MoE 输出形状和输入形状一致。
- `routing_weights` 形状是 `[batch_size * seq_len, top_k]`。
- `selected_experts` 形状是 `[batch_size * seq_len, top_k]`。
- 每个 token 的 top-k 权重和接近 1。
- 所有专家索引都在合法范围内。

```python
def test_moe_router():
    torch.manual_seed(42)
    batch_size, seq_len, hidden_size = 2, 4, 16
    num_experts, top_k = 8, 2

    moe = SparseMoEBlock(hidden_size, num_experts, top_k)
    x = torch.randn(batch_size, seq_len, hidden_size)

    out = moe(x)
    assert out.shape == x.shape

    weights, indices = moe.router(x)
    assert weights.shape == (batch_size * seq_len, top_k)
    assert indices.shape == (batch_size * seq_len, top_k)

    expected = torch.ones(batch_size * seq_len, dtype=weights.dtype)
    assert torch.allclose(weights.sum(dim=-1), expected)

    assert torch.all((indices >= 0) & (indices < num_experts))

    print("All tests passed.")


test_moe_router()
```

## 12. 常见实现错误

### 12.1 先 Top-K Logits 再 Softmax

错误写法：

```python
topk_logits, selected_experts = torch.topk(router_logits, top_k, dim=-1)
routing_weights = F.softmax(topk_logits, dim=-1)
```

问题是只在 top-k 内部做局部 softmax，丢失了所有专家维度上的全局概率信息。

正确写法：

```python
routing_probs = F.softmax(router_logits.float(), dim=-1)
routing_weights, selected_experts = torch.topk(routing_probs, top_k, dim=-1)
routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
```

### 12.2 忘记重归一化

错误写法：

```python
routing_weights, selected_experts = torch.topk(routing_probs, top_k, dim=-1)
```

如果直接使用 `routing_weights`，它们的和通常小于 1，导致 MoE 输出整体尺度变小。

正确写法：

```python
routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
```

### 12.3 Softmax 维度写错

错误写法：

```python
routing_probs = F.softmax(router_logits.float(), dim=0)
```

`dim=0` 会跨 token 做归一化，这是错误的。

正确写法：

```python
routing_probs = F.softmax(router_logits.float(), dim=-1)
```

因为最后一维才是专家维度。

### 12.4 忘记把输入展平成 token 维度

Router 通常对每个 token 独立路由，所以需要把 `[B, S, H]` 展平成 `[B*S, H]`：

```python
hidden_states = hidden_states.view(-1, hidden_size)
```

如果不展平，也可以让 `nn.Linear` 直接作用在最后一维，但后续 `routing_weights`、`selected_experts` 和专家聚合逻辑会更复杂。为了清晰实现，先展平是常见写法。

### 12.5 专家索引和权重位置没有对齐

在聚合阶段，不能只用 `token_idx` 取权重，还必须用 `kth_expert` 指出当前 expert 是该 token 选中的第几个专家：

```python
current_weight = routing_weights[token_idx, kth_expert].unsqueeze(-1)
```

如果写成：

```python
current_weight = routing_weights[token_idx]
```

就会得到 `[num_tokens_for_expert, top_k]`，不仅 shape 不对，也无法对应当前专家的那一个权重。

### 12.6 忽略 dtype

Router softmax 建议用 FP32：

```python
routing_probs = F.softmax(router_logits.float(), dim=-1)
```

但后续和 expert 输出相乘时，通常转回输入 dtype：

```python
routing_weights = routing_weights.to(hidden_states.dtype)
```

这样更符合混合精度训练和推理的常见做法。

## 13. 工业实现中的优化

上面的 `SparseMoEBlock` 用 Python for-loop 遍历专家，主要是为了教学清晰。

真实系统中，MoE 的瓶颈不只在计算，还在 token 分发和专家聚合。工业实现通常会做更复杂的优化：

- Token Sorting：按专家编号把 token 排序，让同一专家的 token 连续排列。
- Batched Expert Computation：同一专家一次性处理一个 token batch，提高 GPU 利用率。
- Expert Parallelism：把不同专家放到不同 GPU 上。
- Capacity Factor：限制单个专家最多接收多少 token，避免某些专家过载。
- Load Balancing Loss：训练时鼓励 token 更均匀地分配到各个专家。
- Dropless Routing：尽量避免 token 因专家容量不足被丢弃。

本篇的代码重点是 Router 的基本数学和张量逻辑，工程系统中还会围绕通信、排序、负载均衡和 kernel 融合继续优化。

## 14. 面试高频问题

### 14.1 MoE 为什么能降低计算量

因为 MoE 只让每个 token 激活少数专家，而不是所有专家。

```text
Dense 模型:
每个 token 经过全部参数。

MoE 模型:
每个 token 只经过 top-k 个专家。
```

所以 MoE 可以增加总参数容量，但不按总参数量线性增加每个 token 的计算成本。

### 14.2 Router 输出的两个张量分别是什么

Router 通常返回：

```python
routing_weights, selected_experts
```

其中：

- `routing_weights`: 每个 token 选择的 top-k 专家的归一化权重。
- `selected_experts`: 每个 token 选择的 top-k 专家索引。

形状都是：

```text
[batch_size * seq_len, top_k]
```

### 14.3 为什么要先 softmax 再 top-k

因为 Router 需要先在所有专家维度上得到全局概率分布，再从全局概率中选择最高的几个专家。

先 top-k logits 再 softmax 只得到局部相对权重，会丢掉 top-k 专家相对于全部专家的置信度。

### 14.4 为什么 top-k 后还要重归一化

因为 top-k 只保留了部分专家概率，保留下来的概率和通常小于 1。

为了让专家输出的加权和保持稳定尺度，需要让 top-k 权重重新加和为 1。

### 14.5 `top_k=1` 和 `top_k=2` 有什么区别

`top_k=1` 时，每个 token 只走一个专家，计算更省，但路由更硬，模型表达和训练稳定性可能较差。

`top_k=2` 时，每个 token 可以融合两个专家的输出，表达更灵活，但计算量和通信量更高。

很多 MoE 模型会选择 `top_k=2` 作为效果和成本之间的折中。

## 15. 工程实现 Checklist

实现 MoE Router 时，可以按下面清单检查：

- `gate` 的输出维度是 `num_experts`。
- 输入 `[B, S, H]` 被展平成 `[B*S, H]`。
- `router_logits` 的形状是 `[B*S, num_experts]`。
- softmax 使用 `dim=-1`，也就是专家维度。
- softmax 前将 logits 临时转成 FP32。
- `torch.topk` 作用在专家维度 `dim=-1`。
- `routing_weights` 和 `selected_experts` 的形状都是 `[B*S, top_k]`。
- top-k 权重经过重归一化，每行和为 1。
- `selected_experts` 的取值范围是 `[0, num_experts)`。
- 聚合时使用 `routing_weights[token_idx, kth_expert]` 对齐当前专家权重。
- 最终输出 reshape 回 `[B, S, H]`。

## 16. 总结

MoE Router 的核心可以记成一条流程：

```text
hidden_states
  -> gate 得到所有专家 logits
  -> 全局 softmax 得到专家概率
  -> top-k 选择专家和权重
  -> top-k 权重重归一化
  -> 专家计算
  -> 按 routing weight 加权聚合
```

最关键的实现点是：

```text
先对所有专家做 softmax，再取 top-k，最后对 top-k 权重重归一化。
```

掌握这条主线，就能清楚解释 MoE 为什么能稀疏激活、Router 如何选择专家、专家输出如何融合，以及 PyTorch 实现中最容易出错的 shape 和 dtype 细节。
