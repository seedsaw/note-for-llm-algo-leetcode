# MoE Load Balancing Loss 知识点整理：路由崩塌、辅助损失与 Top-K 负载均衡

在上一节 [MoE Router](./06_MoE_Router_GitHub_Blog.md) 中，我们实现了 Top-K Routing：每个 token 通过 Router 选择少数几个专家，然后按路由权重聚合专家输出。

但真实训练 MoE 模型时，仅有 Router 还不够。一个很常见的问题是：Router 会逐渐偏向少数几个专家，把大量 token 都送到同一批专家上。这种现象通常叫 **路由崩塌**，Router Collapse。

路由崩塌会带来三个直接后果：

- 少数专家负载过高，容易成为计算瓶颈，甚至导致 OOM。
- 大量专家几乎收不到 token，参数得不到充分训练。
- MoE 退化成“少数专家模型”，失去稀疏扩展参数容量的意义。

为了解决这个问题，Switch Transformer、Mixtral、DeepSeek 等 MoE 模型都会引入一个额外的 **负载均衡辅助损失**，Load Balancing Auxiliary Loss。它会被加到主任务 loss 上，强制 Router 尽量把 token 分散到不同专家。

这篇笔记整理 MoE Load Balancing Loss 在实现和面试中最常被问到的知识点：

- Router Collapse 是什么
- 为什么 MoE 需要辅助损失
- Load Balancing Loss 的数学公式是什么
- $f_i$ 和 $P_i$ 分别表示什么
- Top-K Routing 下如何统计专家负载
- 如何用 PyTorch 实现 `compute_load_balancing_loss`
- 如何验证不均衡路由会得到更大的 loss
- 实际训练中如何把 auxiliary loss 接入总 loss
- 实现时有哪些常见 shape、归一化和梯度问题

## 1. 问题背景：Router Collapse

MoE 的理想状态是：不同 token 根据语义和特征被分配给不同专家。

例如有 8 个专家，`top_k = 2`：

```text
token 0 -> expert 0, expert 3
token 1 -> expert 2, expert 5
token 2 -> expert 1, expert 7
token 3 -> expert 4, expert 6
...
```

这样每个专家都能收到一定数量的 token，整体计算也比较均衡。

但如果没有约束，Router 很可能学出这种分配：

```text
token 0 -> expert 0, expert 1
token 1 -> expert 0, expert 1
token 2 -> expert 0, expert 1
token 3 -> expert 0, expert 1
...
```

这就是路由崩塌。

从优化角度看，Router 这样做并不奇怪。训练早期某几个专家可能因为随机初始化、数据分布或梯度噪声略微表现更好，Router 就会继续把更多 token 发给它们。收到更多 token 的专家又会训练得更充分，于是强者更强，弱者更弱，最后形成自我强化。

因此 MoE 训练通常需要额外约束：

```text
主任务目标：让模型预测更准确。
辅助目标：让 Router 分配 token 时更均衡。
```

## 2. 核心公式

经典的 MoE 负载均衡辅助损失可以写成：

$$
L_{aux} = \alpha \cdot E \sum_{i=1}^{E} f_i \cdot P_i
$$

其中：

- $E$：专家总数，也就是 `num_experts`。
- $f_i$：专家 $i$ 实际被选中的 token 比例。
- $P_i$：专家 $i$ 的平均路由概率或平均路由权重。
- $\alpha$：辅助损失权重，通常是一个很小的数，例如 `0.01`。

在代码里，最终就是：

```python
aux_loss = alpha * num_experts * (f_i * P_i).sum()
```

直观理解：

- 如果某个专家被大量 token 选中，那么它的 $f_i$ 会变大。
- 如果 Router 对某个专家给了很高概率，那么它的 $P_i$ 会变大。
- 当同一个专家同时拥有较大的 $f_i$ 和 $P_i$，点积项 $f_i \cdot P_i$ 会变大。
- 优化器为了降低辅助损失，会倾向于降低这种集中分配，让 token 更均匀地流向所有专家。

## 3. $f_i$：专家实际负载比例

$f_i$ 关注的是 **实际有多少 token 被分配给专家 $i$**。

假设：

```text
num_tokens = 1000
num_experts = 8
top_k = 2
```

那么总的专家选择次数是：

```text
total_assignments = num_tokens * top_k = 2000
```

如果专家 0 被选中了 250 次，那么：

$$
f_0 = \frac{250}{2000} = 0.125
$$

如果所有专家完全均匀，每个专家被选中的次数都应该接近：

```text
2000 / 8 = 250
```

对应：

$$
f_i = \frac{1}{E} = \frac{1}{8}
$$

在 PyTorch 中，可以用 `one_hot` 统计每个专家被选中的次数：

```python
expert_mask = F.one_hot(selected_experts, num_classes=num_experts)
tokens_per_expert = expert_mask.sum(dim=(0, 1)).float()
f_i = tokens_per_expert / (total_tokens * top_k)
```

这里：

```text
selected_experts:  [num_tokens, top_k]
expert_mask:       [num_tokens, top_k, num_experts]
tokens_per_expert: [num_experts]
f_i:               [num_experts]
```

## 4. $P_i$：专家平均路由权重

$P_i$ 关注的是 **Router 在概率层面有多偏向专家 $i$**。

在 Top-K Router 中，我们通常会得到两个张量：

```text
routing_weights:  [num_tokens, top_k]
selected_experts: [num_tokens, top_k]
```

例如：

```text
token 0:
  selected_experts = [3, 1]
  routing_weights  = [0.7, 0.3]

token 1:
  selected_experts = [3, 5]
  routing_weights  = [0.6, 0.4]
```

这表示：

- token 0 选择了专家 3 和专家 1，权重分别是 0.7 和 0.3。
- token 1 选择了专家 3 和专家 5，权重分别是 0.6 和 0.4。

要计算每个专家累计拿到多少路由权重，可以用 `scatter_add_`：

```python
P_i = torch.zeros(num_experts, dtype=routing_weights.dtype, device=routing_weights.device)
P_i.scatter_add_(0, selected_experts.flatten(), routing_weights.flatten())
P_i = P_i / (total_tokens * top_k)
```

关键点是这一行：

```python
P_i.scatter_add_(0, selected_experts.flatten(), routing_weights.flatten())
```

它会把每个位置的 `routing_weights` 累加到对应专家 id 上。

例如：

```text
selected_experts.flatten() = [3, 1, 3, 5]
routing_weights.flatten()  = [0.7, 0.3, 0.6, 0.4]
```

累加结果大致是：

```text
expert 1 += 0.3
expert 3 += 0.7 + 0.6
expert 5 += 0.4
```

最后除以 `total_tokens * top_k`，得到每个专家的平均路由权重。

## 5. Top-K 下的归一化细节

这份实现假设 `routing_weights` 是 Router 取出 top-k 后重新归一化的权重：

```python
routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
```

因此每个 token 的 top-k 权重和为 1：

```text
routing_weights[token].sum() == 1
```

在本教程实现中：

```python
P_i = P_i / (total_tokens * top_k)
f_i = tokens_per_expert / (total_tokens * top_k)
```

也就是说，`f_i` 和 `P_i` 都按总选择次数 `num_tokens * top_k` 进行归一化。

这种写法的理论最小值是：

$$
L_{min} = \frac{\alpha}{K}
$$

其中 $K$ 是 `top_k`。

当 `alpha = 0.01` 且 `top_k = 2` 时：

```text
expected_min = 0.01 / 2 = 0.005
```

这也是后面单元测试中用来验证实现正确性的值。

## 6. 完整实现

下面是一个支持 Top-K Routing 的 MoE Load Balancing Loss 实现：

```python
import torch
import torch.nn.functional as F


def compute_load_balancing_loss(
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
    top_k: int,
    alpha: float = 0.01,
):
    """
    计算 MoE 的负载均衡辅助损失，支持 Top-K Routing。

    Args:
        routing_weights:
            shape 为 [batch_size * seq_len, top_k]。
            每个 token 选中的 K 个专家的路由权重，通常已经在 top-k 内重新归一化。
        selected_experts:
            shape 为 [batch_size * seq_len, top_k]。
            每个 token 选中的 K 个专家索引。
        num_experts:
            专家总数 E。
        top_k:
            每个 token 选择的专家数量 K。
        alpha:
            auxiliary loss 的权重系数。

    Returns:
        aux_loss:
            标量张量，表示负载均衡辅助损失。
    """
    batch_size_x_seq_len, _ = selected_experts.shape
    total_tokens = batch_size_x_seq_len

    # P_i: 每个专家的平均路由权重。
    P_i = torch.zeros(
        num_experts,
        dtype=routing_weights.dtype,
        device=routing_weights.device,
    )
    P_i.scatter_add_(0, selected_experts.flatten(), routing_weights.flatten())
    P_i = P_i / (total_tokens * top_k)

    # f_i: 每个专家实际被选中的比例。
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts)
    tokens_per_expert = expert_mask.sum(dim=(0, 1)).float()
    f_i = tokens_per_expert / (total_tokens * top_k)

    # L_aux = alpha * E * sum_i(f_i * P_i)
    aux_loss = alpha * num_experts * (f_i * P_i).sum()

    return aux_loss
```

## 7. 单元测试：不均衡分配 vs 均匀分配

我们可以构造两个极端 case 来验证：

1. 极度不均衡：所有 token 都选择专家 0 和专家 1。
2. 完全均匀：token 均匀分配给所有专家。

如果实现正确，不均衡 case 的 loss 应该明显大于均匀 case。

```python
def test_aux_loss():
    torch.manual_seed(42)

    num_experts = 8
    top_k = 2
    num_tokens = 1000
    alpha = 0.01

    # 1. 极度不均衡：所有 token 都选专家 0 和专家 1。
    bad_selected = torch.zeros(num_tokens, top_k, dtype=torch.long)
    bad_selected[:, 0] = 0
    bad_selected[:, 1] = 1
    bad_weights = torch.ones(num_tokens, top_k) / top_k

    loss_bad = compute_load_balancing_loss(
        bad_weights,
        bad_selected,
        num_experts,
        top_k,
        alpha,
    )

    # 2. 完全均匀：token 均匀分配给所有专家。
    good_selected = torch.zeros(num_tokens, top_k, dtype=torch.long)
    for i in range(num_tokens):
        good_selected[i, 0] = (i * 2) % num_experts
        good_selected[i, 1] = (i * 2 + 1) % num_experts

    good_weights = torch.ones(num_tokens, top_k) / top_k

    loss_good = compute_load_balancing_loss(
        good_weights,
        good_selected,
        num_experts,
        top_k,
        alpha,
    )

    print(f"极度不均衡的 Loss: {loss_bad.item():.4f}")
    print(f"绝对均匀的 Loss  : {loss_good.item():.4f}")

    expected_min = alpha / top_k

    assert torch.allclose(
        loss_good,
        torch.tensor(expected_min),
        atol=1e-4,
    ), (
        f"理论最小 Loss 计算错误，"
        f"期望 {expected_min:.4f}，实际 {loss_good.item():.4f}"
    )

    assert loss_bad > loss_good * 2, (
        "惩罚项没有对不均衡分布产生足够大的 Loss"
    )

    print("All tests passed.")


test_aux_loss()
```

运行结果：

```text
极度不均衡的 Loss: 0.0200
绝对均匀的 Loss  : 0.0050
All tests passed.
```

可以看到：

```text
loss_bad  = 0.0200
loss_good = 0.0050
```

不均衡路由的 auxiliary loss 是均匀路由的 4 倍，这说明该损失确实会惩罚 Router 把 token 集中送到少数专家的行为。

## 8. 为什么极度不均衡的 loss 更大

继续看上面的极端例子。

### 8.1 均匀分配

当 8 个专家完全均匀时：

$$
f_i = \frac{1}{8}
$$

由于 `routing_weights = 1 / top_k = 1 / 2`，并且本实现按 `total_tokens * top_k` 归一化，均匀情况下：

$$
P_i = \frac{1}{8 \cdot 2} = \frac{1}{16}
$$

所以：

$$
\sum_{i=1}^{8} f_i P_i
= 8 \cdot \frac{1}{8} \cdot \frac{1}{16}
= \frac{1}{16}
$$

辅助损失为：

$$
L_{aux}
= 0.01 \cdot 8 \cdot \frac{1}{16}
= 0.005
$$

### 8.2 极度不均衡

如果所有 token 都只选专家 0 和专家 1：

```text
f_0 = 1 / 2
f_1 = 1 / 2
其他专家 f_i = 0
```

同时：

```text
P_0 = 1 / 4
P_1 = 1 / 4
其他专家 P_i = 0
```

于是：

$$
\sum_i f_i P_i
= \frac{1}{2} \cdot \frac{1}{4}
+ \frac{1}{2} \cdot \frac{1}{4}
= \frac{1}{4}
$$

辅助损失为：

$$
L_{aux}
= 0.01 \cdot 8 \cdot \frac{1}{4}
= 0.02
$$

这正好对应测试结果。

## 9. 如何接入 MoE 训练

在真实训练中，MoE 模型的总损失通常是：

```python
total_loss = ce_loss + aux_loss
```

其中：

- `ce_loss` 是语言模型主任务的 Cross Entropy Loss。
- `aux_loss` 是 MoE Router 的负载均衡损失。

一个简化的训练片段如下：

```python
logits, routing_weights, selected_experts = model(input_ids)

ce_loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    labels.view(-1),
)

aux_loss = compute_load_balancing_loss(
    routing_weights=routing_weights,
    selected_experts=selected_experts,
    num_experts=num_experts,
    top_k=top_k,
    alpha=0.01,
)

loss = ce_loss + aux_loss
loss.backward()
optimizer.step()
```

如果模型有多层 MoE，通常每一层都会产生自己的 Router 结果。可以对每层的 auxiliary loss 求和或求平均：

```python
aux_losses = []

for layer_router_weights, layer_selected_experts in moe_router_outputs:
    aux_losses.append(
        compute_load_balancing_loss(
            layer_router_weights,
            layer_selected_experts,
            num_experts=num_experts,
            top_k=top_k,
            alpha=0.01,
        )
    )

aux_loss = torch.stack(aux_losses).mean()
loss = ce_loss + aux_loss
```

## 10. 与完整 Router 概率版本的区别

这篇教程中的函数输入是：

```text
routing_weights:  [num_tokens, top_k]
selected_experts: [num_tokens, top_k]
```

也就是已经经过 `topk` 之后的稀疏路由结果。

在一些源码实现中，Load Balancing Loss 会直接使用完整 Router softmax 概率：

```text
router_probs: [num_tokens, num_experts]
```

此时 $P_i$ 可以写成：

```python
P_i = router_probs.mean(dim=0)
```

而 $f_i$ 仍然来自实际选中的 top-k 专家：

```python
expert_mask = F.one_hot(selected_experts, num_classes=num_experts)
f_i = expert_mask.float().mean(dim=(0, 1))
```

两类实现都在表达同一个目标：让 Router 的概率偏好和实际专家负载都更均衡。

区别在于：

- 稀疏版本只依赖 top-k 后的 `routing_weights` 和 `selected_experts`。
- 完整概率版本还需要保留 `router_probs`，能让未被 top-k 选中的专家也通过概率项影响 auxiliary loss。

如果你在实现一个接近论文或开源大模型的 MoE 层，通常建议保留完整的 `router_probs` 来计算辅助损失；如果你只在练习 Top-K 路由后的负载统计，本教程的稀疏版本更直观。

## 11. 常见实现错误

### 11.1 忘记乘 `num_experts`

公式中有：

$$
L_{aux} = \alpha \cdot E \sum_i f_i P_i
$$

如果忘记乘 `num_experts`，loss 的尺度会随着专家数量变化而明显变小，不利于设置稳定的 `alpha`。

错误写法：

```python
aux_loss = alpha * (f_i * P_i).sum()
```

正确写法：

```python
aux_loss = alpha * num_experts * (f_i * P_i).sum()
```

### 11.2 归一化分母写错

Top-K Routing 下，每个 token 会产生 `top_k` 次专家选择。因此统计专家被选中比例时，分母应该包含 `top_k`：

```python
f_i = tokens_per_expert / (total_tokens * top_k)
```

如果写成：

```python
f_i = tokens_per_expert / total_tokens
```

那么所有专家的 $f_i$ 之和会变成 `top_k`，loss 尺度会被放大。

### 11.3 `one_hot` 的 dtype 没处理好

`F.one_hot` 返回整数类型张量。后面要计算比例和 loss，最好显式转成浮点：

```python
tokens_per_expert = expert_mask.sum(dim=(0, 1)).float()
```

更严格地说，如果你希望 dtype 和 `routing_weights` 完全一致，可以写成：

```python
tokens_per_expert = expert_mask.sum(dim=(0, 1)).to(routing_weights.dtype)
```

### 11.4 忽略 device

如果模型在 GPU 上，而你创建的临时张量在 CPU 上，会直接报 device mismatch。

推荐写法：

```python
P_i = torch.zeros(
    num_experts,
    dtype=routing_weights.dtype,
    device=routing_weights.device,
)
```

不要写成：

```python
P_i = torch.zeros(num_experts)
```

### 11.5 用 Python 循环逐 token 统计

不推荐：

```python
for token_idx in range(total_tokens):
    for k in range(top_k):
        expert_id = selected_experts[token_idx, k]
        P_i[expert_id] += routing_weights[token_idx, k]
```

这种写法在 token 数很大时会非常慢。推荐使用向量化操作：

```python
P_i.scatter_add_(0, selected_experts.flatten(), routing_weights.flatten())
```

## 12. 面试高频问答

### 12.1 为什么 MoE 需要 Load Balancing Loss？

因为 Router 如果只受主任务 loss 驱动，可能会把大部分 token 分给少数专家，导致路由崩塌。Load Balancing Loss 通过惩罚不均匀分配，让所有专家都有机会接收 token 并参与训练。

### 12.2 $f_i$ 和 $P_i$ 有什么区别？

$f_i$ 是实际分配比例，来自 `selected_experts`，表示专家 $i$ 实际被选中了多少次。

$P_i$ 是平均路由权重或平均路由概率，来自 Router 的概率输出，表示 Router 在概率层面对专家 $i$ 的偏好程度。

### 12.3 为什么不能只看 $f_i$？

只看 $f_i$ 只能知道最终选择是否均匀，但不能直接约束 Router 的概率分布。$P_i$ 能提供概率层面的训练信号，让 Router 在 softmax 输出上也趋向均衡。

### 12.4 为什么 `alpha` 通常很小？

Load Balancing Loss 是辅助目标，主目标仍然是语言建模或下游任务。如果 `alpha` 太大，模型会过度追求均匀分配，损害主任务性能；如果太小，又无法有效防止路由崩塌。

### 12.5 Top-K Routing 下分母为什么是 `total_tokens * top_k`？

因为每个 token 会选择 `top_k` 个专家。专家被选中的总次数不是 `total_tokens`，而是：

```text
total_assignments = total_tokens * top_k
```

统计专家选择比例时，需要除以总选择次数。

## 13. 最小可运行代码

下面这段代码可以直接复制到 Python 文件或 notebook 中运行：

```python
import torch
import torch.nn.functional as F


def compute_load_balancing_loss(
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
    top_k: int,
    alpha: float = 0.01,
):
    total_tokens = selected_experts.shape[0]

    P_i = torch.zeros(
        num_experts,
        dtype=routing_weights.dtype,
        device=routing_weights.device,
    )
    P_i.scatter_add_(0, selected_experts.flatten(), routing_weights.flatten())
    P_i = P_i / (total_tokens * top_k)

    expert_mask = F.one_hot(selected_experts, num_classes=num_experts)
    tokens_per_expert = expert_mask.sum(dim=(0, 1)).to(routing_weights.dtype)
    f_i = tokens_per_expert / (total_tokens * top_k)

    return alpha * num_experts * (f_i * P_i).sum()


def test_aux_loss():
    num_experts = 8
    top_k = 2
    num_tokens = 1000
    alpha = 0.01

    bad_selected = torch.zeros(num_tokens, top_k, dtype=torch.long)
    bad_selected[:, 0] = 0
    bad_selected[:, 1] = 1
    bad_weights = torch.ones(num_tokens, top_k) / top_k

    loss_bad = compute_load_balancing_loss(
        bad_weights,
        bad_selected,
        num_experts,
        top_k,
        alpha,
    )

    good_selected = torch.zeros(num_tokens, top_k, dtype=torch.long)
    for i in range(num_tokens):
        good_selected[i, 0] = (i * 2) % num_experts
        good_selected[i, 1] = (i * 2 + 1) % num_experts
    good_weights = torch.ones(num_tokens, top_k) / top_k

    loss_good = compute_load_balancing_loss(
        good_weights,
        good_selected,
        num_experts,
        top_k,
        alpha,
    )

    expected_min = alpha / top_k

    print(f"loss_bad : {loss_bad.item():.4f}")
    print(f"loss_good: {loss_good.item():.4f}")

    assert torch.allclose(loss_good, torch.tensor(expected_min), atol=1e-4)
    assert loss_bad > loss_good * 2


if __name__ == "__main__":
    test_aux_loss()
```

输出应为：

```text
loss_bad : 0.0200
loss_good: 0.0050
```

## 14. 总结

MoE Load Balancing Loss 的核心目标是防止 Router Collapse。

它通过同时约束两个分布来实现负载均衡：

- $f_i$：专家实际被选中的比例。
- $P_i$：专家平均路由概率或平均路由权重。

最终公式是：

$$
L_{aux} = \alpha \cdot E \sum_{i=1}^{E} f_i \cdot P_i
$$

工程实现时要重点注意：

- Top-K 下统计分母应使用 `total_tokens * top_k`。
- `selected_experts` 用于统计实际负载 $f_i$。
- `routing_weights` 或完整 `router_probs` 用于统计概率偏好 $P_i$。
- 临时张量要和输入保持相同 device。
- 使用 `scatter_add_` 和 `one_hot` 做向量化统计，避免 Python 循环。
- 在训练中使用 `loss = ce_loss + aux_loss` 接入主任务。

掌握这部分之后，MoE Router 就不只是“会选专家”，而是具备了真实训练中必须的防崩塌机制。
