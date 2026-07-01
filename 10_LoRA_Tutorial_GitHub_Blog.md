# LoRA 知识点整理：低秩适配、参数高效微调与权重合并

LoRA, Low-Rank Adaptation, 是大模型微调中最常用的 PEFT, Parameter-Efficient Fine-Tuning, 方法之一。

它的核心思想很简单：

```text
冻结原始预训练权重，只训练一个低秩增量矩阵。
```

这篇笔记整理 LoRA 在实现和面试中最常被问到的知识点：

- 为什么全参微调显存开销巨大
- LoRA 为什么能显著减少可训练参数量
- LoRA 的低秩分解公式是什么
- `A`、`B`、`r`、`alpha` 分别代表什么
- 为什么 `B` 通常初始化为 0
- 如何用 PyTorch 实现 `LoRALinear`
- 推理前如何把 LoRA 权重 merge 回原始权重
- LoRA 在训练和部署中的常见坑

## 1. 全参微调的问题

全参微调会更新模型中的全部参数。

如果模型有 7B 参数，使用 AdamW 训练时，显存里通常不只保存参数本身，还要保存：

- 模型参数
- 参数梯度
- Adam 的一阶动量
- Adam 的二阶动量
- 混合精度训练中的 FP32 master weights 或其他状态

所以训练显存远大于模型权重本身。

对于个人开发者或中小团队来说，全参微调大模型通常成本过高。更重要的是，很多下游任务并不需要更新全部权重，只需要对模型行为做一个小幅适配。

LoRA 解决的就是这个问题：

```text
不改动大矩阵 W0，只训练一个小的低秩更新 Delta W。
```

## 2. LoRA 的核心公式

标准线性层可以写成：

$$
h = x W_0^T
$$

其中：

- $x \in \mathbb{R}^{..., k}$
- $W_0 \in \mathbb{R}^{d \times k}$
- 输出 $h \in \mathbb{R}^{..., d}$

LoRA 不直接训练 $W_0$，而是添加一个低秩更新：

$$
W = W_0 + \Delta W
$$

并把 $\Delta W$ 分解成两个小矩阵：

$$
\Delta W = \frac{\alpha}{r} B A
$$

其中：

$$
A \in \mathbb{R}^{r \times k}
$$

$$
B \in \mathbb{R}^{d \times r}
$$

于是前向传播变成：

$$
h = x W_0^T + \frac{\alpha}{r} x A^T B^T
$$

代码里通常写成：

```python
result = self.linear(x)
lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
return result + lora_out
```

其中：

```python
self.scaling = lora_alpha / r
```

## 3. 为什么叫低秩

原始线性层权重是：

```text
W0: [out_features, in_features]
```

如果直接训练一个完整的增量矩阵：

```text
Delta W: [out_features, in_features]
```

参数量是：

```text
out_features * in_features
```

LoRA 把它分解成：

```text
A: [r, in_features]
B: [out_features, r]
```

参数量变成：

```text
r * in_features + out_features * r
= r * (in_features + out_features)
```

当 `r` 远小于 `in_features` 和 `out_features` 时，可训练参数量会小很多。

举个例子：

```text
in_features  = 4096
out_features = 4096
r            = 8
```

全量矩阵参数量：

```text
4096 * 4096 = 16,777,216
```

LoRA 参数量：

```text
8 * (4096 + 4096) = 65,536
```

只占原矩阵的约 0.39%。

## 4. A 和 B 的维度直觉

LoRA 分支的计算顺序是：

```text
x -> A -> B -> output
```

假设输入：

```text
x: [batch_size, seq_len, in_features]
```

第一步降维：

```python
x @ A.T
```

shape 变化：

```text
[B, T, in_features] @ [in_features, r]
= [B, T, r]
```

第二步升维：

```python
(x @ A.T) @ B.T
```

shape 变化：

```text
[B, T, r] @ [r, out_features]
= [B, T, out_features]
```

这样 LoRA 分支输出就能和原始 linear 输出相加。

## 5. 为什么 B 初始化为 0

LoRA 初始化时希望模型行为和原始预训练模型完全一致。

也就是说，一开始应该有：

```text
Delta W = 0
```

这样微调开始时：

```text
W = W0 + Delta W = W0
```

为了做到这一点，常见初始化是：

```python
nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
nn.init.zeros_(self.lora_B)
```

只要 `B = 0`，就有：

```text
B A = 0
```

所以初始 LoRA 分支输出为 0。

如果 A 和 B 都随机初始化，模型一开始的输出就会偏离预训练模型，可能破坏已经学到的能力。

## 6. 为什么要有 alpha / r

LoRA 的缩放项是：

```python
scaling = lora_alpha / r
```

它控制低秩更新对原模型输出的影响幅度。

如果只使用：

```text
Delta W = B A
```

那么改变 rank `r` 会改变更新矩阵的统计尺度，使得不同 rank 下学习率不太可比。

加入 `alpha / r` 后，可以更稳定地控制 LoRA 分支的整体强度。常见设置是：

```text
r = 8, alpha = 16
r = 16, alpha = 32
```

但这不是固定规则，实际要根据任务、模型大小和训练稳定性调整。

## 7. LoRALinear 完整实现

下面是一个最小可用的 `LoRALinear`：

```python
import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: int = 16,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r

        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.weight.requires_grad = False

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.linear(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return result + lora_out

    def merge_weights(self):
        self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
```

这段代码体现了 LoRA 的四个核心：

- 原始 `linear.weight` 冻结。
- `lora_A` 和 `lora_B` 是可训练参数。
- 前向传播时原始分支和 LoRA 分支相加。
- 推理前可以把低秩更新合并回主权重。

## 8. 权重合并为什么成立

LoRA 前向是：

$$
h = x W_0^T + \frac{\alpha}{r} x A^T B^T
$$

由于：

$$
x A^T B^T = x (B A)^T
$$

所以可以写成：

$$
h = x \left(W_0 + \frac{\alpha}{r} B A\right)^T
$$

也就是合并后的权重：

$$
W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A
$$

代码：

```python
self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
```

合并后，推理图里只剩一个普通线性层，不再需要额外计算 A 和 B，因此没有额外推理延迟。

## 9. 如何验证实现正确

可以用三个断言测试：

```python
def test_lora():
    in_dim, out_dim = 128, 256
    batch_size, seq_len = 32, 10
    layer = LoRALinear(in_dim, out_dim, r=8, lora_alpha=16)

    x = torch.randn(batch_size, seq_len, in_dim)

    with torch.no_grad():
        out_lora = layer(x)
        out_base = layer.linear(x)
        assert torch.allclose(out_lora, out_base)

    layer.lora_B.data.normal_(0, 0.02)
    out_trained = layer(x)
    assert not torch.allclose(out_trained, out_base)

    layer.merge_weights()
    out_merged = layer.linear(x)
    assert torch.allclose(out_trained, out_merged, atol=1e-5)
```

三个检查分别对应：

- 初始化时 LoRA 分支输出为 0。
- 修改 LoRA 参数后，输出确实变化。
- merge 后输出与未 merge 的 LoRA 输出一致。

## 10. 哪些层适合加 LoRA

在 Transformer 中，LoRA 通常加在线性层上，尤其是 attention 和 MLP 中的大矩阵。

常见目标模块包括：

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

实践中最常见的是先对 attention 的 `q_proj` 和 `v_proj` 加 LoRA。如果任务复杂或数据量较大，可以扩展到更多投影层。

选择目标层的基本权衡是：

```text
加得越多 -> 表达能力越强 -> 可训练参数和显存越多
加得越少 -> 成本越低 -> 适配能力可能不足
```

## 11. LoRA 训练时哪些参数参与优化

只应该把 LoRA 参数交给 optimizer：

```python
trainable_params = [
    p for p in model.parameters()
    if p.requires_grad
]
optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
```

如果冻结做错，可能出现两类问题：

- 原模型权重也被更新，变成全参微调。
- LoRA 参数没有被加入 optimizer，训练没有效果。

一个简单检查是统计可训练参数量：

```python
num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
num_total = sum(p.numel() for p in model.parameters())
print(num_trainable, num_total, num_trainable / num_total)
```

LoRA 微调时，这个比例通常应该很小。

## 12. 常见错误

### 错误 1：忘记冻结原始权重

错误写法：

```python
self.linear = nn.Linear(in_features, out_features, bias=False)
```

但没有：

```python
self.linear.weight.requires_grad = False
```

这样会导致原始权重也被训练，违背 LoRA 的参数高效目标。

### 错误 2：B 没有初始化为 0

如果 `lora_B` 随机初始化，模型初始输出会偏离原模型。正确做法：

```python
nn.init.zeros_(self.lora_B)
```

### 错误 3：矩阵方向写反

PyTorch 的 `nn.Linear` 权重 shape 是：

```text
[out_features, in_features]
```

所以 LoRA 权重合并应是：

```python
self.lora_B @ self.lora_A
```

shape：

```text
[out_features, r] @ [r, in_features]
= [out_features, in_features]
```

写成 `A @ B` 会 shape 不匹配，或者语义错误。

### 错误 4：重复 merge

如果多次调用：

```python
merge_weights()
merge_weights()
```

LoRA 增量会被重复加到主权重上。

真实工程中通常要维护一个状态位：

```python
self.merged = False
```

merge 后设置为 `True`，防止重复合并。

### 错误 5：merge 后继续训练没有处理

merge 是部署优化。训练阶段通常保持 LoRA 分支独立，方便只更新 A 和 B。

如果 merge 后继续训练，需要明确是否还保留 LoRA 参数，以及是否需要 unmerge。成熟 PEFT 框架会处理这些状态。

## 13. LoRA 和 Adapter 的区别

LoRA 和 Adapter 都属于 PEFT，但插入位置和计算方式不同。

Adapter 通常是在 Transformer block 中插入一个小 MLP：

```text
x -> down projection -> activation -> up projection -> x + adapter_out
```

LoRA 则是修改已有线性层的权重更新：

```text
W = W0 + BA
```

LoRA 的一个重要优势是可以 merge 回原始线性层，因此推理时不增加额外层结构。

## 14. 面试高频问法

### Q1: LoRA 为什么能减少参数量

因为它不训练完整的 `Delta W`，而是用两个低秩矩阵 `B A` 表示更新。参数量从 `out * in` 变成 `r * (in + out)`。

### Q2: 为什么冻结原始权重

LoRA 的目标是在保留预训练模型能力的基础上学习一个小的任务增量。冻结原始权重可以减少显存、优化器状态和灾难性遗忘风险。

### Q3: 为什么 `lora_B` 要初始化为 0

为了让训练开始时 `Delta W = B A = 0`，使 LoRA 模型初始输出等于原始预训练模型输出。

### Q4: `alpha / r` 的作用是什么

它控制 LoRA 更新的缩放幅度，并让不同 rank 下的更新尺度更稳定。

### Q5: LoRA 为什么可以零延迟推理

因为 LoRA 的低秩更新可以合并成一个完整矩阵：

```text
W_merged = W0 + alpha / r * B A
```

合并后推理只需要普通线性层。

## 15. 小结

LoRA 的关键不是“多加两个矩阵”这么简单，而是三个工程设计同时成立：

```text
1. 冻结原始权重 W0
2. 用低秩矩阵 BA 表示可训练增量
3. 训练后把 BA merge 回 W0 实现零延迟推理
```

最核心代码是：

```python
self.linear.weight.requires_grad = False
self.lora_A = nn.Parameter(torch.empty(r, in_features))
self.lora_B = nn.Parameter(torch.empty(out_features, r))
lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
```

理解这些 shape、初始化和 merge 逻辑，就能从原理上掌握 LoRA，而不是只会调用 PEFT 库。
