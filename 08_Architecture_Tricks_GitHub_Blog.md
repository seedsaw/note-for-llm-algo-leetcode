# Architecture Tricks 知识点整理：Qwen 权重绑定与 Gemma RMSNorm 的工程实现

在前面的 LLaMA3 Block 中，我们已经搭建过现代 Decoder-only LLM 的主干结构：RMSNorm、RoPE、GQA、SwiGLU、Pre-Norm 和残差连接。

但真实开源模型并不总是完全照搬 LLaMA 的细节。不同模型家族会在一些局部模块上做工程取舍，例如：

- Qwen / GPT-2 风格的 Tie Word Embeddings，也就是词嵌入和 LM Head 权重绑定。
- Gemma 风格的 RMSNorm 参数化方式，也就是使用 `(1 + weight)` 作为缩放因子。
- 大量线性层和归一化层去掉 bias，减少参数和计算分支。

这些改动看起来都很小，但在大模型里非常重要。因为词表、hidden size、层数都很大，一个小小的架构差异就可能影响数十亿参数、显存占用、训练稳定性和 checkpoint 加载方式。

这篇笔记整理两个常见 Architecture Tricks：

- Qwen / GPT-2 风格的权重绑定是什么
- 为什么 Embedding 和 LM Head 可以共享权重
- 权重绑定能节省多少参数
- PyTorch 中如何实现物理内存级别的权重共享
- Gemma RMSNorm 和标准 RMSNorm 有什么区别
- 为什么 Gemma 使用 `1 + weight`
- 为什么 RMSNorm 内部通常要临时转 FP32
- 如何测试权重绑定和 Gemma RMSNorm 是否实现正确
- 这些技巧在训练、优化器、checkpoint 和工程部署中有哪些坑

## 1. 两个技巧在解决什么问题

这篇 notebook 涉及两个架构技巧：

```text
Trick 1: Gemma RMSNorm 的 1 + weight 缩放
Trick 2: Qwen / GPT-2 风格的 Tie Word Embeddings
```

它们关注的问题不同。

Gemma RMSNorm 关注的是训练稳定性：

```text
标准 RMSNorm:
y = norm(x) * weight

Gemma RMSNorm:
y = norm(x) * (1 + weight)
```

权重绑定关注的是参数量和输入输出语义共享：

```text
不绑定:
embed_tokens.weight 和 lm_head.weight 是两份独立参数

绑定:
embed_tokens.weight 和 lm_head.weight 指向同一块参数
```

一句话概括：

```text
Gemma RMSNorm 改的是归一化层的参数化方式。
Tie Word Embeddings 改的是输入词表矩阵和输出词表矩阵的共享方式。
```

## 2. Trick 1：Gemma RMSNorm 的 `1 + weight`

### 2.1 标准 RMSNorm

RMSNorm 的核心公式是：

$$
\mathrm{RMS}(x) = \sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \epsilon}
$$

$$
y = \frac{x}{\mathrm{RMS}(x)} \cdot w
$$

其中：

- $x$ 是输入 hidden states。
- $d$ 是 hidden size。
- $\epsilon$ 是防止除零的小常数。
- $w$ 是可学习缩放参数。

和 LayerNorm 相比，RMSNorm 不会减去均值，只按均方根做缩放：

```text
LayerNorm:  减均值 + 除标准差 + scale + bias
RMSNorm:    不减均值 + 除 RMS + scale
```

因此 RMSNorm 更简单，也更常见于 LLaMA、Gemma、Qwen 等 LLM 架构中。

### 2.2 Gemma 的改动

Gemma 风格的 RMSNorm 把缩放因子从 `weight` 改成了：

```python
1 + weight
```

也就是：

$$
y = \frac{x}{\mathrm{RMS}(x)} \cdot (1 + w)
$$

对应代码：

```python
output = x_norm * (1 + self.weight)
```

这个改动的关键在初始化。

标准 RMSNorm 常见写法是：

```python
self.weight = nn.Parameter(torch.ones(hidden_size))
```

这样初始化时缩放因子就是 1。

Gemma 写法则是：

```python
self.weight = nn.Parameter(torch.zeros(hidden_size))
```

然后 forward 中使用：

```python
x_norm * (1 + self.weight)
```

初始化时：

```text
weight = 0
1 + weight = 1
```

所以它和标准 RMSNorm 的初始功能一样，都是纯归一化输出。但参数本身以 0 为中心学习，而不是以 1 为中心学习。

### 2.3 为什么 `1 + weight` 有意义

如果一个归一化层初始化后就把输出整体缩得很小，会影响训练早期的信号传播。

Gemma 的设计让初始化时：

```text
y = norm(x) * (1 + 0)
  = norm(x)
```

也就是说，归一化层一开始不会额外放大或缩小 hidden states，只做 RMS 归一化。训练过程中，`weight` 再逐渐学习每个 hidden dimension 的缩放偏移。

这样做的好处：

- 初始化行为清晰，缩放因子从 1 开始。
- 参数值从 0 附近学习，优化器状态更自然。
- 对低精度训练更友好，因为归一化内部可以 FP32 计算，输出再转回原 dtype。

需要注意，这里的 `weight` 不是传统意义上的 bias。它并不是加到输出上，而是作为乘法缩放项的一部分：

```text
不是: y = norm(x) * weight + bias
而是: y = norm(x) * (1 + weight)
```

## 3. Gemma RMSNorm 完整实现

下面是 notebook 中对应的参考实现：

```python
import torch
import torch.nn as nn


class GemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        x_norm = x_f32 * torch.rsqrt(variance + self.eps)

        output = x_norm * (1 + self.weight)

        return output.type_as(x)
```

逐行解释：

```python
x_f32 = x.float()
```

RMSNorm 涉及平方、均值、倒数平方根。FP16/BF16 下直接计算容易产生数值误差，所以通常先临时转成 FP32。

```python
variance = x_f32.pow(2).mean(-1, keepdim=True)
```

沿 hidden dimension 计算均方值。输入如果是 `[batch_size, seq_len, hidden_size]`，那么输出 shape 是：

```text
variance: [batch_size, seq_len, 1]
```

```python
x_norm = x_f32 * torch.rsqrt(variance + self.eps)
```

`torch.rsqrt` 计算的是 reciprocal square root：

```text
rsqrt(z) = 1 / sqrt(z)
```

所以这一行等价于：

```python
x_norm = x_f32 / torch.sqrt(variance + self.eps)
```

但 `rsqrt` 在 GPU 上更常用，也更贴近深度学习框架里的归一化实现。

```python
output = x_norm * (1 + self.weight)
```

这是 Gemma 的关键差异。`self.weight` shape 是 `[hidden_size]`，会自动 broadcast 到 `[batch_size, seq_len, hidden_size]`。

```python
return output.type_as(x)
```

归一化内部使用 FP32 计算，但输出要转回输入 dtype，方便和模型其他模块保持一致。例如输入是 BF16，输出也应该是 BF16。

## 4. Trick 2：Tie Word Embeddings

语言模型通常有两个和词表相关的大矩阵。

第一个是输入端的 Token Embedding：

```python
self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
```

它把 token id 映射成 hidden vector：

```text
input_ids:      [batch_size, seq_len]
hidden_states:  [batch_size, seq_len, hidden_size]
```

权重 shape 是：

```text
embed_tokens.weight: [vocab_size, hidden_size]
```

第二个是输出端的 LM Head：

```python
self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
```

它把 hidden vector 映射回词表 logits：

```text
hidden_states: [batch_size, seq_len, hidden_size]
logits:        [batch_size, seq_len, vocab_size]
```

`nn.Linear(hidden_size, vocab_size)` 的权重 shape 是：

```text
lm_head.weight: [vocab_size, hidden_size]
```

可以看到：

```text
embed_tokens.weight: [vocab_size, hidden_size]
lm_head.weight:      [vocab_size, hidden_size]
```

两者 shape 完全一样，所以可以共享同一个参数。

## 5. 为什么 Embedding 和 LM Head 可以共享

从语义上看，Embedding 和 LM Head 是词表空间和 hidden 空间之间的两个方向。

Embedding 做的是：

```text
token id -> token vector
```

LM Head 做的是：

```text
hidden vector -> vocabulary logits
```

如果把 `embed_tokens.weight` 看成一个词表向量表：

```text
每一行 = 一个 token 的向量表示
```

那么 LM Head 的计算可以理解成：

```text
当前 hidden state 和每个 token 向量做相似度打分
```

也就是：

$$
\mathrm{logits} = h W_{embed}^{T}
$$

PyTorch 的 `nn.Linear(hidden_size, vocab_size, bias=False)` 内部正好使用：

```text
output = input @ weight.T
```

如果：

```python
lm_head.weight = embed_tokens.weight
```

那么：

```text
logits = hidden_states @ embed_tokens.weight.T
```

这就是权重绑定的数学含义。

## 6. 权重绑定能省多少参数

假设：

```text
vocab_size = 150000
hidden_size = 4096
```

一个词表矩阵参数量是：

```text
150000 * 4096 = 614400000
```

也就是约 6.14 亿参数。

如果使用 FP32，单个参数 4 bytes：

```text
614400000 * 4 bytes = 2457600000 bytes
```

大约是：

```text
2.46 GB
```

如果不绑定，Embedding 和 LM Head 各一份：

```text
参数量约 12.29 亿
FP32 权重约 4.92 GB
```

如果绑定：

```text
只保留一份词表矩阵
节省约 6.14 亿参数
FP32 下节省约 2.46 GB 权重显存
```

实际训练中还要考虑梯度、优化器状态、混合精度 master weight 等额外开销，所以权重绑定节省的训练显存可能比单纯权重大小更明显。

## 7. QwenTieEmbeddings 完整实现

下面是 notebook 中的参考实现：

```python
class QwenTieEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        self.lm_head.weight = self.embed_tokens.weight

    def forward_embed(self, input_ids):
        return self.embed_tokens(input_ids)

    def forward_lm_head(self, hidden_states):
        return self.lm_head(hidden_states)
```

关键只有一行：

```python
self.lm_head.weight = self.embed_tokens.weight
```

这不是复制数值。

错误理解：

```text
把 embed_tokens.weight 当前的值复制一份给 lm_head.weight
```

真实含义：

```text
让 lm_head.weight 这个 Parameter 引用指向 embed_tokens.weight 这个 Parameter
```

因此它们共享同一个 `nn.Parameter` 对象，也共享同一块底层存储。

可以用 `data_ptr()` 验证：

```python
ptr_embed = model.embed_tokens.weight.data_ptr()
ptr_head = model.lm_head.weight.data_ptr()

assert ptr_embed == ptr_head
```

如果两个指针相同，说明是物理内存级别的共享。

## 8. 为什么不能用 `.data.copy_`

有些人会写成：

```python
self.lm_head.weight.data.copy_(self.embed_tokens.weight.data)
```

这只是在初始化时复制数值，不是权重绑定。

复制后：

```text
embed_tokens.weight 和 lm_head.weight 仍然是两块不同内存
```

后续训练中：

```text
更新 embed_tokens.weight 不会自动更新 lm_head.weight
更新 lm_head.weight 也不会自动更新 embed_tokens.weight
```

正确写法是直接替换 Parameter 引用：

```python
self.lm_head.weight = self.embed_tokens.weight
```

这样优化器看到的是同一个参数，反向传播时梯度也会累积到同一份权重上。

## 9. 单元测试

notebook 中的测试分成两部分：

- 检查 Gemma RMSNorm 初始化时是否等价于纯 RMS 归一化。
- 检查 Qwen Tie Embeddings 是否真的共享物理内存。

完整测试代码如下：

```python
def test_tricks():
    hidden_size = 64
    vocab_size = 1000

    # 1. 测试 Gemma RMSNorm。
    gemma_norm = GemmaRMSNorm(hidden_size)
    x = torch.randn(2, 10, hidden_size)
    out = gemma_norm(x)

    variance = x.float().pow(2).mean(-1, keepdim=True)
    expected = (x.float() * torch.rsqrt(variance + 1e-6)).to(x.dtype)

    assert torch.allclose(out, expected, atol=1e-4), (
        "Gemma 的 1 + weight 缩放机制实现错误"
    )
    print("Gemma RMSNorm test passed.")

    # 2. 测试 Qwen 权重绑定。
    qwen_model = QwenTieEmbeddings(vocab_size, hidden_size)

    ptr_embed = qwen_model.embed_tokens.weight.data_ptr()
    ptr_head = qwen_model.lm_head.weight.data_ptr()
    assert ptr_embed == ptr_head, "权重未在物理内存级别绑定"

    qwen_model.embed_tokens.weight.data += 1.0

    assert (
        qwen_model.lm_head.weight.data[0, 0]
        == qwen_model.embed_tokens.weight.data[0, 0]
    ), "权重更新未同步"

    print("Tie embeddings test passed.")


test_tricks()
```

如果实现正确，会看到：

```text
Gemma RMSNorm test passed.
Tie embeddings test passed.
```

## 10. 最小可运行代码

下面是一份可以直接运行的完整代码：

```python
import torch
import torch.nn as nn


class GemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        x_norm = x_f32 * torch.rsqrt(variance + self.eps)
        output = x_norm * (1 + self.weight)
        return output.type_as(x)


class QwenTieEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def forward_embed(self, input_ids):
        return self.embed_tokens(input_ids)

    def forward_lm_head(self, hidden_states):
        return self.lm_head(hidden_states)


def test_tricks():
    hidden_size = 64
    vocab_size = 1000

    gemma_norm = GemmaRMSNorm(hidden_size)
    x = torch.randn(2, 10, hidden_size)
    out = gemma_norm(x)

    variance = x.float().pow(2).mean(-1, keepdim=True)
    expected = (x.float() * torch.rsqrt(variance + 1e-6)).to(x.dtype)
    assert torch.allclose(out, expected, atol=1e-4)

    qwen_model = QwenTieEmbeddings(vocab_size, hidden_size)
    assert (
        qwen_model.embed_tokens.weight.data_ptr()
        == qwen_model.lm_head.weight.data_ptr()
    )

    qwen_model.embed_tokens.weight.data += 1.0
    assert (
        qwen_model.lm_head.weight.data[0, 0]
        == qwen_model.embed_tokens.weight.data[0, 0]
    )

    print("All architecture trick tests passed.")


if __name__ == "__main__":
    test_tricks()
```

## 11. 和 LLaMA 风格实现的对比

### 11.1 RMSNorm 参数化对比

LLaMA 风格 RMSNorm 常见写法：

```python
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        x_norm = x_f32 * torch.rsqrt(variance + self.eps)
        return (x_norm * self.weight).type_as(x)
```

Gemma 风格 RMSNorm：

```python
class GemmaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        x_norm = x_f32 * torch.rsqrt(variance + self.eps)
        return (x_norm * (1 + self.weight)).type_as(x)
```

两者初始化时的有效缩放因子都是 1：

```text
LLaMA: weight = 1
Gemma: 1 + weight = 1 + 0 = 1
```

区别是参数本身的中心不同：

```text
LLaMA 参数围绕 1 学习
Gemma 参数围绕 0 学习，但 forward 时加 1
```

### 11.2 LM Head 对比

不绑定权重时：

```python
self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
```

绑定权重时：

```python
self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
self.lm_head.weight = self.embed_tokens.weight
```

从 forward 看，两者使用方式一样：

```python
hidden_states = self.embed_tokens(input_ids)
logits = self.lm_head(hidden_states)
```

区别只在参数是否共享。

## 12. Bias 的取舍

现代 LLM 中，很多线性层会设置：

```python
bias=False
```

例如：

```python
nn.Linear(hidden_size, vocab_size, bias=False)
```

原因包括：

- 减少参数量。
- 降低一点点显存和带宽开销。
- 和归一化层配合时，bias 的收益通常不明显。
- 简化一些 fused kernel 和张量并行实现。

对于 LM Head，权重绑定时通常也不使用 bias：

```python
self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
```

如果 LM Head 有 bias：

```python
self.lm_head = nn.Linear(hidden_size, vocab_size, bias=True)
```

那么即使绑定了 weight，bias 仍然是额外的一份独立参数：

```text
lm_head.weight 共享
lm_head.bias   不共享
```

这不一定错误，但会让结构和很多 LLM 实现不一致。

## 13. 工程注意事项

### 13.1 绑定要在优化器创建前完成

推荐顺序：

```python
model = QwenTieEmbeddings(vocab_size, hidden_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

不要先创建优化器，再绑定权重：

```python
model = QwenTieEmbeddingsWithoutTying(vocab_size, hidden_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
model.lm_head.weight = model.embed_tokens.weight
```

原因是优化器在创建时会捕获参数列表。如果绑定发生在优化器创建之后，优化器内部可能还保留旧参数，导致训练行为混乱。

### 13.2 保存和加载 checkpoint 后要确认绑定关系

如果模型类的 `__init__` 中包含：

```python
self.lm_head.weight = self.embed_tokens.weight
```

那么每次实例化模型都会重新建立绑定关系，加载 checkpoint 后通常没有问题。

但如果你手动改 `state_dict` 或在外部临时绑定，就要小心：

```python
model.load_state_dict(state_dict)
assert model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
```

对权重绑定模型，加载后做一次 `data_ptr()` 检查是很有价值的。

### 13.3 参数统计可能重复计算

如果直接统计：

```python
sum(p.numel() for p in model.parameters())
```

通常 PyTorch 会按参数对象遍历。直接共享同一个 `nn.Parameter` 时，这个参数一般只出现一次。

但如果你的模型结构、包装器或导出工具把同一块 storage 展示成多个名字，参数统计可能重复。更稳妥的唯一参数统计方式是按 `data_ptr()` 去重：

```python
seen = set()
num_params = 0

for p in model.parameters():
    ptr = p.data_ptr()
    if ptr not in seen:
        seen.add(ptr)
        num_params += p.numel()
```

### 13.4 Tensor Parallel 下的权重绑定更复杂

单机单卡里：

```python
self.lm_head.weight = self.embed_tokens.weight
```

很直接。

但在张量并行中，Embedding 和 LM Head 可能按不同维度切分：

```text
Embedding 可能按 vocab dimension 分片
LM Head 也可能按 vocab dimension 分片
```

如果切分策略一致，可以绑定对应 shard；如果不一致，就需要额外通信或不能简单共享同一个 Parameter。

因此分布式训练框架里经常会有专门的 `tie_weights()` 或 `post_init()` 逻辑。

### 13.5 dtype 和 device 要保持一致

Gemma RMSNorm 中临时创建的张量来自输入计算，不需要单独指定 device。但如果你自己创建参数或 buffer，要确保和输入在同一设备。

权重绑定时也要保证两个模块本来就在同一模型里，由 `model.to(device)` 统一迁移。一般推荐：

```python
model = QwenTieEmbeddings(vocab_size, hidden_size)
model = model.to(device)
```

不要手动把 `embed_tokens` 和 `lm_head` 分别迁移到不同设备后再绑定。

## 14. 常见错误

### 14.1 Gemma RMSNorm 忘记加 1

错误写法：

```python
output = x_norm * self.weight
```

因为 `self.weight` 初始化为 0，所以初始化时输出会接近全 0：

```text
weight = 0
output = x_norm * 0 = 0
```

这会破坏信号传播。

正确写法：

```python
output = x_norm * (1 + self.weight)
```

### 14.2 Gemma RMSNorm 忘记转回原 dtype

错误写法：

```python
return output
```

如果输入是 BF16，输出却变成 FP32，后续模块可能出现 dtype 不一致或额外显存开销。

正确写法：

```python
return output.type_as(x)
```

### 14.3 用数值复制代替权重绑定

错误写法：

```python
self.lm_head.weight.data = self.embed_tokens.weight.data
```

或者：

```python
self.lm_head.weight.data.copy_(self.embed_tokens.weight.data)
```

这些写法要么绕开 autograd 的 Parameter 管理，要么只是复制数值。推荐使用：

```python
self.lm_head.weight = self.embed_tokens.weight
```

### 14.4 在 hidden size 不匹配时强行绑定

权重绑定要求：

```text
embed_tokens.weight shape == lm_head.weight shape
```

也就是：

```text
[vocab_size, hidden_size] == [vocab_size, hidden_size]
```

如果模型中间有额外投影，例如 hidden state 输出维度不是 embedding 维度，就不能直接绑定，需要先通过投影层对齐维度。

### 14.5 忘记用 `bias=False`

权重绑定本身只绑定 weight，不绑定 bias。

LLM 的 LM Head 通常写成：

```python
self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
```

如果不小心使用默认的 `bias=True`，会多出一个 `[vocab_size]` 的 bias 参数，并且和常见 LLM 配置不一致。

## 15. 面试高频问答

### 15.1 什么是 Tie Word Embeddings？

Tie Word Embeddings 是让输入端的 token embedding 矩阵和输出端的 LM Head 权重矩阵共享同一份参数。

在 PyTorch 中典型写法是：

```python
self.lm_head.weight = self.embed_tokens.weight
```

这样两个模块不是数值相同，而是指向同一块参数。

### 15.2 权重绑定有什么好处？

主要有两个好处：

- 减少参数量，尤其词表很大时节省明显。
- 输入 token 表示和输出 token 分类器共享语义空间，Embedding 可以从输出预测任务中获得更直接的梯度信号。

### 15.3 权重绑定有什么代价？

代价是表达能力可能受限。

不绑定时，模型可以分别学习：

```text
输入 token 表示空间
输出 token 分类空间
```

绑定后，两者必须使用同一个矩阵。对于一些大模型，解绑可能带来更强的容量表达，但会增加参数量。

### 15.4 如何验证两个权重真的绑定了？

用 `data_ptr()` 检查底层存储地址：

```python
assert model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
```

也可以修改其中一个权重，检查另一个是否同步变化。但测试时不要在真实训练代码中滥用 `.data`，它主要适合做简单验证。

### 15.5 Gemma RMSNorm 和普通 RMSNorm 的区别是什么？

普通 RMSNorm 常见写法：

```python
output = x_norm * weight
```

Gemma RMSNorm 写法：

```python
output = x_norm * (1 + weight)
```

并且 Gemma 的 `weight` 初始化为 0。这样初始化时有效缩放因子仍然是 1。

### 15.6 为什么 RMSNorm 内部要转 FP32？

因为 RMSNorm 需要平方、求均值、开方倒数。低精度下这些操作更容易出现数值误差。临时转 FP32 可以提高稳定性，最后再转回输入 dtype 以保持模型整体精度设置。

### 15.7 `1 + weight` 是不是 bias？

不是。

bias 是加法项：

```text
y = x * scale + bias
```

Gemma RMSNorm 的 `1 + weight` 是乘法缩放因子：

```text
y = norm(x) * (1 + weight)
```

它改变的是 scale 的参数化方式，而不是增加输出 bias。

## 16. 总结

这篇 notebook 的两个架构技巧都很小，但很典型。

Gemma RMSNorm 的核心是：

```python
self.weight = nn.Parameter(torch.zeros(hidden_size))
output = x_norm * (1 + self.weight)
```

它让归一化层初始化时等价于纯 RMSNorm 输出，同时让可学习参数从 0 附近开始优化。

Qwen / GPT-2 风格权重绑定的核心是：

```python
self.lm_head.weight = self.embed_tokens.weight
```

它让输入词嵌入矩阵和输出 LM Head 矩阵共享同一份参数，从而节省大量词表参数，并让输入输出词表空间保持一致。

工程实现时重点检查：

- Gemma RMSNorm 是否使用 `(1 + weight)`。
- RMSNorm 是否内部 FP32 计算、输出转回原 dtype。
- LM Head 是否使用 `bias=False`。
- 权重绑定是否直接共享 `nn.Parameter`，而不是复制数值。
- 绑定是否发生在 optimizer 创建前。
- checkpoint 加载后 `data_ptr()` 是否仍然一致。

掌握这些细节后，就能更准确地阅读 Qwen、Gemma、LLaMA 等模型的源码差异，也能在自己实现小型 LLM 时做出更清晰的架构取舍。
