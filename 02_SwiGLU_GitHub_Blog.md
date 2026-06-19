# SwiGLU 知识点整理：门控机制、8/3 维度推导与工业级实现

SwiGLU 是 LLaMA、Qwen、Mistral、PaLM 等主流大语言模型中常见的 MLP 激活结构。它不是一个单独替换 ReLU/GELU 的普通激活函数，而是把“激活函数”和“门控机制”结合起来，用额外的一路线性投影控制信息流。

这篇文章整理 SwiGLU 在实现和面试中最容易被问到的几个点：

- GLU / SwiGLU 的核心思想是什么
- SwiGLU 和普通 Transformer MLP 有什么区别
- 为什么 LLaMA 的中间层维度常写成 `8 / 3 * hidden_size`
- 为什么还要把中间层维度向上对齐到 `multiple_of`
- 为什么工业实现会融合 `gate_proj` 和 `up_proj`
- PyTorch 中如何写一个标准的融合版 SwiGLU MLP
- 复习时应该重点检查哪些实现细节

## 1. 从标准 MLP 到门控 MLP

Transformer 中的标准 FFN / MLP 通常是两层线性变换，中间接一个非线性激活函数。给定输入：

$$
x \in \mathbb{R}^{d}
$$

标准 MLP 可以写成：

$$
\mathrm{MLP}(x)=W_{down}(\sigma(W_{up}x))
$$

其中：

- `W_up`：把 hidden size 从 `d` 升到中间维度 `h`。
- `sigma`：激活函数，例如 ReLU 或 GELU。
- `W_down`：把中间维度 `h` 降回 `d`。

在 GPT-2 这类传统 Transformer 中，中间层维度通常取：

$$
h = 4d
$$

因此标准 MLP 的结构可以理解为：

```text
x -> Linear(d, 4d) -> GELU/ReLU -> Linear(4d, d)
```

这个结构简单有效，但它只有一路特征变换。GLU 系列的核心改动是：让 MLP 不只产生一组中间特征，而是产生两组中间特征，其中一路作为“内容”，另一路作为“门控”。

## 2. GLU 的核心思想

GLU, Gated Linear Unit, 中文通常叫门控线性单元。它的思想类似 LSTM 里的门：模型不仅要计算“候选信息”，还要计算“哪些信息应该通过”。

给定输入 `x`，GLU 会做两条投影：

$$
a = xW_{up}
$$

$$
g = xW_{gate}
$$

然后对门控分支做激活，并和内容分支逐元素相乘：

$$
\mathrm{GLU}(x) = a \odot \sigma(g)
$$

最后再通过降维矩阵回到原始 hidden size：

$$
\mathrm{Output}(x)=W_{down}(a \odot \sigma(g))
$$

其中 `odot` 表示逐元素乘法，也就是 Hadamard product。

从工程角度看，GLU 相比标准 MLP 多了一路线性投影：

```text
标准 MLP:
x -> up_proj -> activation -> down_proj

GLU:
x -> up_proj   ---- content branch ----*
x -> gate_proj -- activation branch --*
                                     elementwise multiply -> down_proj
```

这一路 gate 让模型可以更细粒度地控制每个 token、每个 channel 的信息是否应该通过。它提升了表达能力，但代价是升维阶段从一个投影矩阵变成了两个投影矩阵。

## 3. 什么是 SwiGLU

SwiGLU 可以看作是 GLU 的一个具体变体：把 GLU 中门控分支的激活函数换成 Swish。

Swish 的公式是：

$$
\mathrm{Swish}(x)=x \cdot \mathrm{Sigmoid}(\beta x)
$$

当 $\beta=1$ 时，Swish 就是 PyTorch 中常用的 `SiLU`：

```python
torch.nn.functional.silu(x)
```

因此 SwiGLU 的公式可以写成：

$$
\mathrm{SwiGLU}(x)=W_{down}(\mathrm{SiLU}(xW_{gate}) \odot xW_{up})
$$

对应到 LLaMA / HuggingFace 风格的命名，就是：

```python
down_proj(silu(gate_proj(x)) * up_proj(x))
```

需要注意：有些资料会把 `gate` 和 `up` 的名字顺序写得不完全一样，但关键不在命名，而在这三个动作：

1. 输入 `x` 同时经过两路线性投影。
2. 其中一路经过 `SiLU` 作为门控分支。
3. 两路结果逐元素相乘后，再通过 `down_proj` 降回 hidden size。

## 4. 为什么中间层维度是 8/3 hidden size

这是 SwiGLU 最经典的面试推导题。

问题通常会这样问：

> 标准 Transformer MLP 的中间层维度通常是 `4 * hidden_size`。为什么 LLaMA 使用 SwiGLU 后，中间层维度变成了大约 `8 / 3 * hidden_size`？

答案是：为了让 SwiGLU MLP 的参数量和标准 MLP 大致对齐。

### 4.1 标准 MLP 的参数量

设输入 hidden size 为：

$$
d
$$

标准 MLP 中间层维度为：

$$
h=4d
$$

标准 MLP 有两个矩阵：

- `up_proj`: $d \to 4d$
- `down_proj`: $4d \to d$

忽略 bias 时，参数量为：

$$
d \cdot 4d + 4d \cdot d = 8d^2
$$

也就是：

$$
\mathrm{Params}_{MLP}=8d^2
$$

### 4.2 SwiGLU MLP 的参数量

SwiGLU 有三个矩阵：

- `gate_proj`: $d \to h_{swiglu}$
- `up_proj`: $d \to h_{swiglu}$
- `down_proj`: $h_{swiglu} \to d$

忽略 bias 时，参数量为：

$$
d \cdot h_{swiglu} + d \cdot h_{swiglu} + h_{swiglu} \cdot d
$$

即：

$$
\mathrm{Params}_{SwiGLU}=3dh_{swiglu}
$$

### 4.3 参数量对齐

为了让 SwiGLU 的参数量和标准 MLP 大致一致，令：

$$
3dh_{swiglu}=8d^2
$$

两边同时除以 $3d$，得到：

$$
h_{swiglu}=\frac{8}{3}d
$$

这就是 LLaMA 源码和很多模型配置中会看到 `int(8 * hidden_size / 3)` 的原因。

一句话总结：

```text
标准 MLP: 2 个矩阵，中间层 4d，总参数量 8d^2
SwiGLU:   3 个矩阵，为保持参数量接近，中间层要缩到 8/3 d
```

SwiGLU 不是简单地在 `4d` 的基础上再多加一个 gate 矩阵。那样参数量会明显变大。`8/3 d` 的设计是在提升表达能力的同时，让整体参数量和计算量尽量接近原来的标准 MLP。

## 5. 为什么还要对齐到 multiple_of

理论推导给出的中间层维度是：

$$
\frac{8}{3}d
$$

但真实模型通常不会直接使用这个小数结果，而是会先取整数，再向上对齐到某个倍数。例如 LLaMA 中常见的 `multiple_of = 256`。

计算规则可以写成：

```python
intermediate_size = int(hidden_size * 8 / 3)
intermediate_size = multiple_of * ceil(intermediate_size / multiple_of)
```

用整数代码实现就是：

```python
aligned_size = ((intermediate_size + multiple_of - 1) // multiple_of) * multiple_of
```

对齐的原因主要有两个。

第一，硬件更喜欢规则的矩阵形状。GPU Tensor Core、矩阵乘法 kernel、内存访问等通常都对维度对齐更友好。中间层维度对齐到 64、128、256 这类倍数，往往能获得更稳定的吞吐。

第二，大模型训练经常使用张量并行。Tensor Parallelism 会把权重矩阵按维度切到多张 GPU 上。如果中间层维度不能被并行度整除，就会带来切分困难，严重时会直接报错。

例如 LLaMA-7B 的 hidden size 是：

```text
hidden_size = 4096
```

理论 SwiGLU 中间维度为：

$$
4096 \times \frac{8}{3}=10922.66...
$$

代码里先取整数得到：

```text
10922
```

再对齐到 `256` 的倍数：

```text
ceil(10922 / 256) = 43
43 * 256 = 11008
```

所以最终中间层维度是：

```text
intermediate_size = 11008
```

参数量也会因为向上对齐略大于理论值：

```text
标准 MLP:
2 * 4096 * 16384 = 134,217,728

LLaMA SwiGLU:
3 * 4096 * 11008 = 135,266,304
```

两者非常接近，但不完全相等。这种差异来自硬件和并行切分需要的向上取整。

## 6. 工业实现为什么要融合 gate_proj 和 up_proj

从数学上看，SwiGLU 可以直接写成三层：

```python
gate = gate_proj(x)
up = up_proj(x)
out = down_proj(F.silu(gate) * up)
```

这个写法清晰，但不是最优的工业实现。

问题在于 `gate_proj(x)` 和 `up_proj(x)` 使用完全相同的输入 `x`。如果分开执行两次线性层，GPU 需要从 HBM 全局显存中读取两次输入 `x`，也会触发两次矩阵乘法相关的调度开销。

大模型中的 MLP 是高频路径，每层、每个 token 都会调用。对于大 batch、大 sequence、大 hidden size 来说，重复读取输入会放大访存压力。

工业框架中常用的优化是矩阵融合：把 `gate_proj` 和 `up_proj` 合并成一个更大的线性层。

原始的两个矩阵：

```text
gate_proj: Linear(hidden_size, intermediate_size)
up_proj:   Linear(hidden_size, intermediate_size)
```

融合后变成：

```text
gate_up_proj: Linear(hidden_size, 2 * intermediate_size)
```

前向传播时只做一次大矩阵乘法：

```python
gate_up = gate_up_proj(x)
gate, up = torch.chunk(gate_up, 2, dim=-1)
out = down_proj(F.silu(gate) * up)
```

这样做的好处是：

- 输入 `x` 只需要读取一次。
- 两个小 GEMM 合成一个更大的 GEMM，通常更利于 GPU 吞吐。
- 减少 kernel launch 和调度开销。
- 更容易和后续激活、逐元素乘法做进一步 fused kernel 优化。

这类优化的本质是减少 memory bound。SwiGLU 的 `SiLU(gate) * up` 是逐元素操作，本身计算量不大，但会读写中间张量。工业实现会尽量把线性层、激活和乘法的中间结果组织得更紧凑，降低显存带宽压力。

## 7. 标准 PyTorch 实现

下面是一个适合学习和面试讲解的融合版 SwiGLU MLP。它保留了工业实现的关键结构：`gate_up_proj` 融合矩阵。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_intermediate_size(hidden_size: int, multiple_of: int = 256) -> int:
    """
    计算 LLaMA 风格 SwiGLU 的中间层维度。

    规则：
    1. 理论维度为 8/3 * hidden_size。
    2. 再向上取整到 multiple_of 的整数倍，方便硬件对齐和张量并行切分。
    """
    intermediate_size = int(hidden_size * 8 / 3)
    aligned_size = ((intermediate_size + multiple_of - 1) // multiple_of) * multiple_of
    return aligned_size


class SwiGLU_MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(
            hidden_size,
            2 * intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = torch.chunk(gate_up, 2, dim=-1)
        return self.down_proj(F.silu(gate) * up)
```

这个实现里有几个细节值得强调。

第一，`gate_up_proj` 的输出维度是 `2 * intermediate_size`。因为它一次性算出两组中间特征：一组给 gate 分支，一组给 up 分支。

第二，`torch.chunk(gate_up, 2, dim=-1)` 必须沿最后一维切分。假设输入形状是 `[batch_size, seq_len, hidden_size]`，那么融合投影后的形状是：

```text
[batch_size, seq_len, 2 * intermediate_size]
```

切分后得到：

```text
gate: [batch_size, seq_len, intermediate_size]
up:   [batch_size, seq_len, intermediate_size]
```

第三，`F.silu(gate) * up` 是逐元素乘法，两者形状必须完全一致。

第四，`down_proj` 的输入维度是 `intermediate_size`，输出维度是 `hidden_size`，这样整个 MLP 的输出形状和输入形状保持一致。

第五，通常不使用 bias。LLaMA 等模型的 MLP 线性层经常设置 `bias=False`，这样可以减少参数、简化融合实现，并和主流开源实现保持一致。

## 8. 快速测试

可以用下面的代码检查维度、参数量和前向传播形状。

```python
def test_swiglu():
    hidden_size = 4096
    aligned_size = calculate_intermediate_size(hidden_size, multiple_of=256)

    assert aligned_size == 11008, aligned_size
    print(f"hidden_size={hidden_size}, intermediate_size={aligned_size}")

    mlp = SwiGLU_MLP(hidden_size, aligned_size)

    assert hasattr(mlp, "gate_up_proj")

    total_params = sum(p.numel() for p in mlp.parameters())
    assert total_params == 135266304, total_params
    print(f"total_params={total_params}")

    x = torch.randn(2, 10, hidden_size)
    out = mlp(x)

    assert out.shape == x.shape
    print(out.shape)


test_swiglu()
```

预期输出大致是：

```text
hidden_size=4096, intermediate_size=11008
total_params=135266304
torch.Size([2, 10, 4096])
```

这里的参数量可以手动核对：

```text
gate_up_proj:
4096 * (2 * 11008) = 90,177,536

down_proj:
11008 * 4096 = 45,088,768

total:
90,177,536 + 45,088,768 = 135,266,304
```

也可以从“三个矩阵”的角度看：

```text
3 * 4096 * 11008 = 135,266,304
```

## 9. 常见实现错误

写 SwiGLU 时，最容易出错的是下面几类问题。

### 9.1 中间层维度仍然使用 4d

如果直接把 SwiGLU 的中间层维度设为 `4 * hidden_size`，参数量会变成：

$$
3d \cdot 4d = 12d^2
$$

这比标准 MLP 的 `8d^2` 多出 50%。这样就不是“在相近参数量下比较结构优劣”，而是直接增加了模型容量和计算量。

正确做法是先用：

```python
int(hidden_size * 8 / 3)
```

再对齐到 `multiple_of` 的倍数。

### 9.2 忘记对齐 intermediate_size

理论维度 `8/3 d` 只是数学推导。真实实现中如果不对齐，可能导致：

- Tensor Core 利用不充分。
- 张量并行切分不均匀。
- 某些分布式配置下权重维度不能被 GPU 数量整除。
- 与开源模型配置不一致，加载权重失败。

所以模型配置中的 `intermediate_size` 通常是对齐后的结果，而不是纯理论值。

### 9.3 `chunk` 的维度写错

融合投影输出形状通常是：

```text
[B, S, 2H]
```

必须沿最后一维切分：

```python
gate, up = torch.chunk(gate_up, 2, dim=-1)
```

如果沿 batch 或 sequence 维度切分，语义和形状都会错。

### 9.4 gate 和 up 的激活顺序混乱

SwiGLU 的核心是：

```python
F.silu(gate) * up
```

不是：

```python
F.silu(gate * up)
```

也不是：

```python
F.silu(up) * gate
```

虽然某些变体可能有不同约定，但 LLaMA 风格的常见写法是对 gate 分支做 `SiLU`，再和 up 分支逐元素相乘。

### 9.5 融合矩阵输出维度写成 intermediate_size

如果使用融合实现：

```python
self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
```

输出维度必须是 `2 * intermediate_size`。如果误写成 `intermediate_size`，后面 `chunk` 后每半只有 `intermediate_size / 2`，整个 MLP 的维度都会不匹配。

## 10. 面试速记

如果面试中被问到 SwiGLU，可以按下面顺序回答。

第一，SwiGLU 是 GLU 的变体。它把 MLP 拆成 gate 和 up 两条分支，gate 分支经过 `SiLU`，再和 up 分支逐元素相乘，最后通过 down projection 回到 hidden size。

第二，公式是：

$$
\mathrm{SwiGLU}(x)=W_{down}(\mathrm{SiLU}(xW_{gate}) \odot xW_{up})
$$

代码上就是：

```python
down_proj(F.silu(gate_proj(x)) * up_proj(x))
```

第三，`8/3` 来自参数量对齐。标准 MLP 中间层是 `4d`，两个矩阵，参数量是 `8d^2`。SwiGLU 有三个矩阵，参数量是 `3dh`。令 `3dh = 8d^2`，得到 `h = 8/3 d`。

第四，真实模型还会把 `8/3 d` 向上对齐到 `multiple_of` 的倍数，常见如 256。这样更利于 Tensor Core、矩阵乘法 kernel 和张量并行切分。

第五，工业实现通常融合 `gate_proj` 和 `up_proj`，把两个矩阵合成一个 `gate_up_proj`，输出 `2 * intermediate_size`，再用 `torch.chunk` 切成 gate 和 up。这样输入只读一次，能减少访存和 kernel launch 开销。

## 11. 实现检查清单

写完 SwiGLU MLP 后，可以用下面的清单快速检查：

- 是否使用 `intermediate_size = int(hidden_size * 8 / 3)`。
- 是否把 `intermediate_size` 向上对齐到 `multiple_of` 的倍数。
- 融合版 `gate_up_proj` 输出维度是否是 `2 * intermediate_size`。
- `down_proj` 输入维度是否是 `intermediate_size`，输出维度是否是 `hidden_size`。
- 线性层是否和目标模型保持一致，例如 `bias=False`。
- `torch.chunk(..., 2, dim=-1)` 是否沿最后一维切分。
- 是否对 gate 分支使用 `F.silu(gate)`。
- 是否执行逐元素乘法 `F.silu(gate) * up`。
- 输出 shape 是否和输入 shape 一致。
- 参数量是否约等于标准 `4d` MLP 的参数量。

一个正确的最小融合版实现如下：

```python
class SwiGLU_MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = torch.chunk(gate_up, 2, dim=-1)
        return self.down_proj(F.silu(gate) * up)
```

## 12. 总结

SwiGLU 的本质不是“换了一个激活函数”这么简单，而是把 MLP 改成了带门控的信息选择结构。它通过 `SiLU(gate) * up` 让模型可以动态调节每个 hidden channel 的信息流，比普通的 ReLU/GELU MLP 表达能力更强。

`8/3 hidden_size` 是理解 SwiGLU 的关键。标准 MLP 使用两个矩阵和 `4d` 中间层，参数量是 `8d^2`；SwiGLU 使用三个矩阵，为了保持参数量接近，需要把中间层降到 `8/3 d`。真实模型中还会把这个维度向上对齐到 `multiple_of` 的倍数，以适配硬件和张量并行。

工程实现时，重点不是只把公式写对，还要理解为什么要融合 `gate_proj` 和 `up_proj`。融合后的 `gate_up_proj` 可以用一次矩阵乘法同时得到 gate 和 up 两个分支，减少对输入张量的重复读取，降低访存压力。这也是 SwiGLU 从数学公式走向大模型工业实现时最重要的优化之一。

相关阅读：如果想继续理解如何进一步用 Triton 融合 `SiLU`、逐元素乘法和矩阵计算，可以看 `../03_CUDA_and_Triton_Kernels/02_Triton_Fused_SwiGLU.ipynb`。
