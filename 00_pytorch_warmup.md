
# PyTorch 自定义算子：算子融合 Linear + ReLU

本仓库实现了一个自定义的 PyTorch Autograd 函数，将**线性层（全连接层 Linear）**和 **ReLU 激活函数**融合进同一个算子中。文档详细记录了前向传播与反向传播的数学梯度推导，以及 PyTorch 在底层架构设计上的核心考量。


## 1. 代码实现

```python
import torch
import torch.nn.functional as F

class LinearReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        """
        融合算子 Linear + ReLU 的前向传播
        
        参数:
            x (Tensor): 输入张量，形状为 (N, d_in)
            weight (Tensor): 权重矩阵，形状为 (d_out, d_in)
            bias (Tensor): 偏置向量，形状为 (d_out,)
        """
        # 线性变换: z = x * W^T + b
        z = F.linear(x, weight, bias)
        # 激活函数: y = max(0, z)
        y = F.relu(z)
        
        # 缓存 mask 和输入张量，用于反向传播
        # 将 bool 类型的掩码转换为 float，以确保反向传播时计算的高效与类型安全
        mask = (z > 0).float()
        ctx.save_for_backward(x, weight, mask)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        """
        使用链式法则计算梯度的反向传播
        
        参数:
            grad_output (Tensor): 损失函数对输出 y 的梯度，形状为 (N, d_out)
        """
        x, weight, mask = ctx.saved_tensors
        
        # 1. 计算对激活前状态（z）的梯度
        # 与 ReLU 的导数掩码（mask）进行按元素相乘（Element-wise）
        grad_z = grad_output * mask
        
        # 2. 利用矩阵维度匹配法，计算对输入和参数的梯度
        grad_x = grad_z @ weight               # 形状: (N, d_out) @ (d_out, d_in) -> (N, d_in)
        grad_weight = grad_z.T @ x             # 形状: (d_out, N) @ (N, d_in) -> (d_out, d_in)
        grad_bias = grad_z.sum(dim=0)          # 在 batch 维度上累加 -> 形状: (d_out,)
        
        return grad_x, grad_weight, grad_bias

```

---

## 2. 数学原理与梯度推导

### 前向传播公式

设 Batch 大小为 $N$，前向传播的数学表达式为：


$$z = x W^T + b$$

$$y = \text{ReLU}(z)$$

已知从上一层传回的梯度为`grad_output`（即 $\frac{\partial L}{\partial y}$），我们通过**链式法则**依次推导各个部分的梯度：

#### 1. 对 $z$ 的梯度 (`grad_z`)

因为 $y = \max(0, z)$，其导数是一个阶跃函数（即掩码 Mask）：


$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial y} \odot \mathbb{I}(z > 0)$$


其中 $\odot$ 表示按元素相乘（Element-wise Product）。

#### 2. 对输入 $x$ 的梯度 (`grad_x`)

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial x} = \frac{\partial L}{\partial z} W$$

* **维度对齐：** `(N, d_out) @ (d_out, d_in) -> (N, d_in)`

#### 3. 对权重 $W$ 的梯度 (`grad_weight`)

$$\frac{\partial L}{\partial W} = \left(\frac{\partial L}{\partial z}\right)^T x$$

* **维度对齐：** `(d_out, N) @ (N, d_in) -> (d_out, d_in)`

#### 4. 对偏置 $b$ 的梯度 (`grad_bias`)

由于偏置 $b$ 在前向传播中通过**广播机制**作用于整个 Batch 的每一个样本，因此在反向传播时，需要将整个 Batch 产生的梯度全部累加：


$$\frac{\partial L}{\partial b} = \sum_{i=1}^{N} \frac{\partial L}{\partial z_i}$$

* **维度对齐：** 将 `(N, d_out)` 的张量沿着 `dim=0`（Batch 维度）求和，得到 `(d_out,)`。

---

## 3. PyTorch 设计洞察：为什么权重形状是 `(d_out, d_in)`？

在 PyTorch 的 `nn.Linear` 中，权重张量的默认存储形状是 **`(d_out, d_in)`**，而不是直觉上的 `(d_in, d_out)`。这一设计背后包含了以下关键的硬件与架构优化考量：

1. **内存局部性（行连续存储）：** 采用 `(d_out, d_in)` 形状时，负责计算第 $i$ 个输出神经元的所有输入权重在内存中是**行连续存储**的（即 `W[i, :]`）。当硬件（CPU/GPU）读取或对单个通道进行裁剪（Pruning）和量化（Quantization）时，连续内存能极大提升 L1/L2 缓存命中率。
2. **与卷积层（CNN）保持架构高度统一：** 在 PyTorch 中，`nn.Conv2d` 的权重布局为 `(out_channels, in_channels, K, K)`。全连接层设计为 `(out_features, in_features)` 可以确保**输出维度永远位于第 0 维（Axis 0）**。这种一致性极大地简化了底层算子融合以及动静态图转换的逻辑。
3. **硬件加速器友好度：** 底层的矩阵乘法库（如 NVIDIA 的 cuBLAS）能够完美利用这种内存布局。在执行高性能通用矩阵乘法（GEMM）时，该布局更有利于将矩阵乘法与后续的激活函数（如 ReLU）融合进同一个 GPU Kernel 中执行，从而减少显存带宽消耗。

---

## 4. 关键细节：为什么在前向传播中将 Mask 转换为 Float？

代码中有一行关键的操作：`mask = (z > 0).float()`。将布尔型掩码提前转换为浮点型，主要有两点考虑：

* **类型安全：** 反向传播中 `grad_z = grad_output * mask` 需要进行乘法运算。`grad_output` 是浮点型，如果 `mask` 保持 `bool` 类型，在底层算子执行时会引发隐式类型转换或报错。
* **避免运行期开销：** 虽然 `bool` 类型占用的显存更小，但如果在 `backward` 阶段才临时调用 `.float()`，会在反向传播这个计算密集型的过程中带来额外的内存分配与类型转换开销（Type Casting Overhead）。在前向阶段提前转换，能够确保反向传播以最纯粹、最高效的矩阵方式运行。

---

## 5. 如何使用

你可以通过 `.apply` 方法轻松地将该自定义算子嵌入到你的神经网络模型中：

```python
# 模拟输入数据与参数
x = torch.randn(32, 10, requires_grad=True)
weight = torch.randn(5, 10, requires_grad=True)
bias = torch.randn(5, requires_grad=True)

# 前向传播
output = LinearReLUFunction.apply(x, weight, bias)

# 反向传播
loss = output.sum()
loss.backward()

# 打印梯度形状
print(x.grad.shape)       # torch.Size([32, 10])
print(weight.grad.shape)  # torch.Size([5, 10])
print(bias.grad.shape)    # torch.Size([5])

```

```

```
