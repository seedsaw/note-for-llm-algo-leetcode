# SFT Training Loop 知识点整理：数据构造、Prompt Masking 与 Shift Logits

SFT, Supervised Fine-Tuning, 中文通常叫监督微调，是把预训练语言模型训练成“会按照指令回答问题”的关键阶段。

如果说预训练让模型学会通用语言建模，那么 SFT 的目标就是让模型学会这种格式：

```text
用户问题 / 指令 -> 助手回答
```

这篇笔记整理 SFT 训练循环中最容易写错、也最常被面试追问的核心细节：

- SFT 和预训练在 loss 目标上有什么区别
- 为什么 prompt 部分通常不应该计算 loss
- `labels` 为什么要把 prompt 和 padding 位置设成 `-100`
- `CrossEntropyLoss(ignore_index=-100)` 到底忽略了什么
- 自回归语言模型为什么要 shift logits 和 labels
- `input_ids`、`labels`、`logits` 的 shape 如何对齐
- 如何用 PyTorch 实现最小可用的 SFT 数据构造和 loss 计算
- 实现 SFT training loop 时有哪些常见坑

## 1. SFT 训练到底在学什么

预训练和 SFT 都属于自回归语言建模，都在做 next-token prediction。但它们的训练目标不同。

预训练面对的是普通文本：

```text
今天 天气 很 好 ， 我们 去 公园 ...
```

每个 token 都是自然文本的一部分，所以通常每个位置都参与 loss。

SFT 面对的是指令数据：

```text
Prompt:   请解释什么是 LoRA
Response: LoRA 是一种参数高效微调方法 ...
```

训练时输入给模型的是拼接后的完整序列：

```text
[Prompt tokens] + [Response tokens]
```

但真正希望模型学习的是：

```text
在看到 Prompt 之后，生成正确的 Response。
```

因此，SFT 通常只对 response 部分计算 loss。prompt 是条件，不是目标。

## 2. 为什么要 mask prompt

假设一条样本的 token 是：

```text
prompt_ids   = [10, 20, 30]
response_ids = [40, 50, 60, 70]
```

拼接后的 `input_ids` 是：

```text
[10, 20, 30, 40, 50, 60, 70]
```

如果直接把 `labels = input_ids`，模型会被要求预测 prompt 中的 token：

```text
预测 20、30、40、50、60、70
```

这样会带来一个问题：模型不仅在学回答，也在学“复现用户问题”。但 SFT 的目标不是让模型背诵 prompt，而是让模型在 prompt 条件下生成 answer。

所以更合理的 labels 是：

```text
[-100, -100, -100, 40, 50, 60, 70]
```

这里：

- prompt 部分设为 `-100`，不参与 loss。
- response 部分保留原 token id，参与 loss。

在 PyTorch 中，`nn.CrossEntropyLoss` 的默认 `ignore_index` 就是 `-100`。当 label 是 `-100` 时，该位置不会计入平均 loss，也不会产生梯度。

## 3. Padding 也必须 mask

实际训练通常会把 batch 内样本 padding 到同一长度。

例如最大长度是 8：

```text
input_ids = [10, 20, 30, 40, 50, 60, 70, 0]
labels    = [-100, -100, -100, 40, 50, 60, 70, -100]
```

最后一个 `0` 是 `pad_id`。它只是为了凑齐长度，不是真实训练目标，所以对应的 label 也必须是 `-100`。

如果 padding 位置没有 mask，模型会被迫学习预测 padding token。这会浪费训练信号，严重时还会让模型在生成时更容易输出异常的 padding / special token。

## 4. 为什么要 Shift Logits

自回归语言模型在位置 `t` 的输出 logits 用来预测下一个 token，也就是位置 `t + 1` 的 label。

输入序列：

```text
input_ids = [10, 20, 30, 40, 50, 60, 70, 0]
```

模型输出：

```text
logits[0] 预测 input_ids[1]
logits[1] 预测 input_ids[2]
logits[2] 预测 input_ids[3]
...
```

因此计算 loss 时不能直接拿 `logits` 和 `labels` 同位置比较，而要做错位对齐：

```python
shift_logits = logits[..., :-1, :]
shift_labels = labels[..., 1:]
```

也就是：

```text
logits[:, 0] -> labels[:, 1]
logits[:, 1] -> labels[:, 2]
logits[:, 2] -> labels[:, 3]
...
```

最后一个 logits 没有下一个 token 可预测，所以被丢弃。第一个 label 没有前文预测它，所以也被丢弃。

## 5. Shape 对齐

假设：

```text
batch_size = B
seq_len    = T
vocab_size = V
```

模型输出和 labels 的 shape 是：

```text
logits: [B, T, V]
labels: [B, T]
```

shift 之后：

```text
shift_logits: [B, T - 1, V]
shift_labels: [B, T - 1]
```

`CrossEntropyLoss` 需要输入为：

```text
input:  [N, C]
target: [N]
```

其中 `C` 是类别数，也就是 vocab size。所以要展平：

```python
shift_logits = shift_logits.view(-1, shift_logits.size(-1))
shift_labels = shift_labels.view(-1)
```

展平后：

```text
shift_logits: [B * (T - 1), V]
shift_labels: [B * (T - 1)]
```

## 6. SFT 数据构造实现

下面是最小可用版本的单条 SFT 样本构造：

```python
import torch


def build_sft_data(
    prompt_ids: list[int],
    response_ids: list[int],
    pad_id: int = 0,
    max_len: int = 16,
):
    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids

    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
    else:
        pad_len = max_len - len(input_ids)
        input_ids = input_ids + [pad_id] * pad_len
        labels = labels + [-100] * pad_len

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )
```

核心点只有三个：

- `input_ids = prompt + response`
- `labels = [-100] * len(prompt) + response`
- padding 的 label 继续填 `-100`

## 7. SFT Loss 实现

```python
import torch
import torch.nn as nn


def compute_sft_loss(logits: torch.Tensor, labels: torch.Tensor):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)

    return loss_fct(shift_logits, shift_labels)
```

这里有两个工程细节：

第一，切片后调用 `.contiguous()`。切片得到的 tensor 可能不是连续内存，后面直接 `.view()` 可能报错。

第二，`ignore_index=-100` 必须和 labels 中的 mask 值一致。否则 prompt 和 padding 仍然会参与 loss。

## 8. 一个完整测试例子

```python
def test_sft_pipeline():
    prompt = [10, 20, 30]
    response = [40, 50, 60, 70]
    pad_id = 0
    max_len = 8

    input_ids, labels = build_sft_data(prompt, response, pad_id, max_len)

    assert input_ids.tolist() == [10, 20, 30, 40, 50, 60, 70, 0]
    assert labels.tolist() == [-100, -100, -100, 40, 50, 60, 70, -100]

    batch_size = 1
    vocab_size = 100
    logits = torch.randn(batch_size, max_len, vocab_size)

    logits[0, 2, 40] = 50.0
    logits[0, 3, 50] = 50.0
    logits[0, 4, 60] = 50.0
    logits[0, 5, 70] = 50.0

    loss = compute_sft_loss(logits, labels.unsqueeze(0))
    assert loss.item() < 0.01
```

为什么这里是：

```python
logits[0, 2, 40] = 50.0
```

因为 `logits[2]` 用来预测 `labels[3]`，而 `labels[3] = 40`。

这正好验证了 shift 逻辑。

## 9. 截断策略的真实工程问题

上面的示例使用简单的右截断：

```python
input_ids = input_ids[:max_len]
labels = labels[:max_len]
```

真实 SFT 里要更谨慎。

如果 prompt 太长，response 被截没了，那么这条样本几乎没有有效监督信号：

```text
labels 全是 -100
```

这会导致该样本 loss 无意义，甚至在某些自定义 loss 实现中产生 NaN。

常见处理方式：

- 限制 prompt 最大长度，优先保留 response。
- 对超长样本做过滤。
- 使用 chat template 后再统一截断。
- 统计每条样本中有效 label token 数，过滤有效 token 数太少的样本。

一个简单检查是：

```python
num_valid = (labels != -100).sum()
```

如果 `num_valid == 0`，这条样本不应该进入训练。

## 10. Attention Mask 和 Loss Mask 不是一回事

SFT 中经常同时出现两个 mask：

```text
attention_mask
loss_mask / labels == -100
```

它们作用不同。

`attention_mask` 控制模型能不能看见某些 token，通常用于屏蔽 padding：

```text
1 表示真实 token
0 表示 padding token
```

`labels == -100` 控制某个位置是否参与 loss：

```text
普通 token id 表示参与 loss
-100 表示不参与 loss
```

prompt token 通常：

```text
attention_mask = 1
label = -100
```

也就是说，模型可以看见 prompt，但不对 prompt 位置计算 loss。

padding token 通常：

```text
attention_mask = 0
label = -100
```

也就是说，模型既不应该关注 padding，也不应该在 padding 上计算 loss。

## 11. Chat Template 对 SFT 的影响

真实指令数据通常不是简单的 prompt + response，而是经过 chat template 后的多轮对话：

```text
<|user|>
请解释 LoRA
<|assistant|>
LoRA 是一种参数高效微调方法 ...
```

在多轮 SFT 中，通常只对 assistant 的回答部分算 loss，而 user、system、分隔符等位置都 mask 掉。

概念上 labels 可能长这样：

```text
system tokens      -> -100
user tokens        -> -100
assistant answer   -> token ids
user tokens        -> -100
assistant answer   -> token ids
padding            -> -100
```

所以 SFT 数据构造的本质不是“prompt 前缀 mask”，而是：

```text
只有希望模型学习生成的 assistant span 才参与 loss。
```

## 12. 常见错误

### 错误 1：忘记 shift

错误写法：

```python
loss = loss_fct(logits.view(-1, vocab_size), labels.view(-1))
```

这样会让同位置 logits 预测同位置 label。对于 causal LM，这是错位的。

### 错误 2：prompt 没有 mask

错误后果：

- 模型浪费容量学习用户问题。
- loss 看起来更低，但回答能力不一定更好。
- 对长 prompt 数据尤其不合理。

### 错误 3：padding label 填成 pad_id

错误写法：

```python
labels = labels + [pad_id] * pad_len
```

正确写法：

```python
labels = labels + [-100] * pad_len
```

### 错误 4：只 mask 了 input，没有 mask labels

`input_ids` 中的 padding token 只是输入占位。是否计算 loss 由 labels 决定。

### 错误 5：`view` 前没有 `.contiguous()`

切片后的 tensor 可能不是 contiguous：

```python
shift_logits = logits[..., :-1, :]
```

后续 `.view()` 前最好加：

```python
shift_logits = shift_logits.contiguous()
```

或者使用 `.reshape()`。

## 13. 面试高频问法

### Q1: SFT 为什么不对 prompt 算 loss

因为 prompt 是条件，response 才是目标。对 prompt 算 loss 会让模型学习复现用户输入，而不是专注学习如何根据指令生成回答。

### Q2: `-100` 有什么特殊含义

`-100` 是 PyTorch `CrossEntropyLoss` 默认的 `ignore_index`。label 为 `-100` 的位置不会参与 loss，也不会产生梯度。

### Q3: 为什么 logits 要 shift

Causal LM 在位置 `t` 的 logits 用于预测位置 `t + 1` 的 token。所以计算 loss 时需要 `logits[..., :-1, :]` 对齐 `labels[..., 1:]`。

### Q4: prompt token 被 mask 后，模型还能看到 prompt 吗

能。loss mask 只影响是否计算损失，不影响 attention。prompt 的 `attention_mask` 仍然是 1，所以 response 位置仍然可以 attend 到 prompt。

### Q5: 如果一条样本 labels 全是 `-100` 会怎样

这条样本没有有效监督信号。取决于具体 loss 实现，可能产生无意义 loss，甚至在自定义平均逻辑中出现除零。工程上应该过滤。

## 14. 小结

SFT training loop 的核心不是复杂模型结构，而是正确处理 token 对齐和 loss mask。

最关键的三行逻辑是：

```python
labels = [-100] * len(prompt_ids) + response_ids
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
```

理解这三行，就理解了 SFT 数据构造和 loss 计算的核心。

在真实工程中，还要额外关注 chat template、多轮 assistant span、padding、超长截断和有效 label token 数。很多 SFT 训练问题不是 optimizer 或模型结构的问题，而是 labels 构造错了。
