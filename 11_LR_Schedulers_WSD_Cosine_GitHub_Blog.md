# LR Scheduler 知识点整理：Warmup、Cosine Decay 与 WSD 调度器

学习率调度器, Learning Rate Scheduler, 是大模型训练中非常关键的稳定性组件。

在小模型训练中，学习率可能只是一个普通超参数。但在 LLM 预训练、持续预训练和 SFT 中，学习率曲线经常直接决定训练是否稳定、loss 是否 spike、最终模型是否收敛。

这篇笔记整理大模型训练中常见的学习率调度知识点：

- 为什么大模型训练通常需要 warmup
- Cosine Annealing with Warmup 是什么
- WSD, Warmup-Stable-Decay, 为什么适合现代大模型训练
- WSD 和 Cosine 的关键区别
- 如何手写一个 PyTorch `LRScheduler`
- `get_lr()` 中 step 如何分段
- 如何验证学习率曲线是否正确
- 实现 scheduler 时有哪些常见坑

## 1. 为什么不能一开始就用最大学习率

训练初期最容易不稳定。

原因主要有两个。

第一，模型参数和梯度状态还不稳定。对于随机初始化或刚进入新训练阶段的模型，梯度方向可能很嘈杂。如果一开始就使用最大学习率，参数更新可能过大，导致 loss spike 甚至 NaN。

第二，AdamW 这类自适应优化器在刚开始时，二阶动量估计还没有积累足够信息。二阶动量会出现在更新分母中，如果估计不稳定，实际更新步长也会不稳定。

Warmup 的作用就是给模型和优化器一个缓冲期：

```text
学习率从很小逐步升到 max_lr
```

常见线性 warmup：

$$
\eta_t = \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}}
$$

其中：

- $\eta_t$ 是第 $t$ 步学习率。
- $\eta_{\max}$ 是最大学习率。
- $T_{\text{warmup}}$ 是 warmup 步数。

## 2. Cosine Annealing with Warmup

Cosine with Warmup 是很多 LLM 预训练中常见的学习率曲线。

它通常分成两段：

```text
Warmup: 学习率线性上升
Cosine Decay: 学习率按余弦曲线逐渐下降
```

直观图像：

```text
lr
^
|      /\
|     /  \__
|    /      \___
|   /           \____
|__/__________________> step
```

cosine decay 的常见形式：

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})
\left(1 + \cos(\pi p)\right)
$$

其中：

$$
p = \frac{t - T_{\text{warmup}}}{T_{\text{decay}}}
$$

`p` 表示 decay 阶段进度，从 0 到 1。

当 `p = 0`：

```text
cos(0) = 1
lr = max_lr
```

当 `p = 1`：

```text
cos(pi) = -1
lr = min_lr
```

## 3. Cosine 的问题

Cosine 调度器的核心假设是：

```text
训练总步数一开始就确定。
```

如果总步数固定，这很合理。学习率从高到低平滑下降，训练最后进入低学习率收敛阶段。

但现代大模型训练经常遇到持续预训练, Continued Pre-training：

- 中途加入新数据。
- 数据配比发生变化。
- 训练预算增加。
- 想在已有 checkpoint 上继续吃更多 token。

这时 Cosine 有一个麻烦：

```text
如果学习率已经衰减得很低，模型继续学习新数据的能力会变弱。
```

也就是说，Cosine 的学习率曲线过早地把模型带入“收尾状态”。如果后面还想继续训练，可能不得不重新设计 scheduler 或手动重启学习率。

## 4. WSD 的核心思想

WSD 是 Warmup-Stable-Decay 的缩写。

它把训练分成三段：

```text
1. Warmup: 学习率从 0 增长到 max_lr
2. Stable: 长时间保持 max_lr
3. Decay: 最后阶段快速衰减到 min_lr
```

直观图像：

```text
lr
^
|      _____________
|     /             \
|    /               \__
|   /                   \__
|__/________________________> step
```

WSD 的关键是 stable 阶段。

模型在绝大多数训练 token 上都保持较高学习率，只有在真正接近训练结束时才开始 decay。

这样如果想继续加数据，只需要延长 stable 阶段：

```text
Warmup -> Stable -> Stable -> Stable -> Decay
```

不用在一开始就把总训练步数锁死。

## 5. WSD 的数学分段

设：

```text
num_warmup_steps = Tw
num_stable_steps = Ts
num_decay_steps  = Td
base_lr          = eta_max
min_lr           = eta_min
```

总步数：

```text
T = Tw + Ts + Td
```

### Warmup 阶段

当：

```text
step < Tw
```

学习率线性上升：

$$
\eta_t = \eta_{\max} \cdot \frac{t}{T_w}
$$

### Stable 阶段

当：

```text
Tw <= step < Tw + Ts
```

学习率保持不变：

$$
\eta_t = \eta_{\max}
$$

### Decay 阶段

当：

```text
step >= Tw + Ts
```

使用余弦衰减：

$$
p = \frac{step - T_w - T_s}{T_d}
$$

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi p))
$$

工程上通常要把 `p` clamp 到 `[0, 1]`，避免训练超过预期步数后学习率反弹或异常。

## 6. WSD Scheduler 完整实现

下面是一个继承 PyTorch `LRScheduler` 的最小实现：

```python
import math
from torch.optim.lr_scheduler import LRScheduler


class WSD_Scheduler(LRScheduler):
    def __init__(
        self,
        optimizer,
        num_warmup_steps,
        num_stable_steps,
        num_decay_steps,
        min_lr_ratio=0.1,
        last_epoch=-1,
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_stable_steps = num_stable_steps
        self.num_decay_steps = num_decay_steps
        self.min_lr_ratio = min_lr_ratio
        self.total_steps = num_warmup_steps + num_stable_steps + num_decay_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self._step_count - 1

        lrs = []
        for base_lr in self.base_lrs:
            min_lr = base_lr * self.min_lr_ratio

            if step < self.num_warmup_steps:
                if step == 0:
                    current_lr = 0.0
                else:
                    current_lr = base_lr * step / self.num_warmup_steps

            elif step < (self.num_warmup_steps + self.num_stable_steps):
                current_lr = base_lr

            else:
                decay_step = step - self.num_warmup_steps - self.num_stable_steps
                decay_ratio = decay_step / self.num_decay_steps
                cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_ratio))
                current_lr = min_lr + (base_lr - min_lr) * cosine_decay

            lrs.append(current_lr)

        return lrs
```

这段代码的核心是：

```python
if step < warmup:
    ...
elif step < warmup + stable:
    ...
else:
    ...
```

也就是按当前训练步数判断属于哪个阶段。

## 7. 更稳健的 decay 写法

上面的实现适合教学，但真实工程中建议对 `decay_ratio` 做截断：

```python
decay_ratio = min(max(decay_ratio, 0.0), 1.0)
```

完整 decay 片段：

```python
decay_step = step - self.num_warmup_steps - self.num_stable_steps
decay_ratio = decay_step / self.num_decay_steps
decay_ratio = min(max(decay_ratio, 0.0), 1.0)

cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_ratio))
current_lr = min_lr + (base_lr - min_lr) * cosine_decay
```

这样即使训练步数超过预期，也会保持在 `min_lr`，不会继续沿余弦周期反弹。

## 8. 如何验证学习率曲线

可以构造一个 dummy model 和 optimizer，模拟训练步数并记录 lr：

```python
import torch


dummy_model = torch.nn.Linear(2, 2)
max_lr = 3e-4
optimizer = torch.optim.AdamW(dummy_model.parameters(), lr=max_lr)

warmup = 1000
stable = 7000
decay = 2000
total = warmup + stable + decay

scheduler = WSD_Scheduler(
    optimizer,
    num_warmup_steps=warmup,
    num_stable_steps=stable,
    num_decay_steps=decay,
    min_lr_ratio=0.1,
)

lrs = []
for _ in range(total):
    lrs.append(optimizer.param_groups[0]["lr"])
    optimizer.step()
    scheduler.step()
```

关键断言：

```python
assert lrs[0] == 0.0
assert abs(lrs[warmup] - max_lr) < 1e-8
assert abs(lrs[warmup + stable - 1] - max_lr) < 1e-8
assert abs(lrs[-1] - (max_lr * 0.1)) < 1e-8
```

这四个点分别验证：

- 初始学习率为 0。
- warmup 结束后达到最大学习率。
- stable 阶段保持最大学习率。
- decay 结束后达到最小学习率。

## 9. 可视化代码

```python
import matplotlib.pyplot as plt


plt.figure(figsize=(10, 5))
plt.plot(lrs, label="Learning Rate", color="blue", linewidth=2)
plt.axvline(x=warmup, color="red", linestyle="--", alpha=0.5, label="End Warmup")
plt.axvline(
    x=warmup + stable,
    color="green",
    linestyle="--",
    alpha=0.5,
    label="Start Decay",
)
plt.title("WSD Scheduler")
plt.xlabel("Training Steps")
plt.ylabel("Learning Rate")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
```

可视化 scheduler 很重要，因为学习率 bug 不一定会立刻报错，但会直接影响训练效果。

## 10. Optimizer 和 Scheduler 的调用顺序

PyTorch 中常见训练循环是：

```python
loss.backward()
optimizer.step()
scheduler.step()
optimizer.zero_grad()
```

先 `optimizer.step()`，再 `scheduler.step()`。

如果顺序写反，第一步学习率可能和预期不同，PyTorch 也可能给出 warning。

完整示例：

```python
for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss

    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

如果使用 gradient accumulation，通常应该在真正执行 optimizer update 的时候再 step scheduler，而不是每个 micro batch 都 step。

## 11. Gradient Accumulation 下的步数

假设：

```text
gradient_accumulation_steps = 8
```

每 8 个 micro batch 才执行一次：

```python
optimizer.step()
scheduler.step()
```

那么 scheduler 的 step 数应该对应 optimizer update 次数，而不是 dataloader iteration 次数。

如果你把 warmup 设置为 1000，但每个 micro batch 都调用 scheduler：

```text
真实 warmup 只持续 1000 micro steps
= 125 optimizer updates
```

这会让 warmup 过短。

正确理解是：

```text
num_warmup_steps 通常指 optimizer update steps。
```

## 12. 多参数组学习率

PyTorch optimizer 可以有多个 parameter group：

```python
optimizer = torch.optim.AdamW([
    {"params": decay_params, "lr": 3e-4},
    {"params": no_decay_params, "lr": 3e-4},
])
```

`get_lr()` 返回的是一个列表：

```python
return lrs
```

每个 parameter group 对应一个学习率。

因此实现 scheduler 时不能只返回一个 float，而要遍历：

```python
for base_lr in self.base_lrs:
    ...
    lrs.append(current_lr)
```

## 13. Cosine 和 WSD 怎么选

### 适合 Cosine 的场景

- 总训练步数非常确定。
- 训练任务是一次性收敛。
- 不太需要中途继续加数据。
- 希望学习率全程平滑下降。

### 适合 WSD 的场景

- 大模型预训练或持续预训练。
- 数据规模可能变化。
- 希望主体训练阶段保持较大学习率。
- 只在最后短时间做退火收敛。

可以粗略理解为：

```text
Cosine: 一开始就计划好完整训练路线。
WSD: 先长期学习，最后再明确收尾。
```

## 14. 常见错误

### 错误 1：warmup 步数按 batch 算错

如果使用 gradient accumulation，warmup 应该按 optimizer update step 算，而不是 micro batch 数。

### 错误 2：scheduler.step 调用太频繁

在 gradient accumulation 中，每个 micro batch 都 step scheduler 会让学习率变化过快。

### 错误 3：decay_ratio 不 clamp

训练超过预期步数后，余弦函数可能继续周期变化，导致学习率反弹。真实工程中最好 clamp 到 `[0, 1]`。

### 错误 4：min_lr 设成 0 后继续训练

如果学习率降到 0，后续继续训练等于几乎不再更新。持续预训练场景通常会保留一个非零 `min_lr`。

### 错误 5：保存 checkpoint 时没有保存 scheduler state

断点续训时不仅要保存 model 和 optimizer，还要保存 scheduler：

```python
torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "step": global_step,
}, path)
```

否则恢复训练后学习率阶段可能错乱。

## 15. 面试高频问法

### Q1: 为什么需要 warmup

训练初期梯度和 AdamW 动量估计都不稳定。warmup 让学习率逐步升高，降低 loss spike 和 NaN 风险。

### Q2: Cosine decay 的公式是什么

核心是：

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi p))
$$

其中 `p` 是 decay 阶段进度。

### Q3: WSD 和 Cosine 的最大区别是什么

Cosine 在 warmup 后立即持续衰减；WSD 在 warmup 后有一个长 stable 阶段，最后才 decay。

### Q4: WSD 为什么适合持续预训练

因为 stable 阶段可以延长。如果训练中途增加数据，不需要让模型在低学习率状态下继续学习。

### Q5: Scheduler 的 step 数应该对应什么

通常对应 optimizer update step，而不是 micro batch step。使用 gradient accumulation 时尤其要注意。

## 16. 小结

大模型训练中，学习率曲线不是简单的附属配置，而是训练稳定性的核心部分。

Warmup 解决训练初期不稳定：

```text
低学习率 -> 逐步升高 -> max_lr
```

Cosine 适合总步数固定的一次性训练：

```text
Warmup -> 平滑衰减
```

WSD 更适合现代大模型预训练和持续预训练：

```text
Warmup -> Stable -> Decay
```

手写 scheduler 时最重要的是：

```python
step = self._step_count - 1
if step < num_warmup_steps:
    ...
elif step < num_warmup_steps + num_stable_steps:
    ...
else:
    ...
```

再配合可视化和关键点断言，就能比较可靠地发现学习率调度中的 off-by-one、调用频率和阶段切换问题。
