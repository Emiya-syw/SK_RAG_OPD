# `entropy_jsd` 系数计算快照（2026-10-18）

此文记录当前实现的逐 token 系数计算方式，便于后续修改时对照。代码基线提交为 `a20ee5b`；实现位于 `verl-runtime/verl/trainer/distillation/distribution_losses.py` 的 `distribution_loss`，默认参数位于 `scripts/run_opd.sh`。

## 熵估计

教师和学生分别使用各自完整 softmax 的 top-`ENTROPY_TOPK` 原始概率（默认 64），并把其余概率合为一个尾部桶。对任一模型，令 top-k 概率为 `p_i`，尾部质量为 `r = clamp(1 - Σ_i p_i, 0, 1)`，则近似熵为

```text
H = -Σ_i p_i log(p_i) - r log(r)   （单位：nat；0 log 0 按 0 处理）
```

记师生近似熵为 `H_T`、`H_S`，尾部质量为 `r_T`、`r_S`。熵及下述门控系数停止梯度。

## 逐 token 系数

```text
α_raw = ε + (1 - 2ε) · sigmoid((H_T - H_S) / τ)
c     = clamp(1 - max(r_T, r_S) / δ, 0, 1)
α     = 0.5 + c · (α_raw - 0.5)
β     = 1 - α
```

默认 `ε = 0.05`、`τ = 1.0`、`δ = 0.2`。`β` 是混合分布中教师的系数，`α` 是学生的系数。`H_T = H_S` 时 `β = 0.5`；任一尾部质量达到或超过 0.2 时 `c = 0`，也得到 `β = 0.5`。当尾部质量足够小时，`H_T > H_S` 会使 `β < 0.5`。系数范围由 `ε` 约束为 `(0.05, 0.95)`，极限值可趋近但 sigmoid 不会恰好达到。

## 系数在损失中的用法

在**教师 top-`DISTILLATION_TOPK` token 集合**（默认 16）上，分别把教师与学生的概率重新归一化为 `T`、`S`，然后计算

```text
M = βT + (1 - β)S
L = β KL(T || M) + (1 - β) KL(S || M)
```

教师分布和 `β` 停止梯度；学生在 `S` 和 `M` 中的贡献保留梯度。默认 `ENTROPY_JSD_NORMALIZE=false`，不再除以 `β(1-β)`。损失计算中的离散支持**没有尾部桶**：尾部桶只影响熵估计和门控。学生移到教师 top-16 集合之外的概率质量，JSD 项本身可能感知不到。

`scripts/run_opd.sh` 默认 `DISTILLATION_LOSS_MODE=k3`；需显式设置为 `entropy_jsd` 才会启用以上计算。`JSD_BETA` 是固定权重 `jsd` 模式的参数，不参与 `entropy_jsd` 的逐 token `β` 计算。
