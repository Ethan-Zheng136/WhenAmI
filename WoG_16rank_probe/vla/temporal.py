"""
temporal.py  ===zgt===

时序状态表示 —— "When am I?" 的地基。

与 WoG 的根本差异
-----------------
WoG 的 QFormer 把 4 帧未来观测（[B, 4*64, 1024]）拍平成一条序列，用 16 个可学
query 一次性问出一个摘要 [B, 16, 32]。这个摘要没有时间维：问不出"t+3 时刻是
什么状态"，因为时间在 flatten 那一步就被碾平了。它也没有不确定性，只是 512
个浮点数。

我们需要的是一条时间线：每个时刻一个状态，且每个状态带着"这个估计有多可信"。
理由有三：
  1. 消息要能递推 —— α 从 s_{k-1} 推到 s_k，前提是 s_{k-1} 单独存在
  2. "消歧"要有载体 —— 消歧即减少不确定性，点估计里没有不确定性可减
  3. 融合要有原理 —— 精度加权是高斯下的乘法，直接对应 two-filter 分解

数学背景
--------
平滑后验分解为三项（Bresler 1986 / Kitagawa 1994）：

    p(s_k | 全部观测)  ∝   α_k   ×   ℓ(o_k)   ×   β_k
                          过去       当下        未来

在高斯假设下，乘法 = 精度相加：

    Λ* = Λ_α + Λ_ℓ + Λ_β
    μ* = (Λ_α μ_α + Λ_ℓ μ_ℓ + Λ_β μ_β) / Λ*

于是"谁更可信"自动化：遮挡时观测的 Λ_ℓ 变小、未来证据的 Λ_β 相对变大，
未来主导融合；画面清晰时反过来。不需要人工设计门控。

非对称性不是设计选择，是数学结构：α 是概率分布（对 s 归一化），
β 是似然函数（不归一化）。落到代码里，就是 ForwardState 和 BackwardMessage
的字段数不同。

维度约定（实测自 WoG 数据流）
--------------------------
    单帧图像             [B, 3, 224, 224]
    DINOv2 patch         [B, 256, 1024]        16x16 patch, ViT-L hidden
    经 pool_size=8 池化   [B, T, 64, 1024]      64 = 8x8 空间位置
    vla._encode_frames    [B, T*64, 1024]       T 帧拍平

    本模块输入            [B, T*64, 1024]
    本模块输出            每帧一个 GaussianState，mu/log_lam 均为 [B, T, N, D]

N/D 的取值见 StateSpec。
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------- 状态表示


@dataclass(frozen=True)
class StateSpec:
    """状态空间的形状与数值范围。

    n_tokens / dim 的取法与 WoG 对齐（每个时刻 16x32 = 512 个数，
    与 WoG 整个摘要等量），这样跨方法对比时"容量"不是混淆变量。
    若后续验证发现容量不足，优先加 n_tokens —— 它不触及 per_attn 的
    投影维度（inj_channels=32 写死在 action_model.py:36），代价为零。

    log_lam_min / max 是数值护栏，不是超参：
      精度 Lambda = exp(log_lam)，融合时要做除法 mu = sum(Lambda*mu)/sum(Lambda)。
      log_lam 无界会让 exp 溢出、或让分母趋零导致梯度爆炸。
      钳在 [-8, 8] 对应 Lambda in [3e-4, 3e3]，跨 7 个数量级，
      足够表达"几乎无信息"到"高度确信"，同时 fp16/bf16 下不会溢出。
    """

    n_tokens: int = 16
    dim: int = 32
    log_lam_min: float = -8.0
    log_lam_max: float = 8.0


@dataclass
class GaussianState:
    """对角高斯状态：mu 是估计，log_lam 是对数精度（方差的倒数）。

    为什么存精度而非方差
        融合是加法（Λ* = ΣΛ），比方差的调和平均数值稳定得多。
    为什么存 log
        保证 Λ > 0 无需额外约束，且梯度尺度合理。
    为什么对角
        全协方差要 D x D 且需求逆；对角近似把融合降为逐元素运算，
        在 D=32、9 个节点的规模下这是唯一可行的选择。
    """

    mu: Tensor       # [B, T, N, D]
    log_lam: Tensor  # [B, T, N, D]

    @property
    def precision(self) -> Tensor:
        return self.log_lam.exp()

    def __post_init__(self):
        if self.mu.shape != self.log_lam.shape:
            raise ValueError(f"mu {tuple(self.mu.shape)} and log_lam {tuple(self.log_lam.shape)} must have the same shape")


@dataclass
class BackwardMessage:
    """后向消息 β：未来观测对当前状态的似然。

    与 ForwardState（即 GaussianState）**字段不同**，这是 two-filter 的数学事实：
    前向 α 是概率分布，对 s 归一化；后向 β 是似然函数，不归一化
    （见 Briers-Doucet-Maskell 2010；Tronarp 2025 的全部贡献就是把 β
    当似然而非分布来处理）。

    这里仍保留 mu：β 需要指出"未来证据倾向于哪个方向"，但它的权重
    完全由 log_lam 决定，且不参与归一化。β 缺席时 log_lam = -inf
    （精度为 0），融合公式无需分支即自动退化 —— 这是部署时的关键性质。
    """

    mu: Tensor       # [B, T, N, D]
    log_lam: Tensor  # [B, T, N, D]

    @classmethod
    def absent(cls, like: GaussianState) -> "BackwardMessage":
        """构造一个"不提供任何约束"的 β：精度为 0。

        部署时没有未来观测，走这条路径。融合公式 Λ* = Λ_α + Λ_ℓ + Λ_β
        在 Λ_β = 0 时自然退化为只用过去和当下，训练与推理共享同一段代码。
        """
        return cls(
            mu=torch.zeros_like(like.mu),
            log_lam=torch.full_like(like.log_lam, float("-inf")),
        )

    @property
    def precision(self) -> Tensor:
        return self.log_lam.exp()   # -inf -> 0


# ---------------------------------------------------------------- 观测编码


class ObsEncoder(nn.Module):
    """把单帧的 patch 特征编码为高斯状态 —— two-filter 里的 ℓ(o_k)。

    这是整个方法唯一直接接触像素的地方，因此也是不确定性的源头：
    遮挡、模糊、过曝的帧应当输出更小的精度 Λ，从而在融合时让位给
    其他时刻的证据。这条性质是可以在训练前单独验证的（见 exp 脚本），
    若不成立则说明高斯假设在 latent 空间不适用，融合方式需要重新设计。

    结构上借用 WoG QFormer 的 query 机制：一组可学 query 去 cross-attend
    每一帧的 patch，把 64 个空间位置压成 N 个状态 token。区别在于
    **逐帧独立编码**（保留时间维），而非把所有帧拍平成一条序列。

    关于 vae_latent
        WoG 的 QFormer 会把 Wan VAE 的时序隐变量拼进 cross-attention 的 KV
        （qformer.py:25-27）。DINOv2 逐帧给的是静态语义，VAE 给的是跨帧运动，
        两者互补，丢掉后者是实打实的信息损失。

        但 VAE 隐变量是**整段一份**（[B, 16*T', 784]，T' 随输入帧数变化），
        没有逐帧对应关系。若直接拼进逐帧的 KV，每一帧都会看到同一份全局运动
        信息 —— 这会污染"这一帧自己有多可信"这个量，破坏逐帧独立性。

        因此这里把它作为**可选的全局上下文**接入：拼进 KV 但走独立的投影，
        且默认关闭（use_vae=False）。跨帧动态的主要承载者是 TransitionNet，
        VAE 在此处只做补充。是否开启由实验决定。
    """

    def __init__(
        self,
        spec: StateSpec,
        patch_dim: int = 1024,
        n_heads: int = 8,
        n_layers: int = 2,
        init_log_lam: float = 0.0,
        vae_dim: int = 784,
    ) -> None:
        super().__init__()
        self.spec = spec

        hidden = patch_dim
        self.query = nn.Parameter(torch.randn(1, spec.n_tokens, hidden) * hidden**-0.5)
        self.blocks = nn.ModuleList(
            [
                nn.MultiheadAttention(hidden, n_heads, batch_first=True)
                for _ in range(n_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])

        # VAE 全局上下文的投影，与 patch 走不同的入口（见 class docstring）
        self.vae_proj = nn.Linear(vae_dim, hidden)

        self.to_mu = nn.Linear(hidden, spec.dim)
        self.to_log_lam = nn.Linear(hidden, spec.dim)

        # 精度头零初始化 + 常数偏置：训练初期所有帧的 Λ 相同（= exp(init_log_lam)），
        # 融合退化为等权平均。让"哪些帧更可信"由梯度学出来，而不是由随机初始化决定。
        # 同一思路见 WoG 对 per_attn 的零初始化（action_model/models.py:141-145）。
        nn.init.zeros_(self.to_log_lam.weight)
        nn.init.constant_(self.to_log_lam.bias, init_log_lam)

    def forward(
        self,
        patch_features: Tensor,
        n_frames: int,
        vae_latent: Optional[Tensor] = None,
    ) -> GaussianState:
        """
        Args:
            patch_features: [B, T*P, C]  来自 vla._encode_frames，T 帧已拍平
            n_frames:       T，用于恢复时间维
            vae_latent:     [B, L, vae_dim]  可选的全局运动上下文（整段一份）

        Returns:
            GaussianState，mu / log_lam 均为 [B, T, N, D]
        """
        B, TP, C = patch_features.shape
        if TP % n_frames != 0:
            raise ValueError(f"patch count {TP} is not divisible by frame count {n_frames}")
        P = TP // n_frames

        # 恢复时间维后逐帧独立编码 —— 与 WoG 的关键差异：时间不被碾平
        x = patch_features.view(B * n_frames, P, C)

        if vae_latent is not None:
            # 整段一份的全局上下文，广播给每一帧
            ctx = self.vae_proj(vae_latent)                                  # [B, L, C]
            ctx = ctx.unsqueeze(1).expand(-1, n_frames, -1, -1)              # [B, T, L, C]
            ctx = ctx.reshape(B * n_frames, ctx.shape[-2], C)
            x = torch.cat([x, ctx], dim=1)                                   # [B*T, P+L, C]

        q = self.query.expand(B * n_frames, -1, -1)
        for attn, norm in zip(self.blocks, self.norms):
            out, _ = attn(norm(q), x, x)
            q = q + out

        mu = self.to_mu(q).view(B, n_frames, self.spec.n_tokens, self.spec.dim)
        log_lam = self.to_log_lam(q).view(B, n_frames, self.spec.n_tokens, self.spec.dim)
        return GaussianState(mu=mu, log_lam=log_lam)


# ---------------------------------------------------------------- 转移与递推


class TransitionNet(nn.Module):
    """转移算子 T(s, a) -> s'  —— 整个方法的枢纽。

    为什么必须是**一个**算子
        two-filter 的两条递推里出现的是同一个转移核 p(s_{k+1} | s_k, a_k)：

            alpha_k = ell(o_k) * ∫ T(s_k | s_{k-1}, a_{k-1}) alpha_{k-1}
            beta_k  =            ∫ T(s_{k+1} | s_k, a_k) ell(o_{k+1}) beta_{k+1}

        区别只在积分掉谁（前向边缘化过去、后向边缘化未来），核本身不变。
        若前向后向各用一个网络，两支就活在不同的状态空间里，"在当下相遇"
        只能靠 loss 硬凑，塌缩风险随之而来。共享 T 让相遇成为结构性的。

    为什么它同时是部署期的能力
        训练时 T 用于前向递推与后向传播；部署时没有未来观测，同一个 T
        直接用来 rollout。不需要另训一个外推头 —— "latent 里推理未来的
        潜力"就是 T 本身。

    不确定性怎么传播
        转移会引入过程噪声，故输出精度必然低于输入（信息只会在传播中损失）。
        这里用 log_lam - softplus(decay) 保证单调下降，decay 可学。
    """

    def __init__(self, spec: StateSpec, action_dim: int = 7, hidden: int = 256) -> None:
        super().__init__()
        self.spec = spec

        self.act_embed = nn.Linear(action_dim, spec.dim)
        self.net = nn.Sequential(
            nn.Linear(spec.dim * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, spec.dim),
        )
        # 过程噪声：转移后精度的衰减量，逐维可学
        self.log_decay = nn.Parameter(torch.zeros(spec.dim))

        # 零初始化最后一层：初始时 T 是恒等映射（s' = s），状态不会被随机
        # 扰动带偏。与 WoG 对 per_attn 的零初始化同一思路
        # （action_model/models.py:141-145）。
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: GaussianState, action: Tensor) -> GaussianState:
        """把状态推进一步。

        Args:
            state:  GaussianState，mu/log_lam 为 [B, N, D]（单个时刻）
            action: [B, action_dim]

        Returns:
            推进一步后的 GaussianState，形状不变
        """
        a = self.act_embed(action).unsqueeze(1).expand_as(state.mu)     # [B, N, D]
        delta = self.net(torch.cat([state.mu, a], dim=-1))
        mu = state.mu + delta                                           # 残差形式，配合零初始化 = 恒等起步

        # 过程噪声让精度单调下降：信息在传播中只会损失，不会凭空增加
        log_lam = state.log_lam - F.softplus(self.log_decay)
        return GaussianState(mu=mu, log_lam=log_lam)


def fuse(*parts) -> GaussianState:
    """高斯乘法 = 精度加权平均。

        Lambda* = sum(Lambda_i)
        mu*     = sum(Lambda_i * mu_i) / Lambda*

    这正是 two-filter 分解 p(s|all) ∝ alpha * ell * beta 在高斯假设下的形式。
    "谁更可信"由各自的 Lambda 自动决定：遮挡时观测精度低，未来证据主导；
    画面清晰时反过来。不需要人工门控。

    精度为 0 的成分（如缺席的 beta）不贡献任何约束，公式自动退化 ——
    训练与部署因此共享同一段代码。
    """
    lam = sum(p.precision for p in parts)
    weighted = sum(p.precision * p.mu for p in parts)
    mu = weighted / lam.clamp_min(1e-8)
    return GaussianState(mu=mu, log_lam=lam.clamp_min(1e-8).log())


def _index(state: GaussianState, k: int) -> GaussianState:
    """取出时间维上的第 k 个时刻。"""
    return GaussianState(mu=state.mu[:, k], log_lam=state.log_lam[:, k])


def _stack(states) -> GaussianState:
    """把逐时刻的状态沿时间维堆回去。"""
    return GaussianState(
        mu=torch.stack([s.mu for s in states], dim=1),
        log_lam=torch.stack([s.log_lam for s in states], dim=1),
    )


def forward_pass(obs: GaussianState, actions: Tensor, T: TransitionNet) -> GaussianState:
    """前向消息 alpha：因果方向，左 -> 右。

        alpha_k = ell(o_k) ⊗ T(alpha_{k-1}, a_{k-1})

    起点 alpha_0 就是第 0 帧的观测似然（无更早的信息可用）。
    之后每步先用 T 把上一时刻推过来，再与本时刻的观测融合。

    带动作：a_{k-1} 是**已经执行过**的动作，部署时机器人自己知道，
    因此对 student 不构成特权。

    Args:
        obs:     [B, K, N, D]  逐帧观测似然 ell(o_k)
        actions: [B, K-1, A]   a_{k-1}，第 k 步转移用的动作
        T:       共享转移算子

    Returns:
        alpha，[B, K, N, D]
    """
    K = obs.mu.shape[1]
    out = [_index(obs, 0)]
    for k in range(1, K):
        pred = T(out[-1], actions[:, k - 1])
        out.append(fuse(pred, _index(obs, k)))
    return _stack(out)


def backward_pass(obs: GaussianState, actions: Tensor, T: TransitionNet) -> BackwardMessage:
    """后向消息 beta：反因果方向，右 -> 左。

        beta_k = ∫ T(s_{k+1} | s_k, a_k) ell(o_{k+1}) beta_{k+1}

    与前向的三点不同 —— 这些不是设计选择，是 two-filter 的结构：

      1. 方向相反：从最远的未来一层层回溯到 k，正向算不出这个量。
      2. 起点不同：beta_{K-1} 没有更晚的证据，是"无信息"，精度为 0。
         对比 alpha_0 = 观测似然本身。
      3. 类型不同：beta 是**似然**而非分布，不对 s 归一化
         （Briers-Doucet-Maskell 2010；Tronarp 2025 的贡献正是把 beta
         当似然处理）。落到代码里就是返回 BackwardMessage 而非 GaussianState。

    这里复用同一个 T：转移核在两条递推里本就是同一个，只是积分变量不同。

    Args:
        obs:     [B, K, N, D]  逐帧观测似然
        actions: [B, K-1, A]   a_k，第 k 步转移用的动作
        T:       共享转移算子（与 forward_pass 是同一个实例）

    Returns:
        beta，[B, K, N, D]
    """
    K = obs.mu.shape[1]
    last = _index(obs, K - 1)
    out = [BackwardMessage(mu=torch.zeros_like(last.mu),
                           log_lam=torch.full_like(last.log_lam, float("-inf")))]

    for k in range(K - 2, -1, -1):
        # 把 (k+1) 处的证据（观测 + 更远的 beta）合并后，经 T 回传到 k
        evidence = fuse(
            _index(obs, k + 1),
            GaussianState(mu=out[-1].mu, log_lam=out[-1].log_lam),
        )
        back = T(evidence, actions[:, k])
        out.append(BackwardMessage(mu=back.mu, log_lam=back.log_lam))

    out.reverse()
    return BackwardMessage(
        mu=torch.stack([m.mu for m in out], dim=1),
        log_lam=torch.stack([m.log_lam for m in out], dim=1),
    )


# ---------------------------------------------------------------- 汇聚


class TemporalConverge(nn.Module):
    """把 past / present / future 三路证据汇聚成一条状态时间线。

    替代 WoG 的 QFormer 所在的位置，但做的事不同：QFormer 输出一个无时间维的
    摘要 [B, 16, 32]；这里输出 [B, K, 16, 32] —— K 个时刻，每个时刻一个状态。

    信息流
        past  ──alpha(因果, 带已执行动作)──┐
                                          ├──> fuse ──> s*_k  (逐时刻)
        future ──beta(反因果, 似然)───────┘
                          ^
                    ell(o_k) 锚定当下

        融合即 two-filter 分解 p(s|all) ∝ alpha * ell * beta 的高斯形式。
        注意 alpha 内部已经吸收了 ell（见 forward_pass），故此处只需
        alpha ⊗ beta，不重复乘观测项。

    训练与部署共用一条代码路径
        部署时没有未来观测，beta 缺席即精度为 0，融合公式自动退化为
        "只用过去和当下"。无需 if 分支，也就没有训练/推理行为不一致的风险。

    数值稳定性（这些不是可选项，是训练能否收敛的前提）
        1. log_lam 钳位  —— 防 exp 溢出与除法分母趋零
        2. 输出 LayerNorm —— 让 s* 的尺度与 WoG 的 O^c 可比，
                            否则接进 per_attn 后 DiT 收不住
        3. 零初始化输出投影 —— 训练初期 s* 恒为 0，等价于 per_attn 不起作用，
                            模型退化为无条件的 DiT，不破坏 OpenVLA 预训练行为。
                            与 WoG 对 per_attn 的零初始化同一思路。
    """

    def __init__(
        self,
        spec: Optional[StateSpec] = None,
        patch_dim: int = 1024,
        action_dim: int = 7,
        n_heads: int = 8,
        n_enc_layers: int = 2,
    ) -> None:
        super().__init__()
        self.spec = spec or StateSpec()

        self.obs_encoder = ObsEncoder(
            self.spec, patch_dim=patch_dim, n_heads=n_heads, n_layers=n_enc_layers
        )
        self.transition = TransitionNet(self.spec, action_dim=action_dim)

        # 输出侧：归一化定尺度，投影定起点。
        #
        # 关于初始化尺度（实测结论，勿改回零初始化）
        #   WoG 在 DiT 的 per_attn（action_model/models.py:141-145）用零初始化，
        #   那里安全的原因是它处在**残差分支**上：
        #       x = x + per_attn(...)
        #   主干 x 直通，分支即便输出为零，自身仍从主干收到梯度并脱离零点。
        #
        #   本层不同：out_proj 是 converge 的**唯一出口**，没有并行主干。
        #   零初始化会让输出恒为零，梯度 grad_out * input 随之恒为零，
        #   整个 converge（obs_encoder + transition + out_proj，32 个参数）
        #   永远收不到梯度。实测对照：
        #       残差分支零初始化  -> 有梯度 2/2
        #       串联出口零初始化  -> 有梯度 0/4
        #   线上也已印证：whenami_v2 从 step 5000 到 10000，converge 的
        #   32 个参数逐字节未变，条件恒为零向量，训练实际退化成无条件生成。
        #
        #   改为小尺度正态初始化：既保留"起步扰动小、不冲击 OpenVLA 预训练
        #   行为"的意图，又保证梯度能回流。bias 仍可置零（不阻断梯度）。
        self.out_norm = nn.LayerNorm(self.spec.dim)
        self.out_proj = nn.Linear(self.spec.dim, self.spec.dim)
        nn.init.normal_(self.out_proj.weight, std=1e-3)
        nn.init.zeros_(self.out_proj.bias)

    def _clamp(self, state: GaussianState) -> GaussianState:
        return GaussianState(
            mu=state.mu,
            log_lam=state.log_lam.clamp(self.spec.log_lam_min, self.spec.log_lam_max),
        )

    def forward(
        self,
        patch_features: Tensor,
        actions: Tensor,
        n_frames: int,
        use_future: bool = True,
        n_past: Optional[int] = None,
        vae_latent: Optional[Tensor] = None,
    ) -> GaussianState:
        """
        Args:
            patch_features: [B, K*P, C]  全部 K 帧的 patch 特征（past+now+future 已拼好）
            actions:        [B, K-1, A]  相邻帧之间的动作，actions[:, k] 用于 k -> k+1
            n_frames:       K
            use_future:     False 时切断 beta（部署 / Stage 2 的 student 路径）
            n_past:         past 帧数。use_future=False 时只保留 [0, n_past] 段，
                            避免把未来帧的观测偷偷喂给 alpha
            vae_latent:     可选的全局运动上下文；student 路径下强制丢弃，
                            因为它由**包含未来帧**的整段视频编码而来（见 FutureEnc），
                            留着就是泄漏

        Returns:
            s*，[B, K, n_tokens, dim]
        """
        if not use_future:
            vae_latent = None   # 见 docstring：VAE 上下文含未来信息

        obs = self._clamp(self.obs_encoder(patch_features, n_frames, vae_latent))

        if not use_future and n_past is not None:
            # student 路径：物理上截断到当下为止，而不是仅仅关掉 beta。
            # 否则 alpha 仍会吸收未来帧的观测 —— 那是泄漏，不是特权撤除。
            keep = n_past + 1
            obs = GaussianState(mu=obs.mu[:, :keep], log_lam=obs.log_lam[:, :keep])
            actions = actions[:, : keep - 1]
            n_frames = keep

        alpha = self._clamp(forward_pass(obs, actions, self.transition))

        if use_future:
            beta = backward_pass(obs, actions, self.transition)
            beta_state = GaussianState(
                mu=beta.mu,
                log_lam=beta.log_lam.clamp(self.spec.log_lam_min, self.spec.log_lam_max),
            )
        else:
            absent = BackwardMessage.absent(alpha)
            beta_state = GaussianState(mu=absent.mu, log_lam=absent.log_lam)

        # alpha 已含 ell(o_k)，此处只与 beta 相乘，避免观测项被计入两次
        fused = fuse(alpha, beta_state)
        return GaussianState(
            mu=self.out_proj(self.out_norm(fused.mu)),
            log_lam=fused.log_lam,
        )
