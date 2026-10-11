"""[zgt] Masked 时间线汇聚 —— WhenAmI 的第二版实现。

与同目录 temporal.py 的递推版并存，用 `converge_type` 选择。递推版的
probe 结果（步 10000，fractal 267 样本，held-out R²）暴露了四个问题：

    探测目标        WoG O^c   s*_t    仅 alpha   beta 净贡献
    gt_togo          0.171   0.051     0.147      -0.096   ❌
    gt_eef           0.505   0.427     0.379      +0.049
    progress         0.339   0.509     0.497      +0.013
    gt_grip          0.374   0.461     0.608      -0.147   ❌

  1. 有损压缩：每帧 64 个 patch 先被压成 [16,32] 才进递推，
     gt_togo 掉到 0.051 很可能源于此（WoG 保留全部 patch，得 0.171）。
  2. 递推误差累积：alpha 要走 4 步才到当下，每步都引入近似。
  3. 反向算子不严格：backward 复用同一个 T 反着用，隐含假设它可逆，
     而学出来的网络没有这个保证。
  4. Lambda 无监督：唯一信号是穿过 DiT 的动作损失，太弱 ——
     实测 log_decay 的 std 只有 2.4e-4，各维几乎没分化，
     融合退化成固定权重的混合，于是 beta 在 gt_grip 上把 alpha
     已经编好的信息稀释了 0.147。

本模块的四处对应改动：

  * 三路互斥的 cross-attention，不做递推
        Q_past   只见 o_{t-4..t-1}
        Q_now    只见 o_t
        Q_future 只见 o_{t+1..t+4}
    互斥由架构保证（各分支只收到自己那段 patch），不依赖 attention mask
    写得对不对。这同时让 two-filter 分解 p ∝ alpha * ell * beta 要求的
    条件独立性成立 —— 递推版让 o_t 同时进入 alpha 和 beta，证据被
    重复计数，Lambda* 会虚高。

  * 全部 patch 直接进 cross-attention，不预先压缩（对应问题 1）

  * 时序嵌入：每个 patch 加上它所属帧的可学向量。WoG 的 QFormer 把 5 帧
    拍平且 cross-attention 不带位置编码，patch 在它眼里是无序集合；
    TimeChat（CVPR 2024）的消融显示去掉时间戳感知会让时序定位 F1 掉 2.3。

  * Lambda 直接监督（对应问题 4）：精度的定义就是误差方差的倒数，
        Lambda = 1 / sigma^2
    所以让每个分支去重建当下帧的观测，用重建误差的倒数作为 Lambda 的
    目标。这不是外加的正则，而是把 Lambda 的定义本身变成可训练目标。
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import BertConfig, BertModel

from vla.temporal import GaussianState, StateSpec, fuse


@dataclass
class CalibAux:
    """分支自评产生的辅助量，供 forward 之外的损失使用。

    recon：重建误差驱动的 Lambda 校准损失（变体 B，WhenAmI 特有 ——
           它测的是"我对当下有多清楚"）。
    act：  动作预测误差驱动的校准损失（变体 A，任何任务都适用 ——
           它测的是"我知道该做什么"）。两者权重可独立配置，
           天然构成消融。
    """

    recon: Tensor
    act: Tensor

    @classmethod
    def zeros(cls, like: Tensor) -> "CalibAux":
        z = torch.zeros((), device=like.device, dtype=like.dtype)
        return cls(recon=z, act=z)


class TemporalBranch(nn.Module):
    """一路证据源：若干帧的 patch -> (mu, log_lam)。

    骨架沿用 WoG 的 QFormer（BertModel + cross-attention + 可学 query），
    因为那部分已在 OXE 上验证可用；改动在两处：

      * 输出不只 mu，还有 log_lam —— 这一路"有多确定"。WoG 的 O^c 只有
        均值，它无法区分"我看得很清楚"和"我在猜"。
      * 自带重建头与动作读出头，为 log_lam 提供直接的学习信号。

    to_log_lam 零初始化 + 常数偏置：训练初期各帧的 Lambda 相同，融合退化为
    等权平均，让"谁更可信"由梯度学出来而非由随机初始化决定。这里零初始化
    是安全的 —— to_log_lam 与 to_mu 并联，不在 mu 的通路上，不会像
    temporal.py 那次 out_proj 零初始化一样把整条路的梯度掐死。
    """

    def __init__(
        self,
        patch_dim: int,
        spec: StateSpec,
        n_layers: int,
        hidden: int = 1024,
        n_heads: int = 8,
        ff: int = 2048,
        recon_dim: int = 64,
        n_patch: int = 64,
        action_dim: int = 7,
        n_action: int = 16,
        init_log_lam: float = 0.0,
        with_recon: bool = True,
    ) -> None:
        super().__init__()
        self.spec = spec
        cfg = BertConfig(
            hidden_size=hidden,
            num_attention_heads=n_heads,
            intermediate_size=ff,
            num_hidden_layers=n_layers,
            add_cross_attention=True,
            is_decoder=True,
        )
        # add_pooling_layer=False：我们取 last_hidden_state，不用 pooler。
        # 留着它就是一组永远收不到梯度的参数，FSDP 下会浪费分片并可能报
        # unused-parameter。
        self.bert = BertModel(config=cfg, add_pooling_layer=False)
        # 我们喂的是 inputs_embeds，word_embeddings 永不参与前向。显式关掉
        # 梯度，让"无用"成为声明而非隐患。
        self.bert.embeddings.word_embeddings.weight.requires_grad_(False)
        self.query = nn.Parameter(torch.randn(1, spec.n_tokens, hidden) * 0.02)
        self.in_proj = nn.Linear(patch_dim, hidden)

        self.to_mu = nn.Linear(hidden, spec.dim)
        self.to_log_lam = nn.Linear(hidden, spec.dim)
        nn.init.zeros_(self.to_log_lam.weight)
        nn.init.constant_(self.to_log_lam.bias, init_log_lam)

        # 校准头按需创建 —— 用不到的头就是永远收不到梯度的参数，
        # FSDP 下会浪费分片。两种校准的适用范围不同：
        #   recon：让分支去重建**它看不到的**当下帧。对 past/future 成立，
        #          对 now 是平凡的（它刚看过那一帧，重建等于恒等映射，
        #          误差恒为 0，Lambda_ell 会虚高到主导融合）。
        #   act：  让分支预测动作。对三路都有意义 —— 遮挡时单靠当下那一帧
        #          预测不出动作，这正是 Lambda_ell 该小的时候。
        flat = spec.n_tokens * spec.dim
        # 输出 [B, n_patch*recon_dim]，与 (now_patches @ recon_proj).flatten(1)
        # 对齐 —— 保留空间结构，遮挡本身就是空间性的。
        self.recon_head = nn.Linear(flat, n_patch * recon_dim) if with_recon else None
        self.act_head = nn.Linear(flat, n_action * action_dim)
        self.recon_dim = recon_dim
        self.n_patch = n_patch
        self.n_action = n_action
        self.action_dim = action_dim

    def forward(self, patches: Tensor) -> Tuple[GaussianState, Tensor]:
        """
        Args:
            patches: [B, F*P, C]  本分支可见的那几帧（已含时序嵌入）

        Returns:
            GaussianState，mu / log_lam 均为 [B, n_tokens, dim]
            以及展平的 mu，供校准头复用
        """
        src = self.in_proj(patches)
        q = self.query.expand(src.shape[0], -1, -1)
        out = self.bert(inputs_embeds=q, encoder_hidden_states=src,
                        return_dict=True).last_hidden_state
        mu = self.to_mu(out)
        log_lam = self.to_log_lam(out).clamp(self.spec.log_lam_min, self.spec.log_lam_max)
        return GaussianState(mu=mu, log_lam=log_lam), mu.flatten(1)


class MaskedConverge(nn.Module):
    """三路互斥汇聚：past -> now <- future。

    输出只有当下那一个状态 s*_t（[B, n_tokens, dim]），因此 DiT 只跑一次
    —— 与 WoG 同等成本。递推版为了给 9 个节点提供监督要跑 9 次 DiT
    （实测 4.3 s/step vs WoG 2.8 s/step），而评测只看当下那一个节点，
    容量被摊薄了 9 倍。

    Q_now 单独成一路而非并入 Q_past，有两个理由：
      * two-filter 的三项 alpha * ell * beta 要求基于互不重叠的观测；
        把 o_t 同时给 alpha 和 beta 会重复计数。
      * Lambda_ell 由此有了独立语义 —— 它直接表示"当下这一帧清不清楚"，
        正是遮挡的度量。若把当下混进 alpha，就没有单独代表可见度的变量了。
      它只看 1 帧，任务简单，故层数可以远少于另两路。
    """

    def __init__(
        self,
        patch_dim: int,
        spec: Optional[StateSpec] = None,
        n_layers_side: int = 6,
        n_layers_now: int = 2,
        hidden: int = 1024,
        n_heads: int = 8,
        ff: int = 2048,
        recon_dim: int = 64,
        n_patch: int = 64,        # pool_size^2，每帧的 patch 数
        action_dim: int = 7,
        n_action: int = 16,
        max_frames: int = 16,
        init_log_lam: float = 0.0,
    ) -> None:
        super().__init__()
        self.spec = spec or StateSpec()

        branch = lambda n, rec: TemporalBranch(  # noqa: E731
            patch_dim, self.spec, n_layers=n, hidden=hidden, n_heads=n_heads, ff=ff,
            recon_dim=recon_dim, n_patch=n_patch, action_dim=action_dim,
            n_action=n_action, init_log_lam=init_log_lam, with_recon=rec,
        )
        self.q_past = branch(n_layers_side, True)
        self.q_now = branch(n_layers_now, False)     # 重建当下对它是平凡的
        self.q_future = branch(n_layers_side, True)

        # 时序嵌入：每个 patch 加上所属帧的可学向量。索引用**相对偏移**
        # （t-4 -> 0, t -> 4, t+4 -> 8），这样同一个嵌入在不同 n_past 下
        # 含义一致；max_frames 留出余量以支持更长的时间线。
        self.frame_embed = nn.Parameter(torch.randn(max_frames, patch_dim) * 0.02)

        # 重建目标用固定随机投影降维：既保留空间结构（JL 引理保距），
        # 又避免"可学投影 + 重建"退化成把目标压成零的平凡解。
        self.register_buffer(
            "recon_proj", torch.randn(patch_dim, recon_dim) / patch_dim ** 0.5
        )
        self.recon_dim = recon_dim

        # 输出侧：小尺度初始化，不能为零。
        # temporal.py 那版把 out_proj 零初始化，导致输出恒零、梯度恒零，
        # 整个模块 32 个参数在线上跑 10000 步逐字节未变（见该文件注释）。
        # 零初始化只在残差分支上安全，这里是唯一出口。
        self.out_norm = nn.LayerNorm(self.spec.dim)
        self.out_proj = nn.Linear(self.spec.dim, self.spec.dim)
        nn.init.normal_(self.out_proj.weight, std=1e-3)
        nn.init.zeros_(self.out_proj.bias)

    def _split(self, patches: Tensor, n_frames: int, n_past: int):
        """按帧切成 past / now / future 三段，并加上时序嵌入。

        互斥性在这里保证：每一段的 patch 只会出现在一个分支的输入里。
        """
        B, TP, C = patches.shape
        if TP % n_frames != 0:
            raise ValueError(f"patch count {TP} is not divisible by frame count {n_frames}")
        P = TP // n_frames
        x = patches.view(B, n_frames, P, C)

        # 相对偏移索引：当下固定映射到 frame_embed 的中间位置，
        # 使嵌入含义不随 n_past 改变。
        centre = self.frame_embed.shape[0] // 2
        idx = torch.arange(n_frames, device=patches.device) - n_past + centre
        idx = idx.clamp(0, self.frame_embed.shape[0] - 1)
        x = x + self.frame_embed[idx].view(1, n_frames, 1, C)

        flat = lambda t: t.reshape(B, -1, C)                    # noqa: E731
        past = flat(x[:, :n_past]) if n_past > 0 else None
        now = flat(x[:, n_past:n_past + 1])
        future = flat(x[:, n_past + 1:]) if n_frames > n_past + 1 else None
        return past, now, future, x[:, n_past]                  # 末项为重建目标

    def _calib(self, branch: TemporalBranch, state: GaussianState, flat_mu: Tensor,
               now_patches: Tensor, actions: Optional[Tensor]) -> CalibAux:
        """用"自评误差"给 log_lam 提供直接监督。

        精度的定义是误差方差的倒数：Lambda = 1 / sigma^2，取对数即
            log Lambda = -log(sigma^2)
        故把 log_lam 的均值向 -log(误差) 对齐。误差一侧 detach ——
        否则模型会走捷径：把误差做大来满足校准项，而不是把 Lambda 学准。

        误差是逐样本的标量，故只约束 log_lam 在 token/dim 上的均值。
        融合权重取决于 Lambda 的整体量级，这一粒度已经决定了"谁说话算数"；
        逐 token 的精度留给动作损失去塑造。
        """
        eps = 1e-6
        # 变体 B：能否还原当下那一帧 —— 直接对应"我对当下有多清楚"。
        # 只对看不到当下的分支（past / future）成立。
        if branch.recon_head is not None:
            target = (now_patches @ self.recon_proj).flatten(1).detach()
            pred = branch.recon_head(flat_mu)
            err_recon = (pred - target).pow(2).mean(dim=1)                  # [B]
        else:
            err_recon = None

        # 变体 A：能否预测动作 —— 通用，但混入了策略的不确定性
        if actions is not None:
            a_tgt = actions[:, : branch.n_action]
            if a_tgt.shape[1] < branch.n_action:                            # 动作不足则跳过
                err_act = None
            else:
                a_pred = branch.act_head(flat_mu).view(
                    -1, branch.n_action, branch.action_dim
                )
                err_act = (a_pred - a_tgt).pow(2).mean(dim=(1, 2))          # [B]
        else:
            err_act = None

        lam_mean = state.log_lam.mean(dim=(1, 2))                           # [B]

        def align(err: Tensor) -> Tensor:
            tgt = -(err.detach() + eps).log()
            tgt = tgt.clamp(self.spec.log_lam_min, self.spec.log_lam_max)
            return (lam_mean - tgt).pow(2).mean()

        # 重建/动作误差本身也要被优化 —— 否则校准头学不出有意义的预测，
        # 误差就只是噪声，Lambda 的目标也就无意义。
        recon = (align(err_recon) + err_recon.mean()) if err_recon is not None \
            else torch.zeros((), device=flat_mu.device, dtype=flat_mu.dtype)
        act = (align(err_act) + err_act.mean()) if err_act is not None \
            else torch.zeros((), device=flat_mu.device, dtype=flat_mu.dtype)
        return CalibAux(recon=recon, act=act)

    def forward(
        self,
        patches: Tensor,
        actions: Optional[Tensor] = None,
        *,
        n_frames: int,
        n_past: int,
        use_future: bool = True,
        vae_latent: Optional[Tensor] = None,   # 接口对齐，本版不使用
    ) -> Tuple[Tensor, CalibAux]:
        """
        Args:
            patches:  [B, n_frames*P, C]  按时间顺序排列的 patch
            actions:  [B, L, A]  用于变体 A 的校准；None 则只用变体 B
            n_past:   当下帧在时间线上的下标
            use_future: False 时（Stage II / 部署）不构建 beta。
                        观测在 vla 侧已物理截断，这里 future 段本就为空；
                        Lambda_beta 缺席后融合公式自动退化为 alpha + ell。

        Returns:
            s*_t [B, n_tokens, dim]，以及校准辅助损失
        """
        past, now, future, now_patches = self._split(patches, n_frames, n_past)

        parts, auxes = [], []

        ell, flat_ell = self.q_now(now)
        parts.append(ell)
        auxes.append(self._calib(self.q_now, ell, flat_ell, now_patches, actions))

        if past is not None:
            alpha, flat_a = self.q_past(past)
            parts.append(alpha)
            auxes.append(self._calib(self.q_past, alpha, flat_a, now_patches, actions))

        if use_future and future is not None:
            beta, flat_b = self.q_future(future)
            parts.append(beta)
            auxes.append(self._calib(self.q_future, beta, flat_b, now_patches, actions))

        fused = fuse(*parts)
        cond = self.out_proj(self.out_norm(fused.mu))

        if auxes:
            aux = CalibAux(
                recon=torch.stack([a.recon for a in auxes]).mean(),
                act=torch.stack([a.act for a in auxes]).mean(),
            )
        else:
            aux = CalibAux.zeros(cond)
        return cond, aux
