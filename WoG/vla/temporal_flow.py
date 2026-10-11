"""[zgt] 双向时间线 —— WhenAmI v2 的核心。

设计依据见 DESIGN_v2.md。与前两版的根本差别：

    v1 递推版   alpha/beta 各推各的，反向把同一个 T 反着用（数学不严格），
                Lambda 靠 log_decay 人为规定衰减 —— 实测 std 仅 2.4e-4，没学动。
    v2 masked   三路互斥 cross-attention，但三者互不知情，只在最后加权平均。
                那是三份快照拼起来，不是一条时间线。
    本模块      同一个算子 T 在时间轴上前后滑动，方向作为条件。
                过去推未来、未来推过去，互为监督。

Lambda 的来源是关键改变：

    v1/v2:  一个 to_log_lam 头去「猜」自己有多确定，只能从动作损失间接学。
            实测（masked_v2 step 5000）跨样本 std 仅 0.14% —— 学不出自适应。
    本模块: Lambda = 1 / 推演误差^2。误差是数据直接给的客观量，不需要猜。

对齐空间的选择有实验依据（exp_align_choice.py）：
    cos_proj  方差比 28.4，MSE 比 186537  —— 余弦只管方向不管长度，尺度失控
    l2_proj   方差比 0.98，MSE 比 1.31    —— 没有塌缩，选它
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from vla.temporal import GaussianState, StateSpec


@dataclass
class FlowAux:
    """双向推演产生的辅助量。

    fwd / bwd：两个方向的对齐损失，互为监督。
    cycle：    往返一致性 T(-)(T(+)(s)) ≈ s —— 检验「方向作为条件」
               学出来的可逆性是不是真的。方向 token 不保证严格可逆
               （那是 coupling layer 的性质），故需要显式检验。
    """

    fwd: Tensor
    bwd: Tensor
    cycle: Tensor

    @classmethod
    def zeros(cls, like: Tensor) -> "FlowAux":
        z = torch.zeros((), device=like.device, dtype=like.dtype)
        return cls(fwd=z, bwd=z, cycle=z)


def _dt_encode(dt: Tensor, t_max: float = 16.0) -> Tensor:
    """把时间跨度编成条件向量：[归一化值, sin, cos]。

    归一化值利于外推到训练未见的跨度；周期编码让插值更稳
    （与 DiT 的 timestep 编码同一套思路）。

    OXE 里节点间隔是 3~4 步而非均匀 1 步，故 dt 必须显式给 ——
    v1 用 `log_lam - softplus(log_decay)` 人为规定精度单调衰减，
    实测 log_decay 根本没学动（std 2.4e-4）。
    """
    w = dt / t_max
    return torch.stack([w, torch.sin(w * np.pi), torch.cos(w * np.pi)], dim=-1)


class FlowOperator(nn.Module):
    """转移算子 T(s, a, dt, dir) —— 时间线上的唯一规则。

    正推与逆推共用同一套参数，方向作为条件。这是「统一的、能相互流动
    的东西」的字面实现：有未来就用它反推确认当下，没未来就用同一个算子
    正推出来 —— Stage 3 因此不是新机制。

    为什么不用严格可逆的 coupling layer：
        文献警告高表达力 INN 在 rollout 上数值不稳（多模态数据尤甚），
        而我们要连推 4 步。方向 token 保留「同一套参数」这个核心，
        不受可逆性对容量的限制；代价是可逆性需要 cycle loss 检验。

    残差形式 + 末层零初始化 = 恒等起步。这里零初始化是安全的 ——
    它在残差分支上（s + net(...)），不像 v1 的 out_proj 那样处在唯一
    出口，不会把整条路的梯度掐死。
    """

    def __init__(self, dim: int, action_dim: int = 7, hidden: int = 512,
                 dir_dim: int = 32) -> None:
        super().__init__()
        self.act_embed = nn.Linear(action_dim, 64)
        # 方向嵌入可学：让模型自己决定「往前」和「往后」在表示上差什么
        self.dir_embed = nn.Parameter(torch.randn(2, dir_dim) * 0.02)
        self.net = nn.Sequential(
            nn.Linear(dim + 64 + 3 + dir_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s: Tensor, a: Tensor, dt: Tensor, forward: bool) -> Tensor:
        """推进一步。

        Args:
            s: [B, N, D] 状态（N 个 token）
            a: [B, action_dim] 这一步的等效动作
            dt: [B] 跨度（步数）
            forward: True 顺推，False 逆推

        Returns:
            推进后的状态，形状不变
        """
        B, N, D = s.shape
        a_e = self.act_embed(a).unsqueeze(1).expand(B, N, -1)
        dt_e = _dt_encode(dt).unsqueeze(1).expand(B, N, 3)
        d_e = self.dir_embed[0 if forward else 1].view(1, 1, -1).expand(B, N, -1)
        return s + self.net(torch.cat([s, a_e, dt_e, d_e], dim=-1))


class _QueryBlock(nn.Module):
    """query 端的完整 Transformer block：self-attn -> cross-attn -> FFN。

    早先只有 cross-attn，缺 FFN 与 query 之间的 self-attn —— 那让编码
    本质停留在「注意力加权求和」，是线性的。WoG 的 QFormer 每层三者都有
    （BertLayer: self + cross + intermediate/output），FFN 1024→2048→1024
    是它 109.3M 参数里的大头，也是非线性表达力的主要来源。

    实测计算代价可忽略：条件模块占总步时 0.1~1.0%，9 帧视觉编码占 99%。
    所以补齐容量几乎不花钱，没有理由省。
    """

    def __init__(self, hidden: int, n_heads: int, ffn_mult: int = 4) -> None:
        super().__init__()
        self.n1 = nn.LayerNorm(hidden)
        self.self_attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.n2 = nn.LayerNorm(hidden)
        self.cross_attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.n3 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * ffn_mult), nn.GELU(),
            nn.Linear(hidden * ffn_mult, hidden),
        )

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        h = self.n1(q)
        q = q + self.self_attn(h, h, h)[0]          # query 之间协调分工
        q = q + self.cross_attn(self.n2(q), kv, kv)[0]
        return q + self.ffn(self.n3(q))


class _FrameMixBlock(nn.Module):
    """帧间 block：self-attn -> FFN，支持因果掩码。

    这一层做的正是「把 9 帧放在一条时间线上看」，与 QFormer 的设计目标
    （综合多个 token 产出摘要）一致，故给它完整的 block 而非裸 attention。

    掩码是必须的：不加时部署路径的 student 会通过帧间注意力看到未来，
    实测泄漏响应 1.3e-2。
    """

    def __init__(self, hidden: int, n_heads: int, ffn_mult: int = 4) -> None:
        super().__init__()
        self.n1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.n2 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * ffn_mult), nn.GELU(),
            nn.Linear(hidden * ffn_mult, hidden),
        )

    def forward(self, q: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        h = self.n1(q)
        q = q + self.attn(h, h, h, attn_mask=mask)[0]
        return q + self.ffn(self.n2(q))


class FrameEncoder(nn.Module):
    """把 9 帧 patch 编码成 9 个状态，帧之间可以互相看见。

    早先的版本逐帧独立编码（`[self.encoder(x[:,k]) for k in range(T)]`），
    信息在进入时间线之前就被压到 [16,32]，帧间交互只剩 T 的逐步递推。
    这与 WoG 差得很远 —— 它把 5 帧的 patch 拍平成 320 个 token 一起送进
    cross-attention，每个 query 能同时看到所有帧的所有 patch。实测条件
    质量的差距（r 0.563 vs 0.302）很可能主要来自这里，而不是融合机制。

    故这里分两段：
      1. 每帧自己的 query 先做 cross-attention 抽取本帧内容
      2. 再让 9 帧的 query 之间做一次 self-attention —— 帧间交互

    "互为监督"要求的是 alpha / beta 基于不重叠的**观测**做推演，
    不是要求编码阶段彼此失明。交互发生在编码，推演仍然分方向。

    vae_latent 一并接回：它是 Wan2.1 视频 VAE 对 [当下+后续] 整段编码出的
    时空隐变量 [B, 32, 784]，是「接下来怎么动」的直接线索。WoG 把它拼进
    QFormer 的 cross-attention 输入，我们此前三版全都丢了。
    """

    def __init__(self, patch_dim: int, spec: StateSpec, n_heads: int = 8,
                 n_layers: int = 3, vae_dim: int = 784, n_cross_layers: int = 2,
                 hidden: int = 768, use_ffn: bool = True) -> None:
        super().__init__()
        self.spec = spec
        self.use_ffn = use_ffn
        self.in_proj = nn.Linear(patch_dim, hidden)
        self.vae_proj = nn.Linear(vae_dim, hidden)
        self.query = nn.Parameter(torch.randn(1, spec.n_tokens, hidden) * 0.02)

        if use_ffn:
            # 帧内：抽取「这一帧是什么情况」。任务比 WoG 的「综合 5 帧产出
            # 摘要」简单，故层数少于它的 6 层，但每层配置完整
            # （self-attn + cross-attn + FFN）。
            self.blocks = nn.ModuleList([
                _QueryBlock(hidden, n_heads) for _ in range(n_layers)
            ])
            # 帧间：把 9 帧放到一条时间线上协调
            self.cross_frame = nn.ModuleList([
                _FrameMixBlock(hidden, n_heads) for _ in range(n_cross_layers)
            ])
        else:
            # 裸 attention 路径：无 FFN、无 query self-attn。保留它有两个
            # 理由 —— (1) flow_v2 是用这个结构训的，键名必须能对上，
            # 否则那个 checkpoint 直接作废；(2) 它是「FFN 值多少」的消融组。
            self.blocks = nn.ModuleList([
                nn.MultiheadAttention(hidden, n_heads, batch_first=True)
                for _ in range(n_layers)
            ])
            self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
            self.cross_frame = nn.ModuleList([
                nn.MultiheadAttention(hidden, n_heads, batch_first=True)
                for _ in range(n_cross_layers)
            ])
            self.cross_norms = nn.ModuleList(
                [nn.LayerNorm(hidden) for _ in range(n_cross_layers)]
            )

        self.to_mu = nn.Linear(hidden, spec.dim)
        self.to_log_lam = nn.Linear(hidden, spec.dim)
        # 零初始化 + 常数偏置：训练初期各帧精度相同，让「谁更可信」由
        # 梯度学出而非随机初始化决定。安全 —— to_log_lam 与 to_mu 并联，
        # 不在 mu 的通路上。
        nn.init.zeros_(self.to_log_lam.weight)
        nn.init.zeros_(self.to_log_lam.bias)

    def forward(self, frames: Tensor, vae_latent: Optional[Tensor] = None,
                causal_upto: Optional[int] = None) -> GaussianState:
        """
        Args:
            frames: [B, T, P, C] 全部帧的 patch（已含 frame_embed）
            vae_latent: [B, L, vae_dim] 整段视频的 VAE 隐变量，可选
            causal_upto: 不为 None 时，第 k 帧只能看到 <= causal_upto 的帧。
                         部署路径（use_future=False）必须给它 —— 否则跨帧
                         self-attention 会让过去帧看到未来，那是泄漏。
                         实测不加时 student 输出对未来帧的扰动响应 1.3e-2。

        Returns:
            GaussianState，mu / log_lam 均为 [B, T, N, D]
        """
        B, T, P, C = frames.shape
        x = self.in_proj(frames.contiguous().view(B * T, P, C))
        if vae_latent is not None:
            # VAE 是整段一份，广播到每帧后拼进 cross-attention 的 K/V
            v = self.vae_proj(vae_latent)                       # [B, L, hidden]
            v = v.unsqueeze(1).expand(B, T, *v.shape[1:]).contiguous().view(B * T, *v.shape[1:])
            x = torch.cat([x, v], dim=1)

        q = self.query.expand(B * T, -1, -1)
        if self.use_ffn:
            for blk in self.blocks:
                q = blk(q, x)
        else:
            for blk, norm in zip(self.blocks, self.norms):
                q = q + blk(norm(q), x, x)[0]

        # 帧间交互：把 T 帧的 query 拼成一条序列做 self-attention
        q = q.contiguous().view(B, T * self.spec.n_tokens, -1)
        mask = None
        if causal_upto is not None:
            # 屏蔽 causal_upto 之后的帧：token i 属于帧 i // n_tokens
            N = self.spec.n_tokens
            fid = torch.arange(T * N, device=frames.device) // N
            mask = (fid.unsqueeze(0) > causal_upto).expand(T * N, T * N)
        if self.use_ffn:
            for blk in self.cross_frame:
                q = blk(q, mask)
        else:
            for blk, norm in zip(self.cross_frame, self.cross_norms):
                h = norm(q)
                q = q + blk(h, h, h, attn_mask=mask)[0]
        q = q.contiguous().view(B, T, self.spec.n_tokens, -1)

        return GaussianState(
            mu=self.to_mu(q),
            log_lam=self.to_log_lam(q).clamp(self.spec.log_lam_min, self.spec.log_lam_max),
        )


class BidirFlow(nn.Module):
    """双向时间线：past -> now <- future，同一个 T 前后滑动。

    训练时两个方向都有真实数据，互为监督；部署时只有过去，用同一个
    T 顺推出未来。「撤掉未来」不是一个训练阶段，而是推理时的自然状态。

    Lambda 来自推演误差而非某个头的自我评价：
        Lambda_alpha = 1 / (顺推到当下的误差)^2
        Lambda_beta  = 1 / (逆推到当下的误差)^2
    误差是数据给的客观量。v1/v2 让 to_log_lam 去猜，实测跨样本 std
    只有 0.14% —— 猜不出来。
    """

    def __init__(
        self,
        patch_dim: int,
        spec: Optional[StateSpec] = None,
        action_dim: int = 7,
        n_enc_layers: int = 3,
        n_cross_layers: int = 2,
        enc_hidden: int = 768,
        enc_use_ffn: bool = True,
        flow_hidden: int = 512,
        readout_dim: int = 64,
        max_frames: int = 16,
    ) -> None:
        super().__init__()
        self.spec = spec or StateSpec()
        self.encoder = FrameEncoder(patch_dim, self.spec, n_layers=n_enc_layers,
                                    n_cross_layers=n_cross_layers, hidden=enc_hidden,
                                    use_ffn=enc_use_ffn)
        self.flow = FlowOperator(self.spec.dim, action_dim, flow_hidden)

        # 时序嵌入服务编码器（这帧来自何时），dt/dir 服务转移算子
        # （这步跨多远、朝哪边）。两者作用对象不同，不是冗余：
        # 没有 frame_embed，编码器分不清 9 帧先后（WoG 正栽在这）。
        self.frame_embed = nn.Parameter(torch.randn(max_frames, patch_dim) * 0.02)

        # 对齐用的读出头：状态 -> 投影后的 patch 空间。
        # 逐 token 读出 —— 每个 token 预测它负责的那部分，误差与 Lambda
        # 天然对应，不需要人为构造「哪个 patch 归哪个 token」的映射。
        # 实测（masked_v2 step 5000）16 个 query 两两余弦均值 -0.005，
        # 几乎正交，确实各司其职。
        self.readout = nn.Linear(self.spec.dim, readout_dim)
        # 固定随机投影：保距（JL 引理）且不可学，避免「可学投影 + 重建」
        # 退化成把目标压成零的平凡解。
        self.register_buffer(
            "align_proj", torch.randn(patch_dim, readout_dim) / patch_dim ** 0.5
        )
        self.readout_dim = readout_dim

        # 输出侧：小尺度初始化，不能为零。
        # v1 把这里零初始化，导致输出恒零、梯度恒零，32 个参数线上跑
        # 10000 步逐字节未变。零初始化只在残差分支上安全，这里是唯一出口。
        self.out_norm = nn.LayerNorm(self.spec.dim)
        self.out_proj = nn.Linear(self.spec.dim, self.spec.dim)
        nn.init.normal_(self.out_proj.weight, std=1e-3)
        nn.init.zeros_(self.out_proj.bias)

    # ---------------- 内部工具 ----------------

    def _split_frames(self, patches: Tensor, n_frames: int, n_past: int):
        """[B, T*P, C] -> [B, T, P, C]，并加上时序嵌入。

        索引用相对偏移（当下恒映射到 frame_embed 的中心），
        这样同一个嵌入在不同 n_past 下含义一致。
        """
        B, TP, C = patches.shape
        if TP % n_frames != 0:
            raise ValueError(
                f"patch count {TP} is not divisible by frame count {n_frames}"
            )
        P = TP // n_frames
        x = patches.view(B, n_frames, P, C)
        centre = self.frame_embed.shape[0] // 2
        idx = (torch.arange(n_frames, device=patches.device) - n_past + centre)
        idx = idx.clamp(0, self.frame_embed.shape[0] - 1)
        return x + self.frame_embed[idx].view(1, n_frames, 1, C)

    def _align(self, state: Tensor, target_patches: Tensor) -> Tensor:
        """对齐损失：逐 token 读出 vs 投影后的 patch，取 L2。

        用 L2 而非余弦有实验依据：余弦只管方向不管长度，实测让输出
        尺度爆到目标的 28 倍（方差比 28.4，MSE 比 186537）。
        L2 + 固定投影的方差比是 0.98 —— 没有塌缩。

        返回逐样本的误差 [B]，供 Lambda 使用。
        """
        tgt = (target_patches @ self.align_proj).mean(1)          # [B, R]
        pred = self.readout(state).mean(1)                        # [B, R]
        return (pred - tgt).pow(2).mean(dim=-1)                   # [B]

    def _align_per_token(self, state: Tensor, target_patches: Tensor) -> Tensor:
        """逐 token 的对齐误差 [B, N] —— 逐 token Lambda 的来源。"""
        tgt = (target_patches @ self.align_proj).mean(1, keepdim=True)   # [B,1,R]
        pred = self.readout(state)                                       # [B,N,R]
        return (pred - tgt).pow(2).mean(dim=-1)                          # [B, N]

    def _roll(self, s0: Tensor, actions: Tensor, dts: Tensor, forward: bool,
              targets: Optional[Tensor] = None):
        """沿一个方向推演，返回途经的所有状态与逐步对齐误差。

        Args:
            s0:      [B, N, D] 起点状态
            actions: [B, S, A] 每一步的等效动作（已按方向排好序）
            dts:     [B, S]    每一步的跨度
            targets: [B, S, P, C] 每一步该落在的真实帧（None 则不算误差）
        """
        states, errs = [], []
        s = s0
        for k in range(actions.shape[1]):
            s = self.flow(s, actions[:, k], dts[:, k], forward)
            states.append(s)
            if targets is not None:
                errs.append(self._align_per_token(s, targets[:, k]))
        return states, (torch.stack(errs, 1) if errs else None)

    # ---------------- 前向 ----------------

    def forward(
        self,
        patches: Tensor,
        actions: Tensor,
        *,
        n_frames: int,
        n_past: int,
        use_future: bool = True,
        vae_latent: Optional[Tensor] = None,
    ) -> Tuple[Tensor, FlowAux]:
        """
        Args:
            patches: [B, n_frames*P, C] 按时间顺序的 patch
            actions: [B, n_frames-1, A] 相邻帧之间的等效动作
                     （delta EEF 在区间内求和即合成位移；gripper 取末值）
            n_past:  当下帧在时间线上的下标
            use_future: False 时（部署）不做逆推，未来一路缺席，
                        融合自动退化为 alpha + ell
            vae_latent: [B, L, 784] Wan2.1 视频 VAE 对整段的编码。
                        WoG 把它拼进 QFormer 的 cross-attention 输入，
                        我们此前三版都丢了 —— 那是「接下来怎么动」的
                        直接线索。student 路径下由调用方置 None（它由
                        含未来帧的整段视频编码而来，留着就是泄漏）。

        Returns:
            s*_t [B, N, D]，以及双向推演的辅助损失
        """
        x = self._split_frames(patches, n_frames, n_past)          # [B,T,P,C]
        B, T, P, C = x.shape
        eps = 1e-6

        # 一次编码全部帧：帧内 cross-attention + 帧间 self-attention。
        # 早先逐帧独立编码，信息在进入时间线前就被压到 [16,32]，
        # 帧间只剩 T 的递推 —— 那是条件质量落后 WoG 的主要嫌疑。
        # 部署路径不能让跨帧注意力看到未来 —— 否则 use_future=False
        # 只是名义上的，编码阶段已经泄漏了。
        st = self.encoder(x, vae_latent,
                          causal_upto=None if use_future else n_past)
        obs = [GaussianState(mu=st.mu[:, k], log_lam=st.log_lam[:, k])
               for k in range(T)]
        ell = obs[n_past]

        # OXE 的节点间隔恒为 4（相邻采样点的 tstep 差），非均匀时
        # 这里应改为按 obs_offsets 计算。
        dt = torch.full((B, max(T - 1, 1)), 4.0, device=patches.device)

        # --- 顺推：从最早一帧走到最后一帧，落点对齐真实帧 ---
        fwd_states, fwd_err = self._roll(
            obs[0].mu, actions[:, : T - 1], dt, True, x[:, 1:]
        )
        # 顺推到当下的那一步（下标 n_past-1，因为 states[0] 对应第 1 帧）
        h_t = fwd_states[n_past - 1] if n_past > 0 else obs[0].mu
        err_a = fwd_err[:, n_past - 1] if (fwd_err is not None and n_past > 0) \
            else torch.full((B, self.spec.n_tokens), eps, device=patches.device)

        parts = [
            (h_t, (1.0 / (err_a + eps)).log()),
            (ell.mu, ell.log_lam.mean(-1)),
        ]

        bwd_loss = torch.zeros((), device=patches.device)
        cyc = torch.zeros((), device=patches.device)

        if use_future and T > n_past + 1:
            # --- 逆推：从最后一帧走回最早一帧 ---
            rev_actions = actions[:, : T - 1].flip(1)
            rev_targets = x[:, :-1].flip(1)
            bwd_states, bwd_err = self._roll(
                obs[T - 1].mu, rev_actions, dt, False, rev_targets
            )
            # 逆推序列是倒序的：第 j 步落在原下标 T-2-j
            j_now = T - 2 - n_past
            g_t = bwd_states[j_now]
            err_b = bwd_err[:, j_now]
            parts.append((g_t, (1.0 / (err_b + eps)).log()))
            bwd_loss = bwd_err.mean()

            # --- 往返一致：顺推一步再逆推回来，应当回到原处 ---
            # 方向 token 不保证严格可逆，这一项是显式检验。
            s0 = obs[n_past].mu
            s1 = self.flow(s0, actions[:, n_past], dt[:, 0], True)
            s_back = self.flow(s1, actions[:, n_past], dt[:, 0], False)
            cyc = (s_back - s0).pow(2).mean()

        # --- 精度加权融合（逐 token）---
        lam = torch.stack([p[1] for p in parts], 0).exp()          # [n_parts, B, N]
        mu = torch.stack([p[0] for p in parts], 0)                 # [n_parts, B, N, D]
        w = (lam / lam.sum(0, keepdim=True).clamp_min(eps)).unsqueeze(-1)
        fused = (w * mu).sum(0)                                    # [B, N, D]

        cond = self.out_proj(self.out_norm(fused))
        aux = FlowAux(
            fwd=fwd_err.mean() if fwd_err is not None else torch.zeros((), device=patches.device),
            bwd=bwd_loss,
            cycle=cyc,
        )
        return cond, aux
