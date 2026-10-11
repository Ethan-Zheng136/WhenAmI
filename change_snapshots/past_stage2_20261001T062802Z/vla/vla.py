from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union, Tuple
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import LlamaTokenizerFast
from .qformer import QFormer
from .temporal import TemporalConverge, StateSpec   # ===zgt===
from .temporal_masked import MaskedConverge         # ===zgt===
from .temporal_flow import BidirFlow                # ===zgt===

from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import VisionBackbone
from prismatic.models.vlms.base_vlm import VLM
from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector
from prismatic.models.backbones.vision.dinosiglip_vit import FutureEnc

from action_model.action_model import ActionModel
from action_model.models import DiT

overwatch = initialize_overwatch(__name__)
IGNORE_INDEX = -100


class WoG(nn.Module):
    def __init__(
        self,
        vlm: PrismaticVLM,
        action_model_type: str = "DiT-B",
        token_size: int = 4096,
        action_dim: int = 7,
        Ta: int = 16,
        scale: int = 4,  # for key frames
        use_ema: bool = False,
        pool_size=8,
        train_venc=False,
        use_temporal: bool = False,   # ===zgt=== 见 __init__ 内条件模块二选一
        converge_type: str = "recurrent",  # ===zgt=== recurrent | masked
        obs_offsets=None,             # ===zgt=== 节点时间偏移表，用于聚合转移动作
        distill_weight: float = 0.0,  # ===zgt=== >0 开启 Stage 2 双路径蒸馏
        nodes_per_step: int = 0,      # ===zgt=== <=0 全采样；实测降 per_device_batch 比降节点更省显存
        calib_recon_weight: float = 0.0,  # ===zgt=== masked 版 Lambda 校准（重建，变体 B）
        calib_act_weight: float = 0.0,    # ===zgt=== masked 版 Lambda 校准（动作，变体 A）
        flow_fwd_weight: float = 1.0,     # ===zgt=== flow 版：顺推对齐
        flow_bwd_weight: float = 1.0,     # ===zgt=== flow 版：逆推对齐
        flow_cycle_weight: float = 0.1,   # ===zgt=== flow 版：往返一致
        flow_enc_layers: int = 3,         # ===zgt=== 帧内编码层数
        flow_cross_layers: int = 2,       # ===zgt=== 帧间交互层数
        flow_enc_hidden: int = 768,       # ===zgt=== 编码器宽度
        flow_enc_use_ffn: bool = True,    # ===zgt=== False 复现 flow_v2 的裸 attention
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.action_model = ActionModel(
            model_type=action_model_type,
            token_size=token_size,
            in_channels=action_dim,
            Ta=16,
            train_venc=train_venc,
        )
        self.vlm = vlm
        self.Ta = Ta
        self.scale = scale
        self.train_venc = train_venc
        self.pool_size = pool_size

        self.v_enc = FutureEnc()
        self.v_enc.eval()
        self.v_enc.requires_grad_(False)

        # ===zgt===
        # 两种条件模块二选一，由 use_temporal 切换：
        #   False -> QFormer      原版 WoG，输出一个无时间维的摘要 [B, 16, 32]
        #   True  -> TemporalConverge  输出一条状态时间线 [B, K, 16, 32]
        # 保留 QFormer 分支不是为了兼容，而是为了 baseline 对比能跑在同一份
        # 代码上 —— 否则"我们赢了"里会混进实现差异这个变量。
        self.use_temporal = use_temporal
        self.obs_offsets = list(obs_offsets) if obs_offsets else None   # ===zgt===
        self.distill_weight = distill_weight                             # ===zgt===
        self.nodes_per_step = nodes_per_step                             # ===zgt===
        # ===zgt===
        # converge_type 选实现：
        #   recurrent —— 第一版，alpha/beta 沿时间递推后逐时刻融合，
        #                输出 9 个节点，DiT 跑 9 次。probe（步 10000）显示
        #                beta 在 gt_togo / gt_grip 上是负贡献，根因见
        #                temporal_masked.py 顶部注释。
        #   masked    —— 第二版，三路互斥 cross-attention + Lambda 直接监督，
        #                只输出当下一个状态，DiT 跑 1 次（与 WoG 同成本）。
        self.converge_type = converge_type
        self.calib_recon_weight = calib_recon_weight
        self.calib_act_weight = calib_act_weight
        self.flow_fwd_weight = flow_fwd_weight
        self.flow_bwd_weight = flow_bwd_weight
        self.flow_cycle_weight = flow_cycle_weight
        self.flow_enc_layers = flow_enc_layers
        self.flow_cross_layers = flow_cross_layers
        self.flow_enc_hidden = flow_enc_hidden
        self.flow_enc_use_ffn = flow_enc_use_ffn
        if use_temporal and converge_type not in ("recurrent", "masked", "flow"):
            raise ValueError(
                f"converge_type must be 'recurrent' | 'masked' | 'flow', "
                f"got {converge_type!r}"
            )
        # ===zgt===
        if use_temporal:
            if converge_type == "flow":
                # 第三版：同一个算子在时间轴上前后滑动，Lambda 来自推演误差。
                # 前两版让 to_log_lam 去猜自己有多确定，实测跨样本 std 仅 0.14%。
                # enc_* 控制帧编码器的容量。WoG 的 QFormer 是 6 层 h=1024
                # 共 109.3M；我们默认 3+2 层 h=768，因为帧内任务（看懂单帧）
                # 比它的「综合 5 帧产出摘要」简单。实测条件模块只占总步时
                # 0.1~1.0%，补容量几乎不花钱。
                self.converge = BidirFlow(
                    patch_dim=1024, spec=StateSpec(), action_dim=action_dim,
                    n_enc_layers=flow_enc_layers, n_cross_layers=flow_cross_layers,
                    enc_hidden=flow_enc_hidden, enc_use_ffn=flow_enc_use_ffn,
                )
            elif converge_type == "masked":
                self.converge = MaskedConverge(
                    patch_dim=1024, spec=StateSpec(), action_dim=action_dim, n_action=Ta
                )
            else:
                self.converge = TemporalConverge(
                    StateSpec(), patch_dim=1024, action_dim=action_dim
                )
            self.qformer = None
        else:
            self.converge = None
            self.qformer = QFormer(
                hidden_dim=1024, nheads=8, dim_feedforward=2048, num_layers=6, num_query_tokens=16
            )
        cond_key = "converge" if use_temporal else "qformer"
        # ===zgt===

        if not self.train_venc:
            print(f"Freeze conditioning module: {cond_key}")
            cond_module = getattr(self, cond_key)
            cond_module.eval()
            cond_module.requires_grad_(False)
            self.oformer = QFormer(
                hidden_dim=token_size, nheads=8, dim_feedforward=2048, num_layers=4, num_query_tokens=16
            )

        self.use_ema = use_ema
        if self.use_ema:
            self.ema_diffusion = deepcopy(self.action_model)
            self.ema_diffusion.requires_grad_(False)
            self.all_module_keys = ["action_model", "ema_diffusion", cond_key]   # ===zgt===
        else:
            self.all_module_keys = ["action_model", cond_key]                     # ===zgt===
        for module_keys in self.vlm.all_module_keys:
            self.all_module_keys.append("vlm." + module_keys)
        self._trainable_module_keys = ["action_model", cond_key]                  # ===zgt===
        self.norm_stats = norm_stats

    @property
    def trainable_module_keys(self) -> List[str]:
        keys = []
        for module_keys in self.vlm.trainable_module_keys:
            keys.append("vlm." + module_keys)
        keys += self._trainable_module_keys
        return keys

    @property
    def llm_backbone(self) -> LLMBackbone:
        return self.vlm.llm_backbone

    @property
    def vision_backbone(self) -> VisionBackbone:
        return self.vlm.vision_backbone

    def freeze_backbones(self, stage):
        self.vlm.freeze_backbones(stage)

    # ===zgt===
    def _encode_frames(self, frame_pixel_values, base_pixel_values):
        """把一组时序帧编码为 patch token 序列。

        原本这段内联在 forward 里、只服务于 future 帧；past 帧需要走完全相同的
        编码路径（同一个 v_enc、同一套 spatial pooling），故抽成方法共用。

        Args:
            frame_pixel_values: {modality: [B, T, C, H, W]} 待编码的时序帧
            base_pixel_values:  {modality: [B, C, H, W]}    当前帧，作为 VAE 的时序起点

        Returns:
            patch_features: [B, T * pool_size**2, D]
            vae_latent:     VAE 的时序隐变量
        """
        B, T = next(iter(frame_pixel_values.values())).shape[:2]
        flat = {k: v.view(B * T, *v.shape[2:]) for k, v in frame_pixel_values.items()}  # [B*T, C, H, W]

        patch_features, vae_latent = self.v_enc(flat, dict(base_pixel_values))

        _BT, N, D = patch_features.shape
        H = W = int(N**0.5)
        x = patch_features.view(B * T, H, W, D).permute(0, 3, 1, 2)
        x = F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size))
        x = x.permute(0, 2, 3, 1).contiguous().view(B, T, self.pool_size * self.pool_size, D)
        return x.view(B, T * (self.pool_size**2), -1).contiguous(), vae_latent

    def _compute_condition(self, pixel_values, future_pixel_values,
                           past_pixel_values, actions):
        """产生注入 DiT 的条件 token。

        Returns:
            cond_now:   [B, n_tokens, 32]        当下节点，注入 DiT
            all_nodes:  [B, K, n_tokens, 32] 或 None
                        完整时间线；QFormer 路径无时间维，返回 None
            n_past:     当下节点在时间线上的下标

        两条路径：

        QFormer（原版 WoG）
            把 future 帧拍平成一条序列，一次性问出一个摘要。时间被碾平，
            没有"哪个数对应哪个时刻"这回事，故 all_nodes 为 None。

        TemporalConverge（本工作）
            past / now / future 拼成完整时间线，双向消息传播后得到
            s*_k —— 每个时刻一个状态。注入 DiT 的是**当下**那一个节点：
            动作从当下出发，取用的自然也是当下的状态。其余节点交给
            多节点 action loss —— 每个节点都要能生成"从我这里出发"的动作。

        当前帧在时间线上的位置由 past 帧数决定，而非写死 —— 与
        RLDSBatchTransform 里 FrameLayout 的推导方式保持一致。
        """
        if not self.use_temporal:
            feats, vae_latent = self._encode_frames(future_pixel_values, pixel_values)
            return self.qformer(feats, vae_latent), None, 0, None

        # 时间线 = past + now + future，按时间顺序拼接
        now_feats, vae_latent = self._encode_frames(
            {k: v.unsqueeze(1) for k, v in pixel_values.items()}, pixel_values
        )
        future_feats, _ = self._encode_frames(future_pixel_values, pixel_values)

        if past_pixel_values:
            past_feats, _ = self._encode_frames(past_pixel_values, pixel_values)
            n_past = next(iter(past_pixel_values.values())).shape[1]
            patch_seq = torch.cat([past_feats, now_feats, future_feats], dim=1)
        else:
            n_past = 0
            patch_seq = torch.cat([now_feats, future_feats], dim=1)

        n_future = next(iter(future_pixel_values.values())).shape[1]
        n_frames = n_past + 1 + n_future

        # 相邻节点之间的驱动动作。
        # 节点未必相邻一步 —— WoG 按动作窗口 1/4 采样（偏移 [0,3,7,11,15]，
        # 间隔 3~4 步），我们镜像该策略后 past 侧同样是 3~4 步间隔。
        # 一个动作驱动不了 4 步转移，故把区间内的动作**求和**作为等效位移
        # （动作是增量式的 delta EEF，相加即合成位移；gripper 维取末值）。
        step_actions = self._aggregate_step_actions(actions, self.obs_offsets, n_past)

        # ===zgt===
        # masked 版只产出当下一个状态，故 all_nodes 返回 None —— forward 会
        # 因此走单点动作损失（与 WoG 同一条路径，DiT 只跑一次）。
        # 校准损失随返回值带出，而不是挂在 self 上：后者是隐式状态，
        # 在梯度累积或多次 forward 时容易读到上一轮的值。
        if self.converge_type == "flow":
            # 双向推演：顺推 + 逆推互为监督，Lambda 从推演误差导出。
            # step_actions 已把区间内的动作求和成等效位移（见
            # _aggregate_step_actions），正是 T 需要的「这一步的驱动」。
            #
            # vae_latent 一并传入 —— Wan2.1 视频 VAE 对 [当下+后续] 整段
            # 的编码，是「接下来怎么动」的直接线索。WoG 把它拼进 QFormer
            # 的输入；我们此前三版都算了却丢弃，白白损失了这路信息。
            cond, aux = self.converge(
                patch_seq, step_actions, n_frames=n_frames, n_past=n_past,
                vae_latent=vae_latent,
            )
            return cond, None, n_past, aux

        if self.converge_type == "masked":
            cond, aux = self.converge(
                patch_seq, actions, n_frames=n_frames, n_past=n_past
            )
            return cond, None, n_past, aux
        # ===zgt===

        states = self.converge(
            patch_seq, step_actions, n_frames=n_frames, vae_latent=vae_latent
        )
        return states.mu[:, n_past], states.mu, n_past, None

    def _student_timeline(self, pixel_values, future_pixel_values,
                          past_pixel_values, actions, n_past):
        """Stage 2 的 student 路径：只有 past + now，没有未来。

        与 teacher 的差别仅在 use_future=False —— 它不只是把 beta 置零，
        还会**物理截断**观测到当下为止（见 TemporalConverge.forward）。
        只关 beta 的话 alpha 仍会在递推中吸收未来帧，那是泄漏而非特权撤除。
        VAE 上下文同样被丢弃，因为它由含未来帧的整段视频编码而来。

        逐节点对齐的意义
            student 在处理 t-3 时，要逼近的是「被 t+4 的观测修正过的
            s*_{t-3}」—— alpha 在流过历史时，每一步都被整条时间线塑造。
            这是 leakage 转化成能力的地方，也是 Stage 2 与 WoG 两段式
            （用未来 -> 猜未来）的实质差别：我们是「用未来 -> 学会不用未来」。

        Returns:
            [B, n_past+1, n_tokens, 32]
        """
        now_feats, _ = self._encode_frames(
            {k: v.unsqueeze(1) for k, v in pixel_values.items()}, pixel_values
        )
        future_feats, _ = self._encode_frames(future_pixel_values, pixel_values)
        if past_pixel_values:
            past_feats, _ = self._encode_frames(past_pixel_values, pixel_values)
            patch_seq = torch.cat([past_feats, now_feats, future_feats], dim=1)
        else:
            patch_seq = torch.cat([now_feats, future_feats], dim=1)

        n_future = next(iter(future_pixel_values.values())).shape[1]
        n_frames = n_past + 1 + n_future
        step_actions = self._aggregate_step_actions(actions, self.obs_offsets, n_past)

        # ===zgt===
        # masked 版：use_future=False 时 Q_future 不参与，融合退化为
        # alpha + ell。互斥性由架构保证 —— Q_past / Q_now 收到的 patch
        # 里本就不含未来帧，无需像递推版那样额外截断观测。
        if self.converge_type == "flow":
            # use_future=False 时不做逆推，融合退化为 alpha + ell。
            # 「撤掉未来」不是训练阶段，而是推理时的自然状态。
            #
            # vae_latent 不传：它由**含未来帧**的整段视频编码而来，
            # student 拿到就是泄漏，不是特权撤除。
            cond, _ = self.converge(
                patch_seq, step_actions, n_frames=n_frames, n_past=n_past,
                use_future=False, vae_latent=None,
            )
            return cond

        if self.converge_type == "masked":
            cond, _ = self.converge(
                patch_seq, actions, n_frames=n_frames, n_past=n_past, use_future=False
            )
            return cond
        # ===zgt===

        return self.converge(
            patch_seq, step_actions, n_frames=n_frames,
            use_future=False, n_past=n_past,
        ).mu

    def _aggregate_step_actions(self, actions, offsets, n_past):
        """把每对相邻节点之间的动作聚合成一个等效转移动作。

        offsets 为 None（或等价于逐步相邻）时退化为直接切片，与间隔 1 的情形一致。

        Args:
            actions: [B, L, A]，actions[i] 对应偏移 (i - n_past)
            offsets: 节点的时间偏移表，如 [-15,-11,-7,-3,0,3,7,11,15]
            n_past:  offsets 中负数的个数

        Returns:
            [B, K-1, A]，第 k 项驱动 节点k -> 节点k+1
        """
        if offsets is None:
            # 节点逐步相邻：actions[k] 恰好驱动 k -> k+1
            n_nodes = actions.shape[1]
            return actions[:, : n_nodes - 1]

        offsets = list(offsets)
        steps = []
        for k in range(len(offsets) - 1):
            lo = offsets[k] + n_past          # 在 actions 中的起始下标
            hi = offsets[k + 1] + n_past      # 结束下标（不含）
            seg = actions[:, lo:hi]           # 该区间内的所有动作
            agg = seg.sum(dim=1)              # delta 累加 = 等效位移
            agg = torch.cat([agg[..., :-1], seg[:, -1, -1:]], dim=-1)   # gripper 取末值
            steps.append(agg)
        return torch.stack(steps, dim=1)

    def _multi_node_action_loss(self, all_nodes, n_past, actions,
                                cognition_features, repeated_diffusion_steps):
        """每个节点都要能生成"从我这里出发"的 16 步动作。

        为什么不只监督当下
            若只有当下节点有 loss，其余节点就是无监督的中间变量，
            会自由漂移 —— 时间线退化成"一个状态 + 八个装饰"。
            逐节点监督让每个 s*_k 都必须真正编码"k 时刻是什么状态"，
            这是 timeline 成立的前提，也是"When am I"能被验证的基础。

        它同时吸收了三项本需单独设计的约束
            防塌缩      常数时间线无法在每个位置生成正确动作
            序列自洽    相邻节点的动作窗口重叠 Ta-1 步，天然要求一致
            动作敏感    监督本身即是

        DiT 是共享的
            独立 9 份 = 3.7B 参数，且意味着每个时刻遵循不同的动力学 ——
            物理上说不通。共享还保证所有节点的表示可被同一读出头解释，
            这才让"时间线是一串同质状态"成立。

        计算量与显存（后者是真正的约束，实测）
            FLOPs 上逐节点监督很便宜：DiT 一次约 5.1 GFLOPs，ViT 一帧约
            80.9 GFLOPs，9 个节点只占新增开销的 6%。
            但**显存不是这样**：每个节点的 DiT 前向都要保留激活到 backward，
            PPU 实测 B=16、repeated_diffusion_steps=8 时

                1 个节点  6.3 GB
                3 个节点 15.6 GB
                9 个节点 43.5 GB     <- 加上 7B 模型本身必然 OOM

            故用 `nodes_per_step` 每步随机采样若干节点。期望上所有节点
            都被均匀监督（与全采样同分布），但显存恒定。这是 dropout 式
            的标准做法，不改变优化目标，只改变每步的估计方差。
            nodes_per_step <= 0 表示全采样（小 batch 调试用）。

        Args:
            all_nodes: [B, K, n_tokens, 32]
            n_past:    当下节点下标；节点 k 对应相对偏移 (k - n_past)
            actions:   [B, L, A]，覆盖偏移 [-n_past, ...]

        Returns:
            被采样节点的平均 action loss
        """
        B, K = all_nodes.shape[:2]
        L = actions.shape[1]

        # 先筛出动作数据足够的节点
        valid = [k for k in range(K) if k + self.Ta <= L]
        if not valid:
            raise ValueError(
                f"no node has enough action data: K={K}, Ta={self.Ta}, "
                f"action length={L}. Increase future_action_window_size to "
                f"at least {K - n_past - 1 + self.Ta - 1}."
            )

        # 采样：当下节点始终保留（它是注入 DiT 的那个），其余随机补足
        if 0 < self.nodes_per_step < len(valid):
            others = [k for k in valid if k != n_past]
            n_extra = max(0, self.nodes_per_step - (1 if n_past in valid else 0))
            perm = torch.randperm(len(others), device=all_nodes.device)[:n_extra]
            chosen = ([n_past] if n_past in valid else []) + [others[i] for i in perm.tolist()]
        else:
            chosen = valid

        losses = []
        for k in chosen:
            node_actions = actions[:, k : k + self.Ta]
            losses.append(
                self.action_model.loss(
                    node_actions.repeat(repeated_diffusion_steps, 1, 1),
                    cognition_features.repeat(repeated_diffusion_steps, 1, 1),
                    all_nodes[:, k].repeat(repeated_diffusion_steps, 1, 1),
                )
            )
        return torch.stack(losses).mean()
    # ===zgt===

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        future_pixel_values: Optional[torch.FloatTensor] = None,
        past_pixel_values: Optional[torch.FloatTensor] = None,   # ===zgt===
        labels: Optional[torch.LongTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        repeated_diffusion_steps: int = 4,
        action_masks=None,
    ) -> Tuple:
        output: CausalLMOutputWithPast = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden = output.hidden_states[-1]

        if self.vlm.vision_backbone.featurizer is not None:
            num_patch = self.vlm.vision_backbone.featurizer.patch_embed.num_patches
        elif (
            hasattr(self.vlm.vision_backbone, "siglip_featurizer")
            and self.vlm.vision_backbone.siglip_featurizer is not None
        ):
            num_patch = self.vlm.vision_backbone.siglip_featurizer.patch_embed.num_patches
        else:
            raise ValueError("No vision backbone found")

        last_hidden = last_hidden[:, num_patch:]
        obs_ins = last_hidden[:, -4:]

        cumulative_sum = attention_mask.cumsum(dim=1)
        last_true_indices = (cumulative_sum == cumulative_sum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)
        expanded_indices = last_true_indices.unsqueeze(-1).expand(-1, last_hidden.size(-1))
        cognition_features = last_hidden.gather(1, expanded_indices.unsqueeze(1))  # [B, 1, D]

        # ===zgt===
        # 条件计算：两条路径二选一，输出统一为 [B, n_tokens, 32]，
        # 下游（per_attn / oformer）完全不感知走了哪条。
        cond_tokens, all_nodes, n_past, calib_aux = self._compute_condition(
            pixel_values, future_pixel_values, past_pixel_values, actions
        )
        # ===zgt===

        # ===zgt===
        # 动作损失：
        #   QFormer 路径 —— 只有当下一个条件，沿用原版单点监督
        #   Temporal 路径 —— 时间线上每个节点都要能生成"从我出发"的动作
        cognition_features_repeated = cognition_features.repeat(
            repeated_diffusion_steps, 1, 1
        )  # [repeated_diffusion_steps*B, 1, D]

        if all_nodes is not None:
            act_loss = self._multi_node_action_loss(
                all_nodes, n_past, actions, cognition_features, repeated_diffusion_steps
            )

            # Stage 2：同一 batch 再跑一次无 beta 的 student 路径，逐节点对齐。
            # 视觉编码已在 _compute_condition 中完成，此处只重跑消息传播，
            # 开销约为总量的 6%（DiT 5.1 vs ViT 80.9 GFLOPs/帧）。
            if self.distill_weight > 0.0:
                student_nodes = self._student_timeline(
                    pixel_values, future_pixel_values, past_pixel_values, actions, n_past
                )
                # student 只覆盖 [0, n_past]，teacher 取同一段做逐节点对齐
                k_stu = student_nodes.shape[1]
                act_loss = act_loss + self._multi_node_action_loss(
                    student_nodes, n_past, actions, cognition_features, repeated_diffusion_steps
                )
                distill = 1.0 - F.cosine_similarity(
                    student_nodes, all_nodes[:, :k_stu].detach(), dim=-1
                ).mean()
                act_loss = act_loss + self.distill_weight * distill
        else:
            future_repeated = cond_tokens.repeat(repeated_diffusion_steps, 1, 1)
            actions_future = actions[:, -self.Ta :, :]
            act_loss = self.action_model.loss(
                actions_future.repeat(repeated_diffusion_steps, 1, 1),
                cognition_features_repeated,
                future_repeated,
            )

            # ===zgt===
            # Stage 2（masked 版）：student 看不到未来，要复现 teacher 那个
            # 「被未来塑造过的当下」。两路共享 Q_past / Q_now 的权重，
            # 差别只在 Q_future 是否参与 —— 这正是"用未来 -> 学会不用未来"。
            # teacher 一侧 detach：它是目标，不该被 student 的梯度拖动。
            if self.use_temporal and self.converge_type == "masked" \
                    and self.distill_weight > 0.0:
                student = self._student_timeline(
                    pixel_values, future_pixel_values, past_pixel_values, actions, n_past
                )
                act_loss = act_loss + self.action_model.loss(
                    actions_future.repeat(repeated_diffusion_steps, 1, 1),
                    cognition_features_repeated,
                    student.repeat(repeated_diffusion_steps, 1, 1),
                )
                distill = 1.0 - F.cosine_similarity(
                    student, cond_tokens.detach(), dim=-1
                ).mean()
                act_loss = act_loss + self.distill_weight * distill
            # ===zgt===

        if self.train_venc:
            obs_loss = torch.zeros_like(act_loss)
        else:
            obs_pre = self.oformer(obs_ins)
            # 对齐目标是冻结的条件模块产出的当下节点。原代码此处引用了
            # 一个已不存在的变量名（重构 _encode_frames 时的残留），
            # 只在 Stage II 路径触发，故此前未暴露。
            target = cond_tokens.detach()
            obs_pre_n = F.normalize(obs_pre, dim=-1)
            target_n = F.normalize(target, dim=-1)
            obs_loss = (1.0 - (obs_pre_n * target_n).sum(dim=-1)).mean()
        # ===zgt===

        loss = act_loss + 0.3 * obs_loss

        # ===zgt===
        # Lambda 校准（仅 masked 版）。精度的定义是误差方差的倒数，
        #     Lambda = 1 / sigma^2
        # 但递推版从没告诉模型 Lambda 该满足这个定义 —— 唯一信号是穿过 DiT
        # 的动作损失，太弱，实测 log_decay 的 std 只有 2.4e-4，融合退化成
        # 固定权重，于是 beta 把 alpha 已编好的信息稀释了（gt_grip 掉 0.147）。
        # 这两项把定义本身变成可训练目标。权重分开可配，构成天然消融：
        #   recon（变体 B）—— "我对当下有多清楚"，WhenAmI 特有
        #   act  （变体 A）—— "我知道该做什么"，任何任务都适用
        if calib_aux is not None:
            if hasattr(calib_aux, "fwd"):
                # flow 版：双向推演的对齐损失。两个方向互为监督 ——
                # 顺推的落点由真实未来帧校验，逆推由真实过去帧校验。
                # cycle 检验「方向作为条件」学出的可逆性是不是真的
                # （方向 token 不像 coupling layer 那样天然可逆）。
                loss = loss + self.flow_fwd_weight * calib_aux.fwd \
                            + self.flow_bwd_weight * calib_aux.bwd \
                            + self.flow_cycle_weight * calib_aux.cycle
            else:
                # masked 版：Lambda 校准
                if self.calib_recon_weight > 0.0:
                    loss = loss + self.calib_recon_weight * calib_aux.recon
                if self.calib_act_weight > 0.0:
                    loss = loss + self.calib_act_weight * calib_aux.act
        # ===zgt===

        return loss, output, act_loss, obs_loss

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        vision_fsdp_wrapping_policy = self.vlm.vision_backbone.get_fsdp_wrapping_policy()
        llm_fsdp_wrapping_policy = self.vlm.llm_backbone.get_fsdp_wrapping_policy()

        # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector` and DiT
        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={LinearProjector, MLPProjector, FusedMLPProjector, DiT},
        )

        return partial(
            _or_policy,
            policies=[
                vision_fsdp_wrapping_policy,
                llm_fsdp_wrapping_policy,
                prismatic_fsdp_wrapping_policy,
            ],
        )

    def load_ema_to_weights(self):
        """Load the EMA state dict to the weights."""
        if self.use_ema:
            self.action_model.load_state_dict(self.ema_diffusion.state_dict())
            del self.ema_diffusion

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        freeze_weights: bool = True,
        action_dim: int = 7,
        Ta: int = 16,
        action_model_type: str = "DiT-B",
        use_ema: bool = False,
        norm_stats=None,
        train_venc=False,
        use_temporal: bool = False,   # ===zgt===
        obs_offsets=None,             # ===zgt===
        distill_weight: float = 0.0,  # ===zgt===
        nodes_per_step: int = 0,      # ===zgt===
        # ===zgt===
        # 下面三个必须显式列出。本函数的 **kwargs 被转发给 PrismaticVLM
        # 而非 WoG，落进 kwargs 的参数会被静默丢弃 —— 实测后果：配置里写
        # converge_type=masked，构造出来的却是 recurrent，checkpoint 里存的是
        # obs_encoder.*（32 个键）而非三路分支（416 个键）。
        converge_type: str = "recurrent",
        calib_recon_weight: float = 0.0,
        calib_act_weight: float = 0.0,
        flow_fwd_weight: float = 1.0,
        flow_bwd_weight: float = 1.0,
        flow_cycle_weight: float = 0.1,
        flow_enc_layers: int = 3,
        flow_cross_layers: int = 2,
        flow_enc_hidden: int = 768,
        flow_enc_use_ffn: bool = True,
        # ===zgt===
        **kwargs,
    ) -> WoG:
        vlm = PrismaticVLM(
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            **kwargs,
        )

        # ===zgt===
        # mmap=True：权重按需从磁盘映射，不把整个 28GB checkpoint 读进内存。
        #
        # 原版 `torch.load(..., map_location="cpu")` 会把整份 ckpt 载入 RAM。
        # 单进程实测 load_vla 完成后 RSS 达 61.8 GB；平台每节点 16 个 rank
        # 各执行一次，峰值约 16 x 62 = 990 GB，叠加数据管线后触发
        # 系统 OOM —— 表现为 exitcode -9 (SIGKILL)，而非 CUDA OOM。
        # 这与 use_temporal 无关，原版 WoG 在 16 rank/节点下同样会挂。
        #
        # 实测 mmap=True：RSS 0.29 GB，加载 0.6s（原本需数十秒 + 数十 GB）。
        model_state_dict = torch.load(
            pretrained_checkpoint, map_location="cpu", mmap=True
        )["model"]
        # ===zgt===
        assert (
            "projector" in model_state_dict and "llm_backbone" in model_state_dict
        ), "PrismaticVLM `from_pretrained` expects checkpoint with keys for `projector` AND `llm_backbone`!"

        vlm.projector.load_state_dict(model_state_dict["projector"], assign=True)
        vlm.llm_backbone.load_state_dict(model_state_dict["llm_backbone"], assign=True)  # ===zgt=== assign=True: 直接接管 mmap 张量，避免拷贝出第二份权重
        if "vision_backbone" in model_state_dict.keys():
            vlm.vision_backbone.load_state_dict(model_state_dict["vision_backbone"])

        # Freeze Weights
        if freeze_weights:
            vlm.requires_grad_(False)
            vlm.eval()

        # Initialize WoG
        wog = WoG(
            vlm,
            token_size=vlm.llm_backbone.llm.lm_head.in_features,
            action_dim=action_dim,
            Ta=Ta,
            action_model_type=action_model_type,
            use_ema=use_ema,
            norm_stats=norm_stats,
            train_venc=train_venc,
            use_temporal=use_temporal,   # ===zgt===
            obs_offsets=obs_offsets,     # ===zgt===
            distill_weight=distill_weight,  # ===zgt===
            nodes_per_step=nodes_per_step,  # ===zgt===
            converge_type=converge_type,            # ===zgt===
            calib_recon_weight=calib_recon_weight,  # ===zgt===
            calib_act_weight=calib_act_weight,      # ===zgt===
            flow_fwd_weight=flow_fwd_weight,        # ===zgt===
            flow_bwd_weight=flow_bwd_weight,        # ===zgt===
            flow_cycle_weight=flow_cycle_weight,    # ===zgt===
            flow_enc_layers=flow_enc_layers,        # ===zgt===
            flow_cross_layers=flow_cross_layers,    # ===zgt===
            flow_enc_hidden=flow_enc_hidden,        # ===zgt===
            flow_enc_use_ffn=flow_enc_use_ffn,      # ===zgt===
        )

        # Load ActionModel from Checkpoint
        if "action_model" in model_state_dict:
            print("\033[92mLoading ActionModel from checkpoint.\033[0m")
            wog.action_model.load_state_dict(model_state_dict["action_model"], strict=False)
            if "ema_diffusion" in model_state_dict and use_ema:
                wog.ema_diffusion.load_state_dict(model_state_dict["ema_diffusion"], strict=False)
            elif use_ema:
                wog.ema_diffusion.load_state_dict(model_state_dict["action_model"], strict=False)
        else:
            overwatch.warning("No ActionModel found in the pretrained checkpoint. Initializing a new one.")

        # ===zgt===
        # 条件模块的权重加载。use_temporal 时 qformer 不存在，checkpoint 里的
        # qformer 权重也无从迁移（结构完全不同），故跳过并显式提示 ——
        # 静默忽略会让人误以为加载成功。
        if wog.use_temporal:
            if "converge" in model_state_dict:
                print("\033[92mLoading TemporalConverge from checkpoint.\033[0m")
                wog.converge.load_state_dict(model_state_dict["converge"], strict=True)
            else:
                print("\033[93mNo TemporalConverge in checkpoint. Initializing a new one "
                      "(QFormer weights, if any, are not transferable).\033[0m")
        elif "qformer" in model_state_dict:
            qformer_state_dict = model_state_dict["qformer"]
            model_keys = set(wog.qformer.state_dict().keys())
            checkpoint_keys = set(qformer_state_dict.keys())
            if model_keys != checkpoint_keys:
                print("\033[93mWARNING: qformer is not match: maybe the version is old\033[0m")
            else:
                print("\033[92mLoading QFormer from checkpoint.\033[0m")
                wog.qformer.load_state_dict(qformer_state_dict, strict=True)
        else:
            print("\033[93mNo QFormer found in the pretrained checkpoint. Initializing a new one.\033[0m")
        # ===zgt===

        return wog

    @torch.inference_mode()
    def _condition_from_frames(self, cond_frames, cond_actions, pixel_values, model_dtype):
        """===zgt=== 推理期从原始帧算出注入 DiT 的条件。

        训练走的是 _compute_condition（输入已由 dataloader 整理成
        {modality: [B, T, C, H, W]}）。这里把 PIL 帧过同一套 image_transform
        后拼成同样的形状，再调用同一个函数 —— 训练与评测因此共用一条
        条件计算路径，不会出现"训练用 A、评测用 B"的偏差。

        帧的划分沿用 FrameLayout 的约定：use_temporal 时按 obs_offsets 把
        序列切成 past / now / future；QFormer 路径则整段作为 future 帧
        （它本就不区分时刻）。

        Args:
            cond_frames:  List[PIL.Image]，按时间顺序，长度须与 obs_offsets 一致
            cond_actions: [T-1, 7] 帧间动作；时间线递推需要，QFormer 路径忽略
            pixel_values: 当前帧的已变换张量，作为 VAE 的时序起点
            model_dtype:  与 action_model 对齐，避免 per_attn 里 dtype 不匹配

        Returns:
            [1, n_tokens, dim] 当下节点的条件
        """
        tf = self.vlm.vision_backbone.image_transform
        device = self.vlm.device

        stacked = {}
        for img in cond_frames:
            px = tf(img)
            items = px.items() if isinstance(px, dict) else [(None, px)]
            for k, v in items:
                stacked.setdefault(k, []).append(v)
        # [1, T, C, H, W]，与 dataloader 产出的形状一致
        frames = {k: torch.stack(v, 0).unsqueeze(0).to(device) for k, v in stacked.items()}

        if not self.use_temporal:
            # QFormer 不区分时刻，整段帧即它的 future 输入
            cond, _, _, _ = self._compute_condition(pixel_values, frames, None, None)
            return cond.to(model_dtype)

        offsets = list(self.obs_offsets)
        if len(cond_frames) != len(offsets):
            raise ValueError(
                f"cond_frames has {len(cond_frames)} frames but obs_offsets "
                f"expects {len(offsets)}"
            )
        now_idx = offsets.index(0)
        take = lambda lo, hi: {k: v[:, lo:hi] for k, v in frames.items()}   # noqa: E731

        past = take(0, now_idx) if now_idx > 0 else None
        future = take(now_idx + 1, len(offsets))

        actions = None
        if cond_actions is not None:
            actions = torch.as_tensor(cond_actions, device=device).unsqueeze(0).float()

        cond, _, _, _ = self._compute_condition(pixel_values, future, past, actions)
        return cond.to(model_dtype)

    def predict_action(
        self,
        image: Image,
        instruction: str,
        unnorm_key: Optional[str] = None,
        calvin_deploy: bool = False,
        cond_frames: Optional[List[Image]] = None,      # ===zgt===
        cond_actions: Optional[np.ndarray] = None,      # ===zgt===
        **kwargs: str,
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action.

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.
        @param cond_frames: ===zgt=== 计算条件所需的完整帧序列。
                            use_temporal 时按 obs_offsets 给出 9 帧（含未来），
                            否则给 WoG 的 5 帧。为 None 时 DiT 回退到
                            uncondition —— 那测的是无条件性能，不是本方法。
        @param cond_actions: ===zgt=== 帧间动作 [T-1, 7]，时间线递推需要。

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        image_transform, tokenizer = self.vlm.vision_backbone.image_transform, self.vlm.llm_backbone.tokenizer

        # Build VLA Prompt
        prompt_builder = self.vlm.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.vlm.device)
        if isinstance(tokenizer, LlamaTokenizerFast):
            # Note: We need to add this special empty token ('') after the colon (':') token in "ASSISTANT:"
            #       insert it to match the inputs seen at training time. The empty token is at index 29871.
            #       We also need to add the special cognition token at index 2 (i.e. the EOS token).
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871, 2]).long(), dim=0).to(self.vlm.device)), dim=1
            )
        else:
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.vlm.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.vlm.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.vlm.llm_backbone.half_precision_dtype

        # Generate cognition feature through vlm
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.vlm.enable_mixed_precision_training):
            # fmt: off
            output = super(PrismaticVLM, self.vlm).generate(
                input_ids=input_ids,                            # Shape: [1, seq]
                pixel_values=pixel_values,                      # Shape: [1, 3, res, res] or Dict[str, ...]
                max_new_tokens=1,
                output_hidden_states=True, 
                return_dict_in_generate=True,
                **kwargs
            )
            # fmt: on

        # Extract cognition feature
        cognition_features = output.hidden_states[0][-1][:, -1, :]
        assert (cognition_features.shape[0], cognition_features.shape[1]) == (
            1,
            4096,
        ), "Batch size must be 1 for action prediction"
        model_dtype = next(self.action_model.net.parameters()).dtype
        B = cognition_features.shape[0]
        cognition_features = cognition_features.unsqueeze(1).to(model_dtype)  # [B, 1, D]

        # ===zgt===
        # 计算条件并注入 DiT。不给 cond_frames 时 per_token 为 None，
        # DiT 回退到 uncondition —— 那测的是"把条件拿掉的 DiT"，
        # 与训练目标（loss 里条件是必传的）不一致。
        #
        # 两条路径共用 _compute_condition，因此 WoG 的 QFormer 与本工作的
        # 时间线在同一前提下比较：都拿到真实未来帧。这正是 Stage I 该被
        # 衡量的方式 —— 它评的是条件质量，而非可部署的策略能力。
        per_token = None
        if cond_frames is not None:
            per_token = self._condition_from_frames(
                cond_frames, cond_actions, pixel_values, model_dtype
            )
            # ===zgt===
            # cond_scale：诊断用的条件幅度缩放，默认 1.0 不改变行为。
            # 实测三版条件的幅度差一个数量级以上（WoG 的 O^c 均值 0.83，
            # 我们的 s*_t 只有 0.017~0.058）—— out_proj 用 std=1e-3 初始化
            # 后长得太慢。per_attn 是 attention，输入幅度太小时它的贡献
            # 会被主干淹没。这个开关用来判定「幅度是不是主因」。
            scale = getattr(self, "cond_scale", 1.0)
            if scale != 1.0:
                per_token = per_token * scale
            # ===zgt===
        samples = self.action_model.inference(cognition_features, per_token=per_token)
        # ===zgt===

        normalized_actions = samples[0].cpu().numpy()
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        if not calvin_deploy:
            normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
        else:
            normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.0, -1, 1)
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions, normalized_actions

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                "Your model was trained on more than one dataset, "
                "please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            "The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key=None):
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
