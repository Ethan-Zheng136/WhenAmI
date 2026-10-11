import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import draccus
import torch
import torch.distributed as dist
import yaml
import wandb

from prismatic.overwatch import initialize_overwatch
from prismatic.util import set_global_seed
from prismatic.vla import get_vla_dataset_and_collator
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from training import VLAMetrics, get_train_strategy
from conf import VLAConfig, VLARegistry
from vla import load, load_vla
from vla import WoG
from vla.past import PastSpec
from prismatic.util.memory_diagnostics import emit_memory
from prismatic.vla.datasets.rlds.runtime_limits import runtime_summary

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class TrainConfig:
    # fmt: off

    # VLAConfig (`conf/vla.py`); override with --vla.type `VLARegistry.<VLA>.vla_id`
    vla: VLAConfig = field(
        default_factory=VLAConfig.get_choice_class(VLARegistry.EXP_WoG_OXE_MAGIC_SOUP_PLUS_MINUS.vla_id)
    )

    # Directory Paths
    data_root_dir: Path = Path(                                     # Path to Open-X dataset directory
        "datasets/open-x-embodiment"
    )
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints

    # Resume Run Parameters
    pretrained_checkpoint: Optional[Union[str, Path]] = None                  # Absolute Path to Checkpoint
    is_resume: bool = True                                          # Whether we are continuing a prior training run
                                                                    # (only applicable given pretrained checkpoint)
    resume_step: Optional[int] = None                               # Global Step to Resume (should match checkpoint)
    resume_epoch: Optional[int] = None                              # Epoch to Resume (should match checkpoint)

    # Run Arguments
    run_id: Optional[str] = None                                    # Run ID for logging, Weights & Biases
    run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases
    save_interval: int = 2500                                       # Interval for saving checkpoints (in steps)
    image_aug: bool = False                                         # Whether to enable image augmentations
    seed: int = 42                                                  # Random seed (for reproducibility)

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token")                  # Environment variable or Path to HF Token

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("jsonl", "wandb")                  # Trackers to initialize (if W&B, add config!)
    #trackers: Tuple[str, ...] = ("jsonl",)                         # Trackers to initialize (if W&B, add config!)
    wandb_project: str = "wog"                                       # Name of W&B project to log to (use default!)
    wandb_entity: str = "selensu-xidian-university"                 # Name of entity to log under
    repeated_diffusion_steps: int = 8                               # Repeated steps for training action model (a diffusion model)
    load_all_data_for_training: bool = True                         # Load all training data 
    train_venc: bool = False
    # ===zgt===
    # 时序时间线相关。use_temporal=False 时全部不生效，行为与原版 WoG 一致。
    #   obs_offsets  采样哪些帧，相对当下的偏移；决定时间线的长度与间隔
    #   n_past       obs_offsets 中负数的个数，由它推导当下节点的位置
    # 注意 faw 必须 >= max(offset) + Ta - 1，否则未来侧节点缺动作监督
    # （_multi_node_action_loss 会报错并给出建议值）。
    use_temporal: bool = False
    obs_offsets: Optional[Tuple[int, ...]] = None
    distill_weight: float = 0.0     # Stage 2: >0 开启双路径蒸馏
    nodes_per_step: int = 0         # <=0 全采样。实测 per_device=4 全节点 12.6GB < per_device=16 采样3 个 15.6GB
    # converge_type: recurrent（第一版，递推 + 9 节点 DiT）
    #             或 masked（第二版，三路互斥 cross-attention + Lambda 校准，
    #                        只输出当下、DiT 跑 1 次）
    converge_type: str = "recurrent"
    # Lambda 校准权重，仅 masked 版生效。分开配置构成消融：
    #   recon —— 让 past/future 去重建当下帧，对应"我对当下有多清楚"
    #   act   —— 让各路去预测动作，对应"我知道该做什么"
    calib_recon_weight: float = 0.0
    calib_act_weight: float = 0.0
    # flow 版的三项自监督权重：顺推对齐 / 逆推对齐 / 往返一致
    flow_fwd_weight: float = 1.0
    flow_bwd_weight: float = 1.0
    flow_cycle_weight: float = 0.1
    # 帧编码器容量。flow_v2 跑的是 2/1/512（无 FFN）；默认值是补齐版
    flow_enc_layers: int = 3
    flow_cross_layers: int = 2
    flow_enc_hidden: int = 768
    # False 复现 flow_v2 的裸 attention（无 FFN），用于消融与旧 ckpt 兼容
    flow_enc_use_ffn: bool = True
    # New naming: Future=1, Past=2. Zero preserves every old experiment.
    whenami_stage: int = 0
    past_init: str = "checkpoint"
    # ===zgt===
    Ta: int = 16                                                    # Action chunking, predicting future actions + current action
    action_model_type: str = 'DiT-B'                                # Action model type, chose from ['DiT-S', 'DiT-B', 'DiT-L']
    use_ema: bool = False                                           # EMA version of action model
    action_dim: int = 7                                             # Dimension of action space

    def __post_init__(self) -> None:
        """Lift optimization parameters from `self.vla` for ease of use =>> validate on `expected_world_size`"""
        if self.whenami_stage not in (0, 1, 2):
            raise ValueError("Only WhenAmI Future=1 and Past=2 are implemented")
        if self.whenami_stage in (1, 2) and (not self.train_venc or self.use_temporal):
            raise ValueError("WhenAmI stages 1/2 require train_venc=True, use_temporal=False")
        if self.whenami_stage == 2:
            self.obs_offsets = PastSpec.from_offsets(self.obs_offsets, self.Ta).offsets
            if self.is_resume:
                raise ValueError(
                    "Past checkpoints are model-only; exact optimizer resume is unavailable. "
                    "Use is_resume=False with a NEW run_id to explicitly warm-start."
                )
            if self.resume_step is not None or self.resume_epoch is not None:
                raise ValueError("Past warm starts begin at step 0; do not supply resume_step/epoch")
            if self.pretrained_checkpoint is None and self.past_init == "checkpoint":
                raise ValueError("past_init=checkpoint requires pretrained_checkpoint")
        self.epochs = self.vla.epochs
        self.max_steps = self.vla.max_steps
        self.global_batch_size = self.vla.global_batch_size
        self.per_device_batch_size = self.vla.per_device_batch_size

        self.learning_rate = self.vla.learning_rate
        self.weight_decay = self.vla.weight_decay
        self.max_grad_norm = self.vla.max_grad_norm
        self.lr_scheduler_type = self.vla.lr_scheduler_type
        self.warmup_ratio = self.vla.warmup_ratio

        self.train_strategy = self.vla.train_strategy

        # [Validate] Assert on `expected_world_size`
        assert (
            self.vla.expected_world_size == overwatch.world_size()
        ), f"Expected World Size = {self.vla.expected_world_size} but Found {overwatch.world_size()} GPUs!"

    # fmt: on


@draccus.wrap()
def train(cfg: TrainConfig) -> None:
    overwatch.info("WoG-VLA Training :: Warming Up")
    emit_memory("imports_complete")

    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    torch.cuda.set_device(device_id := overwatch.local_rank())
    torch.cuda.empty_cache()

    # Configure Unique Run Name & Save Directory
    vla_id = cfg.vla.vla_id
    cfg.run_id = (
        f"{vla_id}+n{cfg.vla.expected_world_size // 8}+b{cfg.per_device_batch_size}+x{cfg.seed}"
        if cfg.run_id is None
        else cfg.run_id
    )
    if cfg.run_id_note is not None:
        cfg.run_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        cfg.run_id += "--image_aug"

    # Start =>> Build Directories and Set Randomness
    overwatch.info('"Do or do not; there is no try."', ctx_level=1)
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    run_dir = cfg.run_root_dir / cfg.run_id
    if cfg.whenami_stage == 2:
        # Fail closed on reused run IDs; never append a new launch to an old
        # completed run or overwrite its architecture config.
        if overwatch.is_rank_zero():
            run_dir.mkdir(parents=True, exist_ok=False)
        dist.barrier()
    else:
        os.makedirs(run_dir, exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)

    # Save Configuration =>> additionally save a JSON version for later HF Integration
    if overwatch.is_rank_zero():
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml", "r") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)
        (run_dir / "input_runtime.json").write_text(json.dumps(runtime_summary(), indent=2))

    dist.barrier()
    emit_memory("before_model_load")
    # Load VLA checkpoint (if resuming from training) or Base VLM otherwise (from `cfg.vla.base_vlm` ID or Path)
    #   =>> Note :: Verifies that all parameters are loaded in FP32 on load!
    overwatch.info(f"Loading Base VLM `{cfg.vla.base_vlm}` from ID/Path")
    if cfg.pretrained_checkpoint is not None:
        # [Validate] Pretrained Checkpoint `step` and `epoch` should match `resume_step` and `resume_epoch`
        #   =>> Note :: We make developers pass in `resume_*` arguments as an extra sanity check!
        if cfg.is_resume:
            assert int(re.search("step-(.+?)-", cfg.pretrained_checkpoint).group(1)) == cfg.resume_step
            assert int(re.search("epoch-(.+?)-", cfg.pretrained_checkpoint).group(1)) == cfg.resume_epoch
        overwatch.info("Loading VLA Checkpoint")
        if cfg.use_ema:
            overwatch.info("Loading EMA of Diffusion")
        vla = load_vla(
            cfg.pretrained_checkpoint,
            hf_token=hf_token,
            load_for_training=True,
            action_model_type=cfg.action_model_type,
            action_dim=cfg.action_dim,
            Ta=cfg.Ta,
            use_ema=cfg.use_ema,
            train_venc=cfg.train_venc,
            use_temporal=cfg.use_temporal,   # ===zgt===
            obs_offsets=cfg.obs_offsets,     # ===zgt===
            distill_weight=cfg.distill_weight,  # ===zgt===
            nodes_per_step=cfg.nodes_per_step,  # ===zgt===
            converge_type=cfg.converge_type,            # ===zgt===
            calib_recon_weight=cfg.calib_recon_weight,  # ===zgt===
            calib_act_weight=cfg.calib_act_weight,      # ===zgt===
            flow_fwd_weight=cfg.flow_fwd_weight,        # ===zgt===
            flow_bwd_weight=cfg.flow_bwd_weight,        # ===zgt===
            flow_cycle_weight=cfg.flow_cycle_weight,    # ===zgt===
            flow_enc_layers=cfg.flow_enc_layers,        # ===zgt===
            flow_cross_layers=cfg.flow_cross_layers,    # ===zgt===
            flow_enc_hidden=cfg.flow_enc_hidden,        # ===zgt===
            flow_enc_use_ffn=cfg.flow_enc_use_ffn,      # ===zgt===
            whenami_stage=cfg.whenami_stage,
            past_init=cfg.past_init,
        )

    else:
        vlm = load(cfg.vla.base_vlm, hf_token=hf_token, load_for_training=True)
        overwatch.info("Creating VLA from Base VLM")
        if cfg.use_ema:
            overwatch.info("Creating EMA for Diffusion")
        vla = WoG(
            vlm,
            action_model_type=cfg.action_model_type,
            action_dim=cfg.action_dim,
            Ta=cfg.Ta,
            use_ema=cfg.use_ema,
            train_venc=cfg.train_venc,
            use_temporal=cfg.use_temporal,   # ===zgt===
            obs_offsets=cfg.obs_offsets,     # ===zgt===
            distill_weight=cfg.distill_weight,  # ===zgt===
            nodes_per_step=cfg.nodes_per_step,  # ===zgt===
            converge_type=cfg.converge_type,            # ===zgt===
            calib_recon_weight=cfg.calib_recon_weight,  # ===zgt===
            calib_act_weight=cfg.calib_act_weight,      # ===zgt===
            flow_fwd_weight=cfg.flow_fwd_weight,        # ===zgt===
            flow_bwd_weight=cfg.flow_bwd_weight,        # ===zgt===
            flow_cycle_weight=cfg.flow_cycle_weight,    # ===zgt===
            flow_enc_layers=cfg.flow_enc_layers,        # ===zgt===
            flow_cross_layers=cfg.flow_cross_layers,    # ===zgt===
            flow_enc_hidden=cfg.flow_enc_hidden,        # ===zgt===
            whenami_stage=cfg.whenami_stage,
            past_init=cfg.past_init,
        )
        # del this variable to avoid bugs. The vlm shouldn't be used anymore
        del vlm

    emit_memory("after_model_load")
    # [Validate] Model should be in Full Precision!
    for param in vla.parameters():
        assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"

    # Determine training "stage" based on frozen vs unfrozen parameters --> supports different fine-tuning schemes!
    if not cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        stage = "full-finetune"  # Full fine-tuning
    elif cfg.vla.freeze_vision_backbone and not cfg.vla.freeze_llm_backbone:
        stage = "finetune"  # Frozen vision encoder
    elif cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone:
        stage = "align"  # Fine-tuning projector
    elif not cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone and cfg.vla.unfreeze_last_llm_layer:
        stage = "vla-sandwich-train"  # Fine-tuning vision encoder, projector, and LLM last layer
    elif cfg.vla.freeze_vision_backbone and cfg.vla.freeze_llm_backbone and cfg.vla.unfreeze_last_llm_layer:
        stage = "vla-last-layer-train"  # Fine-tuning LLM last layer only
    else:
        raise ValueError(
            "Weight freezing configuration not supported. VLA config has the following parameters: "
            f"freeze_vision_backbone: {cfg.vla.freeze_vision_backbone}"
            f"freeze_llm_backbone: {cfg.vla.freeze_llm_backbone}"
            f"unfreeze_last_llm_layer: {cfg.vla.unfreeze_last_llm_layer}"
        )

    # [Explicit] Call to `freeze_backbones` here for clarity =>> will log exactly what is/is not frozen
    overwatch.info(f"Invoking `VLM.freeze_backbones()` for `{vla_id}` => Stage: `{stage}`")
    vla.freeze_backbones(stage)

    # Print number of total/trainable model parameters
    num_params = sum(p.numel() for p in vla.parameters())
    num_trainable_params = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    overwatch.info(
        f"# Parameters (in millions): {num_params / 10**6:.3f} Total, {num_trainable_params / 10**6:.3f} Trainable"
    )

    overwatch.info(f"Creating VLA Open-X Dataset with Mixture `{cfg.vla.data_mix}`")
    # ===zgt===
    # 窗口参数由 obs_offsets 推导，而非写死 —— 三者必须自洽：
    #   past_action_window_size >= max(-offset)   否则取不到最早的 past 帧
    #   future_action_window_size >= max(offset) + Ta - 1
    #                                             否则最晚的节点缺动作监督
    # 不自洽时数据层/损失层会各自报错并给出建议值，不会静默出错。
    offsets = list(cfg.obs_offsets) if cfg.obs_offsets else None
    if offsets:
        past_window = max(0, -min(offsets))
        future_window = max(cfg.Ta - 1, max(offsets) + cfg.Ta - 1)
    else:
        past_window, future_window = 0, cfg.Ta - 1

    emit_memory("before_dataset_build")
    vla_dataset, _, collator = get_vla_dataset_and_collator(
        cfg.data_root_dir,
        cfg.vla.data_mix,
        image_transform=vla.vision_backbone.get_image_transform(),
        tokenizer=vla.llm_backbone.get_tokenizer(),
        prompt_builder_fn=vla.llm_backbone.prompt_builder_fn,
        default_image_resolution=vla.vision_backbone.default_image_resolution,
        shuffle_buffer_size=cfg.vla.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        load_all_data_for_training=cfg.load_all_data_for_training,
        future_action_window_size=future_window,
        past_action_window_size=past_window,
        obs_offsets=offsets,
    )
    # ===zgt===

    emit_memory("after_dataset_build")
    # Save dataset statistics for de-normalization at inference time
    if overwatch.is_rank_zero():
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    dist.barrier()
    # Create Train Strategy
    overwatch.info(f"Initializing Train Strategy `{cfg.train_strategy}`")
    train_strategy = get_train_strategy(
        train_strategy=cfg.train_strategy,
        vlm=vla,
        device_id=device_id,
        stage=stage,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        enable_gradient_checkpointing=cfg.vla.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.vla.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.vla.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
    )
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(vla_dataset))
    emit_memory("after_fsdp_setup")
    if cfg.pretrained_checkpoint is not None and cfg.is_resume:
        train_strategy.load_optimizer_and_scheduler(cfg.pretrained_checkpoint)

    # Create Metrics =>> Handles on the fly tracking, logging to specified trackers (e.g., JSONL, Weights & Biases)
    overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")
    metrics = VLAMetrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        draccus.encode(cfg),
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        resume_step=cfg.resume_step,
        resume_epoch=cfg.resume_epoch,
    )

    # Run VLA Training
    overwatch.info("Starting VLA Training Loop")
    train_strategy.run_vla_training(
        vla_dataset,
        collator,
        metrics,
        save_interval=cfg.save_interval,
        action_model=True,
        repeated_diffusion_steps=cfg.repeated_diffusion_steps if cfg.whenami_stage == 2 else None,
    )

    # Finalize
    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    # And... we're done!
    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    train()
