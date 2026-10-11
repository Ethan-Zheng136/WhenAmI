"""CPU contract tests using actual WoG methods and small trainable backbones.

No pretrained weights, GPU, network, or dataset writes. The heavyweight VLM /
frozen vision encoder are replaced only at construction; QFormer, DiT, loader,
RLDS chunking, batch transform and collator run their actual implementations.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from PIL import Image

from action_model.action_model import ActionModel
from action_model.models import DiT
from prismatic.models.backbones.vision.dinosiglip_vit import FutureEnc
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.datasets.datasets import RLDSBatchTransform
from prismatic.vla.datasets.rlds.traj_transforms import chunk_act_obs
from vla.past import PastSpec, checkpoint_spec, history_frames
from vla.qformer import QFormer
from vla.vla import WoG
import vla.vla as vla_module


torch.set_num_threads(1)


def transform(image):
    return {"dino": torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255}


class TinyVLM(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.projector = nn.Linear(3, 32)
        self.vision_backbone = nn.Module()
        self.vision_backbone.featurizer = SimpleNamespace(patch_embed=SimpleNamespace(num_patches=1))
        self.vision_backbone.image_transform = transform
        self.llm_backbone = nn.Module()
        self.llm_backbone.llm = nn.Module()
        self.llm_backbone.llm.lm_head = nn.Linear(32, 8)
        self.all_module_keys = ["projector", "vision_backbone", "llm_backbone"]
        self.trainable_module_keys = self.all_module_keys
        self.device = torch.device("cpu")

    def forward(self, pixel_values, input_ids=None, attention_mask=None, **kwargs):
        x = pixel_values["dino"].mean((-2, -1))
        h = self.projector(x).unsqueeze(1)
        return SimpleNamespace(hidden_states=[h.expand(-1, attention_mask.shape[1] + 1, -1)])


class TinyFrozenVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 32)
        # Production reloads the same frozen DINO checkpoint on construction.
        # Mimic that instead of silently generating different frozen features.
        with torch.no_grad():
            self.proj.weight.copy_(torch.arange(96).reshape(32, 3) / 96)
            self.proj.bias.zero_()
        self.calls = []

    def forward(self, flat, base, video_pixel_values=None):
        b = base["dino"].shape[0]
        video = video_pixel_values["dino"] if video_pixel_values is not None else torch.cat(
            [base["dino"].unsqueeze(1), flat["dino"].reshape(b, -1, *flat["dino"].shape[1:])], 1
        )
        self.calls.append(video.detach().clone())
        patches = self.proj(flat["dino"].mean((-2, -1))).unsqueeze(1).expand(-1, 16, -1)
        latent = video.mean((2, 3, 4)).mean(1).view(b, 1, 1).expand(-1, 32, 784)
        return patches, latent


class TinyActionModel(ActionModel):
    def __init__(self, token_size=32, in_channels=7, Ta=16, train_venc=True, **kwargs):
        nn.Module.__init__(self)
        self.in_channels = in_channels
        self.train_venc = train_venc
        self.net = DiT(
            token_size=token_size, in_channels=in_channels, inj_channels=32,
            hidden_size=64, depth=2, num_heads=4, future_action_window_size=Ta - 1,
            use_per_attn=train_venc, obs_dropout_prob=0 if train_venc else 1,
        )


@pytest.fixture
def factory(monkeypatch):
    monkeypatch.setattr(vla_module, "FutureEnc", TinyFrozenVision)
    monkeypatch.setattr(vla_module, "ActionModel", TinyActionModel)
    monkeypatch.setattr(vla_module, "PrismaticVLM", TinyVLM)
    monkeypatch.setattr(
        vla_module, "QFormer",
        lambda **kw: QFormer(32, 4, 64, num_layers=2, num_query_tokens=16),
    )

    def make(stage=2, **kwargs):
        return WoG(TinyVLM(), token_size=32, train_venc=True, whenami_stage=stage, **kwargs)
    return make


def inputs():
    torch.manual_seed(4)
    return {
        "pixel_values": {"dino": torch.randn(2, 3, 8, 8)},
        "past_pixel_values": {"dino": torch.randn(2, 4, 3, 8, 8)},
        "future_pixel_values": {},
        "actions": torch.randn(2, 31, 7),
        "action_masks": torch.ones(2, 31, dtype=torch.bool),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "input_ids": torch.ones(2, 3, dtype=torch.long),
    }


@pytest.mark.parametrize("offsets", [
    [-15, -11, -7, -3, 1], [-4, -3, -2, -1, 0, 1],
    [-3, -1, 0], [0], [-4, -3, -3, -1, 0], [0, -3, -7, -11, -15],
])
def test_invalid_history_spec(offsets):
    with pytest.raises(ValueError):
        PastSpec.from_offsets(offsets)


@pytest.mark.parametrize("offsets", [None, [-4, -3, -2, -1, 0], [-20, -14, -8, -2, 0]])
def test_current_actions_use_raw_offset_not_frame_count(offsets):
    s = PastSpec.from_offsets(offsets)
    a = torch.arange(-s.lookback, 16).view(1, -1, 1)
    mask = torch.ones(1, a.shape[1], dtype=torch.bool)
    mask[:, -1] = False
    target, selected = s.action_targets(a, mask)
    assert target.flatten().tolist() == list(range(16))
    assert selected.shape == (1, 16) and not selected[0, -1]


def test_history_sampler_episode_start_and_raw_stride():
    assert history_frames([0]) == [0, 0, 0, 0]
    assert history_frames(list(range(21))) == [5, 9, 13, 17]
    assert history_frames([100, 101]) == [100] * 4


def test_checkpoint_stage_identity_is_not_guessed_from_config():
    spec = PastSpec()
    ck = {"model": {"past_qformer": {}}, "whenami": spec.metadata()}
    assert checkpoint_spec(ck) == (2, list(spec.offsets))
    with pytest.raises(ValueError):
        checkpoint_spec(ck, 0)
    with pytest.raises(ValueError):
        checkpoint_spec(ck, 2, [-4, -3, -2, -1, 0])
    with pytest.raises(ValueError):
        checkpoint_spec({"model": {"past_qformer": {}}})
    assert checkpoint_spec({"model": {"qformer": {}}}, 2)[0] == 2
    assert checkpoint_spec({"model": {"converge": {}}}) == (0, None)


def test_condition_is_causal_and_video_is_chronological(factory):
    m = factory().eval()
    b = inputs()
    c, nodes, n_past, aux = m._compute_condition(
        b["pixel_values"], {}, b["past_pixel_values"], b["actions"]
    )
    assert c.shape == (2, 16, 32) and nodes is None and n_past == 4 and aux is None
    video = m.v_enc.calls[-1]
    assert torch.equal(video[:, :4], b["past_pixel_values"]["dino"])
    assert torch.equal(video[:, 4], b["pixel_values"]["dino"])
    changed = b["actions"] * 100
    other = m._compute_condition(b["pixel_values"], {}, b["past_pixel_values"], changed)[0]
    torch.testing.assert_close(c, other, rtol=0, atol=0)
    with pytest.raises(ValueError, match="future"):
        m._compute_condition(
            b["pixel_values"], {"dino": torch.randn(2, 1, 3, 8, 8)},
            b["past_pixel_values"], b["actions"],
        )


def test_history_reaches_condition_and_encoder_stays_frozen(factory):
    m = factory().eval()
    b = inputs()
    c = m._compute_condition(b["pixel_values"], {}, b["past_pixel_values"], None)[0]
    other = m._compute_condition(
        b["pixel_values"], {}, {"dino": b["past_pixel_values"]["dino"] + 0.3}, None
    )[0]
    assert (c - other).abs().max() > 1e-6
    m.train()
    assert m.past_qformer.training and not m.v_enc.training
    assert not any(p.requires_grad for p in m.v_enc.parameters())


def test_actual_forward_targets_and_repeat_count(factory, monkeypatch):
    m = factory()
    b = inputs()
    b["actions"] = torch.arange(-15, 16).view(1, 31, 1).expand(2, 31, 7).float()
    grabbed = {}

    def loss(x1, z, obs, action_mask=None):
        grabbed.update(x1=x1, z=z, obs=obs, mask=action_mask)
        return obs.square().mean()
    monkeypatch.setattr(m.action_model, "loss", loss)
    m(**b, repeated_diffusion_steps=3)
    assert grabbed["x1"].shape == (6, 16, 7)
    assert grabbed["x1"][0, :, 0].tolist() == list(range(16))
    assert grabbed["mask"].shape == (6, 16)


def test_actual_qformer_dit_backward_and_parameter_update(factory):
    torch.manual_seed(8)
    m = factory().train()
    b = inputs()
    before = m.past_qformer.v_proj.weight.detach().clone()
    optimizer = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    for _ in range(4):
        optimizer.zero_grad()
        loss, _, act, obs = m(**b, repeated_diffusion_steps=1)
        assert torch.isfinite(loss) and obs == 0
        loss.backward()
        optimizer.step()
    assert not torch.equal(before, m.past_qformer.v_proj.weight)
    assert m.past_qformer.v_proj.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in m.v_enc.parameters())


def test_action_mask_math_and_legacy_default():
    m = TinyActionModel().eval()
    x = torch.randn(2, 16, 7)
    z = torch.randn(2, 1, 32)
    c = torch.randn(2, 16, 32)
    torch.manual_seed(1)
    plain = m.loss(x, z, c)
    torch.manual_seed(1)
    masked = m.loss(x, z, c, torch.ones(2, 16, dtype=torch.bool))
    torch.testing.assert_close(plain, masked)
    assert m.loss(x, z, c, torch.zeros(2, 16, dtype=torch.bool)) == 0
    with pytest.raises(ValueError):
        m.loss(x, z, c, torch.ones(2, 31))


def test_wide_queries_preserve_original_forward():
    q = QFormer(32, 4, 64, num_layers=2, num_query_tokens=16).eval()
    x, vae = torch.randn(2, 20, 32), torch.randn(2, 32, 784)
    visual = torch.cat([x, q.vae_proj(vae)], dim=1)
    expected = q.v_proj(q.bert(
        inputs_embeds=q.query_tokens.expand(2, -1, -1),
        encoder_hidden_states=visual, return_dict=True,
    ).last_hidden_state)
    torch.testing.assert_close(q(x, vae), expected, rtol=0, atol=0)
    assert q.encode_queries(x, vae).shape == (2, 16, 32)


def test_visual_encoder_override_preserves_legacy_and_causal_order():
    enc = FutureEnc.__new__(FutureEnc)
    nn.Module.__init__(enc)
    enc.dino_featurizer = nn.Identity()
    enc.register_buffer("mean", torch.zeros(3))
    enc.register_buffer("std", torch.ones(3))
    clips = []

    def encode(videos):
        clips.append(videos[0].clone())
        return [torch.zeros(16, 2, 2, 2)]
    enc.vae = SimpleNamespace(encode=encode)
    hist = torch.arange(4).view(1, 4, 1, 1, 1).expand(1, 4, 3, 2, 2).float() / 10
    now = torch.full((1, 3, 2, 2), 0.4)
    flat = {"dino": hist.flatten(0, 1)}
    enc(flat, {"dino": now})
    torch.testing.assert_close(clips[-1][0, :, 0, 0], torch.tensor([0.4, 0., 0.1, 0.2, 0.3]))
    enc(flat, {"dino": now}, {"dino": torch.cat([hist, now[:, None]], 1)})
    torch.testing.assert_close(clips[-1][0, :, 0, 0], torch.tensor([0., 0.1, 0.2, 0.3, 0.4]))


def save_tiny_checkpoint(path, model, past=False):
    groups = {k: getattr(model, k).state_dict() for k in ["action_model"]}
    groups["past_qformer" if past else "qformer"] = (
        model.past_qformer if past else model.qformer
    ).state_dict()
    groups.update({k: getattr(model.vlm, k).state_dict() for k in model.vlm.all_module_keys})
    payload = {"model": groups}
    if past:
        payload["whenami"] = model.get_whenami_metadata()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def test_actual_from_pretrained_migrates_future_then_reloads_past(factory, tmp_path):
    future = factory(stage=0).eval()
    path = tmp_path / "checkpoints" / "future.pt"
    save_tiny_checkpoint(path, future)
    past = WoG.from_pretrained(
        path, "tiny", None, None, train_venc=True, whenami_stage=2, freeze_weights=False
    ).eval()
    assert past.qformer is None and past.converge is None
    assert "past_qformer" in past.trainable_module_keys
    for k, v in future.qformer.state_dict().items():
        torch.testing.assert_close(v, past.past_qformer.state_dict()[k], rtol=0, atol=0)
    save_tiny_checkpoint(tmp_path / "checkpoints" / "past.pt", past, past=True)
    reloaded = WoG.from_pretrained(
        tmp_path / "checkpoints" / "past.pt", "tiny", None, None, train_venc=True
    )
    b = inputs()
    expected = past._compute_condition(b["pixel_values"], {}, b["past_pixel_values"], None)[0]
    actual = reloaded._compute_condition(b["pixel_values"], {}, b["past_pixel_values"], None)[0]
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert not any(p.requires_grad for p in reloaded.parameters())


def test_full_load_vla_recovers_stage_and_offsets(factory, tmp_path, monkeypatch):
    import importlib
    loader = importlib.import_module("vla.load")
    m = factory().eval()
    path = tmp_path / "checkpoints" / "past.pt"
    save_tiny_checkpoint(path, m, past=True)
    # Deliberately wrong sampling in mutable config: immutable payload wins.
    (tmp_path / "config.json").write_text(json.dumps({
        "vla": {"base_vlm": "tiny"}, "whenami_stage": 0, "obs_offsets": [0, 1, 2, 3, 4],
    }))
    (tmp_path / "dataset_statistics.json").write_text("{}")
    cfg = SimpleNamespace(
        model_id="tiny", vision_backbone_id="tiny", llm_backbone_id="tiny",
        arch_specifier="gelu-mlp", image_resize_strategy="resize-naive", llm_max_length=16,
    )
    monkeypatch.setattr(loader, "ModelConfig", SimpleNamespace(get_choice_class=lambda _: lambda: cfg))
    monkeypatch.setattr(loader, "get_vision_backbone_and_transform", lambda *a, **kw: (None, None))
    monkeypatch.setattr(loader, "get_llm_backbone_and_tokenizer", lambda *a, **kw: (None, None))
    got = loader.load_vla(path, action_model_type="DiT-L")
    assert got.whenami_stage == 2 and got.train_venc is True
    assert got.obs_offsets == [-15, -11, -7, -3, 0]
    with pytest.raises(ValueError):
        loader.load_vla(path, whenami_stage=0)


def test_openvla_start_requires_explicit_random_head(factory, tmp_path):
    source = factory(stage=0)
    path = tmp_path / "checkpoints" / "openvla.pt"
    save_tiny_checkpoint(path, source)
    payload = torch.load(path, weights_only=True)
    del payload["model"]["qformer"], payload["model"]["action_model"]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="warm start"):
        WoG.from_pretrained(path, "tiny", None, None, whenami_stage=2, train_venc=True)
    past = WoG.from_pretrained(
        path, "tiny", None, None, whenami_stage=2, train_venc=True, past_init="random"
    )
    assert past.past_qformer is not None and past.converge is None


def test_raw_rlds_chunk_transform_collator_to_model(factory):
    import tensorflow as tf
    t = 50
    traj = {
        "action": tf.tile(tf.cast(tf.range(t)[:, None], tf.float32), [1, 7]),
        "observation": {"image_primary": tf.broadcast_to(
            tf.cast(tf.range(t)[:, None, None, None], tf.uint8), [t, 8, 8, 3]
        )},
        "task": {"language_instruction": tf.fill([t], "pick the cup")},
        "dataset_name": tf.fill([t], "test"),
    }
    chunk = chunk_act_obs(
        traj, window_size=16, future_action_window_size=15,
        dataset_statistics={"action": {"q01": np.full(7, -1), "q99": np.ones(7)}},
        obs_offsets=list(PastSpec().offsets),
    )
    one = tf.nest.map_structure(lambda x: x[20].numpy(), chunk)
    assert one["observation"]["image_primary"][:, 0, 0, 0].tolist() == [5, 9, 13, 17, 20]

    class Prompt:
        def __init__(self, *args):
            pass
        def add_turn(self, *args, **kwargs):
            pass
        def get_prompt(self):
            return "pick the cup"

    transform_batch = RLDSBatchTransform(
        action_tokenizer=None,
        base_tokenizer=lambda *a, **kw: SimpleNamespace(input_ids=[1, 2, 3]),
        image_transform=transform, prompt_builder_fn=Prompt,
        obs_offsets=list(PastSpec().offsets),
    )
    collator = PaddedCollatorForActionPrediction(16, 0)
    batch = collator([transform_batch(one)])
    assert batch["future_pixel_values"] == {}
    assert batch["past_pixel_values"]["dino"].shape == (1, 4, 3, 8, 8)
    target, _ = PastSpec().action_targets(batch["actions"], batch["action_masks"])
    assert target[0, :, 0].tolist() == list(range(20, 36))
    batch.pop("dataset_names")
    loss = factory()(**batch, repeated_diffusion_steps=1)[0]
    assert torch.isfinite(loss)


def test_train_stage_guards(factory):
    from scripts.train import TrainConfig
    import conf
    cfg = conf.VLAConfig.get_choice_class(
        conf.VLARegistry.EXP_WoG_OXE_MAGIC_SOUP_PLUS_MINUS.vla_id
    )()
    cfg.expected_world_size = 1
    c = TrainConfig(
        vla=cfg, whenami_stage=2, train_venc=True, is_resume=False,
        pretrained_checkpoint="future.pt",
    )
    assert c.obs_offsets == PastSpec().offsets
    with pytest.raises(ValueError, match="resume"):
        TrainConfig(vla=cfg, whenami_stage=2, train_venc=True, is_resume=True)
    with pytest.raises(ValueError):
        TrainConfig(vla=cfg, whenami_stage=2, train_venc=False, is_resume=False)


def test_stage2_missing_history_fails_before_vlm_generation(factory):
    m = factory()
    image = Image.new("RGB", (8, 8))
    with pytest.raises(ValueError, match="past_frames"):
        m.predict_action(image=image, instruction="pick")
    with pytest.raises(ValueError, match="legacy"):
        m.predict_action(image=image, instruction="pick", cond_frames=[image] * 5)


def test_inference_condition_matches_training_condition(factory):
    m = factory().eval()
    frames = [Image.new("RGB", (8, 8), color=(i * 20, i * 20, i * 20)) for i in range(5)]
    current = {k: v[None] for k, v in transform(frames[-1]).items()}
    past = {"dino": torch.stack([transform(im)["dino"] for im in frames[:-1]])[None]}
    expected = m._compute_condition(current, {}, past, None)[0]
    actual = m._condition_from_frames(frames, None, current, torch.float32)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)


@pytest.mark.skipif(os.environ.get("WAI_TEST_GPU") != "1", reason="opt-in single-GPU FSDP checkpoint test")
def test_fsdp_saver_keeps_past_identity(factory, tmp_path):
    import torch.distributed as dist
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig, FullOptimStateDictConfig,
    )
    from training.strategies.fsdp import FSDPStrategy

    dist.init_process_group("nccl", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1)
    try:
        torch.cuda.set_device(0)
        m = factory().cuda()
        keys = m.all_module_keys
        wrapped = FSDP(m, use_orig_params=True, device_id=0)
        runner = FSDPStrategy.__new__(FSDPStrategy)
        runner.vlm = wrapped
        runner.all_module_keys = keys
        runner.trainable_module_keys = keys
        runner.fsdp_state_dict_type = StateDictType.FULL_STATE_DICT
        runner.fsdp_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        runner.fsdp_save_optimizer_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        (tmp_path / "checkpoints").mkdir()
        runner.save_checkpoint(tmp_path, 1, 0, 0.1, only_trainable=False)
        saved = torch.load(next((tmp_path / "checkpoints").glob("*.pt")), weights_only=True)
        assert "past_qformer" in saved["model"] and "qformer" not in saved["model"]
        assert checkpoint_spec(saved) == (2, [-15, -11, -7, -3, 0])
    finally:
        dist.destroy_process_group()
