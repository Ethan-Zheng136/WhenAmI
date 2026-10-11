# WhenAmI Timeline-A

**Phase-A privileged condition learning, not a new deployment Stage4.**
Independent package; does not modify existing Stage3, checkpoints or evaluation runs.

## Pipeline

```text
one real frame/node → frozen DINO + single-frame Wan → local features
                                                      ↓
                                       shared TRAINABLE QFormer
                                                      ↓
                          pooled / independent / timeline routing
                                                      ↓
                             one condition per actual time anchor
                                                      ↓
                         SAME fully TRAINABLE original action DiT
                                                      ↓
                           all-anchor valid-mask action FM loss
```

VLM supplies per-node cognition and remains frozen for first admission. A hidden
node loses BOTH its local features and its image-related cognition; a missing
cognition token replaces the latter. Future actions are labels only, never inputs
to the condition model.

## Three controls

| mode | information path |
|---|---|
| `pooled` | compress all permitted local source states into one common query bank |
| `independent` | each target reads its own visible local node only |
| `timeline` | target explicitly attends to distinct visible source nodes with relative time |

Same parameters, initialization, frames, targets and attention tensor dimensions.
Pooling changes the base condition, so common weights do **not** mean identical
initial policy outputs. The independent/timeline zero residual starts identically.
Pooled is a compression control, not a claim to reproduce published WoG.

Conditioner width256; local states stay1024D until the shared32D action-condition
projection.16 queries per anchor are NOT16 different time points.
Time anchors use actual frame indices and variable1–12 gaps (bounded by length).
This build has no action-conditioned physical transition or reversible/cycle model.

## Evidence boundary

- Visual encoders process each frame independently; no full-video VAE sliced later.
- Local features masked before QFormer; hidden cognition removed on source and reader paths.
- Each target has an evidence cutoff. EVERY relation layer reads immutable local
  evidence, preventing a later query from relaying illegal evidence to an earlier one.
- An explicit null evidence token handles all-hidden cases without NaNs.
- Full evidence at teacher training is privileged; prefix mask support is only a
  wiring contract, not proof of learned causal deployment.

Single-frame visual conditioning differs from the Future endpoint's original
multiframe input. This is an intentional safe initial interface, not a statement
that local single-frame encoding is the best final architecture.

## Existing components reused

- Future Stage1:
  `outputs/stage1/baseline_v2--image_aug/checkpoints/step-020000-epoch-02-loss=0.1076.pt`.
- Original local WoG QFormer, DiT-L, DINO/Wan, VLM/tokenizer.
- Existing OXE27 prepared TRAIN-only episodes; no new dataset download.
- Original FM distribution and action normalization/masking.

I-JEPA reference cloned for review of context/target masking:

```text
outputs/whenami_timeline_a/refs/ijepa
commit 52c1ae95d05f743e000e8f10a1f3a79b10cff048
```

Its CC-BY-NC4.0 code is not copied/imported here; new masking/routing is implemented
locally. Original WoG licensing/notices remain in the reused source tree.
No claim that masking, time tokens or attention are themselves novel.

## Local commands

```bash
cd /home/zhengguantian/WoG_fresh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/mnt/pfs/zhengguantian/WoG_v2/pylibs-train:/home/zhengguantian/WoG_fresh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=3
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/mnt/pfs/zhengguantian/WoG_v2/hf_home

python -m pytest -q whenami_timeline_a/test_model.py

# New output required; do not repeat into completed directories.
python -m whenami_timeline_a.smoke \
  --output /mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_a/<new-smoke-name> \
  --steps 8 --nodes 3
```

Raw train runner (small admission, not recommended full scientific training budget):

```bash
python -m whenami_timeline_a.train --mode timeline \
  --steps 4 --save-every 2 --audit-state \
  --output /mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_a/<new-run-name>
```

Resume into a **new** output, preserving the original run:

```bash
python -m whenami_timeline_a.train --mode timeline \
  --steps 4 --save-every 2 --audit-state \
  --resume-from <old-run>/checkpoints/step-000002 \
  --output <new-resume-run>
```

`--audit-state` hashes every model and optimizer tensor for exact restart admission.
It is expensive and should not be enabled for large training by default.
Output directories and stop points are excluded from recipe identity; scientific
settings/source/world size must match. No silent optimizer reset.

## Outputs and what they mean

```text
outputs/whenami_timeline_a/
  tests_01.log
  smoke_real27_01/
    fixture_manifest.json
    train_fixture.pt
    {pooled,independent,timeline}_updates.json
    {pooled,independent,timeline}_samples.pt
    RESULT.json
  raw_continuous4_01/
  raw_resumed4_01/
```

Real smoke: all27 TRAIN domains encoded/scored,3 actual anchors/domain, each mode8
updates on the same16 domain fixtures.521,146,151 trainable parameters/mode,
including the original action DiT and local QFormer. This is not the old1.26M
frozen-reader timeline experiment.

The small smoke fixture is not a held-out benchmark. Initial/final pure-noise
action outputs are recorded to expose bugs; a decrease is not method validation.
Phase-B student, held-out effectiveness and actual multi-node PPU training remain
separate next tasks. Do not infer that a large training run is admitted merely
because CPU DDP passes.

Final admission summary:
`../WHENAMI_TIMELINE_A_SMOKE_20261004.md`.
The original intermediate/duplicate raw-run checkpoints were removed after full
model/Adam hash parity was verified. Metadata and the cleanup manifest remain in
`outputs/whenami_timeline_a/cleanup_01/`; the resumed final4 checkpoint is retained.
