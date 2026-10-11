# Full-VLM Timeline-A — 2026-10-05

User offers640PPUs. Intended allocation: two independent320PPU/20node jobs,
one timeline and one stronger pooled-query comparator. Not a640-rank job, not
a continuation of the earlier32-rank optimizer, and not an already validated
novel algorithm. Full Phase-B remains unimplemented.

Confirmed earlier frozen-VLM32-rank arms have reached120 updates with save,
new-process restart and27-domain evaluation; their code/weights remain untouched.

## New trainable graph

- VLM vision/projector/LLM + local QFormer + temporal conditioner + FULL original
  pretrained DiT are inside one FSDP root. No detached or cached VLM cognition.
- Frozen standalone DINO/Wan remain outside FSDP, operate per frame (Wan T=1).
- VLM receives zeroed hidden-node pixels; hidden cognition/local features are
  also removed before temporal routing and action reading.
- Target noisy actions enter the action DiT ONLY, never a source-state encoder.
- Every flow target/anchor is masked for valid action labels; noise is generated
  without coupling one anchor's labels to another anchor's flow state.
- All modes use the same source nodes, target query time encoding, local base,
  attention shapes, parameter names/initialization, optimization and data.

## Two arms

`timeline`: each target attends to separately time-addressed visible source nodes.
`pooled_query`: same time-addressed sources compressed to a16-query bank, then each
target uses its OWN local base and explicit target-time query. This is stronger
than the old `pooled` which forced a shared output for identical evidence.
Pooling reduces information by design; neither arm is declared an exact WoG
replication. Need broader trained controls/Phase-B for method claims.

## Draft full recipe (admission precedes long run)

Per arm320PPU,20x16; micro1 trajectory/rank,global320 trajectories (960 anchors),
2 FM draws/anchor;50k updates; cosine decay,1000 warmup; existing pretrained
VLM/Q/DiT LR1e-5, temporal new parameters LR1e-4.
Sharding hybrid ZeRO2, LLM activation checkpointing, save5000/eval2500, early
save/eval100. Same source statistics and data pool, no image augmentation in this
first matched recipe (explicit, not silently claimed identical to old Stage3).

Scheduled admission: tiny SAME-topology FSDP save/resume proof -> real100 updates
-> new-process resume120. PHASE=admit stops here; PHASE=run validates all proof
files and automatically continues. All stages use50k schedule from the beginning.
After a successful admission-only job, PHASE=full may separately continue to50k.
Do not auto-run a long allocation through a failed check. Fresh output IDs only.

## Engineering evidence labels

Local full8B backward smoke can validate gradient/memory/masking but cannot fit
full FP32 Adam state on one96GiB device. Full checkpoint/optimizer code is tested
using the same FSDP classes on small models; actual8B sharded Adam/save/resume must
be admitted on the cluster. Do not misrepresent the tiny proof as full-model
multinode validation.

Evaluation re-encodes current images through UPDATED VLM for every checkpoint,
with equal numbers of root FSDP calls on every rank. Only frozen visual features
may be cached. Both unmasked and masked-node pure-noise actions are evaluated
on all27 existing-dev domains; old training-fixture metrics are not validation.
