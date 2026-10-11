# Timeline-A construction and smoke admission — 2026-10-04

User authorized implementation, primary-source/code reuse, and real smoke tests.
This is Phase A (privileged representation learning), NOT the completed main
Stage3 and NOT a deployed causal student. Phase B remains a separate experiment.

## Implement now

Raw OXE TRAIN episodes -> distinct actual anchors -> per-frame frozen DINO and
single-frame Wan encoding -> shared trainable local QFormer -> temporal routing
-> one shared fully trainable pretrained action DiT at every anchor.

- Start all arms from the SAME Future Stage1 endpoint, not three different bases.
- Freeze VLM/visual foundations in this first admission. Train all local-QFormer
  and DiT parameters, plus the new temporal conditioner. No frozen-reader shortcut.
- Three modes share parameter names/shapes and initialization:
  `pooled` compresses allowed source-anchor tokens into one shared condition bank;
  `independent` only reads its own visible node;
  `timeline` reads separate visible time-addressed source nodes.
- All modes use the same source-frame count, target nodes,16 query tokens/anchor,
  hidden width/layers, action targets, training updates and noise. Routing differs.
- Pooling and local retention necessarily produce different initial conditions;
  common weights do NOT imply identical initial policies. Report initial metrics.
  `pooled` is a compression control, NOT an exact reimplementation of WoG-A/V.
- Local information is encoded from ONE actual frame per node. Wan receives one
  frame, not a full clip sliced after encoding. This prevents a hidden node's
  image leaking through another node's local temporal encoder. It changes the
  Stage1 visual input distribution; smoke is admission, not a quality claim.
- No ground-truth actions enter the conditioner. Real actions appear ONLY in
  labels. This Phase-A build is a temporal-relation model, not an action-conditioned
  physical transition or inverse dynamics model.

## Evidence masking and leakage

- Hidden source images, local features and source VLM cognition are removed from
  all conditioning paths. Local features are masked BEFORE trainable QFormer.
- For hidden target nodes, their image-derived VLM cognition is ALSO replaced by
  a trainable missing-cognition token. Same language is available through visible
  evidence nodes; no task-changing trajectory is selected.
- Reader receives its own image cognition only when that node is visible/legal.
  This is stronger than masking a memory branch while keeping the hidden image in z.
- Per-target evidence cutoff is supported. A fixed local-evidence memory is used
  at EVERY temporal layer; no hidden-to-hidden cross-target communication can
  carry future evidence around a triangular mask.
- Null evidence token handles all-hidden cases without NaNs; that case is only an
  admission edge test, not the default training distribution.
- Time IDs/Δt are metadata, not normalized episode progress or total length.
- Prefix API proves wiring causality, not that a Phase-A teacher works as a Phase-B
  student. Future target-image access remains privileged during Phase-A training.

## Objectives and scope

- Primary loss: valid-action-mask FM at EVERY actual anchor, equal per-node weight.
- Half admission updates fully visible; half hide an interior node. These are
  fixed smoke settings, not chosen by validation performance.
- No free latent-cycle objective, no arbitrary inverse-variance gate, no assertion
  that action loss alone identifies physical states or prevents all collapse.
- Store original pure-noise action samples at initial/final smoke weights, same
  noise for all arms; these training-fixture diagnostics are NOT generalization.
- A full pipeline mode must touch every real raw frame independently, not only
  simulate high-level tensors. Also exercise perturbation of hidden/future frames.

## Admission before any cluster claim

1. CPU shape, masking, cutoff, null evidence, hidden-input gradient, mode count,
   position-shift, singleton and same-weight load tests.
2.27 real OXE domains, distinct anchors, source TRAIN identities only. No dev/test
   images. Keep labels/masks separate from policy inputs.
3. Real shared QFormer/DiT updates in all three modes; action-only gradient into
   temporal input/attention/output, local QFormer and original DiT.
4. Zero temporal residual reproduces local base condition (independent/timeline).
   Pooled base reproduces explicitly computed pooled local queries.
5. One original-backbone full save -> new process restore -> update, with state,
   optimizer and RNG restored. Preserve provenance; do not modify source weights.
6. Generic CPU two-rank DDP if feasible; actual multinode is still unvalidated.

## Reuse and licenses

- WoG local repository modules (MIT; source copyright remains in original files):
  QFormer, DiT, DINO/Wan, VLM/tokenizer and flow-matching behavior.
- Existing project raw OXE shards/reader and per-node action masks; no dataset
  download or new TFRecord preparation.
- I-JEPA official repository cloned and read for context/target evidence separation;
  its CC-BY-NC4.0 code is NOT copied/imported into this implementation.
- Masked temporal prediction and multiple time tokens are not claimed as novel.
  This implementation review does not constitute an exhaustive novelty search.
- New code implements only this experiment's explicit routing/contracts.
