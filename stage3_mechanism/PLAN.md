# Stage3-50k: current-model mechanism diagnostic, 2026-10-04

User authorized local evaluation/experiments. This study is diagnosis of an
existing policy, NOT a novel algorithm and NOT a matched retraining ablation.

## Questions

1. Does the trained K condition correction help or hurt generated actions?
2. Does K benefit from R's sample-specific predicted Future beyond H_P/z?
3. Does actual historical input help versus repeated-current input?
4. Are history dependence and K correction coupled?

## Fixed protocol before observing new interventions

- Fixed committed Stage3-50k checkpoint; no source weights or optimizer updated.
- All1348 existing Stage3-dev episodes /2694 windows /27 domains already used by
  the completed50k comparison. No new split, no sealed test, no cherry-picking.
- Inference seeds42/43/44 and exact stored FP32 Gaussian arrays;9 Euler intervals;
  BF16 policy parameters, FP32 buffers/integration; same action normalization.
- Five conditions:
  - full: unmodified causal student;
  - no_k: final condition becomes C_P, learned VLM/Past/DiT kept unchanged;
  - mean_future: keep H_P/z/C_P, supply standardized zero (=fixed R future_mean)
    to K instead of sample-specific predicted Future;
  - repeat_history: replace four history images with the current image and
    recompute full policy;
  - repeat_history_no_k: repeat history and remove K.
- Store per-window latent norm summaries and conditions, predictions, masks,
  target/noise identities. Atomic per-episode files, resume verified by hashes.
- Replay unmodified full predictions on EVERY episode against the completed50k
  baseline. Abort on meaningful drift; do not splice different protocols.
- Primary clipped-normalized MSE6, also unclipped MSE, first-action MSE,
  gripper, per-domain and episode-level metrics, fixed-model paired intervals.
- Report all domains, full-history vs boundary-padded strata.
- Infer neither causal-training contribution nor closed-loop success from these
  potentially out-of-distribution inference interventions.
- Removing predicted-Future content does NOT remove future-alignment's training
  influence on VLM/Past/R/K. A null inference effect cannot disprove that training
  influence; a harmful intervention cannot establish a full novel mechanism.

## Engineering admission

Free local PPU only. Host RAM is shared with other users. The diagnostic loader
initializes the trained architecture then copies/casts to private accelerator
storage BEFORE loading DCP tensors there; no writes to mmap warm-start tensors.
Strict key/shape/dtype audit and same50k full-output replay are required.
First27-domain observation pass is a wiring/throughput check, not a separately
selected performance test. No change to production code or completed evaluations.
