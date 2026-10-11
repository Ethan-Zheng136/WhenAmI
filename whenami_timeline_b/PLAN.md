# Phase-B controlled transfer pilot — 2026-10-05

User authorized further experiment construction and offers an additional640PPUs.
No claim of guaranteed conference acceptance. New runs must establish transfer,
not merely improve privileged teacher scores. Do not edit running A dependencies.

## Evidence available before defining this experiment

- Actual A5k teacher/student interface smoke passed, independent loaded teacher
  frozen, student action gradient live, future perturbation invariant.
- Fixed A5k on410 DEV episodes/27 domains/3 inference seeds:
  teacher middle-action MSE6=.0465251; same-weight two-node causal-prefix
  MSE6=.0499646; teacher advantage6.88%, paired interval[3.89%,10.14%].
  This is teacher headroom, NOT evidence that distillation will recover it.
- A12.5k teacher is still training; no checkpoint selected by best dev result.

## Fixed pilot recipe

- Pin A timeline step10000, latest COMPLETE saved checkpoint at experiment design.
  Both students start from its EXACT weights. Teacher remains fixed throughout.
  Do not follow newer A checkpoints automatically or resume A Adam/RNG.
- B0 `action_only`: causal action FM. B1 `condition_align`: same action FM
  plus0.1 cosine alignment to the frozen teacher's current16x32 condition.
- Both have the SAME auxiliary linear32->32 identity-initialized head, but B0
  does not train it. It never feeds action decoder; only shared representation
  gradients can transfer its training signal.
- Student only sees the physical two-node prefix (one history anchor, current).
  Teacher sees the three actual nodes (history/current/later observation).
  This first controlled pilot supervises the current middle action; it is NOT
  yet full all-time/horizon distillation or the final deployment history policy.
- Exact same training examples, identities, action/noise draws and update budget.
- 320PPU per arm,global320 trajectories,micro1,20k updates; pretrained body LR1e-5,
  fresh auxiliary head LR1e-4,1000 warmup,cosine floor0.1; save2500/eval1000,
  early100/120 admission. No image augmentation in this matched pilot.
- Teacher and student FSDP model initialization loads A-DCP model tensors only,
  not optimizer/RNG. Sharded Adam/RNG resumes belong only to the new B recipe.
- Both arms construct the same frozen teacher and run its forward during training,
  including B0, to keep compute/evidence-generation work matched. B0 never uses
  teacher target in loss. This intentional control overhead is disclosed.
- Raw OXE27 TRAIN input remains streaming; no trainable feature cache.
- Evaluation uses410 existing-dev middle actions under CAUSAL student inference,
  full27-domain macro and all-domain breakdown, fixed noise; final evaluation
  will use3 noise seeds and paired-episode uncertainty. No sealed-test access.

## Admission and resource boundary

CPU prefix/teacher-detach/decoupling/gradients, CPU multi-rank sharded A->B load,
new B model/Adam/RNG restart, actual single-PPU loaded A10k student and teacher
forward/backward, raw27dev evaluation. Actual320rank update/save/resume is required
before continuing to20k. Resource sufficiency measured there, not guessed.

All full learned body modules train; frozen teacher and DINO/Wan do not.
Single-PPU BF16/SGD smoke proves computation, not distributed fullFP32Adam state.

## What this experiment can and cannot establish

- B1 vs B0 tests incremental teacher-condition supervision at fixed A10k warm start.
- It does NOT isolate all timeline pretraining benefits because both inherit A.
- It does NOT compare a timeline teacher to a pooled teacher; that matched second
  comparison and additional training seeds remain needed.
- Small cosine loss may have limited action impact; log raw/weighted loss and
  auxiliary-head gradient, sample actions, don't treat low alignment as success.
- Teacher can depend on unknowable future; no deterministic target fit guarantee.
- Closed-loop same-initial-state success/recovery tests remain necessary and are
  assigned to the NVIDIA server. Provide an explicit prefix/history/action contract.
