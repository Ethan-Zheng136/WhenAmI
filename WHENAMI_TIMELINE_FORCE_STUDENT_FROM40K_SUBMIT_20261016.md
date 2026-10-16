# Stage II causal U student from A+ step-040000 — submit notes (2026-10-16)

Teacher (frozen, also the student's init):
`/mnt/pfs/zhengguantian/WoG_v2/outputs/whenami_timeline_force/training/timeline_force_Aplus_160_resume5k_01/main/checkpoints/step-040000`
(COMMITTED, world 160, sha256(COMMITTED.json) c9c2b202…).

## Submit (platform: 10 nodes x 16 PPU; WORLD_SIZE=10 node count, RANK=node rank)

```bash
VARIANT=revise RUN_ID=timeline_force_student_revise_from40k_160_01 NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_whenami_timeline_force_student.sh
```
20 nodes x16 also accepted (uses original whenami_timeline_force.train; default RUN_ID ..._320_01).
Default PHASE=run: topology probe -> admission1 (1-100, save100) -> admission2 (new process resume
101-120) -> check_admission -> main 120..50000. Resume: PHASE=resume RESUME_FROM=<student ckpt>, new RUN_ID.

## Recipe (identical to WHENAMI_TIMELINE_FORCE_320_SUBMIT_20261012.md Stage2)
init = teacher weights (1551 tensors) + new executed_adapter/forecast_head (8 tensors);
loss = action FM (3 prefix-current requests, equal) + 0.1 current-C + 0.1 node-cognition + 0.1 revision;
LR 1e-6 old / 1e-5 new, warmup 1000, cosine to 10%; 50k updates; global batch 320, micro 1, draws 2;
FP32 params/Adam, BF16 FSDP compute, hybrid_zero2, LLM activation ckpt; save/DEV every 5k.

## Deviations (only these)
1. Teacher = committed intermediate A+ step-040000, not completed step-050000
   (user decision 2026-10-16). Pinned allow-list in whenami_timeline_force_student/gate.py;
   any other step is rejected. Original loading.source_record still rejects it, unchanged.
2. 160 cards via whenami_timeline_force_160.train (accum 2, virtual world 320, same as the
   running A+ 160 resume). Admission check is the original one with world=160 instead of 320.
No existing file was modified (fingerprinted sources/LOCAL_VERIFICATION unchanged).

## Local verification (single PPU, engineering BF16, gb1)
local/student_from40k_dryrun_01: real step-040000 loads as student (1551+8 new) and frozen
teacher (1551, 0 new); 3 updates finite, all 9 gradient groups >0; save+DEV ok.
Unit tests whenami_timeline_force.test_force/test_launcher: 9/9 OK.
