# Fixed A10k -> causal Phase-B pilot

The additional640PPUs are intended for two **new**320PPU allocations. Running A
continues untouched. This is a controlled transfer feasibility study, not a
guarantee of paper acceptance or the final full-horizon Phase-B architecture.

## Pipeline

```text
Fixed A10k teacher: [history,current,later image] -> current condition target
                                                               |
                                                       auxiliary loss only
                                                               |
Causal student: [history,current] -> current condition -> auxiliary predictor
                                      |
                                      +-> full DiT -> current16-step action
```

Two students have identical A10k model initialization and auxiliary head.

- `action_only`: action FM; teacher forward computed but target unused in loss.
- `condition_align`: action FM +0.1 cosine auxiliary teacher-condition alignment.

The auxiliary prediction is NOT fed back into the action head. Both bodies train
fully; teacher and standalone DINO/Wan freeze. Student observations are physically
truncated before tokenization/image encoding. Ground-truth action goes to the
flow target/noised decoder state only, never into visual/state conditioning.

Current pilot has **one observed history anchor + current anchor**, sampled as the
first two of a three-anchor real clip. It does not silently become the former
Stage3 four-history-frame wrapper. Future/long-horizon/all-node distillation is
not yet implemented. The eventual deployment wrapper must respect this contract.

## Pretraining and optimizer

- Student and teacher load the SAME fixed timeline-A `step-010000/model_dcp`.
- Source A-DCP keys verified against live model; only `forecast_head.*` is new.
- B optimizer and per-rank RNG start fresh. A optimizer not copied.
- Further B segments restore their own B sharded optimizer/RNG exactly.
- A later20k/50k checkpoint never replaces the pinned teacher mid-run.

## Extra640PPU commands

Each command is a separate20node x16PPU task. Platform sets WORLD_SIZE=20,
RANK,MASTER_ADDR; do not manually set WORLD_SIZE=320.

```bash
B_VARIANT=action_only PHASE=run NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_timeline_b_320.sh
```

```bash
B_VARIANT=condition_align PHASE=run NPROC_PER_NODE=16 \
bash /home/zhengguantian/WoG_fresh/run_timeline_b_320.sh
```

`PHASE=run` requires source-bound local verification, then actual320-rank tiny
topology proof -> full B100updates/save/causal27dev -> new-process restore120 ->
validate -> continue20k. Any failure stops. `PHASE=admit` stops after120;
`PHASE=full` later resumes from successful120. Existing run directories are never
overwritten. No cluster jobs are submitted automatically.

## Fixed recipe

Global320 trajectories,current target per trajectory,2 flow draws.
Pretrained VLM/QFormer/timeline/DiT LR1e-5; new auxiliary LR1e-4;
warmup1000/cosine,total20k,save2500,eval1000,early100/120 checks.
Same teacher compute in both arms, even B0, to match budget.
410 fixedDEV episodes/27domains; pure Gaussian causal action sampling,1 noise seed
periodically and3 at final. Store predictions, not only aggregate metrics.
No OXE download or train-feature preparation. No sealed-test access.

Compare completed matching checkpoints:

```bash
python -m whenami_timeline_b.compare \
  --action-only <B0>/main --aligned <B1>/main --step 20000 \
  --output <new-comparison-directory>
```

## Readiness

`LOCAL_VERIFICATION.json` is written only after local tests really pass.
Includes CPU4-rank nested-FSDP A->B model loading and new B optimizer restore.
Real singlePPU loaded-model gradient/causality tests use a one-step SGD smoke,
NOT full8B FP32Adam. Actual320-rank update/save/resume admission remains required.

This experiment tests **extra teacher supervision** given common timeline-A
pretraining. It cannot on its own isolate all timeline representation benefits,
replace the pooled-vs-timeline A->B comparison, establish independent training-seed
stability, or claim improved closed-loop success.
