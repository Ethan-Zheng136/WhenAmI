# Parallel Phase-B preparation, 2026-10-05

User asks whether we can do more than wait for full A training.
Running A teacher jobs and all hash-bound dependencies MUST remain unchanged.

## Work that can happen now

1. Audit existing paired27-domain intermediate A results (no new training).
2. Strict model-only loading of pinned A5k for transfer-interface admission.
3. Physically remove future raw/tensor evidence from the student API.
4. Define a WoG-style decoupled auxiliary predictor: student conditions already
   influence action; predicted teacher condition is supervision-only, never fed
   back as privileged input.
5. CPU leakage/gradient/teacher-detach tests; actual singlePPU loaded-model
   current-decision forward/backward smoke if resources allow.

## First transfer contract, NOT final full B experiment

- Pin A timeline step5000 (latest full checkpoint at start of this work).
- Teacher and student initialized from the SAME source weights for admission.
- Teacher: may see the full sampled three-node clip, frozen/eval/no_grad.
- Student: only observations through current middle anchor; future pixels,
  timestamps and language are physically sliced away before student.forward.
- Current decision at that anchor only in this first interface. It does NOT yet
  implement full multi-node/horizon distillation or a complete B trainer.
- Actions are labels/noised action decoder states only; never used by encoder.
- Student auxiliary head predicts teacher's16x32 current condition; cosine
  alignment adopts the existing WoG-style target principle, no new-method claim.
- B0 control: action FM only. B1: same model/current inputs + auxiliary alignment.
  Both retain same parameters, same initialization and action path.
- Aligning a privileged target cannot make an unobservable outcome predictable.
  Teacher advantage vs the SAME initialized causal policy must be measured;
  deployment success cannot be inferred from target matching.

## Admission and eventual evaluation

- First CPU tests: future perturbation leaves student input/output unchanged;
  prefix input contains no future sequence length/pixels/language; no target
  actions accepted in encoder; teacher target detached; student gradients reach
  encoder, temporal and action reader when appropriate.
- Real smoke uses a legitimate TRAIN episode, not dev feedback for training.
- A5k is a wiring fixture, not the final teacher or chosen best checkpoint.
- B large training must wait for a fixed teacher, matched B0/B1 recipe,
  full optimizer/save/resume admission and a declared dev/closed-loop protocol.
- No running A checkpoint is modified; no large B job is submitted here.
