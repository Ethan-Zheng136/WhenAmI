# Embodied AI / WoG Research Handoff

> **Purpose:** This document is a context handoff for another research/coding agent (e.g. Cursor agent).  
> It should allow the agent to continue the discussion and implementation work **without re-asking basic questions or re-deriving the entire motivation**.
>
> **Primary objective:** Help the user identify a strong, publishable Embodied AI / robot manipulation research idea centered on **World Guidance / future-aware VLA / world-action interaction**, while avoiding superficial module stacking or blindly porting autonomous-driving methods.

---

## 0. User context that matters for this project

The user is transitioning from **autonomous driving research** to **Embodied AI / robot manipulation**.

Relevant prior expertise:

- End-to-end autonomous driving
- BEV / trajectory prediction
- Online map / vector map representation
- Mapping uncertainty and downstream trajectory prediction
- World-model-like predictive representations
- Flow Matching / trajectory generation
- Multimodal / structured representations

Important prior work:

- **Delving into Mapping Uncertainty for Mapless Trajectory Prediction**
  - arXiv: https://arxiv.org/abs/2507.18498
- Xunjiang Gu et al., **Producing and Leveraging Online Map Uncertainty in Trajectory Prediction**
  - arXiv: https://arxiv.org/abs/2403.16439

The user's previous uncertainty work is **NOT** primarily about model confidence / OOD / "knowing when it doesn't know".

It is specifically:

> An upstream perception/reconstruction module outputs a structured spatial representation (e.g. lane points).  
> Reconstruction is inevitably noisy. Instead of outputting only `(x,y)`, each point also has uncertainty parameters such as `sigma_x`, `sigma_y`, Laplace `beta_x`, `beta_y`, or correlation `rho`.  
> This uncertainty is explicitly passed to the downstream trajectory predictor/planner to reduce **cascade error**.

This distinction must be preserved.

However, **do not assume uncertainty must be the final Embodied AI research direction**. It is just one prior expertise and one possible tool.

---

# 1. Papers / projects currently central to the user's interests

The user is particularly interested in:

1. **World Guidance (WoG)**
   - arXiv: https://arxiv.org/abs/2602.22010
   - Project / code should be checked directly when needed.
2. **Dense Policy**
   - User is interested in Dense Policy / Dense Policy v2 and the broader idea of coarse-to-fine / sparse-to-dense action generation.

The strongest research discussion so far has centered on **WoG**.

---

# 2. Correct understanding of WoG

This section is critical. Do **not** regress to the earlier incorrect interpretation.

## 2.1 What WoG is NOT

WoG is **not** simply:

`O_t -> C_future_hat -> A`

at deployment.

It is also **not** a classical inference-time world model that performs explicit rollout:

`s_t, a_t -> s_{t+1} -> s_{t+2} -> ...`

The final deployed policy remains essentially end-to-end.

---

## 2.2 Stage I: learn an action-relevant future condition space

Training data contains current observation, instruction, future observation sequence, and action.

Current path:

`(O_t, L) -> VLM -> z`

Future path:

`O_future -> frozen visual encoders -> Future Encoder / Q-Former -> O^c`

Then the action head receives both:

`(z, O^c) -> A`

The important idea:

> The Future Encoder is trained through its usefulness for **action generation**.

Thus `O^c` is not an arbitrary generic visual feature. It is a compressed future representation selected because it helps predict action.

A useful description is:

`O^c = action-supervised compression of future visual observations`

or:

`action-sufficient future bottleneck`

---

## 2.3 Stage II: internalize future information into the current VLM representation

At deployment, true future observations are unavailable.

So Stage II freezes the Stage-I future-condition teacher and trains the current VLM representation to predict / align with that future condition.

Conceptually:

`(O_t, L) -> z`

then an auxiliary query head predicts:

`O^c_hat = q(z)`

and aligns it with teacher `O^c`.

But **the action head in Stage II does not directly consume `O^c_hat`**.

Instead:

`z -> Action Head -> A`

while in parallel:

`z -> q(z) -> O^c_hat`

with a condition-alignment loss.

So WoG is better understood as:

> **Future-aware representation learning / privileged-future distillation for VLA**

rather than explicit future-condition inference at deployment.

---

## 2.4 Final inference path

Final deployment:

`(O_t, L) -> VLM -> z -> A`

No true future frames.

No Future Encoder.

No explicit `O^c` interface into the action head.

This is an important correction from an earlier mistaken interpretation.

---

# 3. DINOv2 and Wan VAE inside WoG

WoG uses future observations and extracts complementary features.

## DINOv2

Rough role:

- self-supervised ViT representation
- strong patch-level visual semantics
- object / part / local spatial information
- good for "what / where"

A useful shorthand:

`DINOv2 ~= future what/where`

This is a simplification, not a strict disentanglement.

## Wan VAE

WoG does **not** need the full Wan video diffusion model to generate future video.

It uses the **Wan VAE encoder** to encode short video clips containing current/future frames.

Rough role:

- spatiotemporal video representation
- captures temporal evolution / motion / dynamics
- good for "how the scene changes"

A useful shorthand:

`Wan VAE ~= future evolution / motion`

Again, this is only a functional interpretation.

---

# 4. Important unresolved question: what is actually inside O^c?

This is one of the best diagnostic questions discovered so far.

Because `O^c`:

1. is derived from **future visual observations**, but
2. is optimized to help **action prediction**,

it could contain a mixture of:

- object motion
- future scene consequences
- contact-related information
- task progress
- manipulated-object state
- robot / gripper motion
- latent proxies for future action

A critical concern:

> Future images directly reveal the robot arm / gripper trajectory.  
> Therefore the easiest shortcut for the Future Encoder may be to encode **future robot motion**, which is effectively an action proxy.

This does **not** prove that `O^c` is "just action".

The correct current position is:

`O^c = future-derived + action-selected`

Its internal composition is unknown.

Do not claim more than the evidence supports.

---

# 5. First recommended diagnostic experiment

Before inventing a large architecture, test what `O^c` actually uses.

## Robot-only vs world/object-only future information

Construct variants of future observations:

### Full future
`F_full`

### Robot-only
Keep robot/gripper region, suppress object/world context.

`F_robot`

### World/object-only
Mask robot/gripper while preserving manipulated objects / environment.

`F_world`

Then compare:

- Stage-I action prediction
- condition feature similarity
- Stage-II downstream success if retraining is feasible
- ability to decode action vs object state / motion from `O^c`

Possible interpretation:

If `O^c_robot ~= O^c_full`, then WoG's condition may rely heavily on a future-action shortcut.

If `O^c_world ~= O^c_full`, then the "world consequence" interpretation is much stronger.

This experiment is valuable regardless of outcome.

---

# 6. Core research philosophy developed in the discussion

The main lesson is:

> **Do not port an autonomous-driving method into Embodied AI. Port the user's way of asking research questions.**

The user's strongest transferable habit is not "Gaussian uncertainty".

It is:

`Intermediate / predictive representation -> downstream decision`

and asking:

- What information is really necessary?
- What is redundant?
- How is the representation learned?
- What shortcuts can it exploit?
- Does representation error affect decision quality?
- Should world information be explicit or internalized?
- What future information should an action policy imagine?

---

# 7. Current high-level philosophy for world models in manipulation

A useful guiding question:

`For the next action, what does the robot need to know about the future?`

This is preferred over the vague question:

> What should the robot's world model look like?

Key principles:

### 7.1 Do not reconstruct the entire future world unless necessary

Full future RGB contains large amounts of action-irrelevant information.

Prefer a compact but action-sufficient condition.

### 7.2 Prefer training-time privileged future supervision when possible

Future RGB / depth / tracks / contact / 3D / simulator state can be used as teachers.

At inference, ideally retain:

`O_t, L -> z -> A`

unless an explicit intermediate interface provides clear benefit.

### 7.3 "Action relevance" must be tested, not assumed

A representation may be predictive of logged action while failing to represent real physical consequence.

This is why controlled interventions and counterfactual tests matter.

---

# 8. Strongest idea currently: Action-World should not be one-way

This is the most exciting idea from the discussion.

A one-way architecture says either:

`World -> Action`

or:

`Action -> World`

But physical interaction is reciprocal:

> The future world depends on what action is considered, and the quality of the action depends on what future world that action would cause.

Thus:

`Action candidate -> Predicted consequence -> Action refinement`

---

# 9. Important prior-art warning

The broad idea of bidirectional / iterative world-action reasoning is **not completely novel**.

Relevant nearby work includes:

- **DAWN: The DAWN of World-Action Interactive Models**
  - autonomous driving
  - recursively refines world hypothesis and action hypothesis
- **PAD**
  - joint future-image and action denoising
- **STARRY**
  - joint spatiotemporal future latent + action modeling
- **tau_0-WM**
  - candidate action -> world/video simulation -> evaluation / correction
- **FACT**
  - action-conditioned future / task-progress reasoning
- other 2026 world-action model papers

Therefore:

> Do NOT propose a paper whose novelty is merely "make world and action bidirectional".

That novelty claim is too weak.

---

# 10. The more specific promising formulation

The currently favored formulation is:

`C = C(z, a)`

instead of WoG Stage-II style:

`C = C(z)`

Interpretation:

> The condition should answer not merely  
> **"what future is likely from this scene?"**  
> but  
> **"if I take this candidate action, what consequence will occur?"**

This introduces **counterfactual action sensitivity**.

For two candidate actions `a1 != a2`, the model should produce different conditions if their physical consequences differ.

This can be tested explicitly.

---

# 11. Working name: Counterfactual Condition Co-Inference

Current conceptual loop:

`a^(0) -> c^(0)=C(z,a^(0)) -> a^(1) -> c^(1) -> ...`

where:

- `a^(k)`: current action hypothesis
- `c^(k)`: compact action-conditioned future consequence
- `c^(k)` guides refinement of `a^(k+1)`

The desired distinction from existing work is the combination of:

1. **WoG-style learned, compact, action-sufficient condition space**
2. **Action-conditioned / counterfactual future condition**
3. **Inference-time reciprocal action-condition refinement**
4. **Training with alternative / perturbed / failed actions**
5. **Compact latent consequence modeling instead of full RGB video rollout**

The exact novelty still needs careful literature verification.

---

# 12. Why counterfactuality may be more important than simple bidirectionality

A world model can easily learn:

`C(z) ~= typical task progression`

Example:

> "pick mug" usually means the mug moves upward.

But that is not enough.

A true interactive dynamics representation should answer:

`What happens if I choose this specific candidate action?`

For the same current observation:

- candidate action A: grasp left side
- candidate action B: approach from right
- candidate action C: bad grasp / collision trajectory

should produce different predicted future consequences.

Thus a potentially stronger scientific question is:

> **Can we train an action-relevant condition space that is explicitly sensitive to counterfactual actions rather than merely encoding the demonstration's typical future?**

This is one of the most promising formulations so far.

---

# 13. Relation to Dense Policy

Dense Policy is interesting because it treats action generation as coarse-to-fine.

A natural integration is:

`Sparse Action -> Predicted Consequence -> Refined Sparse Action -> Dense Action`

Rather than imagining consequences for every low-level timestep, use a compact sparse/global action hypothesis.

Possible structure:

`A_sparse -> C_future -> A_sparse' -> A_dense`

This is conceptually cleaner than simply attaching a world model to every dense action token.

However, this connection should not be forced unless experiments support it.

---

# 14. Candidate paper directions currently ranked

## Idea A — Counterfactual Condition Co-Inference
**Current top candidate**

Question:

> Can a VLA iteratively refine action proposals using compact, action-conditioned future consequence representations?

Strength:
- directly extends the strongest WoG insight
- gives the condition space an inference-time role
- explicitly models "if I do this, what happens?"
- compatible with flow matching / diffusion action generation

Risk:
- must clearly distinguish from DAWN / STARRY / PAD / FACT / other world-action interactive models

## Idea B — Effect-Centric World Guidance

Question:

> Is WoG's condition representing real world consequence or mostly future embodiment/action proxy?

Potential method:

Learn a condition space focused on **effects on manipulated objects / environment**, while suppressing embodiment-specific shortcut information.

Possible targets:
- object displacement
- contact transition
- relative state change
- task progress
- object-centric future dynamics

Risk:
- must avoid collapsing into existing "object motion" approaches.

## Idea C — Diagnostic paper / analysis first

Question:

> What does WoG's condition actually encode?

Experiments:
- robot mask / world mask
- counterfactual perturbations
- probe action vs object state
- representation sensitivity tests
- stage-I oracle contribution analysis

This can be valuable as a first step even if it does not become the final paper.

## Idea D — Uncertainty-aware structured future guidance

Possible only **after** discovering a structured future representation whose reconstruction error matters.

Example:

`(mu_track, Sigma_track) -> policy`

This is the cleanest way to reuse the user's prior mapping-uncertainty expertise.

Current status:
- **not the top priority**
- do not force this direction before proving the structured representation bottleneck

---

# 15. Things the next agent should NOT do

Avoid the following failure modes:

### 15.1 Do not say WoG inference is:
`O_t -> O^c_hat -> A`

That is incorrect.

### 15.2 Do not describe `O^c` as purely "world state"

It is a learned future visual bottleneck optimized for action prediction.

### 15.3 Do not describe `O^c` as purely "future action"

That is also unproven.

### 15.4 Do not equate the user's old uncertainty work with:
- OOD detection
- policy confidence
- epistemic failure detection
- ask-for-help systems

That is not the same uncertainty notion.

### 15.5 Do not propose:
- "add geometry"
- "add contact"
- "add object tokens"
- "make it bidirectional"
- "add Gaussian uncertainty"

as a final paper idea without checking 2026 prior art.

These are likely too incremental or already crowded.

### 15.6 Do not optimize for "one more module"

The user wants a **formulation-level research idea**.

---

# 16. Immediate next tasks for a Cursor / coding agent

## Task 1 — Audit WoG implementation precisely

Inspect:

- Stage-I Future Encoder input/output
- DINOv2 feature path
- Wan VAE feature path
- Q-Former query count and dimensions
- how `O^c` is injected into DiT
- which modules are frozen / trainable
- Stage-II auxiliary query head
- Stage-II action head input
- inference path

Produce a compact architecture diagram based on code, not only paper prose.

## Task 2 — Locate exact hook points for `O^c`

Find code locations to:
- save / dump `O^c`
- replace future frames
- mask robot regions
- mask manipulated-object regions
- feed counterfactual future clips
- record action-head outputs

This enables the diagnostic experiment.

## Task 3 — Design a minimum diagnostic experiment

Prefer a cheap test before any 7B-scale retraining.

Questions:

1. Does robot-only future preserve much of `O^c`'s action benefit?
2. Does object/world-only future preserve it?
3. Can probes decode:
   - action chunk
   - end-effector trajectory
   - object pose / object motion
   - task progress
   from `O^c`?
4. How sensitive is `O^c` to:
   - background changes
   - robot pose changes
   - object pose changes
   - contact outcome changes?

## Task 4 — Literature audit around `C(z,a)`

Search specifically for:

- action-conditioned latent world models for manipulation
- world-action co-inference
- bidirectional action-world refinement
- counterfactual VLA
- action-conditioned future feature prediction
- latent consequence prediction
- world-action diffusion / flow matching
- action-conditioned future tokens
- candidate action evaluation with compact latent futures

The goal is NOT to find "something similar".  
The goal is to determine the **exact novelty gap** for `C(z,a)` plus reciprocal action refinement.

## Task 5 — Prototype a minimal `C(z,a)` module

Avoid large retraining initially.

Candidate prototype:

`z, a^(k) -> C^(k) -> Delta a^(k)`

Possible implementation:
- cross-attention between action tokens and current VLM features
- compact condition tokens
- consequence prediction supervision from Stage-I WoG teacher `O^c`
- action refinement through residual / denoising / flow update

Test on a small subset first.

---

# 17. Desired scientific standard

A good result should ideally follow:

`Observation -> Failure of current formulation -> Explanation -> New formulation -> Minimal intervention`

rather than:

`New module -> hope metric improves`

The user prefers critical, skeptical reasoning.

Do not oversell novelty before literature verification.

Do not confuse architectural plausibility with scientific evidence.

---

# 18. Current one-sentence research thesis

> **Instead of merely teaching a VLA what future is likely, learn a compact action-relevant condition that answers what future would result from a candidate action, and let predicted consequences iteratively refine the action itself.**

Core loop:

`a^(k) -> C(z, a^(k)) -> a^(k+1)`

This is currently the central direction to investigate.

---

# 19. Companion PDF

The existing PDF memo provides a visual, presentation-style summary of the same line of thought:

**Embodied AI — From WoG to World–Action Co-Inference**

Use both files together:
- PDF = visual overview / research memo
- MD = precise handoff context for agents and implementation work

---

# 20. Final instruction to the next agent

When continuing this project:

1. Treat **WoG's exact implementation** as the factual base.
2. Separate:
   - what the paper proves,
   - what the code confirms,
   - what is only a hypothesis.
3. Investigate `O^c` before replacing it.
4. Prioritize **counterfactual action-conditioned consequence modeling** over generic "world-model enhancement".
5. Keep the final research goal formulation-level, not module-level.
6. Be skeptical of novelty until 2026 literature is checked.
7. Preserve the possibility that the best final idea may **not** involve uncertainty.
