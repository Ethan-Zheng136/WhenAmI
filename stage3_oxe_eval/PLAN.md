# OXE27 current-policy comparison, 2026-10-03

Authorized by user: run the evaluation locally, request resources if needed.
Do not edit production training, its dependencies, source checkpoints, or sealed tests.

## Frozen initial protocol

- Models: published local WoG-A `50k-1229.pt`, original cumulative20k Past,
  Stage3 `oxe27_forecast_zero2_320_50k_save5k_01/checkpoints/step-035000`.
- Dataset: original27 OXE domains, **existing Stage3 dev hash partition only**,
  exact EpisodeDecoder/content-key/registry used by production preparation.
  Not timeline's internal split. No sealed-test evaluation.
- Target64 distinct dev episodes/domain, at most2 uniformly selected legal
  action-window centres/episode. Smaller domains retain all available eligible
  dev episodes, report counts. Deterministic hash-shuffled source-shard scan,
  stopping at quota: a bounded diagnostic sample, not a claim of an unbiased
  all-corpus census. Model outcomes do not influence selection.
- Source records must be parsed to compute the existing partition identity.
  Only dev-role records have pixels decoded or observations/labels retained.
- All selected content keys must be absent from the actual Stage3 training index.
  Base models/public checkpoints may have seen these episodes in prior training.
- Same current images/language; Past and Stage3 receive actual causal history
  at [-15,-11,-7,-3], boundaries repeat earliest frame. WoG-A current-only.
  No future images or expert actions passed to prediction functions.
- 16x7 generated actions, pure FP32 initial Gaussian arrays,9 Euler intervals,
  BF16 policy compute with FP32 sampling integration, inference seeds42/43/44.
  Inference seeds are not independent training seeds.
- Keep each model's training normalization; score in common per-domain
  evaluation coordinates. Episode action padding masked. Primary normalized
  continuous-action MSE6 (clipped to training range), plus unclipped MSE,
  MAE, first-action MSE, gripper accuracy, domain raw r/R²/RMSE.
- Primary aggregate: windows/seeds equally within episode, episodes equally
  within domain, domains equally overall; also report micro and training-mixture
  weighted aggregates. Do not pool incompatible raw robot units.
- Paired stratified episode bootstrap for fixed-model uncertainty; small domains
  flagged. Publish all domains, not only favorable/previously monitored two.
- Loading admission: strict live-state coverage, original-path checks, Stage3
  reproduction on a small fixed old-dev fixture (deployment validation only).
- Atomic per-episode prediction files, deterministic resume, independent summary.
  No task-success or causal-mechanism claim from this performance comparison.

Implementation/admission status must be recorded from actual commands, not this plan.
