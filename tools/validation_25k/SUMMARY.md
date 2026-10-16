# timeline_force A+ teacher: counterfactual-future DEV test (2026-10-15)
DEV force_timeline_fixed_dev3_v1, 410 ep / 27 domains, seeds 42-44, same scoring as evaluate.py; single PPU param_bf16
(reproduces train-time RESULT.json within 1e-4). Future = frame 2 for hindsight_middle (target 1, version 2). Macro mse6 / first_action_mse6.

| condition | 5k mse6 | 5k first | 25k mse6 | 25k first |
|---|---|---|---|---|
| causal_middle (no future) | 0.04491 | 0.03452 | 0.04426 | 0.03473 |
| hindsight true future | 0.04214 | 0.03205 | 0.04137 | 0.03204 |
| hindsight swap (other ep, same domain) | 0.04762 | 0.04001 | 0.04804 | 0.04094 |
| hindsight repeat current frame | 0.04571 | 0.03526 | 0.04531 | 0.03556 |
| hindsight future hidden (visible=False), 25k only | - | - | 0.04360 | 0.03474 |

Paired (episode) swap-true 25k +0.0067 [CI +0.0039,+0.0100]; repeat-true +0.0040 [+0.0026,+0.0055]; hide-true +0.0022 [+0.0013,+0.0033].
causal_middle identical across conditions (leak check 0). Files: cf_step0{05,25}000.json, hide_step025000.json, summarize.py.
