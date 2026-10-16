# Claude Code session agent-aca18b7aaf1fca4ac

- Time (Beijing): 2026-09-16 19:24 → 2026-09-16 19:54
- cwd: `/home/zhengguantian`
- Source: `~/.claude/projects/-home-zhengguantian/c976c298-d71c-4d38-9107-f692d869d083/subagents/agent-aca18b7aaf1fca4ac.jsonl` (neolix_al_2)
- Messages (new, after de-dup): 220
- Note: tool outputs capped at 2000 chars; thinking blocks and images dropped; secrets/private IPs redacted.

---

### USER — 2026-09-16 19:24

First read /tmp/wog/CONTEXT.md in full — it describes the exact target model (WoG) you are finding transplantable mechanisms for, the user's measured diagnostics, hard constraints, and the arXiv search tooling. Follow its evidence discipline strictly.

YOUR ASSIGNMENT — GRAFT POINT 3: change O^c from "encode the future" to "encode the action-induced DIFFERENCE in the future".

Current state in WoG: the QFormer sees the current frame AND 4 future frames and encodes them jointly into O^c. The user measured that **83% of O^c's variance is linearly reconstructible from a current-frame-only version of O^c** — i.e. most of the bottleneck's capacity is spent re-encoding static scene content that the policy already gets from z / the VLM. Yet the causal effect on the DiT is 96%-108% attributable to the future-only component. Conclusion: the useful signal is the delta, but the representation is dominated by the absolute content. So: find mechanisms that explicitly represent Δ (difference / residual / displacement / change) rather than absolute state.

Find 4-6 concrete candidates. Categories to search:
- latent difference / latent displacement decoding; inverse dynamics from latent deltas
- frame-difference or optical-flow-based conditioning replacing raw future frames
- residual/predictive coding architectures where a branch predicts only what the current frame cannot explain
- change detection / dynamics-only representations; "what changed" objectives
- contrastive or predictive objectives that explicitly subtract out static content: e.g. removing static bias in video representation learning, debiasing static appearance shortcuts
- action-conditioned dynamics latents that are provably action-relevant: inverse-dynamics-regularized latents, controllability-aware / action-informative latent objectives, empowerment / mutual-information-with-action regularizers
- bisimulation metrics and value-equivalent / control-relevant state abstraction (they discard task-irrelevant static content by construction)
- point tracking / trajectory-of-points as the intermediate representation instead of future frames (e.g. track-based affordance, ATM-style any-point trajectory) — note whether these need extra annotation, since the user cannot afford heavy labeling (though off-the-shelf trackers count as no manual annotation)

Known leads to verify and anchor on: Delta-JEPA (arXiv 2606.31232) decodes latent differences and inverts them to actions; dWorldEval uses Δ-LPIPS; CoMo replaces future frames with frame differences (find its arXiv id, it is about learning motion/"continuity of motion" latents for VLA — verify carefully, and if you cannot find it, say so explicitly rather than guessing an id). Go well beyond these three.

An important design question to address: WoG feeds the current frame INTO the QFormer alongside the futures. Two obvious surgical variants are (a) remove the current frame from the QFormer input entirely, (b) keep it but subtract/orthogonalize — e.g. train O^c to be orthogonal to the current-frame-only encoding, or add an explicit adversarial/regularization loss that makes O^c unpredictable from the current frame alone. Search for prior work that does exactly this kind of "make the latent independent of the static/current component" (information-theoretic disentangling, HSIC / adversarial static-invariance, gradient-reversal on a static predictor) and report the standard recipe, since that is the cheapest possible graft (a loss term, tens of lines, Stage I retrain only).

For EACH candidate deliver:
1. arXiv id + URL + submission date + venue from abs page Comments (say "(none)" if absent) + code URL if public
2. Precise mechanism
3. HOW TO GRAFT INTO WoG concretely: what changes in the QFormer input (frames, DINOv2 patches, Wan VAE latents), what loss term is added and in what exact form, whether Stage I needs retraining, whether Stage II's distillation target changes, and the estimated change size
4. What the diagnostics predict: would this raise the fraction of O^c capacity that is causally useful, and would it make Stage-II distillation harder or easier (recall Stage II currently only reaches 0.177 predictability of the future-only residual from z)? Note honestly: if O^c becomes purely delta, it becomes LESS predictable from the current observation, which is exactly what Stage II must predict from — discuss this tension.

Do not write any report files. Return findings directly as your final message, in Chinese, structured and information-dense. Rank by expected usefulness for THIS model and flag anything unverified.

### ASSISTANT — 2026-09-16 19:24

I'll start by reading the context file in full.

**tool call `Read`**
```
{"file_path": "/tmp/wog/CONTEXT.md"}
```

<details><summary>tool result</summary>

```
1	# WoG target model — exact structure (given as fact by the user, who read the source)
2	
3	WoG = "World Guidance: World Modeling in Condition Space for Action Generation",
4	arXiv 2602.22010 (25 Feb 2026), ICML 2026, ByteDance Seed + HKU, code github.com/Selen-Suyue/WoG
5	
6	## Stage I (train_venc=True)
7	- Future Encoder = QFormer: 6 layers, hidden 1024, **16 learnable queries**
8	- Input: 5 frames at indices [0, Ta//4, Ta//2, 3Ta//4, Ta] (index 0 = CURRENT frame),
9	  each through frozen DINOv2 penultimate-layer patches (adaptive-pooled to 8x8=64 tok/frame)
10	  + frozen Wan VAE latent (current frame concatenated BEFORE future frames, temporal dim squeezed to 2)
11	- Output O^c: v_proj -> 32 dims => **[B, 16, 32]** = 512 numbers total
12	- O^c enters DiT-L as **cross-attention K/V in EVERY one of the 24 blocks**; action tokens are Q;
13	  the four `per_attn` weights are **zero-init** (ControlNet style)
14	- z (VLM cognition feature, 4096-d) takes a DIFFERENT path: c = t + z is concatenated as token 0 of
15	  the action sequence; at the end x[:, 1:] discards it
16	- loss = **act_loss only** (rectified-flow velocity MSE, t ~ Beta(1.5, 1.0)); obs_loss hard-zeroed
17	- action output shape (16, 7)
18	
19	## Stage II (train_venc=False)
20	- qformer frozen + .detach(); new `oformer` (4 layers, hidden 4096) predicts O^c_hat from `obs_ins`
21	  (the VLM's last 4 hidden tokens; z is one of them)
22	- obs_loss = (1 - cosine(normalize(O^c_hat), normalize(O^c.detach()))).mean()
23	- loss = act_loss + 0.3 * obs_loss
24	- SIMULTANEOUSLY obs_dropout_prob=1 AND use_per_attn=False => the DiT per_attn pathway does not exist
25	  at the weight level (checkpoint: action_model 350 tensors vs 204 tensors)
26	- oformer is discarded after training; present in no checkpoint
27	
28	## Deployment
29	O_t, L -> VLM -> z -> DiT -> A.  O^c does not participate at all.
30	
31	# User's own measured diagnostics (use these to judge which mechanisms would help)
32	- O^c effective rank only 18 (out 
... [truncated 2195 chars]
```
</details>

### ASSISTANT — 2026-09-16 19:25

I have the context. Let me verify the tooling and start with the known leads.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; cat /tmp/wog/axs.py | head -60", "description": "Inspect the arXiv search tool"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py abs 2606.31232 2602.22010 2>&1 | head -60", "description": "Verify Delta-JEPA and WoG abs pages"}
```

<details><summary>tool result</summary>

```
#!/usr/bin/env python3
"""arXiv search + abs verify helper.
Usage:
  axs.py search "QUERY_STRING" [max]     # QUERY uses arXiv API syntax, e.g. abs:"latent action" AND cat:cs.RO
  axs.py abs ID [ID...]                  # verify title/date/comments from abs page
"""
import sys, re, html, urllib.parse, subprocess, time

def fetch(url, tries=3):
    for i in range(tries):
        p = subprocess.run(["curl","-sL","--max-time","45",url], capture_output=True, text=True)
        if p.returncode==0 and len(p.stdout)>200: return p.stdout
        time.sleep(4)
    return p.stdout if p.returncode==0 else ""

def search(q, mx=25, sort="relevance"):
    url=("https://export.arxiv.org/api/query?search_query="+urllib.parse.quote(q)
         +f"&start=0&max_results={mx}&sortBy={'relevance' if sort=='relevance' else 'submittedDate'}&sortOrder=descending")
    t=fetch(url)
    ents=re.findall(r'<entry>(.*?)</entry>', t, re.S)
    out=[]
    for e in ents:
        def g(p, d='N/A'):
            m=re.search(p,e,re.S); return html.unescape(re.sub(r'\s+',' ',re.sub(r'<[^>]+>','',m.group(1)))).strip() if m else d
        aid=g(r'<id>http[s]?://arxiv.org/abs/(.*?)</id>')
        out.append((aid, g(r'<title>(.*?)</title>'), g(r'<published>(.*?)T'), g(r'<arxiv:comment[^>]*>(.*?)</arxiv:comment>','-')[:180], g(r'<summary>(.*?)</summary>')[:300]))
    for a,ti,pu,co,su in out:
        print(f"{a} | {pu} | {ti}\n    COMMENT: {co}\n    ABS: {su}\n")
    print(f"[{len(out)} results for: {q}]")

def verify(ids):
    for i in ids:
        t=fetch(f"https://arxiv.org/abs/{i}")
        def g(p,d='N/A'):
            m=re.search(p,t,re.S); return html.unescape(re.sub(r'\s+',' ',re.sub(r'<[^>]+>','',m.group(1)))).strip() if m else d
        print(f"=== {i} ===")
        print("TITLE :", g(r'<meta name="citation_title" content="(.*?)"'))
        print("SUBMIT:", g(r'\[Submitted on (.*?)\]')[:60], "| citdate:", g(r'<meta name="citation_date" content="(.*?)"'))
        print("COMM  :", g(r'<td class="table
... [truncated 432 chars]
```
</details>

<details><summary>tool result</summary>

```
=== 2606.31232 ===
TITLE : Delta-JEPA: Learning Action-Sensitive World Models via Latent Difference Decoding
SUBMIT: 30 Jun 2026 | citdate: 2026/06/30
COMM  : (none)
SUBJ  : Artificial Intelligence (cs.AI)
DOI   : (none)

=== 2602.22010 ===
TITLE : World Guidance: World Modeling in Condition Space for Action Generation
SUBMIT: 25 Feb 2026 | citdate: 2026/02/25
COMM  : (none)
SUBJ  : Robotics (cs.RO); Computer Vision and Pattern Recognition (cs.CV)
DOI   : (none)
```
</details>

### ASSISTANT — 2026-09-16 19:25

Both anchors verified. Now let me run broad searches across the categories in parallel.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'abs:\"latent difference\" AND (cat:cs.RO OR cat:cs.CV OR cat:cs.LG)' 20 2>&1 | tail -80", "description": "Search latent difference"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'abs:\"frame difference\" AND abs:\"vision-language-action\"' 20 2>&1 | tail -60", "description": "Search frame difference VLA (CoMo)"}
```

<details><summary>tool result</summary>

```
2602.00440v1 | 2026-01-31 | DISK: Dynamic Inference SKipping for World Models
    COMMENT: -
    ABS: We present DISK, a training-free adaptive inference method for autoregressive world models. DISK coordinates two coupled diffusion transformers for video and ego-trajectory via dual-branch controllers with cross-modal skip decisions, preserving motion-appearance consistency without retraining. We ex

2605.15582v1 | 2026-05-15 | LDGuid: A Framework for Robust Change Detection via Latent Difference Guidance
    COMMENT: Accepted to IGARSS 2026. Code is available at: https://github.com/zjxyoyo/LDGuid
    ABS: Modern deep learning models for change detection (CD) often struggle to explicitly represent task-relevant semantic differences. This paper proposes the Latent Difference Guidance (LDGuid) framework that explicitly learns and injects semantic differences into CD models. LDGuid deploys adversarial au

2505.07073v2 | 2025-05-11 | Discovering Concept Directions from Diffusion-based Counterfactuals via Latent Clustering
    COMMENT: Accepted at Pattern Recognition Letters Journal (14 Pages)
    ABS: Concept-based explanations have emerged as an effective approach within Explainable Artificial Intelligence, enabling interpretable insights by aligning model decisions with human-understandable concepts. However, existing methods rely on computationally intensive procedures and struggle to efficien

2410.09873v1 | 2024-10-13 | Training-Free Adaptive Diffusion with Bounded Difference Approximation Strategy
    COMMENT: Accepted by NeurIPS 2024, Homepage: https://jiakangyuan.github.io/AdaptiveDiffusion-project-page/ The code is available at https://github.com/UniModal4Reasoning/AdaptiveDiffusion
    ABS: Diffusion models have recently achieved great success in the synthesis of high-quality images and videos. However, the existing denoising techniques in diffusion models are commonly based on step-by-step noise predictions, which suffers from high computation cost, resulting
... [truncated 530 chars]
```
</details>

<details><summary>tool result</summary>

```
[0 results for: abs:"frame difference" AND abs:"vision-language-action"]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'ti:\"CoMo\" AND (cat:cs.RO OR cat:cs.CV)' 20 2>&1 | tail -70", "description": "Search CoMo by title"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'abs:\"continuity of motion\" OR abs:\"motion latent\" AND cat:cs.RO' 20 2>&1 | tail -70", "description": "Search continuity of motion"}
```

<details><summary>tool result</summary>

```
1806.07908v1 | 2018-06-20 | Como funciona o Deep Learning
    COMMENT: Book chapter, in Portuguese, 31 pages
    ABS: Deep Learning methods are currently the state-of-the-art in many problems which can be tackled via machine learning, in particular classification problems. However there is still lack of understanding on how those methods work, why they work and what are the limitations involved in using them. In th

2404.03531v2 | 2024-04-04 | COMO: Compact Mapping and Odometry
    COMMENT: Website: https://edexheim.github.io/como/
    ABS: We present COMO, a real-time monocular mapping and odometry system that encodes dense geometry via a compact set of 3D anchor points. Decoding anchor point projections into dense geometry via per-keyframe depth covariance functions guarantees that depth maps are joined together at visible anchor poi

2403.13900v2 | 2024-03-20 | CoMo: Controllable Motion Generation through Language Guided Pose Code Editing
    COMMENT: -
    ABS: Text-to-motion models excel at efficient human motion generation, but existing approaches lack fine-grained controllability over the generation process. Consequently, modifying subtle postures within a motion or inserting new actions at specific moments remains a challenge, limiting the applicabilit

2101.10775v1 | 2021-01-26 | CoMo: A novel co-moving 3D camera system
    COMMENT: -
    ABS: Motivated by the theoretical interest in reconstructing long 3D trajectories of individual birds in large flocks, we developed CoMo, a co-moving camera system of two synchronized high speed cameras coupled with rotational stages, which allow us to dynamically follow the motion of a target flock. Wit

2510.23007v1 | 2025-10-27 | CoMo: Compositional Motion Customization for Text-to-Video Generation
    COMMENT: -
    ABS: While recent text-to-video models excel at generating diverse scenes, they struggle with precise motion control, particularly for complex, multi-subject motions. Although methods for single-motion cust
... [truncated 3399 chars]
```
</details>

<details><summary>tool result</summary>

```
2403.20048v1 | 2024-03-29 | On Fractional Kinetic Equations Involving Srivastava Polynomial
    COMMENT: 10 pages
    ABS: Kinetic equations hold a very important place in physics and further their fractional generalization enhances the scope of their applicability and significance in describing the continuity of motion in materials. After the development of generalized form of fractional kinetic equations, many researc

2603.07949v2 | 2026-03-09 | RAPID: Redundancy-Aware and Compatibility-Optimal Edge-Cloud Partitioned Inference for Diverse VLA Models
    COMMENT: -
    ABS: Vision Language Action (VLA) models are mainstream in embodied intelligence but face high inference costs. Edge-Cloud Collaborative (ECC) inference offers an effective fix by easing edge-device computing pressure to meet real-time needs. However, existing ECC frameworks are suboptimal for VLA models

1207.7254v1 | 2012-07-31 | Crofton measures and Minkowski valuations
    COMMENT: -
    ABS: A description of continuous rigid motion compatible Minkowski valuations is established. As an application, we present a Brunn-Minkowski type inequality for intrinsic volumes of these valuations.

1010.2398v1 | 2010-10-12 | An All-Electric Single-Molecule Motor
    COMMENT: 15 pages, 7 figures
    ABS: Many types of molecular motors have been proposed and synthesized in recent years, displaying different kinds of motion, and fueled by different driving forces such as light, heat, or chemical reactions. We propose a new type of molecular motor based on electric field actuation and electric current 

0805.1443v3 | 2008-05-10 | The Computational Status of Physics: A Computable Formulation of Quantum Theory
    COMMENT: To appear, Natural Computing (2009). 21 pages, 1 fig, 44 refs. Content revised, new material added (hypercomputation, digital physics, cosmological computing, open questions)
    ABS: According to the Church-Turing Thesis (CTT), effective formal behaviours can be simulated by Turing machines; th
... [truncated 5596 chars]
```
</details>

### ASSISTANT — 2026-09-16 19:25

Found CoMo (2505.17006). Let me verify it and launch broad parallel searches.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py abs 2505.17006 2>&1; echo \"-----FULL ABS-----\"; curl -sL --max-time 45 https://arxiv.org/abs/2505.17006 | python3 -c \"\nimport sys,re,html\nt=sys.stdin.read()\nm=re.search(r'<blockquote class=\\\"abstract mathjax\\\">(.*?)</blockquote>',t,re.S)\nprint(html.unescape(re.sub(r'\\s+',' ',re.sub(r'<[^>]+>','',m.group(1)))) if m else 'NOABS')\n\"", "description": "Verify CoMo abstract"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'abs:\"static bias\" AND (cat:cs.CV OR cat:cs.LG)' 20 2>&1 | tail -60", "description": "Search static bias video representation"}
```

<details><summary>tool result</summary>

```
=== 2505.17006 ===
TITLE : CoMo: Learning Continuous Latent Motion from Internet Videos for Scalable Robot Learning
SUBMIT: 22 May 2025 (v1), last revised 18 Jun 2026 (this version, v3 | citdate: 2025/05/22
COMM  : (none)
SUBJ  : Computer Vision and Pattern Recognition (cs.CV); Robotics (cs.RO)
DOI   : (none)

-----FULL ABS-----
 Abstract:Unsupervised learning of latent motion from Internet videos is crucial for robot learning. Existing discrete methods generally mitigate the shortcut learning caused by extracting excessive static backgrounds through vector quantization with a small codebook size. However, they suffer from information loss and struggle to capture more complex and fine-grained dynamics. Moreover, there is an inherent gap between the distribution of discrete latent motion and continuous robot action, which hinders the joint learning of a unified policy. We propose CoMo, which aims to learn more precise continuous latent motion from internet-scale videos. CoMo employs an early temporal difference (Td) mechanism to increase the shortcut learning difficulty and explicitly enhance motion cues. Additionally, to ensure latent motion better captures meaningful foregrounds, we further propose a temporal contrastive learning (Tcl) scheme. Specifically, positive pairs are constructed with a small future frame temporal offset, while negative pairs are formed by directly reversing the temporal direction. The proposed Td and Tcl work synergistically and effectively ensure that the latent motion focuses better on the foreground and reinforces motion cues. Critically, CoMo exhibits strong zeroshot generalization, enabling it to generate effective pseudo action labels for unseen videos. Extensive simulated and real-world experiments show that policies co-trained with CoMo pseudo action labels achieve superior performance with both diffusion and auto-regressive architectures.
```
</details>

<details><summary>tool result</summary>

```
2508.02981v1 | 2025-08-05 | MoExDA: Domain Adaptation for Edge-based Action Recognition
    COMMENT: 7 pages
    ABS: Modern action recognition models suffer from static bias, leading to reduced generalization performance. In this paper, we propose MoExDA, a lightweight domain adaptation between RGB and edge information using edge frames in addition to RGB frames to counter the static bias issue. Experiments demons

2509.23009v1 | 2025-09-27 | Disentangling Static and Dynamic Information for Reducing Static Bias in Action Recognition
    COMMENT: in Proc. of ICCV2025 Workshop and Challenge on Disentangled Representation Learning for Controllable Generation (DRL4Real)
    ABS: Action recognition models rely excessively on static cues rather than dynamic human motion, which is known as static bias. This bias leads to poor performance in real-world applications and zero-shot action recognition. In this paper, we propose a method to reduce static bias by separating temporal 

2211.12883v3 | 2022-11-23 | Mitigating and Evaluating Static Bias of Action Representations in the Background and the Foreground
    COMMENT: -
    ABS: In video action recognition, shortcut static features can interfere with the learning of motion features, resulting in poor out-of-distribution (OOD) generalization. The video background is clearly a source of static bias, but the video foreground, such as the clothing of the actor, can also provide

2501.12681v1 | 2025-01-22 | Can masking background and object reduce static bias for zero-shot action recognition?
    COMMENT: In proc. of MMM2025
    ABS: In this paper, we address the issue of static bias in zero-shot action recognition. Action recognition models need to represent the action itself, not the appearance. However, some fully-supervised works show that models often rely on static appearances, such as the background and objects, rather th

2107.10161v2 | 2021-07-21 | Evidential Deep Learning for Open Set Action Recognition
    COMMENT: I
... [truncated 4059 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; curl -sL --max-time 45 https://arxiv.org/abs/2505.17006 | grep -A4 -i 'comments' | head -20; echo \"=== code links ===\"; curl -sL --max-time 45 https://arxiv.org/abs/2505.17006 | grep -oE 'https?://[a-zA-Z0-9./_-]*github[a-zA-Z0-9./_-]*' | sort -u | head", "description": "Check CoMo comments and code URL"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'abs:\"inverse dynamics\" AND abs:\"latent action\" AND cat:cs.RO' 25 2>&1 | tail -100", "description": "Search inverse dynamics latent action"}
```

<details><summary>tool result</summary>

```
          <td class="tablecell label">Comments:</td>
          <td class="tablecell comments mathjax">CVPR 2026</td>
        </tr>
<tr>
          <td class="tablecell label">Subjects:</td>
          <td class="tablecell subjects">
=== code links ===
```
</details>

<details><summary>tool result</summary>

```
2605.12167v1 | 2026-05-12 | From Imagined Futures to Executable Actions: Mixture of Latent Actions for Robot Manipulation
    COMMENT: ICML 2026
    ABS: Video generation models offer a promising imagination mechanism for robot manipulation by predicting long-horizon future observations, but effectively exploiting these imagined futures for action execution remains challenging. Existing approaches either condition policies on predicted frames or dire

2602.21736v1 | 2026-02-25 | Joint-Aligned Latent Action: Towards Scalable VLA Pretraining in the Wild
    COMMENT: CVPR2026
    ABS: Despite progress, Vision-Language-Action models (VLAs) are limited by a scarcity of large-scale, diverse robot data. While human manipulation videos offer a rich alternative, existing methods are forced to choose between small, precisely-labeled datasets and vast in-the-wild footage with unreliable 

2605.16412v1 | 2026-05-13 | SCAR: Self-Supervised Continuous Action Representation Learning
    COMMENT: -
    ABS: Despite the central role of action in embodied intelligence, learning transferable action representations from visual transitions remains a fundamental challenge, particularly when world models must generalize across embodiments under limited data. We argue that action is not merely an auxiliary con

2512.13030v2 | 2025-12-15 | Motus: A Unified Latent Action World Model
    COMMENT: -
    ABS: While a general embodied agent must function as a unified system, current methods are built on isolated models for understanding, world modeling, and control. This fragmentation prevents unifying multimodal generative capabilities and hinders learning from large-scale, heterogeneous data. In this pa

2604.16391v1 | 2026-03-27 | Disentangled Robot Learning via Separate Forward and Inverse Dynamics Pretraining
    COMMENT: ICLR 2026
    ABS: Vision-language-action (VLA) models have shown great potential in building generalist robots, but still face a dilemma-misalignment of 2D image forecast
... [truncated 1725 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'abs:\"bisimulation\" AND (cat:cs.LG OR cat:cs.RO)' 25 2>&1 | tail -100", "description": "Search bisimulation"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'abs:\"gradient reversal\" AND abs:\"static\"' 20 2>&1 | tail -60", "description": "Search gradient reversal static"}
```

<details><summary>tool result</summary>

```
    COMMENT: Accepted to Transactions on Machine Learning Research (TMLR)
    ABS: Bisimulation metrics define a distance measure between states of a Markov decision process (MDP) based on a comparison of reward sequences. Due to this property they provide theoretical guarantees in value function approximation (VFA). In this work we first prove that bisimulation and $π$-bisimulati

2110.03726v2 | 2021-10-07 | Bisimulations for Neural Network Reduction
    COMMENT: -
    ABS: We present a notion of bisimulation that induces a reduced network which is semantically equivalent to the given neural network. We provide a minimization algorithm to construct the smallest bisimulation equivalent network. Reductions that construct bisimulation equivalent neural networks are limite

2202.01214v1 | 2022-02-02 | Approximate Bisimulation Relations for Neural Networks and Application to Assured Neural Network Compression
    COMMENT: -
    ABS: In this paper, we propose a concept of approximate bisimulation relation for feedforward neural networks. In the framework of approximate bisimulation relation, a novel neural network merging method is developed to compute the approximate bisimulation error between two neural networks based on reach

2310.17139v1 | 2023-10-26 | Understanding and Addressing the Pitfalls of Bisimulation-based Representations in Offline Reinforcement Learning
    COMMENT: NeurIPS 2023
    ABS: While bisimulation-based approaches hold promise for learning robust state representations for Reinforcement Learning (RL) tasks, their efficacy in offline RL tasks has not been up to par. In some instances, their performance has even significantly underperformed alternative methods. We aim to under

2403.16369v3 | 2024-03-25 | Learning Action-based Representations Using Invariance
    COMMENT: Published at the Reinforcement Learning Conference 2024
    ABS: Robust reinforcement learning agents using high-dimensional observations must be able to identify relevant state fe
... [truncated 9607 chars]
```
</details>

<details><summary>tool result</summary>

```
2504.21383v1 | 2025-04-30 | FAST-Q: Fast-track Exploration with Adversarially Balanced State Representations for Counterfactual Action Estimation in Offline Reinforcement Learning
    COMMENT: -
    ABS: Recent advancements in state-of-the-art (SOTA) offline reinforcement learning (RL) have primarily focused on addressing function approximation errors, which contribute to the overestimation of Q-values for out-of-distribution actions, a challenge that static datasets exacerbate. However, high stakes

[1 results for: abs:"gradient reversal" AND abs:"static"]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'abs:\"optical flow\" AND abs:\"manipulation policy\"' 20 2>&1 | tail -70", "description": "Search optical flow conditioning policy"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'abs:\"point trajectory\" AND cat:cs.RO' 25 2>&1 | tail -90", "description": "Search point trajectory robotics"}
```

<details><summary>tool result</summary>

```
2111.05623v3 | 2021-11-10 | FabricFlowNet: Bimanual Cloth Manipulation with a Flow-based Policy
    COMMENT: CoRL 2021
    ABS: We address the problem of goal-directed cloth manipulation, a challenging task due to the deformability of cloth. Our insight is that optical flow, a technique normally used for motion estimation in video, can also provide an effective representation for corresponding cloth poses across observation 

2505.24156v1 | 2025-05-30 | Towards a Generalizable Bimanual Foundation Policy via Flow-based Video Prediction
    COMMENT: -
    ABS: Learning a generalizable bimanual manipulation policy is extremely challenging for embodied agents due to the large action space and the need for coordinated arm movements. Existing approaches rely on Vision-Language-Action (VLA) models to acquire bimanual policies. However, transferring knowledge f

[2 results for: abs:"optical flow" AND abs:"manipulation policy"]
```
</details>

<details><summary>tool result</summary>

```
2303.09037v3 | 2023-03-16 | Homography matrix based trajectory planning method for robot uncalibrated visual servoing
    COMMENT: -
    ABS: In view of the classical visual servoing trajectory planning method which only considers the camera trajectory, this paper proposes one homography matrix based trajectory planning method for robot uncalibrated visual servoing. Taking the robot-end-effector frame as one generic case, eigenvalue decom

2309.11413v1 | 2023-09-20 | Enhancing motion trajectory segmentation of rigid bodies using a novel screw-based trajectory-shape representation
    COMMENT: This work has been submitted to the IEEE International Conference on Robotics and Automation (ICRA) for possible publication
    ABS: Trajectory segmentation refers to dividing a trajectory into meaningful consecutive sub-trajectories. This paper focuses on trajectory segmentation for 3D rigid-body motions. Most segmentation approaches in the literature represent the body's trajectory as a point trajectory, considering only its tr

1612.00192v2 | 2016-12-01 | Flight Dynamics-based Recovery of a UAV Trajectory using Ground Cameras
    COMMENT: -
    ABS: We propose a new method to estimate the 6-dof trajectory of a flying object such as a quadrotor UAV within a 3D airspace monitored using multiple fixed ground cameras. It is based on a new structure from motion formulation for the 3D reconstruction of a single moving point with known motion dynamics

2401.00025v3 | 2023-12-28 | Any-point Trajectory Modeling for Policy Learning
    COMMENT: 18 pages, 15 figures
    ABS: Learning from demonstration is a powerful method for teaching robots new skills, and having more demonstration data often improves policy learning. However, the high cost of collecting demonstration data is a significant bottleneck. Videos, as a rich data source, contain knowledge of behaviors, phys

2207.07289v1 | 2022-07-15 | A Simulation Study of Functional Electrical Stimulation for An Upper Limb Rehabilitatio
... [truncated 8469 chars]
```
</details>

### ASSISTANT — 2026-09-16 19:26

Good leads accumulating. Let me now run the targeted searches on the disentangling/orthogonalization recipe and controllability-aware latents.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'abs:\"controllability\" AND abs:\"representation\" AND cat:cs.LG AND abs:\"agent\"' 20 2>&1 | tail -70", "description": "Search controllability-aware representation"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'abs:\"HSIC\" AND abs:\"disentangle\"' 20 2>&1 | tail -50; echo \"########\"; python3 /tmp/wog/axs.py search 'abs:\"orthogonality\" AND abs:\"disentangled\" AND cat:cs.CV' 15 2>&1 | tail -50", "description": "Search HSIC disentangle"}
```

<details><summary>tool result</summary>

```
2503.12358v4 | 2025-03-16 | IPCGRL: Language-Instructed Reinforcement Learning for Procedural Level Generation
    COMMENT: 9 pages, 9 figures, 3 tables, accepted to Conference on Games 2025
    ABS: Recent research has highlighted the significance of natural language in enhancing the controllability of generative models. While various efforts have been made to leverage natural language for content generation, research on deep reinforcement learning (DRL) agents utilizing text-based instructions

2011.12690v3 | 2020-11-25 | DeepKoCo: Efficient latent planning with a task-relevant Koopman representation
    COMMENT: -
    ABS: This paper presents DeepKoCo, a novel model-based agent that learns a latent Koopman representation from images. This representation allows DeepKoCo to plan efficiently using linear control methods, such as linear model predictive control. Compared to traditional agents, DeepKoCo learns task-relevan

2607.05001v2 | 2026-07-06 | TACTIC-KG: Toward Small Agent Teams for Cyber Threat Intelligence Knowledge Graph Construction
    COMMENT: 20 pages, 2 figures, 10 tables
    ABS: Cyber Threat Intelligence (CTI) reports are predominantly unstructured, heterogeneous, and noisy, which limits their direct usability for automated analysis and reasoning. Cybersecurity Knowledge Graphs (CSKGs) provide a structured representation of adversarial entities, actions, and relations, but 

2607.04409v1 | 2026-07-05 | Learning Task-Sufficient World Models by Synergizing Agentic Exploration and Structured Modeling
    COMMENT: -
    ABS: Learning and planning in imagination using world models provides an effective paradigm for training agents for decision-making. However, existing approaches often rely on high-dimensional latent spaces or generic visual embeddings that retain many factors irrelevant to control, limiting efficiency a

2605.30656v1 | 2026-05-28 | Learning to Perceive the World Through Control: Empowerment-Based Representation Learning
    COMMENT: -
  
... [truncated 6158 chars]
```
</details>

<details><summary>tool result</summary>

```
2208.03563v1 | 2022-08-06 | HSIC-InfoGAN: Learning Unsupervised Disentangled Representations by Maximising Approximated Mutual Information
    COMMENT: MICCAI MAD Workshop 2022
    ABS: Learning disentangled representations requires either supervision or the introduction of specific model designs and learning constraints as biases. InfoGAN is a popular disentanglement framework that learns unsupervised disentangled representations by maximising the mutual information between latent

2110.15415v1 | 2021-10-28 | On the Use of CSI for the Generation of RF Fingerprints and Secret Keys
    COMMENT: -
    ABS: This paper presents a systematic approach to use channel state information for authentication and secret key distillation for physical layer security (PLS). We use popular machine learning (ML) methods and signal processing-based approaches to disentangle the large scale fading and be used as a sour

1911.11430v1 | 2019-11-26 | Independence Promoted Graph Disentangled Networks
    COMMENT: -
    ABS: We address the problem of disentangled representation learning with independent latent factors in graph convolutional networks (GCNs). The current methods usually learn node representation by describing its neighborhood as a perceptual whole in a holistic manner while ignoring the entanglement of th

1901.06618v1 | 2019-01-20 | Modeling the Biological Pathology Continuum with HSIC-regularized Wasserstein Auto-encoders
    COMMENT: -
    ABS: A crucial challenge in image-based modeling of biomedical data is to identify trends and features that separate normality and pathology. In many cases, the morphology of the imaged object exhibits continuous change as it deviates from normality, and thus a generative model can be trained to model th

2604.13409v1 | 2026-04-15 | CausalDisenSeg: A Causality-Guided Disentanglement Framework with Counterfactual Reasoning for Robust Brain Tumor Segmentation Under Missing Modalities
    COMMENT: -
    ABS: In clinical practice, the robust
... [truncated 8120 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'ti:\"dWorldEval\" OR abs:\"dWorldEval\"' 10 2>&1 | tail -30; echo \"#########\"; python3 /tmp/wog/axs.py search 'abs:\"Delta-LPIPS\" OR abs:\"LPIPS\" AND abs:\"world model\" AND abs:\"evaluation\" AND cat:cs.RO' 15 2>&1 | tail -50", "description": "Search dWorldEval"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'abs:\"latent action\" AND (abs:\"static\" OR abs:\"appearance\") AND cat:cs.RO' 25 2>&1 | tail -90", "description": "Search latent action static appearance"}
```

<details><summary>tool result</summary>

```
2604.22152v1 | 2026-04-24 | dWorldEval: Scalable Robotic Policy Evaluation via Discrete Diffusion World Model
    COMMENT: -
    ABS: Evaluating robotics policies across thousands of environments and thousands of tasks is infeasible with existing approaches. This motivates the need for a new methodology for scalable robotics policy evaluation. In this paper, we propose dWorldEval, which uses a discrete diffusion world model as a s

[1 results for: ti:"dWorldEval" OR abs:"dWorldEval"]
#########
2605.03821v1 | 2026-05-05 | RoboAlign-R1: Distilled Multimodal Reward Alignment for Robot Video World Models
    COMMENT: -
    ABS: Existing robot video world models are typically trained with low-level objectives such as reconstruction and perceptual similarity, which are poorly aligned with the capabilities that matter most for robot decision making, including instruction following, manipulation success, and physical plausibil

2609.03613v1 | 2026-09-03 | TileGS: Tile-Local Depth Binning for Gaussian Splatting Rasterization
    COMMENT: -
    ABS: Real-time 3D Gaussian Splatting (3DGS) achieves high rendering quality, but standard rasterization still traverses a globally sorted tile stream that creates long per-tile ranges and heavy geometry-attribute traffic. We present TileGS, a tile-local reorganization of Gaussian splatting. TileGS turns 

[2 results for: abs:"Delta-LPIPS" OR abs:"LPIPS" AND abs:"world model" AND abs:"evaluation" AND cat:cs.RO]
```
</details>

<details><summary>tool result</summary>

```
2602.20231v2 | 2026-02-23 | UniLACT: Depth-Aware RGB Latent Action Learning for Vision-Language-Action Models
    COMMENT: https://manishgovind.github.io/unilact-vla/
    ABS: Latent action representations learned from unlabeled videos have recently emerged as a promising paradigm for pretraining vision-language-action (VLA) models without explicit robot action supervision. However, latent actions derived solely from RGB observations primarily encode appearance-driven dyn

2607.11397v1 | 2026-07-13 | WALA Learning Executable Latent Actions from Action-Labeled Demonstrations and Action-Free Videos
    COMMENT: Project page: https://liujiahao2077.github.io/WALA.github.io
    ABS: Generalizable robot policies typically rely on action-labeled robot demonstrations, which are expensive to collect and difficult to scale. In contrast, large-scale human and robot videos contain rich physical interactions but often lack executable robot action labels. We present WALA, a framework fo

2601.02994v1 | 2026-01-06 | Learning to Act Robustly with View-Invariant Latent Actions
    COMMENT: Website: https://joon-stack.github.io/VILA/
    ABS: Vision-based robotic policies often struggle with even minor viewpoint changes, underscoring the need for view-invariant visual representations. This challenge becomes more pronounced in real-world settings, where viewpoint variability is unavoidable and can significantly disrupt policy performance.

2609.17099v1 | 2026-09-15 | GeoLAM: Learning Geometry-Grounded Latent Actions from Unlabeled Human Videos
    COMMENT: 8 pages, 6 figures, 4 tables
    ABS: Human videos provide rich manipulation experience, but extracting action representations that preserve useful motion remains challenging. Visual reconstruction alone can entangle manipulation-related motion with appearance changes and camera movement. We present GeoLAM, a framework for learning geom

2601.04061v2 | 2026-01-07 | CLAP: Contrastive Latent Action Pretraining for Learning Vision-Lang
... [truncated 4473 chars]
```
</details>

### ASSISTANT — 2026-09-16 19:27

Excellent leads. Let me get full abstracts for the top candidates.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nfor id in 2602.10098 2601.04061 2606.31232 2602.00557; do\necho \"=========== $id ===========\"\ncurl -sL --max-time 45 \"https://arxiv.org/abs/$id\" | python3 -c \"\nimport sys,re,html\nt=sys.stdin.read()\ndef g(p,d='(none)'):\n    m=re.search(p,t,re.S); return html.unescape(re.sub(r'\\s+',' ',re.sub(r'<[^>]+>','',m.group(1)))).strip() if m else d\nprint('TITLE:',g(r'<meta name=\\\"citation_title\\\" content=\\\"(.*?)\\\"'))\nprint('SUBMIT:',g(r'\\[Submitted on (.*?)\\]')[:80])\nprint('COMMENTS:',g(r'<td class=\\\"tablecell comments[^\\\"]*\\\">(.*?)</td>')[:300])\nprint('SUBJ:',g(r'<td class=\\\"tablecell subjects\\\">(.*?)</td>')[:150])\nm=re.search(r'<blockquote class=\\\"abstract mathjax\\\">(.*?)</blockquote>',t,re.S)\nprint('ABSTRACT:',html.unescape(re.sub(r'\\s+',' ',re.sub(r'<[^>]+>','',m.group(1)))).strip() if m else 'NOABS')\nimport re as r2\nprint('LINKS:', sorted(set(r2.findall(r'https?://(?:github\\.com|[a-z0-9-]+\\.github\\.io)[^\\\"<>\\s]*', t)))[:6])\n\"\ndone", "description": "Fetch full details for 4 key candidates"}
```

<details><summary>tool result</summary>

```
=========== 2602.10098 ===========
TITLE: VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model
SUBMIT: 10 Feb 2026 (v1), last revised 14 Feb 2026 (this version, v2)
COMMENTS: (none)
SUBJ: Robotics (cs.RO); Computer Vision and Pattern Recognition (cs.CV)
ABSTRACT: Abstract:Pretraining Vision-Language-Action (VLA) policies on internet-scale video is appealing, yet current latent-action objectives often learn the wrong thing: they remain anchored to pixel variation rather than action-relevant state transitions, making them vulnerable to appearance bias, nuisance motion, and information leakage. We introduce VLA-JEPA, a JEPA-style pretraining framework that sidesteps these pitfalls by design. The key idea is leakage-free state prediction: a target encoder produces latent representations from future frames, while the student pathway sees only the current observation -- future information is used solely as supervision targets, never as input. By predicting in latent space rather than pixel space, VLA-JEPA learns dynamics abstractions that are robust to camera motion and irrelevant background changes. This yields a simple two-stage recipe -- JEPA pretraining followed by action-head fine-tuning -- without the multi-stage complexity of prior latent-action pipelines. Experiments on LIBERO, LIBERO-Plus, SimplerEnv and real-world manipulation tasks show that VLA-JEPA achieves consistent gains in generalization and robustness over existing methods.
LINKS: []
=========== 2601.04061 ===========
TITLE: CLAP: Contrastive Latent Action Pretraining for Learning Vision-Language-Action Models from Human Videos
SUBMIT: 7 Jan 2026 (v1), last revised 14 Jun 2026 (this version, v2)
COMMENTS: The code is available at: this https URL
SUBJ: Robotics (cs.RO); Computer Vision and Pattern Recognition (cs.CV)
ABSTRACT: Abstract:Generalist Vision-Language-Action models remain constrained by the scarcity of robotic data relative to the abundance of human video demonstrations. Exi
... [truncated 4734 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'abs:\"information bottleneck\" AND abs:\"latent action\"' 20 2>&1 | tail -60; echo \"########\"; python3 /tmp/wog/axs.py search 'abs:\"mutual information\" AND abs:\"latent action\" AND cat:cs.RO' 20 2>&1 | tail -60", "description": "Search info bottleneck latent action"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'abs:\"adversarial\" AND abs:\"static\" AND abs:\"dynamic\" AND abs:\"disentangle\" AND (cat:cs.CV OR cat:cs.LG)' 20 2>&1 | tail -70", "description": "Search adversarial static dynamic disentangle"}
```

<details><summary>tool result</summary>

```
2508.08743v1 | 2025-08-12 | Boosting Action-Information via a Variational Bottleneck on Unlabelled Robot Videos
    COMMENT: -
    ABS: Learning from demonstrations (LfD) typically relies on large amounts of action-labeled expert trajectories, which fundamentally constrains the scale of available training data. A promising alternative is to learn directly from unlabeled video demonstrations. However, we find that existing methods te

2609.15189v1 | 2026-09-14 | Reconstructing Is Not Acting: Action-Centric Latent Dynamics Modeling
    COMMENT: Tech report, 26 pages
    ABS: Latent action models (LAMs) learn action representations from unlabeled videos by inferring latent actions from visual transitions and reconstructing future states. However, we identify a fundamental $\textbf{reconstruction-action mismatch}$: lower reconstruction error does not necessarily yield bet

2511.16175v2 | 2025-11-20 | Mantis: A Versatile Vision-Language-Action Model with Disentangled Visual Foresight
    COMMENT: -
    ABS: Recent advances in Vision-Language-Action (VLA) models demonstrate that visual signals can effectively complement sparse action supervisions. However, letting VLA directly predict high-dimensional visual states can distribute model capacity and incur prohibitive training cost, while compressing visu

[3 results for: abs:"information bottleneck" AND abs:"latent action"]
########
2508.08743v1 | 2025-08-12 | Boosting Action-Information via a Variational Bottleneck on Unlabelled Robot Videos
    COMMENT: -
    ABS: Learning from demonstrations (LfD) typically relies on large amounts of action-labeled expert trajectories, which fundamentally constrains the scale of available training data. A promising alternative is to learn directly from unlabeled video demonstrations. However, we find that existing methods te

2602.03668v3 | 2026-02-03 | MVP-LAM: Learning Action-Centric Latent Action via Cross-Viewpoint Reconstruction
    COMMENT: -
    ABS: Latent actions learned from d
... [truncated 807 chars]
```
</details>

<details><summary>tool result</summary>

```
2502.02690v3 | 2025-02-04 | Controllable Video Generation with Provable Disentanglement
    COMMENT: -
    ABS: Controllable video generation remains a significant challenge, despite recent advances in generating high-quality and consistent videos. Most existing methods for controlling video generation treat the video as a whole, neglecting intricate fine-grained spatiotemporal relationships, which limits bot

2208.07365v3 | 2022-08-15 | Unsupervised Video Domain Adaptation for Action Recognition: A Disentanglement Perspective
    COMMENT: NeurIPS 2023; 20 pages, 9 figures, 10 tables; Code at https://github.com/ldkong1205/TranSVAE
    ABS: Unsupervised video domain adaptation is a practical yet challenging task. In this work, for the first time, we tackle it from a disentanglement view. Our key idea is to handle the spatial and temporal domain divergence separately through disentanglement. Specifically, we consider the generation of c

2205.05871v2 | 2022-05-12 | Towards Robust Unsupervised Disentanglement of Sequential Data -- A Case Study Using Music Audio
    COMMENT: The paper is accepted to IJCAI 2022
    ABS: Disentangled sequential autoencoders (DSAEs) represent a class of probabilistic graphical models that describes an observed sequence with dynamic latent variables and a static latent variable. The former encode information at a frame rate identical to the observation, while the latter globally gover

2510.05717v1 | 2025-10-07 | DiffSDA: Unsupervised Diffusion Sequential Disentanglement Across Modalities
    COMMENT: -
    ABS: Unsupervised representation learning, particularly sequential disentanglement, aims to separate static and dynamic factors of variation in data without relying on labels. This remains a challenging problem, as existing approaches based on variational autoencoders and generative adversarial networks 

2210.10592v2 | 2022-10-19 | DyTed: Disentangled Representation Learning for Discrete-time Dynamic Graph
    COMMENT: -
    ABS: Unsupe
... [truncated 1286 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nfor id in 2609.15189 2508.08743 2511.16175 2602.03668; do\necho \"=========== $id ===========\"\ncurl -sL --max-time 45 \"https://arxiv.org/abs/$id\" | python3 -c \"\nimport sys,re,html\nt=sys.stdin.read()\ndef g(p,d='(none)'):\n    m=re.search(p,t,re.S); return html.unescape(re.sub(r'\\s+',' ',re.sub(r'<[^>]+>','',m.group(1)))).strip() if m else d\nprint('TITLE:',g(r'<meta name=\\\"citation_title\\\" content=\\\"(.*?)\\\"'))\nprint('SUBMIT:',g(r'\\[Submitted on (.*?)\\]')[:80])\nprint('COMMENTS:',g(r'<td class=\\\"tablecell comments[^\\\"]*\\\">(.*?)</td>')[:300])\nprint('SUBJ:',g(r'<td class=\\\"tablecell subjects\\\">(.*?)</td>')[:150])\nm=re.search(r'<blockquote class=\\\"abstract mathjax\\\">(.*?)</blockquote>',t,re.S)\nprint('ABSTRACT:',html.unescape(re.sub(r'\\s+',' ',re.sub(r'<[^>]+>','',m.group(1)))).strip() if m else 'NOABS')\nprint('LINKS:', sorted(set(re.findall(r'https?://(?:github\\.com|[a-z0-9-]+\\.github\\.io)[^\\\"<>\\s]*', t)))[:6])\n\"\ndone", "description": "Fetch details for 4 more candidates"}
```

<details><summary>tool result</summary>

```
=========== 2609.15189 ===========
TITLE: Reconstructing Is Not Acting: Action-Centric Latent Dynamics Modeling
SUBMIT: 14 Sep 2026
COMMENTS: Tech report, 26 pages
SUBJ: Computer Vision and Pattern Recognition (cs.CV)
ABSTRACT: Abstract:Latent action models (LAMs) learn action representations from unlabeled videos by inferring latent actions from visual transitions and reconstructing future states. However, we identify a fundamental $\textbf{reconstruction-action mismatch}$: lower reconstruction error does not necessarily yield better latent dynamics or downstream performance. We attribute this mismatch to two underconstrained aspects of reconstruction-based latent dynamics modeling: (i) the inverse dynamics model (IDM) is not explicitly encouraged to distinguish action-related transitions from nuisance appearance, and (ii) the forward dynamics model (FDM) can underutilize the inferred latent action by exploiting predictive shortcuts from the current state. To address both limitations, we propose $\textbf{ACT-LAM}$, a lightweight action-centric framework that strengthens both action extraction and action utilization. Specifically, its Action Query IDM (AQ-IDM) employs learnable action queries and gated aggregation to selectively extract rich action-related transition cues without strong information bottlenecks. And its Action Token FDM (AT-FDM) projects latent actions into action tokens that progressively interact with evolving state representations, enabling continuous state-aware action conditioning. ACT-LAM further streamlines feature processing to concentrate model capacity on latent dynamics modeling. Extensive experiments on several robotic datasets and the VP$^2$ benchmark demonstrate stronger latent action consistency, forward dynamics, and downstream visual planning performance with fewer trainable parameters and lower computational overhead. In particular, ACT-LAM surpasses the previous state of the art by $\textbf{7.6%}$ on the aggregated VP$^2$ success r
... [truncated 5119 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nfor id in 2510.05057 2401.00025 2609.09210 2602.20231; do\necho \"=========== $id ===========\"\ncurl -sL --max-time 45 \"https://arxiv.org/abs/$id\" | python3 -c \"\nimport sys,re,html\nt=sys.stdin.read()\ndef g(p,d='(none)'):\n    m=re.search(p,t,re.S); return html.unescape(re.sub(r'\\s+',' ',re.sub(r'<[^>]+>','',m.group(1)))).strip() if m else d\nprint('TITLE:',g(r'<meta name=\\\"citation_title\\\" content=\\\"(.*?)\\\"'))\nprint('SUBMIT:',g(r'\\[Submitted on (.*?)\\]')[:80])\nprint('COMMENTS:',g(r'<td class=\\\"tablecell comments[^\\\"]*\\\">(.*?)</td>')[:300])\nprint('SUBJ:',g(r'<td class=\\\"tablecell subjects\\\">(.*?)</td>')[:150])\nm=re.search(r'<blockquote class=\\\"abstract mathjax\\\">(.*?)</blockquote>',t,re.S)\nprint('ABSTRACT:',html.unescape(re.sub(r'\\s+',' ',re.sub(r'<[^>]+>','',m.group(1)))).strip() if m else 'NOABS')\nprint('LINKS:', sorted(set(re.findall(r'https?://(?:github\\.com|[a-z0-9-]+\\.github\\.io)[^\\\"<>\\s]*', t)))[:6])\n\"\ndone", "description": "Fetch StaMo, ATM, and 2 more"}
```

<details><summary>tool result</summary>

```
=========== 2510.05057 ===========
TITLE: StaMo: Unsupervised Learning of Generalizable Robot Motion from Compact State Representation
SUBMIT: 6 Oct 2025 (v1), last revised 12 Apr 2026 (this version, v2)
COMMENTS: (none)
SUBJ: Robotics (cs.RO); Computer Vision and Pattern Recognition (cs.CV)
ABSTRACT: Abstract:A fundamental challenge in embodied intelligence is developing expressive and compact state representations for efficient world modeling and decision making. However, existing methods often fail to achieve this balance, yielding representations that are either overly redundant or lacking in task-critical information. We propose an unsupervised approach that learns a highly compressed two-token state representation using a lightweight encoder and a pre-trained Diffusion Transformer (DiT) decoder, capitalizing on its strong generative prior. Our representation is efficient, interpretable, and integrates seamlessly into existing VLA-based models, improving performance by 14.3% on LIBERO and 30% in real-world task success with minimal inference overhead. More importantly, we find that the difference between these tokens, obtained via latent interpolation, naturally serves as a highly effective latent action, which can be further decoded into executable robot actions. This emergent capability reveals that our representation captures structured dynamics without explicit supervision. We name our method StaMo for its ability to learn generalizable robotic Motion from compact State representation, which is encoded from static images, challenging the prevalent dependence to learning latent action on complex architectures and video data. The resulting latent actions also enhance policy co-training, outperforming prior methods by 10.4% with improved interpretability. Moreover, our approach scales effectively across diverse data sources, including real-world robot data, simulation, and human egocentric video.
LINKS: []
=========== 2401.00025 ===========
TITLE: Any-point Tr
... [truncated 4671 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'abs:\"adversarial\" AND abs:\"invariant\" AND abs:\"nuisance\" AND abs:\"representation\" AND cat:cs.LG' 20 2>&1 | tail -60", "description": "Search adversarial nuisance invariance"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'abs:\"empowerment\" AND abs:\"representation learning\"' 15 2>&1 | tail -50; echo \"########\"; python3 /tmp/wog/axs.py abs 2605.30656 2403.16369 2>&1", "description": "Search empowerment representation"}
```

<details><summary>tool result</summary>

```
1805.08097v1 | 2018-05-21 | Invariant Representations from Adversarially Censored Autoencoders
    COMMENT: 9 pages, 5 figures
    ABS: We combine conditional variational autoencoders (VAE) with adversarial censoring in order to learn invariant representations that are disentangled from nuisance/sensitive variations. In this method, an adversarial network attempts to recover the nuisance variable from the representation, which the V

2007.01255v3 | 2020-07-02 | AutoBayes: Automated Bayesian Graph Exploration for Nuisance-Robust Inference
    COMMENT: 24 pages, 11 figures, under review in ICLR2021
    ABS: Learning data representations that capture task-related features, but are invariant to nuisance variations remains a key challenge in machine learning. We introduce an automated Bayesian inference framework, called AutoBayes, that explores different graphical models linking classifier, encoder, deco

2209.06827v1 | 2022-09-15 | Weakly Supervised Invariant Representation Learning Via Disentangling Known and Unknown Nuisance Factors
    COMMENT: -
    ABS: Disentangled and invariant representations are two critical goals of representation learning and many approaches have been proposed to achieve either one of them. However, those two goals are actually complementary to each other so that we propose a framework to accomplish both of them simultaneousl

1809.10083v1 | 2018-09-26 | Unsupervised Adversarial Invariance
    COMMENT: To appear in Proceedings of NIPS 2018
    ABS: Data representations that contain all the information about target variables but are invariant to nuisance factors benefit supervised learning algorithms by preventing them from learning associations between these factors and the targets, thus reducing overfitting. We present a novel unsupervised in

1905.03629v2 | 2019-05-07 | Unified Adversarial Invariance
    COMMENT: In submission to T-PAMI. Some results updated. arXiv admin note: substantial text overlap with arXiv:1809.10083
    ABS: We pres
... [truncated 3005 chars]
```
</details>

<details><summary>tool result</summary>

```
2106.01404v1 | 2021-06-02 | Variational Empowerment as Representation Learning for Goal-Based Reinforcement Learning
    COMMENT: Accepted at International Conference on Machine Learning (ICML) 2021
    ABS: Learning to reach goal states and learning diverse skills through mutual information (MI) maximization have been proposed as principled frameworks for self-supervised reinforcement learning, allowing agents to acquire broadly applicable multitask policies with minimal reward engineering. Starting fr

1905.09334v1 | 2019-05-22 | The Journey is the Reward: Unsupervised Learning of Influential Trajectories
    COMMENT: ICML'19 ERL Workshop
    ABS: Unsupervised exploration and representation learning become increasingly important when learning in diverse and sparse environments. The information-theoretic principle of empowerment formalizes an unsupervised exploration objective through an agent trying to maximize its influence on the future sta

2507.08210v1 | 2025-07-10 | From Curiosity to Competence: How World Models Interact with the Dynamics of Exploration
    COMMENT: -
    ABS: What drives an agent to explore the world while also maintaining control over the environment? From a child at play to scientists in the lab, intelligent agents must balance curiosity (the drive to seek knowledge) with competence (the drive to master and control the environment). Bridging cognitive 

2009.01791v3 | 2020-09-03 | Action and Perception as Divergence Minimization
    COMMENT: Website: https://danijar.com/apd
    ABS: To learn directed behaviors in complex environments, intelligent agents need to optimize objective functions. Various objectives are known for designing artificial agents, including task rewards and intrinsic motivation. However, it is unclear how the known objectives relate to each other, which obj

[4 results for: abs:"empowerment" AND abs:"representation learning"]
########
=== 2605.30656 ===
TITLE : Learning to Perceive the World Through Control: Empowermen
... [truncated 434 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nfor id in 2403.16369 2605.30656; do\necho \"=========== $id ===========\"\ncurl -sL --max-time 45 \"https://arxiv.org/abs/$id\" | python3 -c \"\nimport sys,re,html\nt=sys.stdin.read()\ndef g(p,d='(none)'):\n    m=re.search(p,t,re.S); return html.unescape(re.sub(r'\\s+',' ',re.sub(r'<[^>]+>','',m.group(1)))).strip() if m else d\nprint('TITLE:',g(r'<meta name=\\\"citation_title\\\" content=\\\"(.*?)\\\"'))\nprint('COMMENTS:',g(r'<td class=\\\"tablecell comments[^\\\"]*\\\">(.*?)</td>')[:300])\nm=re.search(r'<blockquote class=\\\"abstract mathjax\\\">(.*?)</blockquote>',t,re.S)\nprint('ABSTRACT:',html.unescape(re.sub(r'\\s+',' ',re.sub(r'<[^>]+>','',m.group(1)))).strip() if m else 'NOABS')\nprint('LINKS:', sorted(set(re.findall(r'https?://(?:github\\.com|[a-z0-9-]+\\.github\\.io)[^\\\"<>\\s]*', t)))[:6])\n\"\ndone", "description": "Recheck comments for ACRO and empowerment paper"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'abs:\"flow\" AND abs:\"vision-language-action\" AND abs:\"intermediate representation\"' 20 2>&1|tail -50; echo \"#####\"; python3 /tmp/wog/axs.py search 'ti:\"Track2Act\" OR ti:\"Im2Flow2Act\" OR ti:\"General Flow\"' 10 2>&1 | tail -40", "description": "Search flow-based intermediate representations"}
```

<details><summary>tool result</summary>

```
=========== 2403.16369 ===========
TITLE: Learning Action-based Representations Using Invariance
COMMENTS: Published at the Reinforcement Learning Conference 2024
ABSTRACT: Abstract:Robust reinforcement learning agents using high-dimensional observations must be able to identify relevant state features amidst many exogeneous distractors. A representation that captures controllability identifies these state elements by determining what affects agent control. While methods such as inverse dynamics and mutual information capture controllability for a limited number of timesteps, capturing long-horizon elements remains a challenging problem. Myopic controllability can capture the moment right before an agent crashes into a wall, but not the control-relevance of the wall while the agent is still some distance away. To address this we introduce action-bisimulation encoding, a method inspired by the bisimulation invariance pseudometric, that extends single-step controllability with a recursive invariance constraint. By doing this, action-bisimulation learns a multi-step controllability metric that smoothly discounts distant state features that are relevant for control. We demonstrate that action-bisimulation pretraining on reward-free, uniformly random data improves sample efficiency in several environments, including a photorealistic 3D simulation domain, Habitat. Additionally, we provide theoretical analysis and qualitative results demonstrating the information captured by action-bisimulation.
LINKS: []
=========== 2605.30656 ===========
TITLE: Learning to Perceive the World Through Control: Empowerment-Based Representation Learning
COMMENTS: (none)
ABSTRACT: Abstract:In many practical reinforcement learning environments, observations are far higher-dimensional than the variables that matter for control. In this work, we ask: can we learn representations that capture only control-relevant features of the environment? We study this question through the empowerment objecti
... [truncated 722 chars]
```
</details>

<details><summary>tool result</summary>

```
2604.21241v2 | 2026-04-23 | CorridorVLA: Explicit Spatial Constraints for Generative Action Heads via Sparse Anchors
    COMMENT: Accepted to the 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2026)
    ABS: Vision--Language--Action (VLA) models often use intermediate representations to connect multimodal inputs with continuous control, yet spatial guidance is often injected implicitly through latent features. We propose CorridorVLA, which predicts sparse spatial anchors as incremental physical changes 

[1 results for: abs:"flow" AND abs:"vision-language-action" AND abs:"intermediate representation"]
#####
    COMMENT: ECCV 2024. Last 3 authors contributed equally
    ABS: We seek to learn a generalizable goal-conditioned policy that enables zero-shot robot manipulation: interacting with unseen objects in novel scenes without test-time adaptation. While typical approaches rely on a large amount of demonstration data for such generalization, we propose an approach that

2106.01707v2 | 2021-06-03 | Parametric Geometry of Numbers with General Flow
    COMMENT: -
    ABS: We extend results of parametric geometry of numbers to a general diagonal flow on the space of lattices. Moreover, we compute the Hausdorff dimension of the set of trajectories with every given behavior, with respect to a nonstandard metric and thereby attain bounds on the standard ones.

2401.11439v2 | 2024-01-21 | General Flow as Foundation Affordance for Scalable Robot Learning
    COMMENT: https://general-flow.github.io
    ABS: We address the challenge of acquiring real-world manipulation skills with a scalable framework. We hold the belief that identifying an appropriate prediction target capable of leveraging large-scale datasets is crucial for achieving efficient and universal learning. Therefore, we propose to utilize 

2505.03561v1 | 2025-05-06 | Ergodic Generative Flows
    COMMENT: 20 pages, 5 figures, 1 table, accepted at ICML 2025
    ABS: Generative Flow 
... [truncated 2814 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'ti:\"Track2Act\"' 5 2>&1|tail -20; echo \"#####\"; python3 /tmp/wog/axs.py search 'abs:\"residual\" AND abs:\"conditioning\" AND abs:\"world model\" AND cat:cs.RO' 20 2>&1|tail -60", "description": "Search Track2Act and residual conditioning"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'abs:\"what changed\" OR (abs:\"change\" AND abs:\"delta\" AND abs:\"visual foresight\" AND cat:cs.RO)' 20 2>&1|tail -60", "description": "Search what-changed objectives"}
```

<details><summary>tool result</summary>

```
2405.01527v2 | 2024-05-02 | Track2Act: Predicting Point Tracks from Internet Videos enables Generalizable Robot Manipulation
    COMMENT: ECCV 2024. Last 3 authors contributed equally
    ABS: We seek to learn a generalizable goal-conditioned policy that enables zero-shot robot manipulation: interacting with unseen objects in novel scenes without test-time adaptation. While typical approaches rely on a large amount of demonstration data for such generalization, we propose an approach that

[1 results for: ti:"Track2Act"]
#####
2607.02403v1 | 2026-07-02 | ACID: Action Consistency via Inverse Dynamics for Planning with World Models
    COMMENT: Project Page: [this https URL](https://gawon1224.github.io/ACID/)
    ABS: Decision-time planning with action-conditioned world models has become a popular paradigm for embodied control. However, the standard planning cost judges a candidate solely by how close its predicted terminal state lies to the goal, leaving the realizability of the intermediate transitions unchecke

2603.11110v1 | 2026-03-11 | ResWM: Residual-Action World Model for Visual RL
    COMMENT: Submit KDD2026
    ABS: Learning predictive world models from raw visual observations is a central challenge in reinforcement learning (RL), especially for robotics and continuous control. Conventional model-based RL frameworks directly condition future predictions on absolute actions, which makes optimization unstable: th

2606.09640v1 | 2026-06-08 | Physics-Aware Sparse Learning and Selective Online Adaptation for Euler-Lagrange Robot Dynamics
    COMMENT: -
    ABS: Accurate dynamics models are essential for model-based robotic control, yet nominal Euler--Lagrange models often become inaccurate in the presence of payload variation, unmodeled coupling, friction, aerodynamic effects, and changing operating conditions. Most learning-based correction methods improv

2606.13817v1 | 2026-06-11 | FlowMo-WM: A World Model with Object Momentum and Hidden Ambient Drift
    COM
... [truncated 5213 chars]
```
</details>

<details><summary>tool result</summary>

```
    COMMENT: -
    ABS: Traditional change detection identifies where changes occur, but does not explain what changed in natural language. Existing remote sensing change captioning datasets typically describe overall image-level differences, leaving fine-grained localized semantic reasoning largely unexplored. To close th

2003.03492v1 | 2020-03-07 | Semantic Change Pattern Analysis
    COMMENT: 17 pages, 7 figures
    ABS: Change detection is an important problem in vision field, especially for aerial images. However, most works focus on traditional change detection, i.e., where changes happen, without considering the change type information, i.e., what changes happen. Although a few works have tried to apply semantic

1307.5098v2 | 2013-07-18 | An electromagnetic analog of gravitational wave memory
    COMMENT: -
    ABS: We present an electromagnetic analog of gravitational wave memory. That is, we consider what change has occurred to a detector of electromagnetic radiation after the wave has passed. Rather than a distortion in the detector, as occurs in the gravitational wave case, we find a residual velocity (a "k

2607.01935v2 | 2026-07-02 | A-TMA: Decoupling State-Aware Memory Failures in Long-Term Agent Memory
    COMMENT: -
    ABS: Long term memory lets LLM agents act as persistent assistants, but user facts change. A useful memory system must know what is true now, what used to be true, and what changed. We study \emph{ghost memory}, a state coordination failure in which old, current, and transition facts coexist in the memor

2107.09124v1 | 2021-07-19 | Decoherence in the three-state quantum walk
    COMMENT: -
    ABS: Quantum walks are dynamic systems with a wide range of applications in quantum computation and quantum simulation of analog systems, therefore it is of common interest to understand what changes from an isolated process to one embedded in an environment. In the present work, we analyze the decoheren

2003.07912v2 | 2020-03-17 | Predictin
... [truncated 4706 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nfor id in 2608.02990 2603.11110 2604.16391; do\necho \"=========== $id ===========\"\ncurl -sL --max-time 45 \"https://arxiv.org/abs/$id\" | python3 -c \"\nimport sys,re,html\nt=sys.stdin.read()\ndef g(p,d='(none)'):\n    m=re.search(p,t,re.S); return html.unescape(re.sub(r'\\s+',' ',re.sub(r'<[^>]+>','',m.group(1)))).strip() if m else d\nprint('TITLE:',g(r'<meta name=\\\"citation_title\\\" content=\\\"(.*?)\\\"'))\nprint('SUBMIT:',g(r'\\[Submitted on (.*?)\\]')[:80])\nprint('COMMENTS:',g(r'<td class=\\\"tablecell comments[^\\\"]*\\\">(.*?)</td>')[:300])\nm=re.search(r'<blockquote class=\\\"abstract mathjax\\\">(.*?)</blockquote>',t,re.S)\nprint('ABSTRACT:',html.unescape(re.sub(r'\\s+',' ',re.sub(r'<[^>]+>','',m.group(1)))).strip() if m else 'NOABS')\nprint('LINKS:', sorted(set(re.findall(r'https?://(?:github\\.com|[a-z0-9-]+\\.github\\.io)[^\\\"<>\\s]*', t)))[:6])\n\"\ndone", "description": "Check EmbodiedVAE, ResWM, disentangled fwd/inv"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'abs:\"predictive coding\" AND abs:\"video prediction\" AND abs:\"residual\"' 15 2>&1|tail -40; echo \"#####\"; python3 /tmp/wog/axs.py search 'abs:\"subtract\" AND abs:\"static\" AND abs:\"latent\" AND cat:cs.RO' 15 2>&1|tail -40", "description": "Search predictive coding residual"}
```

<details><summary>tool result</summary>

```
=========== 2608.02990 ===========
TITLE: EmbodiedVAE: Disentangled Video VAE for Efficient and Controllable Embodied Manipulation
SUBMIT: 4 Aug 2026
COMMENTS: ECCV 2026
ABSTRACT: Abstract:Latent diffusion models (LDMs) have recently significantly advanced embodied learning in constructing powerful embodied manipulation world models. However, despite the remarkable performance, existing LDMs predominantly rely on Variational Autoencoders (VAEs) optimized for natural scenes while failing to account for the unique characteristics of embodied manipulation scenarios, yielding latent representations that are neither compact nor controllable, thereby hindering efficient training of LDMs and precise robotic control. To solve this problem, we present EmbodiedVAE, a novel video VAE that provides compact yet controllable latent representations tailored for the robotic manipulation world models. Specifically, EmbodiedVAE adopts a dual-encoder, single-decoder architecture with an asymmetric spatio-temporal compression module, which automatically disentangles the robot arm's motion from background environment, resulting in overall compactness while providing explicit embodied latent to support fine-grained action control. To further preserve the temporal consistency of learned robotic motion latent, we introduce an optimal-transport-based consistency module that explicitly enforces motion fidelity and inter-frame coherence. Extensive experiments demonstrate that our proposed EmbodiedVAE achieves superior reconstruction quality with high compression rate, while enabling more precise action control in robotic manipulation scenarios with an average of 2dB PSNR improvement over state-of-the-art video VAEs.
LINKS: []
=========== 2603.11110 ===========
TITLE: ResWM: Residual-Action World Model for Visual RL
SUBMIT: 11 Mar 2026
COMMENTS: Submit KDD2026
ABSTRACT: Abstract:Learning predictive world models from raw visual observations is a central challenge in reinforcement learning (RL),
... [truncated 3289 chars]
```
</details>

<details><summary>tool result</summary>

```
[0 results for: abs:"predictive coding" AND abs:"video prediction" AND abs:"residual"]
#####
[0 results for: abs:"subtract" AND abs:"static" AND abs:"latent" AND cat:cs.RO]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\necho \"=== CoMo v3 HTML full text search for github ===\"\ncurl -sL --max-time 60 \"https://arxiv.org/abs/2505.17006v3\" | grep -oiE 'https?://[a-z0-9./_-]*(github|hf\\.co|huggingface)[a-z0-9./_-]*' | sort -u | head\necho \"=== try searching github for CoMo latent motion ===\"\ncurl -sL --max-time 45 \"https://api.github.com/search/repositories?q=CoMo+latent+motion\" | python3 -c \"\nimport sys,json\nd=json.load(sys.stdin)\nfor r in d.get('items',[])[:8]: print(r['full_name'],'|',r['description'],'|',r['html_url'])\n\"", "description": "Find CoMo code repo"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'abs:\"information leakage\" AND abs:\"latent action\"' 15 2>&1|tail -40; echo \"#####\"; python3 /tmp/wog/axs.py search 'abs:\"shortcut\" AND abs:\"latent action\" AND cat:cs.RO' 20 2>&1|tail -60", "description": "Search leakage and shortcut latent action"}
```

<details><summary>tool result</summary>

```
=== CoMo v3 HTML full text search for github ===
https://huggingface.co/docs/hub/spaces
https://huggingface.co/huggingface
=== try searching github for CoMo latent motion ===
MCG-NJU/CoMo | [CVPR 2026] CoMo: Learning Continuous Latent Motion from Internet Videos for Scalable Robot Learning | https://github.com/MCG-NJU/CoMo
```
</details>

<details><summary>tool result</summary>

```
2602.10098v2 | 2026-02-10 | VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model
    COMMENT: -
    ABS: Pretraining Vision-Language-Action (VLA) policies on internet-scale video is appealing, yet current latent-action objectives often learn the wrong thing: they remain anchored to pixel variation rather than action-relevant state transitions, making them vulnerable to appearance bias, nuisance motion,

[1 results for: abs:"information leakage" AND abs:"latent action"]
#####
2602.00557v1 | 2026-01-31 | ConLA: Contrastive Latent Action Learning from Human Videos for Robotic Manipulation
    COMMENT: -
    ABS: Vision-Language-Action (VLA) models achieve preliminary generalization through pretraining on large scale robot teleoperation datasets. However, acquiring datasets that comprehensively cover diverse tasks and environments is extremely costly and difficult to scale. In contrast, human demonstration v

2601.22467v1 | 2026-01-30 | CARE: Multi-Task Pretraining for Latent Continuous Action Representation in Robot Control
    COMMENT: Accepted to 2026 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2026)
    ABS: Recent advances in Vision-Language-Action (VLA) models have shown promise for robot control, but their dependence on action supervision limits scalability and generalization. To address this challenge, we introduce CARE, a novel framework designed to train VLA models for robotic task execution. Unli

2601.15197v7 | 2026-01-21 | LangForce: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries
    COMMENT: ICML 2026
    ABS: Vision-Language-Action (VLA) models have shown promise in robot manipulation but often struggle to generalize to new instructions or complex multi-task scenarios. We identify a critical pathology in current training paradigms where goal-driven data collection creates a dataset bias. In such datasets

[3 results for: abs:"shortcut" AND abs:"latent action" AND 
... [truncated 10 chars]
```
</details>

### ASSISTANT — 2026-09-16 19:31

CoMo code confirmed at MCG-NJU/CoMo. Let me verify the repo exists and inspect its Td mechanism, plus check remaining leads.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ncurl -s --max-time 45 \"https://api.github.com/repos/MCG-NJU/CoMo\" | python3 -c \"\nimport sys,json; d=json.load(sys.stdin)\nprint('repo:',d.get('full_name'),'| stars:',d.get('stargazers_count'),'| pushed:',d.get('pushed_at'),'| lic:',(d.get('license') or {}).get('spdx_id'))\n\"\necho \"=== file tree ===\"\ncurl -s --max-time 45 \"https://api.github.com/repos/MCG-NJU/CoMo/git/trees/main?recursive=1\" | python3 -c \"\nimport sys,json\nd=json.load(sys.stdin)\nps=[t['path'] for t in d.get('tree',[]) if t['type']=='blob']\nprint(len(ps),'files')\nfor p in ps[:60]: print(' ',p)\n\"", "description": "Verify CoMo repo and files"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'abs:\"variance\" AND abs:\"invariance\" AND abs:\"covariance\" AND abs:\"VICReg\"' 15 2>&1|tail -40; echo \"#####\"; python3 /tmp/wog/axs.py search 'abs:\"Barlow Twins\" OR abs:\"redundancy reduction\" AND abs:\"representation\"' 15 2>&1|tail -40", "description": "Search VICReg/Barlow decorrelation"}
```

<details><summary>tool result</summary>

```
repo: MCG-NJU/CoMo | stars: 16 | pushed: 2026-05-15T14:05:08Z | lic: None
=== file tree ===
0 files
```
</details>

<details><summary>tool result</summary>

```
    COMMENT: -
    ABS: Self-supervised learning methods prevent embedding collapse via modeling heuristics or explicit regularization of the embedding space. Among the latter, VICReg decomposes regularization into variance and covariance objectives, offering flexibility and interpretability. However, covariance captures o

2405.18320v2 | 2024-05-28 | Self-Supervised Learning Based Handwriting Verification
    COMMENT: 8 pages, 2 figures, 2 tables, Accepted at Irish Machine Vision and Image Processing Conference 2024
    ABS: We present SSL-HV: Self-Supervised Learning approaches applied to the task of Handwriting Verification. This task involves determining whether a given pair of handwritten images originate from the same or different writer distribution. We have compared the performance of multiple generative, contras

2607.16681v1 | 2026-07-18 | A Framework for Early Sepsis Prediction via Self-Supervised (JEPA) and Federated Representation Learning
    COMMENT: 11 pages, 6 figures, Submitted to arXiv
    ABS: Early sepsis prediction from electronic health records is challenged by irregular sampling, high missingness, and class imbalance. We systematically compare four modeling paradigms -- self-supervised Joint Embedding Predictive Architecture (JEPA) via masked latent prediction, self-supervised VICReg 

2410.19560v1 | 2024-10-25 | Connecting Joint-Embedding Predictive Architecture with Contrastive Self-supervised Learning
    COMMENT: -
    ABS: In recent advancements in unsupervised visual representation learning, the Joint-Embedding Predictive Architecture (JEPA) has emerged as a significant method for extracting visual features from unlabeled imagery through an innovative masking strategy. Despite its success, two primary limitations hav

2607.16350v1 | 2026-07-17 | Joint-Embedding Predictive Architecture for Sensor-based Activity Recognition
    COMMENT: -
    ABS: Sensor-based human activity recognition (HAR) has achieved significant progressed in fully s
... [truncated 7242 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ncurl -s --max-time 45 \"https://api.github.com/repos/MCG-NJU/CoMo\" | python3 -c \"import sys,json;d=json.load(sys.stdin);print('default_branch:',d.get('default_branch'),'size',d.get('size'))\"\nfor b in master main; do\necho \"--- branch $b ---\"\ncurl -s --max-time 45 \"https://api.github.com/repos/MCG-NJU/CoMo/git/trees/$b?recursive=1\" | python3 -c \"\nimport sys,json\nd=json.load(sys.stdin)\nps=[t['path'] for t in d.get('tree',[]) if t['type']=='blob']\nprint(len(ps),'files'); [print(' ',p) for p in ps[:50]]\nprint(d.get('message',''))\n\"\ndone", "description": "Check CoMo repo branch and tree"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py abs 1809.10083 1805.09458 1409.7495 2>&1", "description": "Verify adversarial invariance classics"}
```

<details><summary>tool result</summary>

```
default_branch: master size 215740
--- branch master ---
1222 files
  DP/como_policy/dataloader/__init__.py
  DP/como_policy/dataloader/base_dataset.py
  DP/como_policy/dataloader/bc_dataloader.py
  DP/como_policy/dataloader/utils.py
  DP/como_policy/policy/RoboticsDiffusionTransformer/.gitignore
  DP/como_policy/policy/RoboticsDiffusionTransformer/LICENSE
  DP/como_policy/policy/RoboticsDiffusionTransformer/README.md
  DP/como_policy/policy/RoboticsDiffusionTransformer/assets/head.png
  DP/como_policy/policy/RoboticsDiffusionTransformer/configs/base.yaml
  DP/como_policy/policy/RoboticsDiffusionTransformer/configs/calvin_rel_traj_location_bounds_task_ABC_D.json
  DP/como_policy/policy/RoboticsDiffusionTransformer/configs/dataset_control_freq.json
  DP/como_policy/policy/RoboticsDiffusionTransformer/configs/dataset_img_keys.json
  DP/como_policy/policy/RoboticsDiffusionTransformer/configs/dataset_stat.json
  DP/como_policy/policy/RoboticsDiffusionTransformer/configs/finetune_datasets.json
  DP/como_policy/policy/RoboticsDiffusionTransformer/configs/finetune_sample_weights.json
  DP/como_policy/policy/RoboticsDiffusionTransformer/configs/pretrain_datasets.json
  DP/como_policy/policy/RoboticsDiffusionTransformer/configs/pretrain_sample_weights.json
  DP/como_policy/policy/RoboticsDiffusionTransformer/configs/state_vec.py
  DP/como_policy/policy/RoboticsDiffusionTransformer/configs/zero2.json
  DP/como_policy/policy/RoboticsDiffusionTransformer/data/.gitignore
  DP/como_policy/policy/RoboticsDiffusionTransformer/data/agilex/hdf5totfrecords.py
  DP/como_policy/policy/RoboticsDiffusionTransformer/data/aloha/hdf5totfrecords.py
  DP/como_policy/policy/RoboticsDiffusionTransformer/data/aloha/unzip_data.sh
  DP/como_policy/policy/RoboticsDiffusionTransformer/data/bridgev2/bridgedata_numpy_to_tfrecord.py
  DP/como_policy/policy/RoboticsDiffusionTransformer/data/bridgev2/bridgedata_raw_to_numpy.py
  DP/como_policy/policy/RoboticsDiffusionTransformer/data/bridgev2/download.sh

... [truncated 2287 chars]
```
</details>

<details><summary>tool result</summary>

```
=== 1809.10083 ===
TITLE : Unsupervised Adversarial Invariance
SUBMIT: 26 Sep 2018 | citdate: 2018/09/26
COMM  : (none)
SUBJ  : Machine Learning (cs.LG); Machine Learning (stat.ML)
DOI   : (none)

=== 1805.09458 ===
TITLE : Invariant Representations without Adversarial Training
SUBMIT: 24 May 2018 (v1), last revised 2 Dec 2019 (this version, v4) | citdate: 2018/05/24
COMM  : (none)
SUBJ  : Machine Learning (cs.LG); Machine Learning (stat.ML)
DOI   : (none)

=== 1409.7495 ===
TITLE : Unsupervised Domain Adaptation by Backpropagation
SUBMIT: 26 Sep 2014 (v1), last revised 27 Feb 2015 (this version, v2 | citdate: 2014/09/26
COMM  : (none)
SUBJ  : Machine Learning (stat.ML); Machine Learning (cs.LG); Neural and Evolutionary Computing (cs.NE)
DOI   : (none)
```
</details>

### ASSISTANT — 2026-09-16 19:32

CoMo code is confirmed public. Let me inspect CoMo's Td implementation to give the exact graft recipe.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ncurl -s --max-time 45 \"https://api.github.com/repos/MCG-NJU/CoMo/git/trees/master?recursive=1\" | python3 -c \"\nimport sys,json,re\nd=json.load(sys.stdin)\nps=[t['path'] for t in d.get('tree',[]) if t['type']=='blob']\nps=[p for p in ps if 'RoboticsDiffusionTransformer' not in p and 'preprocess_scripts' not in p]\nprint(len(ps))\nfor p in ps:\n    if p.endswith('.py') or p.endswith('.md') or p.endswith('.yaml'): print(' ',p)\n\" | head -80", "description": "List CoMo core source files"}
```

<details><summary>tool result</summary>

```
1077
  DP/como_policy/dataloader/__init__.py
  DP/como_policy/dataloader/base_dataset.py
  DP/como_policy/dataloader/bc_dataloader.py
  DP/como_policy/dataloader/utils.py
  DP/como_policy/policy/__init__.py
  DP/como_policy/policy/rdt_libero_policy.py
  DP/como_policy/utils/env_utils.py
  DP/como_policy/utils/log_utils.py
  DP/como_policy/utils/lr_scheduler.py
  DP/como_policy/utils/train_utils.py
  DP/como_policy/utils/video_utils.py
  DP/como_policy/utils/visualization_utils.py
  DP/conf/train_policy/libero_rdt.yaml
  DP/engine/__init__.py
  DP/engine/eval_policy.py
  DP/engine/train_policy.py
  DP/engine/utils.py
  DP/libero/__init__.py
  DP/libero/benchmark/__init__.py
  DP/libero/benchmark/libero_suite_task_map.py
  DP/libero/benchmark/mu_creation.py
  DP/libero/envs/__init__.py
  DP/libero/envs/arenas/__init__.py
  DP/libero/envs/arenas/coffee_table_arena.py
  DP/libero/envs/arenas/empty_arena.py
  DP/libero/envs/arenas/kitchen_arena.py
  DP/libero/envs/arenas/living_room_arena.py
  DP/libero/envs/arenas/study_arena.py
  DP/libero/envs/arenas/style.py
  DP/libero/envs/arenas/table_arena.py
  DP/libero/envs/base_object.py
  DP/libero/envs/bddl_base_domain.py
  DP/libero/envs/bddl_utils.py
  DP/libero/envs/env_wrapper.py
  DP/libero/envs/object_states/__init__.py
  DP/libero/envs/object_states/base_object_states.py
  DP/libero/envs/objects/__init__.py
  DP/libero/envs/objects/articulated_objects.py
  DP/libero/envs/objects/google_scanned_objects.py
  DP/libero/envs/objects/hope_objects.py
  DP/libero/envs/objects/site_object.py
  DP/libero/envs/objects/target_zones.py
  DP/libero/envs/objects/turbosquid_objects.py
  DP/libero/envs/predicates/__init__.py
  DP/libero/envs/predicates/base_predicates.py
  DP/libero/envs/problems/__init__.py
  DP/libero/envs/problems/libero_coffee_table_manipulation.py
  DP/libero/envs/problems/libero_floor_manipulation.py
  DP/libero/envs/problems/libero_kitchen_tabletop_manipulation.py
  DP/libero/envs/problems/libero_living_room_t
... [truncated 1119 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ncurl -s --max-time 45 \"https://api.github.com/repos/MCG-NJU/CoMo/git/trees/master?recursive=1\" | python3 -c \"\nimport sys,json\nd=json.load(sys.stdin)\nps=[t['path'] for t in d.get('tree',[]) if t['type']=='blob']\nps=[p for p in ps if 'libero/' not in p and 'RoboticsDiffusionTransformer' not in p and (p.endswith('.py') or p.endswith('.md'))]\nfor p in ps: print(p)\n\" | grep -viE '^DP/' | head -60", "description": "Find CoMo latent motion model source"}
```

<details><summary>tool result</summary>

```
README.md
idm/CoMo/latent_motion_tokenizer/src/models/latent_motion_decoder.py
idm/CoMo/latent_motion_tokenizer/src/models/latent_motion_tokenizer.py
idm/CoMo/latent_motion_tokenizer/src/models/m_former.py
idm/CoMo/latent_motion_tokenizer/src/models/modeling_lmd_vit.py
idm/CoMo/latent_motion_tokenizer/src/models/vector_quantizer.py
idm/__init__.py
idm/idm_model/__init__.py
idm/idm_model/data_libero.py
idm/idm_model/data_real.py
idm/idm_model/idm_trainer3.py
idm/idm_model/io_utils.py
idm/idm_model/optimizer.py
idm/idm_model/utils_ceph.py
idm/io_utils.py
idm/process.py
idm/train_como.py
idm/utils_ceph.py
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ncurl -s --max-time 45 \"https://raw.githubusercontent.com/MCG-NJU/CoMo/master/idm/CoMo/latent_motion_tokenizer/src/models/latent_motion_tokenizer.py\" | head -160", "description": "Read CoMo tokenizer to find Td mechanism"}
```

<details><summary>tool result</summary>

```
"""The IDM."""

import torch
import torch.nn.functional as F
from torch import nn
import lpips
from einops import rearrange
from transformers import ViTMAEModel
from PIL import Image
from torchvision import transforms as T
import time
from collections import OrderedDict

class LatentMotionTokenizer(nn.Module):
    def __init__(
            self,
            image_encoder,
            m_former,
            vector_quantizer,
            decoder,
            codebook_dim=32,
            commit_loss_w=1.,
            recon_loss_w=1.,
            perceptual_loss_w=1.,
            contrastive_loss_w=0.004,  # contrastive loss wieight 
            temperature=1.0            # infonce temperature
    ):
        super().__init__()

        codebook_embed_dim = codebook_dim
        decoder_hidden_size = decoder.config.hidden_size

        if isinstance(image_encoder, ViTMAEModel):
            image_encoder.config.mask_ratio = 0.0

        self.image_encoder = image_encoder.requires_grad_(False).eval()
        self.m_former = m_former
        
        #The vq is not using in como
        self.vector_quantizer = vector_quantizer
        
        self.vq_down_resampler = nn.Sequential(
            nn.Linear(decoder_hidden_size, decoder_hidden_size),
            nn.Tanh(),
            nn.Linear(decoder_hidden_size, codebook_embed_dim)
        )
        self.vq_up_resampler = nn.Sequential(
            nn.Linear(codebook_embed_dim, codebook_embed_dim),
            nn.Tanh(),
            nn.Linear(codebook_embed_dim, decoder_hidden_size)
        )
        self.decoder = decoder
        self.commit_loss_w = commit_loss_w
        self.recon_loss_w = recon_loss_w
        self.perceptual_loss_w = perceptual_loss_w
        self.contrastive_loss_w = contrastive_loss_w
        self.temperature = temperature

        self.loss_fn_lpips = lpips.LPIPS(net='vgg').requires_grad_(False).eval()

    @property
    def device(self):
        return next(self.parameters()).device

    def get_state_
... [truncated 4233 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ncurl -s --max-time 45 \"https://raw.githubusercontent.com/MCG-NJU/CoMo/master/idm/CoMo/latent_motion_tokenizer/src/models/latent_motion_tokenizer.py\" | sed -n '160,260p'", "description": "Read CoMo forward continuation"}
```

<details><summary>tool result</summary>

```
        query_num = self.m_former.query_num

        latent_motion_tokens = [REDACTED]
            cond_hidden_states=cond_hidden_states,
            target_hidden_states=motion_td_features).last_hidden_state[:, :query_num]
        
        
        # Apply downsampling after extraction
        latent_motion_tokens_down = [REDACTED])

        if tests:
            return latent_motion_tokens_down


        indices = 0
        commit_loss = torch.tensor(0.0, device=latent_motion_tokens_down.device, requires_grad=True)
        
        ####################### vq baseline
        #quant, indices, commit_loss = self.vector_quantizer(latent_motion_tokens_down)

        if return_motion_token_ids_only:
            [REDACTED] indices  # (bs, motion_query_num)

        
        # Apply an additional upsampling step.
        ######################## vq baseline
        #latent_motion_tokens_up = [REDACTED])
        latent_motion_tokens_up = [REDACTED])


        # The decoder uses this motion representation to reconstruct low-level details of the future frame.
        recons_pixel_values = self.decoder(
            cond_input=img_1,
            latent_motion_tokens=[REDACTED]
        )
        
        if return_recons_only:
            return {
                "recons_pixel_values": recons_pixel_values,
                "indices": indices
            }

        # Compute reconstruction loss
        outputs = {
            "loss": torch.zeros_like(commit_loss),
            "commit_loss": commit_loss,
            "recons_loss": torch.zeros_like(commit_loss),
            "perceptual_loss": torch.zeros_like(commit_loss)
        }

        recons_loss = F.mse_loss(img_2, recons_pixel_values)
        outputs["recons_loss"] = recons_loss

        if self.perceptual_loss_w > 0:
            with torch.no_grad():
                perceptual_loss = self.loss_fn_lpips.forward(
              
... [truncated 2132 chars]
```
</details>

### ASSISTANT — 2026-09-16 19:33

This is a near-exact architectural match to WoG's QFormer. Let me check m_former structure and continue searching in parallel.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ncurl -s --max-time 45 \"https://raw.githubusercontent.com/MCG-NJU/CoMo/master/idm/CoMo/latent_motion_tokenizer/src/models/m_former.py\" | head -120", "description": "Read CoMo m_former"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'abs:\"orthogonal\" AND abs:\"projection\" AND abs:\"remove\" AND abs:\"component\" AND cat:cs.CL' 15 2>&1|tail -40; echo \"####\"; python3 /tmp/wog/axs.py search 'abs:\"concept erasure\" OR abs:\"linear concept erasure\"' 15 2>&1|tail -50", "description": "Search concept erasure / orthogonal projection"}
```

<details><summary>tool result</summary>

```
from transformers.models.vit.modeling_vit import (
    ViTConfig,
    ViTPreTrainedModel,
    ViTEncoder
)
from torch import nn
import torch
from typing import Optional, Dict, List, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPooling


class MFormerEmbeddings(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        query_num = config.query_num
        self.query_num = query_num
        self.latent_motion_token = [REDACTED], query_num, config.hidden_size))
        self.sep_token = [REDACTED], 1, config.hidden_size))
        self.projection = nn.Linear(config.input_hidden_size, config.hidden_size, bias=True)
        self.position_embeddings = nn.Parameter(torch.randn(1, config.num_patches*2 + 1 + query_num, config.hidden_size))
        self.token_type_embeddings = [REDACTED], config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(
        self,
        cond_hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, per_seq_length = cond_hidden_states.shape[:2]

        cond_embeddings = self.projection(cond_hidden_states)

        latent_motion_tokens = [REDACTED], -1, -1)
        sep_tokens = [REDACTED], -1, -1)
        cond_embeddings = torch.cat((latent_motion_tokens, cond_embeddings, sep_tokens), dim=1)

        target_embeddings = self.projection(target_hidden_states)
        embeddings = torch.cat((cond_embeddings, target_embeddings), dim=1)

        # add positional encoding to each token
        embeddings = embeddings + self.position_embeddings

        # add token type encoding to each token
        cond_token_type_embeddings = [REDACTED], per_seq_length + self.query_num + 1, -1)
... [truncated 3383 chars]
```
</details>

<details><summary>tool result</summary>

```
2109.04727v1 | 2021-09-10 | A Simple and Effective Method To Eliminate the Self Language Bias in Multilingual Representations
    COMMENT: Accepted to the 2021 Conference on Empirical Methods in Natural Language Processing
    ABS: Language agnostic and semantic-language information isolation is an emerging research direction for multilingual representations models. We explore this problem from a novel angle of geometric algebra and semantic space. A simple but highly effective method "Language Information Removal (LIR)" facto

[1 results for: abs:"orthogonal" AND abs:"projection" AND abs:"remove" AND abs:"component" AND cat:cs.CL]
####

2510.27285v4 | 2025-10-31 | Rethinking Robust Adversarial Concept Erasure in Diffusion Models
    COMMENT: -
    ABS: Concept erasure methods aim to remove specific unsafe target concepts in diffusion models while preserving image generation utility. To address the vulnerability that erased concepts can be easily recovered under adversarial attacks, adversarial concept erasure methods integrate adversarial optimiza

2605.25574v1 | 2026-05-25 | Mosaic: Compositional Multi-Concept Erasure via Vector Field Blending
    COMMENT: -
    ABS: Concept erasure has emerged as a key research direction for ensuring safe and ethical image synthesis in Text-to-Image (T2I) models. While existing studies have explored concept erasure across multiple concepts, they typically assume only a single target concept per image, a limitation increasingly 

2502.14896v2 | 2025-02-17 | A Comprehensive Survey on Concept Erasure in Text-to-Image Diffusion Models
    COMMENT: -
    ABS: Text-to-Image (T2I) models have made remarkable progress in generating high-quality, diverse visual content from natural language prompts. However, their ability to reproduce copyrighted styles, sensitive imagery, and harmful content raises significant ethical and legal concerns. Concept erasure off

2505.15450v3 | 2025-05-21 | Comprehensive Evaluation and Analysis for NSFW Conce
... [truncated 4165 chars]
```
</details>

### ASSISTANT — 2026-09-16 19:33

Found the key architectural match. Let me search the remaining categories and verify the "make latent independent of current frame" recipe.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'ti:\"Null It Out\" OR abs:\"iterative nullspace projection\"' 10 2>&1|tail -30; echo \"####\"; python3 /tmp/wog/axs.py search 'abs:\"linear guardedness\" OR (abs:\"concept erasure\" AND abs:\"least-squares\" AND abs:\"linear\")' 10 2>&1|tail -40", "description": "Search INLP and LEACE"}
```

<details><summary>tool result</summary>

```
2601.02400v1 | 2025-12-30 | Detecting and Mitigating Treatment Leakage in Text-Based Causal Inference: Distillation and Sensitivity Analysis
    COMMENT: -
    ABS: Text-based causal inference increasingly employs textual data as proxies for unobserved confounders, yet this approach introduces a previously undertheorized source of bias: treatment leakage. Treatment leakage occurs when text intended to capture confounding information also contains signals predic

2602.12659v1 | 2026-02-13 | IndicFairFace: Balanced Indian Face Dataset for Auditing and Mitigating Geographical Bias in Vision-Language Models
    COMMENT: -
    ABS: Vision-Language Models (VLMs) are known to inherit and amplify societal biases from their web-scale training data with Indian being particularly misrepresented. Existing fairness-aware datasets have significantly improved demographic balance across global race and gender groups, yet they continue to

2606.13720v2 | 2026-06-11 | Refusal Beyond a Single Direction: A Preliminary Comparison of Diff-in-Means and INLP
    COMMENT: -
    ABS: Arditi et al. (2024) has shown that refusal in safety fine-tuned chat models is mediated by a single linear direction in the residual stream, recoverable by a difference-in-means (DiM) of harmful and harmless activations. We compare DiM-based interventions (activation addition and directional ablati

2212.04273v1 | 2022-12-08 | Better Hit the Nail on the Head than Beat around the Bush: Removing Protected Attributes with a Single Projection
    COMMENT: EMNLP 2022
    ABS: Bias elimination and recent probing studies attempt to remove specific information from embedding spaces. Here it is important to remove as much of the target information as possible, while preserving any other information present. INLP is a popular recent method which removes specific information t

2510.22084v1 | 2025-10-24 | Compositional Bias Control in Large Language Models: Preference Learning Fails, Supervision Succeeds
    COMMENT: 20 p
... [truncated 2727 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nfor id in 2605.12167 2601.15197 2607.11397; do\necho \"=========== $id ===========\"\ncurl -sL --max-time 45 \"https://arxiv.org/abs/$id\" | python3 -c \"\nimport sys,re,html\nt=sys.stdin.read()\ndef g(p,d='(none)'):\n    m=re.search(p,t,re.S); return html.unescape(re.sub(r'\\s+',' ',re.sub(r'<[^>]+>','',m.group(1)))).strip() if m else d\nprint('TITLE:',g(r'<meta name=\\\"citation_title\\\" content=\\\"(.*?)\\\"'))\nprint('SUBMIT:',g(r'\\[Submitted on (.*?)\\]')[:80])\nprint('COMMENTS:',g(r'<td class=\\\"tablecell comments[^\\\"]*\\\">(.*?)</td>')[:250])\nm=re.search(r'<blockquote class=\\\"abstract mathjax\\\">(.*?)</blockquote>',t,re.S)\nprint('ABSTRACT:',html.unescape(re.sub(r'\\s+',' ',re.sub(r'<[^>]+>','',m.group(1)))).strip()[:1400] if m else 'NOABS')\nprint('LINKS:', sorted(set(re.findall(r'https?://(?:github\\.com|[a-z0-9-]+\\.github\\.io)[^\\\"<>\\s]*', t)))[:5])\n\"\ndone", "description": "Check MoLA, LangForce, WALA"}
```

<details><summary>tool result</summary>

```
=========== 2605.12167 ===========
TITLE: From Imagined Futures to Executable Actions: Mixture of Latent Actions for Robot Manipulation
SUBMIT: 12 May 2026
COMMENTS: ICML 2026
ABSTRACT: Abstract:Video generation models offer a promising imagination mechanism for robot manipulation by predicting long-horizon future observations, but effectively exploiting these imagined futures for action execution remains challenging. Existing approaches either condition policies on predicted frames or directly decode generated videos into actions, both suffering from a mismatch between visual realism and control relevance. As a result, predicted observations emphasize perceptual fidelity rather than action-centric causes of state transitions, leading to indirect and unstable control. To address this gap, we propose MoLA (Mixture of Latent Actions), a control-oriented interface that transforms imagined future videos into executable representations. Instead of passing predicted frames directly to the policy, MoLA leverages a mixture of pretrained inverse dynamics models to infer a mixture of latent actions implied by generated visual transitions. These modality-aware inverse dynamics models capture complementary semantic, depth, and flow cues, providing a structured and physically grounded action representation that bridges video imagination and policy execution. We evaluate our approach on simulated benchmarks (LIBERO, CALVIN, and LIBERO-Plus) and real-world robot manipulation tasks, achieving consistent gains in task success, temporal consistency, and generalization.
LINKS: []
=========== 2601.15197 ===========
TITLE: LangForce: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries
SUBMIT: 21 Jan 2026 (v1), last revised 29 May 2026 (this version, v7)
COMMENTS: ICML 2026
ABSTRACT: Abstract:Vision-Language-Action (VLA) models have shown promise in robot manipulation but often struggle to generalize to new instructions or complex multi-task scenarios. We id
... [truncated 2901 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py abs 2306.03819 2004.07667 2405.01527 2401.00025 2>&1", "description": "Verify LEACE, INLP, Track2Act, ATM"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'abs:\"DINO\" AND abs:\"feature\" AND abs:\"difference\" AND cat:cs.RO' 20 2>&1|tail -60", "description": "Search DINO feature difference robotics"}
```

<details><summary>tool result</summary>

```
=== 2306.03819 ===
TITLE : LEACE: Perfect linear concept erasure in closed form
SUBMIT: 6 Jun 2023 (v1), last revised 3 Apr 2025 (this version, v4) | citdate: 2023/06/06
COMM  : (none)
SUBJ  : Machine Learning (cs.LG); Computation and Language (cs.CL); Computers and Society (cs.CY)
DOI   : (none)

=== 2004.07667 ===
TITLE : Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection
SUBMIT: 16 Apr 2020 (v1), last revised 28 Apr 2020 (this version, v2 | citdate: 2020/04/16
COMM  : (none)
SUBJ  : Computation and Language (cs.CL); Machine Learning (cs.LG)
DOI   : (none)

=== 2405.01527 ===
TITLE : Track2Act: Predicting Point Tracks from Internet Videos enables Generalizable Robot Manipulation
SUBMIT: 2 May 2024 (v1), last revised 8 Aug 2024 (this version, v2) | citdate: 2024/05/02
COMM  : (none)
SUBJ  : Robotics (cs.RO); Computer Vision and Pattern Recognition (cs.CV)
DOI   : (none)

=== 2401.00025 ===
TITLE : Any-point Trajectory Modeling for Policy Learning
SUBMIT: 28 Dec 2023 (v1), last revised 12 Jul 2024 (this version, v3 | citdate: 2023/12/28
COMM  : (none)
SUBJ  : Robotics (cs.RO); Computer Vision and Pattern Recognition (cs.CV)
DOI   : (none)
```
</details>

<details><summary>tool result</summary>

```
2608.14650v1 | 2026-07-30 | Paired Exact-Reset Evaluation of a Prediction-Derived Medium-to-Full World-Model Cascade
    COMMENT: 20 pages, 6 figures, 6 tables
    ABS: Existing adaptive-inference and world-action-model systems use cheap-stage outputs or predicted futures to allocate additional computation. We study a narrower question: under paired exact-reset physical outcomes, can a Medium-derived interface predict when switching to a separately frozen Full pred

2510.03104v1 | 2025-10-03 | Geometry Meets Vision: Revisiting Pretrained Semantics in Distilled Fields
    COMMENT: -
    ABS: Semantic distillation in radiance fields has spurred significant advances in open-vocabulary robot policies, e.g., in manipulation and navigation, founded on pretrained semantics from large vision models. While prior work has demonstrated the effectiveness of visual-only semantic features (e.g., DIN

2505.11680v1 | 2025-05-16 | Grounded Task Axes: Zero-Shot Semantic Skill Generalization via Task-Axis Controllers and Visual Foundation Models
    COMMENT: -
    ABS: Transferring skills between different objects remains one of the core challenges of open-world robot manipulation. Generalization needs to take into account the high-level structural differences between distinct objects while still maintaining similar low-level interaction control. In this paper, we

2409.11941v1 | 2024-09-18 | GauTOAO: Gaussian-based Task-Oriented Affordance of Objects
    COMMENT: 6 pages
    ABS: When your robot grasps an object using dexterous hands or grippers, it should understand the Task-Oriented Affordances of the Object(TOAO), as different tasks often require attention to specific parts of the object. To address this challenge, we propose GauTOAO, a Gaussian-based framework for Task-O

2505.06111v3 | 2025-05-09 | UniVLA: Learning to Act Anywhere with Task-centric Latent Actions
    COMMENT: Accepted to RSS 2025. Code is available at https://github.com/OpenDriveLab/UniVLA
    ABS: A generalist 
... [truncated 1286 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nfor id in 2505.06111 2604.22152; do\necho \"=========== $id ===========\"\ncurl -sL --max-time 45 \"https://arxiv.org/abs/$id\" | python3 -c \"\nimport sys,re,html\nt=sys.stdin.read()\ndef g(p,d='(none)'):\n    m=re.search(p,t,re.S); return html.unescape(re.sub(r'\\s+',' ',re.sub(r'<[^>]+>','',m.group(1)))).strip() if m else d\nprint('TITLE:',g(r'<meta name=\\\"citation_title\\\" content=\\\"(.*?)\\\"'))\nprint('SUBMIT:',g(r'\\[Submitted on (.*?)\\]')[:80])\nprint('COMMENTS:',g(r'<td class=\\\"tablecell comments[^\\\"]*\\\">(.*?)</td>')[:250])\nm=re.search(r'<blockquote class=\\\"abstract mathjax\\\">(.*?)</blockquote>',t,re.S)\nprint('ABSTRACT:',html.unescape(re.sub(r'\\s+',' ',re.sub(r'<[^>]+>','',m.group(1)))).strip()[:1600] if m else 'NOABS')\nprint('LINKS:', sorted(set(re.findall(r'https?://(?:github\\.com|[a-z0-9-]+\\.github\\.io)[^\\\"<>\\s]*', t)))[:5])\n\"\ndone", "description": "Check UniVLA and dWorldEval details"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'abs:\"task-irrelevant dynamics\" OR (abs:\"decouple\" AND abs:\"latent action\" AND abs:\"task-centric\")' 15 2>&1|tail -40; echo \"####\"; python3 /tmp/wog/axs.py search 'abs:\"CoTracker\" AND cat:cs.RO' 15 2>&1|tail -45", "description": "Search task-irrelevant dynamics, CoTracker"}
```

<details><summary>tool result</summary>

```
=========== 2505.06111 ===========
TITLE: UniVLA: Learning to Act Anywhere with Task-centric Latent Actions
SUBMIT: 9 May 2025 (v1), last revised 3 Nov 2025 (this version, v3)
COMMENTS: Accepted to RSS 2025. Code is available at this https URL
ABSTRACT: Abstract:A generalist robot should perform effectively across various environments. However, most existing approaches heavily rely on scaling action-annotated data to enhance their capabilities. Consequently, they are often limited to single physical specification and struggle to learn transferable knowledge across different embodiments and environments. To confront these limitations, we propose UniVLA, a new framework for learning cross-embodiment vision-language-action (VLA) policies. Our key innovation is to derive task-centric action representations from videos with a latent action model. This enables us to exploit extensive data across a wide spectrum of embodiments and perspectives. To mitigate the effect of task-irrelevant dynamics, we incorporate language instructions and establish a latent action model within the DINO feature space. Learned from internet-scale videos, the generalist policy can be deployed to various robots through efficient latent action decoding. We obtain state-of-the-art results across multiple manipulation and navigation benchmarks, as well as real-robot deployments. UniVLA achieves superior performance over OpenVLA with less than 1/20 of pretraining compute and 1/10 of downstream data. Continuous performance improvements are observed as heterogeneous data, even including human videos, are incorporated into the training pipeline. The results underscore UniVLA's potential to facilitate scalable and efficient robot policy learning.
LINKS: ['https://github.com/OpenDriveLab/UniVLA']
=========== 2604.22152 ===========
TITLE: dWorldEval: Scalable Robotic Policy Evaluation via Discrete Diffusion World Model
SUBMIT: 24 Apr 2026
COMMENTS: (none)
ABSTRACT: Abstract:Evaluating robotics policies acr
... [truncated 1454 chars]
```
</details>

<details><summary>tool result</summary>

```
2505.06111v3 | 2025-05-09 | UniVLA: Learning to Act Anywhere with Task-centric Latent Actions
    COMMENT: Accepted to RSS 2025. Code is available at https://github.com/OpenDriveLab/UniVLA
    ABS: A generalist robot should perform effectively across various environments. However, most existing approaches heavily rely on scaling action-annotated data to enhance their capabilities. Consequently, they are often limited to single physical specification and struggle to learn transferable knowledge

[1 results for: abs:"task-irrelevant dynamics" OR (abs:"decouple" AND abs:"latent action" AND abs:"task-centric")]
####
2605.09513v1 | 2026-05-10 | QueST: Persistent Queries as Semantic Monitors for Drift Suppression in Long-Horizon Tracking
    COMMENT: -
    ABS: Tracking points in videos is typically formulated as frame-to-frame correspondence, where each point is matched locally to the next frame. While this works over short horizons, errors accumulate under articulation, occlusion, and viewpoint change, leading to silent semantic drift that existing track

[1 results for: abs:"CoTracker" AND cat:cs.RO]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\necho \"### Mantis github ###\"\ncurl -s --max-time 45 \"https://api.github.com/search/repositories?q=Mantis+disentangled+visual+foresight+VLA\" | python3 -c \"\nimport sys,json;d=json.load(sys.stdin)\nfor r in d.get('items',[])[:6]: print(r['full_name'],'|',r['description'])\n\"\necho \"### UniVLA repo ###\"\ncurl -s --max-time 45 \"https://api.github.com/repos/OpenDriveLab/UniVLA\" | python3 -c \"import sys,json;d=json.load(sys.stdin);print(d.get('full_name'),d.get('stargazers_count'),d.get('pushed_at'))\"\necho \"### ACT-LAM repo ###\"\ncurl -s --max-time 45 \"https://api.github.com/repos/DingjieFu/ACT-LAM\" | python3 -c \"import sys,json;d=json.load(sys.stdin);print(d.get('full_name'),d.get('stargazers_count'),d.get('pushed_at'),d.get('message',''))\"\necho \"### OpenCLAP ###\"\ncurl -s --max-time 45 \"https://api.github.com/repos/LinShan-Bin/OpenCLAP\" | python3 -c \"import sys,json;d=json.load(sys.stdin);print(d.get('full_name'),d.get('stargazers_count'),d.get('pushed_at'),d.get('message',''))\"", "description": "Verify code repos exist"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'abs:\"residual connection\" AND abs:\"current\" AND abs:\"future\" AND abs:\"latent action\"' 15 2>&1|tail -40; echo \"####\"; python3 /tmp/wog/axs.py search 'abs:\"action-relevant\" AND abs:\"bottleneck\" AND cat:cs.RO' 15 2>&1|tail -50", "description": "Search residual conditioning latent action"}
```

<details><summary>tool result</summary>

```
### Mantis github ###
### UniVLA repo ###
OpenDriveLab/UniVLA 1130 2025-11-19T03:31:25Z
### ACT-LAM repo ###
DingjieFu/ACT-LAM 1 2026-09-15T02:24:41Z 
### OpenCLAP ###
LinShan-Bin/OpenCLAP 42 2026-08-18T03:19:50Z
```
</details>

<details><summary>tool result</summary>

```
[0 results for: abs:"residual connection" AND abs:"current" AND abs:"future" AND abs:"latent action"]
####
2508.08743v1 | 2025-08-12 | Boosting Action-Information via a Variational Bottleneck on Unlabelled Robot Videos
    COMMENT: -
    ABS: Learning from demonstrations (LfD) typically relies on large amounts of action-labeled expert trajectories, which fundamentally constrains the scale of available training data. A promising alternative is to learn directly from unlabeled video demonstrations. However, we find that existing methods te

2605.05714v1 | 2026-05-07 | TriRelVLA: Triadic Relational Structure for Generalizable Embodied Manipulation
    COMMENT: -
    ABS: Vision-language-action (VLA) models perform well on training-seen robotic tasks but struggle to generalize to unseen scenes and objects. A key limitation lies in their implicit visual representations, which entangle object appearance, background, and scene layout. This makes policies sensitive to vi

2607.09185v1 | 2026-07-10 | Causally Debiased Latent Action Model for Embodied Action Conditioned World Models
    COMMENT: -
    ABS: Action-conditioned world models (ACWMs) aim to simulate future observations conditioned on embodied actions, offering a promising foundation for robot planning, policy evaluation, and data augmentation. However, learning controllable ACWMs requires large-scale action-labeled data, which remains cost

[3 results for: abs:"action-relevant" AND abs:"bottleneck" AND cat:cs.RO]
```
</details>

### ASSISTANT — 2026-09-16 19:35

That last one is highly relevant. Let me check it and a few final searches.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nfor id in 2607.09185 2607.11397; do\necho \"=========== $id ===========\"\ncurl -sL --max-time 45 \"https://arxiv.org/abs/$id\" | python3 -c \"\nimport sys,re,html\nt=sys.stdin.read()\ndef g(p,d='(none)'):\n    m=re.search(p,t,re.S); return html.unescape(re.sub(r'\\s+',' ',re.sub(r'<[^>]+>','',m.group(1)))).strip() if m else d\nprint('TITLE:',g(r'<meta name=\\\"citation_title\\\" content=\\\"(.*?)\\\"'))\nprint('SUBMIT:',g(r'\\[Submitted on (.*?)\\]')[:80])\nprint('COMMENTS:',g(r'<td class=\\\"tablecell comments[^\\\"]*\\\">(.*?)</td>')[:250])\nm=re.search(r'<blockquote class=\\\"abstract mathjax\\\">(.*?)</blockquote>',t,re.S)\nprint('ABSTRACT:',html.unescape(re.sub(r'\\s+',' ',re.sub(r'<[^>]+>','',m.group(1)))).strip()[:1800] if m else 'NOABS')\nprint('LINKS:', sorted(set(re.findall(r'https?://(?:github\\.com|[a-z0-9-]+\\.github\\.io)[^\\\"<>\\s]*', t)))[:5])\n\"\ndone", "description": "Check causally debiased LAM"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'abs:\"minimal sufficient\" AND abs:\"action\" AND abs:\"representation\" AND cat:cs.RO' 15 2>&1|tail -40; echo \"####\"; python3 /tmp/wog/axs.py search 'abs:\"delta\" AND abs:\"DINOv2\" AND (cat:cs.RO OR cat:cs.CV)' 15 2>&1|tail -40", "description": "Search minimal sufficient, DINOv2 delta"}
```

<details><summary>tool result</summary>

```
=========== 2607.09185 ===========
TITLE: Causally Debiased Latent Action Model for Embodied Action Conditioned World Models
SUBMIT: 10 Jul 2026
COMMENTS: (none)
ABSTRACT: Abstract:Action-conditioned world models (ACWMs) aim to simulate future observations conditioned on embodied actions, offering a promising foundation for robot planning, policy evaluation, and data augmentation. However, learning controllable ACWMs requires large-scale action-labeled data, which remains costly to collect in the real world. Latent action models (LAMs) mitigate this bottleneck by inferring latent actions from unlabeled videos, but existing LAMs are typically trained with reconstruction-only objectives and therefore entangle action-relevant dynamics with action-irrelevant visual factors such as backgrounds and untouched objects. In this work, we identify this action-irrelevant bias as a key obstacle to controllable ACWMs and introduce evaluation metrics to measure latent-action bias, action following, and robustness. We propose CD-LAM, a causally debiased framework for LAM-based ACWMs. CD-LAM introduces three efficient fine-tuning objectives: embodiment-centric reconstruction, action-centric contrastive learning, and latent space calibration, which together encourage embodiment-focused, action-aware, and calibrated non-collapsed latent action representations. Experiments on 2B and 14B ACWM backbones show that CD-LAM substantially improves latent-action controllability, downstream robot-action following, visual fidelity, and adaptation efficiency, requiring only 6k fine-tuning steps and more than 12$\times$ fewer robot-action adaptation updates than the baseline.
LINKS: []
=========== 2607.11397 ===========
TITLE: WALA Learning Executable Latent Actions from Action-Labeled Demonstrations and Action-Free Videos
SUBMIT: 13 Jul 2026
COMMENTS: Project page: this https URL
ABSTRACT: Abstract:Generalizable robot policies typically rely on action-labeled robot demonstrations, which are expen
... [truncated 1494 chars]
```
</details>

<details><summary>tool result</summary>

```
[0 results for: abs:"minimal sufficient" AND abs:"action" AND abs:"representation" AND cat:cs.RO]
####
2606.29173v2 | 2026-06-28 | TacGen: Touch Is a Necessary Dimension of Physical-World Representation -- Addressing Tactile Data Scarcity with Scalable Vision-to-Touch Alignment and Generation
    COMMENT: 49 pages, 29 figures
    ABS: Touch resolves the physical-property ambiguity left by vision: exploratory contact recovers shape, texture, compliance, and material, and visuo-haptic object representations converge in ventral visual cortex. We ask whether representation learning can reproduce this grounding. TacGen mitigates the t

2601.06067v1 | 2025-12-29 | HyperTopo-Adapters: Geometry- and Topology-Aware Segmentation of Leaf Lesions on Frozen Encoders
    COMMENT: 13 pages, 8 figures. Code available at https://github.com/ChimdiWalter/HyperTopo-Adapters
    ABS: Leaf-lesion segmentation is topology-sensitive: small merges, splits, or false holes can be biologically meaningful descriptors of biochemical pathways, yet they are weakly penalized by standard pixel-wise losses in Euclidean latents. I explore HyperTopo-Adapters, a lightweight, parameter-efficient 

[2 results for: abs:"delta" AND abs:"DINOv2" AND (cat:cs.RO OR cat:cs.CV)]
```
</details>

### ASSISTANT — 2026-09-16 19:35

I have strong coverage. Let me do a final round on a few gaps: Mantis code, VLA-JEPA venue update, and point-track conditioning as O^c replacement.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\necho \"### Mantis code (arXiv says released) ###\"\ncurl -sL --max-time 45 \"https://arxiv.org/abs/2511.16175v2\" | grep -oiE 'https?://[a-z0-9./_-]*(github|huggingface)[a-z0-9./_-]*' | sort -u | head\ncurl -s --max-time 45 \"https://api.github.com/search/repositories?q=Mantis+VLA+visual+foresight+in:name,description\" | python3 -c \"\nimport sys,json;d=json.load(sys.stdin)\nfor r in d.get('items',[])[:8]: print(r['full_name'],'|',r['description'])\n\"\necho \"### search github Mantis-VLA ###\"\ncurl -s --max-time 45 \"https://api.github.com/search/repositories?q=Mantis+vision-language-action\" | python3 -c \"\nimport sys,json;d=json.load(sys.stdin)\nfor r in d.get('items',[])[:8]: print(r['full_name'],'|',r['description'])\n\"", "description": "Find Mantis code repo"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'abs:\"track\" AND abs:\"condition\" AND abs:\"diffusion policy\" AND cat:cs.RO' 20 2>&1|tail -50", "description": "Search track-conditioned diffusion policy"}
```

<details><summary>tool result</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: bc5n7fgz2). Output is being written to: /tmp/claude-1001/-home-zhengguantian/c976c298-d71c-4d38-9107-f692d869d083/tasks/bc5n7fgz2.output. You will be notified when it completes. To check interim output, use Read on that file path.
```
</details>

<details><summary>tool result</summary>

```
2604.03404v1 | 2026-04-03 | Diffusion Policy with Bayesian Expert Selection for Active Multi-Target Tracking
    COMMENT: -
    ABS: Active multi-target tracking requires a mobile robot to balance exploration for undetected targets with exploitation of uncertain tracked ones. Diffusion policies have emerged as a powerful approach for capturing diverse behavioral strategies by learning action sequences from expert demonstrations. 

2510.09786v1 | 2025-10-10 | Enhancing Diffusion Policy with Classifier-Free Guidance for Temporal Robotic Tasks
    COMMENT: 7 pages, 7 figures
    ABS: Temporal sequential tasks challenge humanoid robots, as existing Diffusion Policy (DP) and Action Chunking with Transformers (ACT) methods often lack temporal context, resulting in local optima traps and excessive repetitive actions. To address these issues, this paper introduces a Classifier-Free G

2504.20359v3 | 2025-04-29 | PRISM-DP: Spatial Pose-based Observations for Diffusion-Policies via Segmentation, Mesh Generation, and Pose Tracking
    COMMENT: -
    ABS: Diffusion policies generate robot motions by learning to denoise action-space trajectories conditioned on observations. These observations are commonly streams of RGB images, whose high dimensionality includes substantial task-irrelevant information, requiring large models to extract relevant patter

2408.04198v1 | 2024-08-08 | F1tenth Autonomous Racing With Offline Reinforcement Learning Methods
    COMMENT: -
    ABS: Autonomous racing serves as a critical platform for evaluating automated driving systems and enhancing vehicle mobility intelligence. This work investigates offline reinforcement learning methods to train agents within the dynamic F1tenth racing environment. The study begins by exploring the challen

2603.09574v1 | 2026-03-10 | SCDP: Learning Humanoid Locomotion from Partial Observations via Mixed-Observation Distillation
    COMMENT: 6 pages, 8 figures, 5 tables, iRos
    ABS: Distilling humanoid locomotion co
... [truncated 2173 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ncurl -sL --max-time 40 \"https://arxiv.org/abs/2511.16175\" 2>/dev/null | grep -oiE 'https?://[a-z0-9./_-]*(github\\.io|github\\.com)[a-z0-9./_-]*' | sort -u | head -5\necho \"--- api search ---\"\ncurl -s --max-time 40 \"https://api.github.com/search/repositories?q=Mantis+DVF+robot\" 2>/dev/null | python3 -c \"\nimport sys,json\ntry:\n  d=json.load(sys.stdin)\n  for r in d.get('items',[])[:6]: print(r['full_name'],'|',r.get('description'))\nexcept Exception as e: print('err',e)\n\"", "description": "Retry Mantis code lookup", "timeout": 110000}
```

<details><summary>tool result</summary>

```
--- api search ---
```
</details>

**tool call `Read`**
```
{"file_path": "/tmp/claude-1001/-home-zhengguantian/c976c298-d71c-4d38-9107-f692d869d083/tasks/bc5n7fgz2.output"}
```

<details><summary>tool result</summary>

```
1	### Mantis code (arXiv says released) ###
2	https://huggingface.co/docs/hub/spaces
3	https://huggingface.co/huggingface
4	### search github Mantis-VLA ###
5	SJTU-DENG-Lab/Mantis | [CVPR 2026] Mantis: A Versatile Vision-Language-Action Model with Disentangled Visual Foresight
6	
7	[exited with code 0]
8	
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ncurl -s --max-time 40 \"https://api.github.com/repos/SJTU-DENG-Lab/Mantis\" | python3 -c \"import sys,json;d=json.load(sys.stdin);print(d.get('full_name'),'stars',d.get('stargazers_count'),'pushed',d.get('pushed_at'))\"\necho \"=== tree (py files) ===\"\ncurl -s --max-time 40 \"https://api.github.com/repos/SJTU-DENG-Lab/Mantis/git/trees/main?recursive=1\" | python3 -c \"\nimport sys,json\nd=json.load(sys.stdin)\nps=[t['path'] for t in d.get('tree',[]) if t['type']=='blob' and t['path'].endswith('.py')]\nprint(len(ps))\nfor p in ps[:40]:print(' ',p)\nprint(d.get('message',''))\n\"", "description": "Verify Mantis repo"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'abs:\"latent action\" AND abs:\"orthogonal\" ' 15 2>&1|tail -40", "description": "Search latent action orthogonal"}
```

<details><summary>tool result</summary>

```
SJTU-DENG-Lab/Mantis stars 94 pushed 2026-06-05T09:04:00Z
=== tree (py files) ===
33
  dataset.py
  experiments/__init__.py
  experiments/aloha/client.py
  experiments/aloha/mantis_vla_utils.py
  experiments/libero/libero_utils.py
  experiments/libero/mantis_vla_utils.py
  experiments/libero/run_libero_eval.py
  models/__init__.py
  models/action_model/__init__.py
  models/action_model/action_model.py
  models/action_model/diffusion_utils.py
  models/action_model/gaussian_diffusion.py
  models/action_model/models.py
  models/action_model/respace.py
  models/action_model/timestep_sampler.py
  models/mantis.py
  models/model.py
  models/transformer_encoder.py
  train.py
  trainer.py
  trainer_utils.py
  utils/convert_multimodal_to_json.py
  utils/droid_sample_image.py
  utils/generate_text.py
  utils/get_droid_success_episodes.py
  utils/get_norm_stats.py
  utils/multimodal_eval/realworldqa/answer_compare_realworldqa.py
  utils/multimodal_eval/realworldqa/convert_realworldqa.py
  utils/multimodal_eval/realworldqa/eval_realworldqa_mantis.py
  utils/multimodal_eval/realworldqa/eval_realworldqa_qwen.py
  utils/save_simplified_model.py
  utils/ssv2_sample_image.py
  utils/vid2img_sthv2.py
```
</details>

<details><summary>tool result</summary>

```
[0 results for: abs:"latent action" AND abs:"orthogonal" ]
```
</details>

### ASSISTANT — 2026-09-16 19:39

Mantis's DVF residual mechanism is exactly design variant (b). Let me confirm it in code.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ncurl -s --max-time 45 \"https://raw.githubusercontent.com/SJTU-DENG-Lab/Mantis/main/models/action_model/models.py\" | grep -n -iE 'residual|resid|cond|current|meta|\\+ *x|skip' | head -50", "description": "Look for residual connection in Mantis DiT"}
```

<details><summary>tool result</summary>

```
4:# Copyright (c) Meta Platforms, Inc. and affiliates.
25:#               Embedding Layers for Timesteps and conditions                 #
69:    Embeds conditions into vector representations. Also handles label dropout for classifier-free guidance.
71:    def __init__(self, in_size, hidden_size, dropout_prob=0.1, conditions_shape=(1, 1, 2048)):
76:            self.uncondition = nn.Parameter(torch.empty(conditions_shape[1:]))
78:    def token_drop(self, conditions, force_drop_ids=None):
80:        Drops conditions to enable classifier-free guidance.
83:            drop_ids = torch.rand(conditions.shape[0], device=conditions.device) < self.dropout_prob
86:        conditions = torch.where(drop_ids.unsqueeze(1).unsqueeze(1).expand(conditions.shape[0], *self.uncondition.shape), self.uncondition, conditions)
87:        return conditions
90:    def forward(self, conditions, train, force_drop_ids=None):
93:            conditions = self.token_drop(conditions, force_drop_ids)
94:        embeddings = self.linear(conditions)
125:    A DiT block with self-attention conditioning.
196:        # +2, one for the conditional token, and one for the current action prediction
225:            nn.init.normal_(self.z_embedder.uncondition, std=0.02)
242:        z: (N, 1, D) tensor of conditions
260:        Forward pass of Diffusion, but also batches the unconditional forward pass for classifier-free guidance.
269:        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
270:        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ncurl -s --max-time 45 \"https://raw.githubusercontent.com/SJTU-DENG-Lab/Mantis/main/models/mantis.py\" | grep -n -iE 'residual|current_|meta_quer|foresight|dvf|next_state|\\bvisual_' | head -60", "description": "Find DVF residual in mantis.py"}
```

<details><summary>tool result</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: bxby6tu3y). Output is being written to: /tmp/claude-1001/-home-zhengguantian/c976c298-d71c-4d38-9107-f692d869d083/tasks/bxby6tu3y.output. You will be notified when it completes. To check interim output, use Read on that file path.
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; python3 /tmp/wog/axs.py search 'abs:\"latent action\" AND abs:\"identifiability\"' 15 2>&1|tail -35", "description": "Search latent action identifiability"}
```

<details><summary>tool result</summary>

```
Command did not complete within its 120s timeout and was moved to the background (ID: bl154h31n). Output is being written to: /tmp/claude-1001/-home-zhengguantian/c976c298-d71c-4d38-9107-f692d869d083/tasks/bl154h31n.output. You will be notified when it completes. To check interim output, use Read on that file path.
```
</details>

**tool call `Read`**
```
{"file_path": "/tmp/claude-1001/-home-zhengguantian/c976c298-d71c-4d38-9107-f692d869d083/tasks/bxby6tu3y.output"}
```

<details><summary>tool result</summary>

```
<system-reminder>Warning: the file exists but the contents are empty.</system-reminder>
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ntimeout 60 curl -s --max-time 50 \"https://raw.githubusercontent.com/SJTU-DENG-Lab/Mantis/main/models/mantis.py\" -o /tmp/mantis.py; wc -l /tmp/mantis.py 2>/dev/null; grep -n -iE 'residual|current_|meta_quer|foresight|dvf|next_state' /tmp/mantis.py | head -40", "description": "Download and grep mantis.py", "timeout": 90000}
```

<details><summary>tool result</summary>

```
606 /tmp/mantis.py
```
</details>

**tool call `Bash`**
```
{"command": "grep -n -iE 'skip|shortcut|\\bcat\\(|clean_|latent_|vision_token|img_emb|query' /tmp/mantis.py | head -50", "description": "Grep mantis.py structure"}
```

<details><summary>tool result</summary>

```
349:        latent_size = self.config.input_size
350:        latent_channels = self.config.in_channels
355:                latent_channels,
356:                latent_size,
357:                latent_size,
416:            latent_model_input = torch.cat([latents] * (len(input_ids) // len(latents)))
417:            latent_model_input = latent_model_input.to(prompt_embeds.dtype)
419:                latent_model_input = self.scheduler.scale_model_input(
420:                    latent_model_input, t
425:                x=latent_model_input,
427:                .expand(latent_model_input.shape[0])
547:            noise = torch.cat([noise, noise], 0)
551:            z = torch.cat([cognition_features, uncondition], 0)
```
</details>

**tool call `Read`**
```
{"file_path": "/tmp/claude-1001/-home-zhengguantian/c976c298-d71c-4d38-9107-f692d869d083/tasks/bl154h31n.output"}
```

<details><summary>tool result</summary>

```
1	    ABS: Latent action models (LAMs) learn action representations from unlabeled videos by inferring latent actions from visual transitions and reconstructing future states. However, we identify a fundamental $\textbf{reconstruction-action mismatch}$: lower reconstruction error does not necessarily yield bet
2	
3	2602.24275v1 | 2026-02-27 | Hierarchical Action Learning for Weakly-Supervised Action Segmentation
4	    COMMENT: -
5	    ABS: Humans perceive actions through key transitions that structure actions across multiple abstraction levels, whereas machines, relying on visual features, tend to over-segment. This highlights the difficulty of enabling hierarchical reasoning in video understanding. Interestingly, we observe that lowe
6	
7	2509.26251v2 | 2025-09-30 | Seeing Space and Motion: Enhancing Latent Actions with Geometric and Dynamic Awareness for Vision-Language-Action Models
8	    COMMENT: 8 pages, correct errors, clarify details
9	    ABS: Latent Action Models (LAMs) enable Vision- Language-Action (VLA) systems to learn semantic action representations from large-scale unannotated data. Yet, we identify two bottlenecks of LAMs: 1) the commonly adopted end-to-end trained image encoder suffers from poor spatial understanding; 2) LAMs can
10	
11	2604.04974v3 | 2026-04-04 | From Video to Control: A Survey of Learning Manipulation Interfaces from Temporal Visual Data
12	    COMMENT: -
13	    ABS: Video is a scalable observation of physical dynamics: it captures how objects move, how contact unfolds, and how scenes evolve under interaction -- all without requiring robot action labels. Yet translating this temporal structure into reliable robotic control remains an open challenge, because vide
14	
15	2605.16054v1 | 2026-05-15 | Ada-Diffuser: Latent-Aware Adaptive Diffusion for Decision-Making
16	    COMMENT: ICLR 2026
17	    ABS: Recent work has framed decision-making as a sequence modeling problem using generative models such as diffusion models. Although promi
... [truncated 2176 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ntimeout 100 python3 /tmp/wog/axs.py abs 2104.13712 2509.26251 2>&1\necho \"=== Mantis repo README head ===\"\ntimeout 60 curl -s --max-time 50 \"https://raw.githubusercontent.com/SJTU-DENG-Lab/Mantis/main/README.md\" | head -30", "description": "Verify Barlow-HSIC and Mantis README", "timeout": 180000}
```

<details><summary>tool result</summary>

```
=== 2104.13712 ===
TITLE : A Note on Connecting Barlow Twins with Negative-Sample-Free Contrastive Learning
SUBMIT: 28 Apr 2021 | citdate: 2021/04/28
COMM  : (none)
SUBJ  : Machine Learning (cs.LG)
DOI   : (none)

=== 2509.26251 ===
TITLE : Seeing Space and Motion: Enhancing Latent Actions with Geometric and Dynamic Awareness for Vision-Language-Action Models
SUBMIT: 30 Sep 2025 (v1), last revised 11 Mar 2026 (this version, v2 | citdate: 2025/09/30
COMM  : (none)
SUBJ  : Computer Vision and Pattern Recognition (cs.CV)
DOI   : (none)

=== Mantis repo README head ===

<p align="center">
    <img src="assets/mantis.png?raw=true" width="120" style="margin-bottom: 0.1;"/>
<p>

<h3 align="center">
  <a href="https://github.com/SJTU-DENG-Lab/Mantis" style="color:#567053"> [CVPR 2026] Mantis: A Versatile Vision-Language-Action Model<br>with Disentangled Visual Foresight</a>
</h3>

<h4 align="center"> 
  If you find our project helpful, please give us a star ⭐ to support us 🙏🙏
</h4>

<p align="center">
  <a href="https://arxiv.org/pdf/2511.16175"><b>📄 Paper</b></a> |
  <a href="https://huggingface.co/collections/Yysrc/mantis"><b>🤗 Checkpoints</b></a> |
  <a href="./LICENSE"><b>📜 License</b></a>
</p>

![head](assets/head.png)

- **Disentangled Visual Foresight** automatically capture the *latent actions* that delineate the visual trajectory without overburdening the backbone.
- **Progressive Training** introduces modalities in stages, preserving the language understanding and reasoning capabilities of the VLM backbone.
- **Adaptive Temporal Ensemble** dynamically adjusts temporal ensembling strength, reducing inference cost while maintaining stable control.


## 📘 Contents
- [Demos](#-demos)
- [Introduction](#-introduction)
- [Models & Datasets](#-models--datasets)
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ntimeout 110 python3 /tmp/wog/axs.py search 'abs:\"unpredictable\" AND abs:\"current\" AND abs:\"latent\"' 12 2>&1|tail -30", "description": "Search unpredictable-from-current latent", "timeout": 130000}
```

<details><summary>tool result</summary>

```
2601.15333v2 | 2026-01-20 | Empowering LLMs for Structure-Based Drug Design via Exploration-Augmented Latent Inference
    COMMENT: -
    ABS: Large Language Models (LLMs) possess strong representation and reasoning capabilities, but their application to structure-based drug design (SBDD) is limited by insufficient understanding of protein structures and unpredictable molecular generation. To address these challenges, we propose Exploratio

2505.06218v1 | 2025-05-09 | Let Humanoids Hike! Integrative Skill Development on Complex Trails
    COMMENT: CVPR 2025. Project page: https://lego-h-humanoidrobothiking.github.io/
    ABS: Hiking on complex trails demands balance, agility, and adaptive decision-making over unpredictable terrain. Current humanoid research remains fragmented and inadequate for hiking: locomotion focuses on motor skills without long-term goals or situational awareness, while semantic navigation overlooks

2601.07251v5 | 2026-01-12 | MeepleLM: A Virtual Playtester Simulating Diverse Subjective Experiences
    COMMENT: ACL 2026 MainConference
    ABS: Recent advancements have expanded the role of Large Language Models in board games from playing agents to creative co-designers. However, a critical gap remains: current systems lack the capacity to offer constructive critique grounded in the emergent user experience. Bridging this gap is fundamenta

2212.12735v1 | 2022-12-24 | An Adaptive Deep RL Method for Non-Stationary Environments with Piecewise Stable Context
    COMMENT: NeurIPS 2022
    ABS: One of the key challenges in deploying RL to real-world applications is to adapt to variations of unknown environment contexts, such as changing terrains in robotic tasks and fluctuated bandwidth in congestion control. Existing works on adaptation to unknown environment contexts either assume the co

2609.16589v1 | 2026-09-15 | Do LLMs Have Values? A Quantitative Analysis and Alignment Framework for Values in Large Language Models
    COMMENT: Preprint. 9 aut
... [truncated 387 chars]
```
</details>

### ASSISTANT — 2026-09-16 19:46

Let me do a final verification pass on the remaining IDs I plan to cite.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ntimeout 110 python3 /tmp/wog/axs.py abs 2509.23009 2211.12883 1409.7495 2>&1", "description": "Verify static-bias papers", "timeout": 130000}
```

<details><summary>tool result</summary>

```
=== 2509.23009 ===
TITLE : Disentangling Static and Dynamic Information for Reducing Static Bias in Action Recognition
SUBMIT: 27 Sep 2025 | citdate: 2025/09/27
COMM  : (none)
SUBJ  : Computer Vision and Pattern Recognition (cs.CV)
DOI   : (none)

=== 2211.12883 ===
TITLE : Mitigating and Evaluating Static Bias of Action Representations in the Background and the Foreground
SUBMIT: 23 Nov 2022 (v1), last revised 27 Sep 2023 (this version, v3 | citdate: 2022/11/23
COMM  : (none)
SUBJ  : Computer Vision and Pattern Recognition (cs.CV)
DOI   : (none)

=== 1409.7495 ===
TITLE : Unsupervised Domain Adaptation by Backpropagation
SUBMIT: 26 Sep 2014 (v1), last revised 27 Feb 2015 (this version, v2 | citdate: 2014/09/26
COMM  : (none)
SUBJ  : Machine Learning (stat.ML); Machine Learning (cs.LG); Neural and Evolutionary Computing (cs.NE)
DOI   : (none)
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nfor id in 2509.23009 2211.12883 2306.03819 2004.07667 2606.31232 2604.22152; do\necho -n \"$id : \"\ntimeout 50 curl -sL --max-time 45 \"https://arxiv.org/abs/$id\" | python3 -c \"\nimport sys,re,html\nt=sys.stdin.read()\nm=re.search(r'<td class=\\\"tablecell comments[^\\\"]*\\\">(.*?)</td>',t,re.S)\nprint(html.unescape(re.sub(r'\\s+',' ',re.sub(r'<[^>]+>','',m.group(1)))).strip()[:200] if m else '(none)')\n\"\ndone", "description": "Recheck comments fields robustly", "timeout": 300000}
```

<details><summary>tool result</summary>

```
2509.23009 : in Proc. of ICCV2025 Workshop and Challenge on Disentangled Representation Learning for Controllable Generation (DRL4Real)
2211.12883 : (none)
2306.03819 : (none)
2004.07667 : Accepted as a long paper in ACL 2020
2606.31232 : (none)
2604.22152 : (none)
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\necho \"=== 2509.23009 full abs ===\"\ntimeout 50 curl -sL --max-time 45 \"https://arxiv.org/abs/2509.23009\" | python3 -c \"\nimport sys,re,html\nt=sys.stdin.read()\nm=re.search(r'<blockquote class=\\\"abstract mathjax\\\">(.*?)</blockquote>',t,re.S)\nprint(html.unescape(re.sub(r'\\s+',' ',re.sub(r'<[^>]+>','',m.group(1)))).strip() if m else 'NOABS')\nprint(sorted(set(re.findall(r'https?://(?:github\\.com|[a-z0-9-]+\\.github\\.io)[^\\\"<>\\s]*',t)))[:4])\n\"\necho \"=== 2211.12883 code ===\"\ntimeout 50 curl -sL --max-time 45 \"https://arxiv.org/abs/2211.12883\" | python3 -c \"\nimport sys,re,html\nt=sys.stdin.read()\nm=re.search(r'<blockquote class=\\\"abstract mathjax\\\">(.*?)</blockquote>',t,re.S)\nprint(html.unescape(re.sub(r'\\s+',' ',re.sub(r'<[^>]+>','',m.group(1)))).strip()[:1200] if m else 'NOABS')\nprint(sorted(set(re.findall(r'https?://(?:github\\.com|[a-z0-9-]+\\.github\\.io)[^\\\"<>\\s]*',t)))[:4])\n\"", "description": "Get static-bias method details and code", "timeout": 150000}
```

<details><summary>tool result</summary>

```
=== 2509.23009 full abs ===
Abstract:Action recognition models rely excessively on static cues rather than dynamic human motion, which is known as static bias. This bias leads to poor performance in real-world applications and zero-shot action recognition. In this paper, we propose a method to reduce static bias by separating temporal dynamic information from static scene information. Our approach uses a statistical independence loss between biased and unbiased streams, combined with a scene prediction loss. Our experiments demonstrate that this method effectively reduces static bias and confirm the importance of scene prediction loss.
[]
=== 2211.12883 code ===
Abstract:In video action recognition, shortcut static features can interfere with the learning of motion features, resulting in poor out-of-distribution (OOD) generalization. The video background is clearly a source of static bias, but the video foreground, such as the clothing of the actor, can also provide static bias. In this paper, we empirically verify the existence of foreground static bias by creating test videos with conflicting signals from the static and moving portions of the video. To tackle this issue, we propose a simple yet effective technique, StillMix, to learn robust action representations. Specifically, StillMix identifies bias-inducing video frames using a 2D reference network and mixes them with videos for training, serving as effective bias suppression even when we cannot explicitly extract the source of bias within each video frame or enumerate types of bias. Finally, to precisely evaluate static bias, we synthesize two new benchmarks, SCUBA for static cues in the background, and SCUFO for static cues in the foreground. With extensive experiments, we demonstrate that StillMix mitigates both types of static bias and improves video representations for downstrea
['https://github.com/lihaoxin05/StillMix', 'https://github.com/lihaoxin05/StillMix.']
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ntimeout 60 curl -s --max-time 50 \"https://raw.githubusercontent.com/SJTU-DENG-Lab/Mantis/main/models/model.py\" -o /tmp/mm.py; wc -l /tmp/mm.py; grep -n -iE 'residual|foresight|dvf|meta|\\+ *cond|current' /tmp/mm.py | head -40", "description": "Grep Mantis model.py for DVF residual", "timeout": 90000}
```

<details><summary>tool result</summary>

```
533 /tmp/mm.py
29:        num_metaqueries: int = 9,
41:        self.num_metaqueries = num_metaqueries
75:            + config.num_metaqueries + 2
92:        self.tokenizer.num_metaqueries = config.num_metaqueries
110:        if config.num_metaqueries > 0:
114:                    + [f"<img{i}>" for i in range(self.tokenizer.num_metaqueries)]
245:        if not add_generation_prompt or tokenizer.num_metaqueries <= 0:
250:                + "".join([f"<img{i}>" for i in range(tokenizer.num_metaqueries)])
259:                + "".join([f"<img{i}>" for i in range(tokenizer.num_metaqueries)])
457:        if self.tokenizer.num_metaqueries > 0:
496:                # Inference action without metaqueries
```
</details>

**tool call `Bash`**
```
{"command": "grep -n -iE 'clean_latent|noise|dit|foresight|cond_latent|torch.cat|img_latent|vae' /tmp/mantis.py | head -40; echo \"==== trainer ====\"; export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; timeout 60 curl -s --max-time 50 \"https://raw.githubusercontent.com/SJTU-DENG-Lab/Mantis/main/trainer.py\" -o /tmp/mt.py; grep -n -iE 'residual|foresight|clean|cond_input|\\+ *latent|next_' /tmp/mt.py|head -30", "description": "Find DVF residual connection code", "timeout": 90000}
```

<details><summary>tool result</summary>

```
29:        vae_id: str = "Efficient-Large-Model/Sana_600M_512px_diffusers",
32:        vae_downsample_f: int = 32,
33:        noise_scheduler_id: str = "Efficient-Large-Model/Sana_600M_512px_diffusers",
45:        self.vae_id = vae_id
48:        self.vae_downsample_f = vae_downsample_f
49:        self.noise_scheduler_id = noise_scheduler_id
66:        self.vae = AutoencoderDC.from_pretrained(config.vae_id, subfolder="vae", use_safetensors=True)
70:            self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
71:                config.noise_scheduler_id, subfolder="scheduler", use_safetensors=True
74:            self.noise_scheduler = DDPMScheduler.from_pretrained(
75:                config.noise_scheduler_id, subfolder="scheduler", use_safetensors=True
125:        sigmas = self.noise_scheduler.sigmas.to(device=device, dtype=dtype)
126:        schedule_timesteps = self.noise_scheduler.timesteps.to(device)
158:            latents = self.vae.encode(target).latent
161:                "shift_factor" in self.vae.config
162:                and self.vae.config.shift_factor is not None
164:                latents = latents - self.vae.config.shift_factor
165:            latents = latents * self.vae.config.scaling_factor
176:            noise = torch.randn_like(latents, device=latents.device)
186:            indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
187:            timesteps = self.noise_scheduler.timesteps[indices].to(
194:            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
195:            prompt_embeds, attention_mask = self.model.encode_condition(
208:            target = noise - latents
222:            action_prompt_embeds = self.model.encode_condition_action(
261:        latents = latents / self.vae.config.scaling_factor
263:            "shift_factor" in self.vae.config
264:            and self.vae.config.shift_factor is not None
266:            latents = latents + self.vae.config.shift_factor
267:      
... [truncated 658 chars]
```
</details>

**tool call `Bash`**
```
{"command": "grep -n -B3 -A12 'def forward' /tmp/mantis.py | head -60", "description": "Read mantis forward signature"}
```

<details><summary>tool result</summary>

```
138-    def get_tokenize_fn(self):
139-        return self.model.get_tokenize_fn()
140-
141:    def forward(
142-        self, target, pixel_values=None, input_ids=None, attention_mask=None, **kwargs
143-    ):
144-        mllm_output = self.model.mllm_backbone(
145-            input_ids=input_ids,
146-            pixel_values=pixel_values,
147-            image_grid_thw=kwargs.get("image_sizes", None),
148-            attention_mask=attention_mask,
149-            output_hidden_states=True,
150-        )
151-
152-        image_loss = None
153-        action_loss = None
```
</details>

**tool call `Bash`**
```
{"command": "sed -n '150,235p' /tmp/mantis.py", "description": "Read Mantis forward body"}
```

<details><summary>tool result</summary>

```
        )

        image_loss = None
        action_loss = None
        language_loss = None

        if "image" in self.training_mode:

            latents = self.vae.encode(target).latent

            if (
                "shift_factor" in self.vae.config
                and self.vae.config.shift_factor is not None
            ):
                latents = latents - self.vae.config.shift_factor
            latents = latents * self.vae.config.scaling_factor

            bsz = latents.shape[0]

            if (
                pixel_values is not None
                and hasattr(self.model, "mllm_type")
                and self.model.mllm_type == "qwenvl"
            ):
                pixel_values = pixel_values.squeeze(0)

            noise = torch.randn_like(latents, device=latents.device)

            weighting_scheme = "uniform"
            u = compute_density_for_timestep_sampling(
                weighting_scheme=weighting_scheme,
                batch_size=bsz,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )
            indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
            timesteps = self.noise_scheduler.timesteps[indices].to(
                device=latents.device
            )

            sigmas = self.get_sigmas(
                timesteps, latents.device, n_dim=latents.ndim, dtype=latents.dtype
            )
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
            prompt_embeds, attention_mask = self.model.encode_condition(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mllm_output=mllm_output,
            )

            model_pred = self.model(
                x=noisy_latents,
                timestep=timesteps,
                prompt_embeds=prompt_embeds,
                attention_mask=attention_mask,
            )

            target = noise - latents
            weighting = compute_los
... [truncated 1162 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ntimeout 110 python3 /tmp/wog/axs.py search 'abs:\"predictable\" AND abs:\"unpredictable\" AND abs:\"decompose\" AND abs:\"future\" AND cat:cs.LG' 12 2>&1|tail -35", "description": "Search predictable/unpredictable decomposition", "timeout": 130000}
```

<details><summary>tool result</summary>

```
[0 results for: abs:"predictable" AND abs:"unpredictable" AND abs:"decompose" AND abs:"future" AND cat:cs.LG]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\necho \"--- 2511.16175 comments ---\"\ntimeout 50 curl -sL --max-time 45 \"https://arxiv.org/abs/2511.16175\" | python3 -c \"\nimport sys,re,html\nt=sys.stdin.read()\nfor lbl in ['comments','subjects']:\n    m=re.search(r'<td class=\\\"tablecell '+lbl+r'[^\\\"]*\\\">(.*?)</td>',t,re.S)\n    print(lbl.upper(),':',html.unescape(re.sub(r'\\s+',' ',re.sub(r'<[^>]+>','',m.group(1)))).strip()[:200] if m else '(none)')\n\"\necho \"--- LEACE code repo ---\"\ntimeout 40 curl -s --max-time 35 \"https://api.github.com/repos/EleutherAI/concept-erasure\" | python3 -c \"import sys,json;d=json.load(sys.stdin);print(d.get('full_name'),d.get('stargazers_count'),d.get('description'))\"\necho \"--- ATM code repo ---\"\ntimeout 40 curl -s --max-time 35 \"https://api.github.com/repos/Large-Trajectory-Model/ATM\" | python3 -c \"import sys,json;d=json.load(sys.stdin);print(d.get('full_name'),d.get('stargazers_count'),d.get('message',''))\"\necho \"--- StillMix ---\"\ntimeout 40 curl -s --max-time 35 \"https://api.github.com/repos/lihaoxin05/StillMix\" | python3 -c \"import sys,json;d=json.load(sys.stdin);print(d.get('full_name'),d.get('stargazers_count'))\"", "description": "Final code and comments verification", "timeout": 200000}
```

<details><summary>tool result</summary>

```
--- 2511.16175 comments ---
COMMENTS : (none)
SUBJECTS : Computer Vision and Pattern Recognition (cs.CV); Artificial Intelligence (cs.AI)
--- LEACE code repo ---
EleutherAI/concept-erasure 268 Erasing concepts from neural representations with provable guarantees
--- ATM code repo ---
Large-Trajectory-Model/ATM 279 
--- StillMix ---
lihaoxin05/StillMix 10
```
</details>

### ASSISTANT — 2026-09-16 19:48

Let me verify UniVLA's disentangling mechanism precisely since it's a strong candidate with substantial code.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ntimeout 60 curl -s --max-time 50 \"https://raw.githubusercontent.com/OpenDriveLab/UniVLA/main/README.md\" | grep -iE 'task-irrelevant|task-centric|latent action|two.stage|DINO' | head -15\necho \"=== LAM source files ===\"\ntimeout 50 curl -s --max-time 45 \"https://api.github.com/repos/OpenDriveLab/UniVLA/git/trees/main?recursive=1\" | python3 -c \"\nimport sys,json\nd=json.load(sys.stdin)\nps=[t['path'] for t in d.get('tree',[]) if t['type']=='blob' and t['path'].endswith('.py')]\nfor p in ps:\n    if 'lam' in p.lower() or 'latent' in p.lower() or 'genie' in p.lower(): print(' ',p)\n\"", "description": "Check UniVLA task-irrelevant LAM", "timeout": 150000}
```

<details><summary>tool result</summary>

```
- A novel approach for extracting task-centric latent actions from cross-embodiment videos.
  - [Task-centric Latent Action Learning](#one-task-centric-latent-action-learning)
    <td> The stage-1 latent action model trained on OpenX and Ego4D. </td></td>
    <td> The stage-2 latent action model trained on OpenX and Ego4D. (Generate task-centric latent actions.)</td>
    <td><a href="https://huggingface.co/TRI-ML/prismatic-vlms/tree/main/prism-dinosiglip-224px%2B7b">TRI-ML/prismatic-vlms/prism-dinosiglip-224px+7b</a></td>
    <td><a href="https://huggingface.co/TRI-ML/prismatic-vlms/tree/main/prism-dinosiglip-224px%2B7b">TRI-ML/prismatic-vlms/prism-dinosiglip-224px+7b</a></td>
    <td><a href="https://huggingface.co/TRI-ML/prismatic-vlms/tree/main/prism-dinosiglip-224px%2B7b">TRI-ML/prismatic-vlms/prism-dinosiglip-224px+7b</a></td>
### :one: Task-centric Latent Action Learning
> We hightly recommond directly using our pre-trained latent action model ckeckpoints to save your time and compute.
> Our latent action model is trained on a comprehensive data collection, encompassing multiple robotic manipulation and navigation datasets from Open X-Embodiment, along with a curated subset of the Ego4D dataset (detailed data construction procedures are provided in the appendix of our [paper](https://www.roboticsproceedings.org/rss21/p014.pdf)).
The latent action model is implemented based on [VQ-VAE](https://arxiv.org/abs/1711.00937).
We train the latent action model on the collection of dataset comprising robot manipulation, navigation and human videos. In stage-1 training, we use an overall batch size of 512 and 100k optimization steps to construct the task-irrelevant latent actions:
The following stage-2 then focuses on learning task-centric latent actions on the basis of stage-1 results. Please modify the ʼʼʼstage_one_ckptʼʼʼ in ʼʼʼlatent_action_model/config/lam-stage-2.yamlʼʼʼ to your local path of stage-1 checkpoint, then run training with:
- **Latent Action Pseudo-Labe
... [truncated 941 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ntimeout 60 curl -s --max-time 50 \"https://raw.githubusercontent.com/OpenDriveLab/UniVLA/main/latent_action_model/genie/modules/lam.py\" -o /tmp/lam.py; wc -l /tmp/lam.py\ngrep -n -iE 'stage|task_aware|lang|dino|residual|second|delta|minus|-' /tmp/lam.py | grep -iE 'stage|lang|task|residual|dino' | head -30", "description": "Read UniVLA LAM two-stage code", "timeout": 90000}
```

<details><summary>tool result</summary>

```
313 /tmp/lam.py
20:class UncontrolledDINOLatentActionModel(nn.Module):
37:        super(UncontrolledDINOLatentActionModel, self).__init__()
42:        self.dino_transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
43:        self.dino_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
44:        self.dino_encoder.requires_grad_(False)
46:        dino_dim = 768
49:        self.action_latent = nn.Parameter(torch.empty(1, 1, self.num_codes, dino_dim))    # TODO: num of codes
52:            in_dim=dino_dim,
69:        self.patch_up = nn.Linear(dino_dim, model_dim)
74:            out_dim=dino_dim,        # Dim of DINOv2-Base
83:        self.lang_proj = nn.Linear(768, model_dim)
88:    def encode_text(self, lang: List):
90:        encoding = self.tokenizer(lang, return_tensors="pt", padding=True).to(self.device) 
105:    def vq_encode(self, videos: Tensor, lang_embed: Tensor = None, attention_mask: Tensor = None) -> Dict:
109:        videos = self.dino_transform(videos)
110:        dion_features = self.dino_encoder.forward_features(videos)['x_norm_patchtokens']
117:        z = self.encoder(padded_patches, lang_embed, attention_mask) 
139:        lang_embed, attention_mask = self.encode_text(batch["task_instruction"])
140:        lang_embed = self.lang_proj(lang_embed)
145:        outputs = self.vq_encode(batch["videos"], repeat(lang_embed, 'b l d -> b T l d', T=T), attention_mask.repeat(T, 1)) 
151:        video_recon = self.decoder(video_action_patches, lang_embed.unsqueeze(1), attention_mask)
169:class ControllableDINOLatentActionModel(nn.Module):
186:        super(ControllableDINOLatentActionModel, self).__init__()
191:        self.dino_transform = transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
192:        self.dino_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
193:        self.dino_encoder.requires_grad_(False)
195:        dino_dim = 768
198:        self.acti
... [truncated 185 chars]
```
</details>

**tool call `Bash`**
```
{"command": "sed -n '240,313p' /tmp/lam.py", "description": "Read ControllableDINOLatentActionModel forward"}
```

<details><summary>tool result</summary>

```
        self.vq.requires_grad_(False)


    def vq_encode(self, videos: Tensor, lang_embed: Tensor = None, attention_mask: Tensor = None) -> Dict:
        # Preprocess videos
        B, T = videos.shape[:2]
        videos = rearrange(videos, "b T c h w -> (b T) c h w")
        videos = self.dino_transform(videos)
        dion_features = self.dino_encoder.forward_features(videos)['x_norm_patchtokens']
        dion_features = rearrange(dion_features, "(b T) l d -> b T l d", T=2)

        action_pad = self.action_latent.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad, dion_features], dim=2)
        action_pad_controllable = self.action_latent_controllable.expand(B, T, -1, -1)
        padded_patches = torch.cat([action_pad_controllable, padded_patches], dim=2)

        # Encode
        z = self.encoder(padded_patches) 
      
        # Get 'uncotrollable' latent action for all future frames
        z_uncontrol = self.to_codebook_uncontrol(z[:, 1:, self.num_codes : self.num_codes * 2])

        # Vector quantize
        z_uncontrol = z_uncontrol.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q_uncontrol, z_uncontrol, emb_uncontrol, indices_uncontrol = self.vq(z_uncontrol)
        z_q_uncontrol = z_q_uncontrol.reshape(B, T - 1, self.num_codes, self.latent_dim)

        # Get 'cotrollable' latent action for all future frames
        z_action = self.to_codebook(z[:, 1:, :self.num_codes])  # (B, T-1, n, E)

        # Vector quantize
        z_action = z_action.reshape(B * (T - 1), self.num_codes, self.latent_dim)
        z_q, z, emb, indices = self.vq_action(z_action)
        z_q = z_q.reshape(B, T - 1, self.num_codes, self.latent_dim)

        return {
            "patches": dion_features,
            "z_q": z_q,
            "z": z,
            "emb": emb,
            "z_q_uncontrol": z_q_uncontrol,
            "z_uncontrol": z_uncontrol,
            "emb_uncontrol": emb_uncontrol,
            "indices": indices,
            "indices_unc
... [truncated 985 chars]
```
</details>

### ASSISTANT — 2026-09-16 19:49

I have comprehensive material. Let me do a final consolidated verification of every ID I will cite.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nfor id in 2606.31232 2505.17006 2604.22152 2602.10098 2609.15189 2511.16175 2505.06111 2607.09185 2306.03819 2004.07667 2509.23009 2211.12883 2603.11110 2607.11397 2510.05057 2401.00025; do\ntimeout 45 curl -sL --max-time 40 \"https://arxiv.org/abs/$id\" | python3 -c \"\nimport sys,re,html\nt=sys.stdin.read()\ndef g(p,d='(none)'):\n    m=re.search(p,t,re.S); return html.unescape(re.sub(r'\\s+',' ',re.sub(r'<[^>]+>','',m.group(1)))).strip() if m else d\nprint('$id | '+g(r'\\[Submitted on (.*?)\\]')[:22]+' | '+g(r'<meta name=\\\"citation_title\\\" content=\\\"(.*?)\\\"')[:70]+' | COMM: '+g(r'<td class=\\\"tablecell comments[^\\\"]*\\\">(.*?)</td>')[:70])\n\"\ndone", "description": "Final batch verification of all cited IDs", "timeout": 600000}
```

<details><summary>tool result</summary>

```
2606.31232 | 30 Jun 2026 | Delta-JEPA: Learning Action-Sensitive World Models via Latent Differen | COMM: (none)
2505.17006 | 22 May 2025 (v1), last | CoMo: Learning Continuous Latent Motion from Internet Videos for Scala | COMM: CVPR 2026
2604.22152 | 24 Apr 2026 | dWorldEval: Scalable Robotic Policy Evaluation via Discrete Diffusion  | COMM: (none)
2602.10098 | 10 Feb 2026 (v1), last | VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Mod | COMM: (none)
2609.15189 | 14 Sep 2026 | Reconstructing Is Not Acting: Action-Centric Latent Dynamics Modeling | COMM: Tech report, 26 pages
2511.16175 | 20 Nov 2025 (v1), last | Mantis: A Versatile Vision-Language-Action Model with Disentangled Vis | COMM: (none)
2505.06111 | 9 May 2025 (v1), last  | UniVLA: Learning to Act Anywhere with Task-centric Latent Actions | COMM: Accepted to RSS 2025. Code is available at this https URL
2607.09185 | 10 Jul 2026 | Causally Debiased Latent Action Model for Embodied Action Conditioned  | COMM: (none)
2306.03819 | 6 Jun 2023 (v1), last  | LEACE: Perfect linear concept erasure in closed form | COMM: (none)
2004.07667 | 16 Apr 2020 (v1), last | Null It Out: Guarding Protected Attributes by Iterative Nullspace Proj | COMM: Accepted as a long paper in ACL 2020
2509.23009 | 27 Sep 2025 | Disentangling Static and Dynamic Information for Reducing Static Bias  | COMM: in Proc. of ICCV2025 Workshop and Challenge on Disentangled Representa
2211.12883 | 23 Nov 2022 (v1), last | Mitigating and Evaluating Static Bias of Action Representations in the | COMM: (none)
2603.11110 | 11 Mar 2026 | ResWM: Residual-Action World Model for Visual RL | COMM: Submit KDD2026
2607.11397 | 13 Jul 2026 | WALA Learning Executable Latent Actions from Action-Labeled Demonstrat | COMM: Project page: this https URL
2510.05057 | 6 Oct 2025 (v1), last  | StaMo: Unsupervised Learning of Generalizable Robot Motion from Compac | COMM: (none)
2401.00025 | 28 Dec 2023 (v1), last | Any-point Trajectory Modeling fo
... [truncated 46 chars]
```
</details>

### ASSISTANT — 2026-09-16 19:54

All IDs verified. Here are my findings.

---

# GRAFT POINT 3：把 O^c 从"编码未来"改成"编码动作引起的未来差异 Δ"

## 0. 先说结论：诊断数据指向的唯一自洽解释

用户的三个数字放在一起，含义是唯一的：

| 诊断 | 含义 |
|---|---|
| 83% O^c 方差可由 current-only 版本线性重建 | **表示**被绝对内容主导 |
| 因果效应 96%–108% 归于 future-only 分量 | **有用信号**几乎纯粹是 Δ |
| effective rank 18 / top-8 主方向复现 97.7% 动作方向 | 512 个数里真正干活的 ≈ 8–18 维 |

也就是说：QFormer 学到的是 `O^c ≈ A·(current) + B·(Δ)`，其中 `A` 占 83% 能量但因果贡献 ≈ 0，`B` 占 17% 能量却承担 ~100% 因果。**Graft Point 3 的本质是把 A 项砍掉，让 16×32 的容量全部让给 B 项。**

但有一个致命张力必须先摊开（后面每个候选都会回到它）：

> **Stage II 的输入是 `obs_ins`（VLM 的最后 4 个 hidden token，只看当前观测 O_t + 语言 L）。当前 Stage II 只能把 future-only 残差的可预测性从 0.061 拉到 0.177。**
> 如果 O^c 变成纯 Δ，那 83% 那部分"白送的、Stage II 免费能猜对的"内容就消失了 —— **Stage II 的余弦目标会从"猜一个 83% 可从当前帧推出的东西"变成"猜一个 0% 可从当前帧推出的东西"，obs_loss 会显著变差。**

这不是 bug，是必然。我在每个候选的第 4 节都会给出具体判断，并在最后 §7 给出对策。**排序时我把"能同时缓解这个张力的机制"排在前面。**

---

## 候选 1（最推荐）：CoMo — early Temporal-difference (Td) + 时间反转对比 (Tcl)

| | |
|---|---|
| arXiv | **2505.17006**（v1 2025-05-22，v3 2026-06-18） |
| URL | https://arxiv.org/abs/2505.17006 |
| Comments/venue | **CVPR 2026**（abs 页 Comments 字段确认） |
| 代码 | **公开**，https://github.com/MCG-NJU/CoMo （master 分支，1222 files，最后 push 2026-05-15，16 star，无 license 文件） |

### 为什么排第一：架构是 WoG QFormer 的近亲，可以逐行对照移植

我读了 `idm/CoMo/latent_motion_tokenizer/src/models/latent_motion_tokenizer.py` 和 `m_former.py`。CoMo 的 M-Former 结构与 WoG 的 QFormer 几乎同构：**learnable query tokens 拼在前面（`latent_motion_token`，`query_num` 个）+ cond 帧 patches + sep token + target patches，过 ViTEncoder，取 `last_hidden_state[:, :query_num]`，再 `vq_down_resampler` 降到 codebook_dim=32。** WoG 是 16 queries → 32 dims；CoMo 也是 query tokens → codebook_dim=32。这是我在整个搜索里找到的**唯一一个和 WoG 的 O^c 通路可以做逐行对齐的公开实现**。

### 精确机制（两个可分离的部件）

**(a) early temporal difference (Td)** — 代码原文：

```python
with torch.no_grad():
    cond_hidden_states   = self.image_encoder(img_1).last_hidden_state   # 当前帧
    target_hidden_states = self.image_encoder(img_2).last_hidden_state   # 未来帧
# our early temporal difference
motion_td_features = target_hidden_states - cond_hidden_states           # ← 关键一行

latent_motion_tokens = [REDACTED]
    cond_hidden_states=cond_hidden_states,        # 当前帧 patches 仍然进去（做 context）
    target_hidden_states=motion_td_features       # 但未来帧位置喂的是 Δ，不是绝对值
).last_hidden_state[:, :query_num]
```

注意这个设计精确回答了你问的设计问题：**它不是 (a) 完全去掉当前帧，也不是 (b) 事后正交化，而是第三条路 —— 当前帧保留在 cond 位置作为 context，但 target 位置只喂 Δ。** 这在 encoder 输入端就物理性地消除了"重复编码静态内容"的捷径，因为静态内容在 Δ 里已经被减掉了。作者的动机是"increase the shortcut learning difficulty and explicitly enhance motion cues"，与你的 83% 诊断是同一个病。

**(b) temporal contrastive learning (Tcl)** — 正负样本构造非常巧妙，代码原文：

```python
reversed_motion_features        = cond_hidden_states - target_hidden_states  # 时间反转 = 负样本
positive_jitter_motion_features = latent_4 - cond_hidden_states              # 未来帧小抖动 = 正样本
```
- **正样本**：把未来帧索引抖动一点（`img_4`），Δ 应该几乎不变
- **负样本**：**把时间方向反转**（`cond - target`，且 m_former 的 cond/target 也对调）

InfoNCE，temperature 默认 1.0（构造函数），`contrastive_loss_w=0.004`。反转作负样本这一招对 WoG 特别对症：**静态内容在时间反转下完全不变，只有 Δ 会翻符号。所以"能区分正向 vs 反向"这个能力在数学上不可能由静态分量承载 —— 这是一个 O(10 行) 的、可证明只惩罚静态分量的损失。**

仓库里还有一个未在损失中启用的 `l_cov(S)`（off-diagonal 协方差平方和 / d，VICReg 风格去相关），可以直接拿来打 effective rank 18 / 16-query 冗余那两个诊断。

### 如何移植进 WoG（具体到改哪些张量）

**改 QFormer 输入（3 处，约 15–25 行）：**
1. DINOv2 分支：现在是 5 帧 × 64 tok。改成 **当前帧 64 tok 原样 + 4 个未来帧各自减去当前帧**：`dino[i] = dino[i] - dino[0]` for i=1..4。索引 0 保留绝对值当 context。这一步保持 token 数、序列长度、position embedding 全不变 —— **零架构改动**。
2. Wan VAE 分支：当前帧 latent 在未来帧之前 concat、temporal 维压到 2。同样把未来部分改成减去当前帧的 latent。（若 temporal squeeze 使得对齐麻烦，第一版可以只改 DINOv2 分支 —— 因为 DINOv2 是语义主通路，8×8 pooling 后的 patch 差本身就是"哪个区域动了"的强信号。）
3. 建议加一个 token-type embedding 区分"absolute" vs "delta" 两类 token（CoMo 的 `token_type_embeddings` 就是干这个的；顺便说，**CoMo 代码里这里有一个作者自己标注的 bug**：`target_token_type_embeddings` 用了 `token_type_embeddings[0]` 而不是 `[1]`，注释写着 "BUG: should be self.token_type_embeddings[1], to be fixed" —— 移植时不要照抄这个错误）。

**加损失（约 30–40 行）：**

```
L_tcl = InfoNCE( pos = <O^c(0,T_a), O^c(0,T_a±jitter)>,
                 neg = { <O^c(0,T_a), O^c_reversed>, ... } )       # O^c_reversed: 交换 cond/target 且 Δ 取负
loss = act_loss + λ_tcl · L_tcl                                     # λ_tcl ≈ 0.004（CoMo 默认）
```
反转分支在 CoMo 里是 `torch.no_grad()` 下算 m_former 前向、只对 down_resampler 之后回传梯度 —— 这个细节直接照抄，省显存。

**Stage 需要重训吗：** Stage I **必须**重训（QFormer 输入分布变了）。Stage II 的**代码**不需要改一行 —— `obs_loss` 仍是 `1 - cos(O^c_hat, O^c.detach())`，只是 target 的语义变了。但 Stage II 需要重训，且见下节。

**改动量估计：** QFormer 输入 15–25 行 + Tcl 损失 30–40 行 + 数据加载器多取一个 jitter 帧 ~10 行 ≈ **60–75 行**，外加 Stage I 一次完整重训。

### 诊断预测

- **83% → 期望大幅下降。** Td 是在输入端做的，不是靠损失"劝说"模型别学静态内容 —— 静态内容在 target 位置**物理上不存在**。残余的 current-only 可重建性只能来自 cond 位置那 64 个绝对 token，可以通过调低 cond 分支的权重/token 数进一步压。
- **因果有用容量占比 → 应显著上升。** 现在 16×32 里不再有 83% 的能量花在因果贡献 ≈0 的分量上。配合 `l_cov` 去相关，effective rank 18 有希望往上走，16-query 冗余（单 token R²=0.62 vs 全 16 个 0.645）也应缓解。
- **Stage II 会变难 —— 但这是本候选相对其他候选最轻的一个。** 关键理由：Td 保留了 cond 位置的绝对当前帧，所以 O^c 不是"纯 Δ"，而是"以当前帧为条件的 Δ"。Stage II 的 `obs_ins` 也看得到当前观测，**两边条件化在同一个东西上**，所以 Stage II 要预测的是 `p(Δ | O_t, L)` 而不是 `p(Δ | nothing)`。这仍然比现在难（0.177 那个数会先掉），但它是 well-posed 的：给定当前帧和语言指令，专家会怎么动，本来就是策略要学的东西。**如果 Stage II 在这个设置下无法把 Δ 预测好，那不是 graft 失败，而是直接证明了 O^c 通路在部署时本来就无法传递信息 —— 这本身是对 WoG 设计的一个强诊断结论，值得作为实验结果报告。**

---

## 候选 2（最便宜，几十行，只改损失）："让 O^c 无法从当前帧预测"的标准配方

这是你问的设计变体 (b)。我找到了三个层次的标准配方，**按"越靠前越便宜、越靠后越彻底"排列**。它们不是同一篇论文，而是同一个技术家族；我把它们放在一个候选里，因为对 WoG 而言它们共用同一个 graft 点（只加损失，QFormer 输入完全不动）。

### 2A. 梯度反转 / 对抗式静态不变性

| | |
|---|---|
| 奠基 | **1409.7495** Unsupervised Domain Adaptation by Backpropagation（2014-09-26，Comments: **(none)**）— gradient reversal layer 的原始出处 |
| 视频版对症实现 | **2509.23009** Disentangling Static and Dynamic Information for Reducing Static Bias in Action Recognition（2025-09-27，Comments: **in Proc. of ICCV2025 Workshop and Challenge on Disentangled Representation Learning for Controllable Generation (DRL4Real)**）；代码：**未在 abs 页找到公开链接** |
| 同族基线 | **2211.12883** Mitigating and Evaluating Static Bias of Action Representations in the Background and the Foreground（2022-11-23，Comments: **(none)**）；代码 **公开** https://github.com/lihaoxin05/StillMix （10 star） |

**标准配方**（2509.23009 的表述：*"a statistical independence loss between biased and unbiased streams, combined with a scene prediction loss"*）：

```
# 1) 训一个 static predictor：从 current-frame-only 特征预测 O^c
O_hat_static = StaticHead( DINOv2(frame_0) )            # 2~3 层 MLP，几百行以内
L_static     = || O_hat_static - O^c.detach() ||^2      # StaticHead 自己的目标：尽量猜中

# 2) 对 QFormer 施加梯度反转：让 O^c 反过来最大化 StaticHead 的误差
loss = act_loss + λ_adv · ( - L_static_through_GRL )
```
GRL 实现是 5 行的 `torch.autograd.Function`（`backward` 里返回 `-λ*grad`）。**注意 2509.23009 强调 scene prediction loss 是必需的配套项** —— 单纯对抗容易塌，需要一个 biased stream 显式地去承接静态信息（在 WoG 里对应：让 StaticHead 有自己的、非对抗的训练信号）。

**注意**：`abs:"gradient reversal" AND abs:"static"` 这个检索只返回 1 条不相关结果，说明"梯度反转 + 静态偏置"这个组合**没有一篇专门的、可引用的 robotics 论文**。它是从 DANN（1409.7495）+ 视频静态偏置文献（2509.23009 / 2211.12883）拼出来的标准工程做法。**这一点请视为"配方可靠但无单一 anchor 论文"，不要当成有现成 baseline 可对比。**

### 2B. HSIC / 协方差惩罚（无对抗，更稳，我更推荐先试这个）

| | |
|---|---|
| Barlow↔HSIC 的桥 | **2104.13712** A Note on Connecting Barlow Twins with Negative-Sample-Free Contrastive Learning（2021-04-28，Comments: **(none)**）— 明确论证 Barlow Twins 的 off-diagonal 惩罚就是 HSIC |

**配方**：不训对抗网络，直接惩罚 O^c 与 current-only 编码之间的 cross-covariance：

```
S = flatten(O^c)                      # [B, 512]
C = StaticEnc(DINOv2(frame_0))        # [B, d]，可以直接用 pooled DINOv2，甚至不需要新参数
Sn = (S - S.mean(0)) / S.std(0);  Cn = (C - C.mean(0)) / C.std(0)
L_hsic = ( (Sn.T @ Cn) / (B-1) ).pow(2).sum() / 512      # cross-covariance 的 Frobenius 平方
loss = act_loss + λ_hsic · L_hsic
```
**约 10 行，零新参数**（若 `StaticEnc` 直接用冻结的 DINOv2 pooled 特征）。CoMo 仓库里的 `l_cov` 就是这个东西的自协方差版本，可以直接改造。

对 WoG 的适配性：你的 83% 是**线性**可重建性。HSIC/cross-cov 惩罚正是直接打这个线性可预测性的 —— **诊断指标和损失函数是同一个量**，这是极难得的对应关系。可以把 83% 这个数直接当训练监控指标看它掉到多少。

### 2C. LEACE / INLP：闭式线性擦除（最彻底，可做"不训练的上界实验"）

| | |
|---|---|
| LEACE | **2306.03819** LEACE: Perfect linear concept erasure in closed form（v1 2023-06-06，v4 2025-04-03，Comments: **(none)**）；代码 **公开** https://github.com/EleutherAI/concept-erasure （268 star） |
| INLP | **2004.07667** Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection（2020-04-16，Comments: **Accepted as a long paper in ACL 2020**） |

**机制**：LEACE 给出闭式的仿射变换 `r ↦ P(r - μ) + μ`，使得**任何**线性预测器都无法从 `r` 预测目标概念，且在所有满足该约束的变换中对 `r` 的改动（协方差意义下）最小。INLP 是它的迭代前身（反复求线性探针的 null space 做投影）。

**对 WoG 最高价值的用法：一个不需要重训的诊断实验（我认为应该第一个做）。**
1. 在已有 Stage I checkpoint 上，收集一批 `(O^c, current-only-O^c)` 对；
2. 用 LEACE 闭式求出把 current-frame 分量擦掉的投影 `P`；
3. **推理时**把 `P` 插在 `v_proj` 之后、DiT cross-attn 之前，重跑你已有的因果效应度量。

如果动作性能不掉（甚至上升），你就用**几十行、零训练**证明了那 83% 是纯冗余，从而为整个 Graft Point 3 提供 go/no-go 依据。如果掉了，说明 83% 里其实混着有用信息，Graft 3 需要更保守的版本。**这是整个任务里投入产出比最高的一步。**

### Stage / 改动量总结（候选 2）

- QFormer 输入：**完全不改**（frames / DINOv2 patches / Wan VAE latents 全部原样）
- 2B 加损失约 **10 行**；2A 加 GRL + StaticHead 约 **40–60 行**；2C 诊断实验约 **30 行且零训练**
- Stage I：2A/2B 需重训；2C 不需要
- Stage II：代码不变；蒸馏目标语义改变

### 诊断预测（候选 2）

- **83% → 可以直接被压到任意低**（λ 是旋钮），这是它相对候选 1 的独特优势：**你可以做 λ 扫描，画出"83% 可重建性" vs "因果有用容量" vs "Stage-II 0.177" 的三维 trade-off 曲线。** 对一篇论文来说这条曲线比任何单点结果都有说服力。
- **Stage II 会变难，且这里的张力最尖锐。** 因为 2B/2A 是**事后正交化**：O^c 被强行做成与当前帧统计独立。而 Stage II 的 `obs_ins` **只有当前观测和语言**。极限情况下（λ→∞，完美独立），`O^c` 与当前帧互信息为 0，那么 `O^c_hat` 能从 `obs_ins` 学到的只剩语言 `L` 那部分信息 —— **obs_loss 的可达上界被 λ 直接掐住。**
  - 但要点在于：**语言 L 恰恰不是当前帧的函数。** 所以"与当前帧独立"≠"与 obs_ins 独立"。正交化应当针对 **current frame**，**绝不能**针对整个 `obs_ins`（那会把 L 也擦掉）。这个区分是移植成败的关键，实现上就是 `StaticEnc` 只吃 `DINOv2(frame_0)`，不吃 `z`。
  - 另一个可用信号：*"Under real z, O^c's causal influence is 36% LOWER than under random z"* —— 说明 `z` 和 `O^c` 目前**信息重叠严重**（真 z 已经供给了 O^c 想说的一部分，所以 O^c 边际贡献下降）。做 current-frame 正交化会**减少**这种重叠，可能反而让 36% 这个数缩小、O^c 通路变得更有边际价值。这是候选 2 一个额外的、可测量的预期收益。

---

## 候选 3：VLA-JEPA — leakage-free：未来帧只做监督目标，绝不做输入

| | |
|---|---|
| arXiv | **2602.10098**（v1 2026-02-10，v2 2026-02-14） |
| URL | https://arxiv.org/abs/2602.10098 |
| Comments/venue | **(none)** |
| 代码 | **未找到公开链接**（abs 页无 github/项目页链接）→ **未验证有代码** |

### 精确机制

原文表述：*"leakage-free state prediction: a target encoder produces latent representations from future frames, while the student pathway sees only the current observation — future information is used solely as supervision targets, never as input."* 明确点名它要解决的病是 *"appearance bias, nuisance motion, and information leakage"* 和 *"remain anchored to pixel variation rather than action-relevant state transitions"*。

### 为什么对 WoG 特别值得注意：它其实是在质疑 WoG 的两阶段结构本身

WoG 现在的结构是：Stage I 让 O^c **看见**未来 → Stage II 再让 `oformer` 从只看当前的 `obs_ins` **追**这个 O^c → 部署时 O^c 完全不参与。VLA-JEPA 的论断是：**这个"先泄漏、再蒸馏、最后丢弃"的三段式是多余的**，应该一步到位 —— 未来帧从头到尾只出现在 loss 的 target 侧。

对照 WoG 的诊断，这个论断有直接证据支持：`0.061 → 0.177` 意味着**Stage II 的蒸馏只成功了很小一部分**，而部署时 O^c 根本不参与。也就是说 WoG 现在为 O^c 付出的全部代价，最终只以 0.177 的保真度传给了部署路径。VLA-JEPA 的做法把这个损耗环节直接消掉。

### 如何移植进 WoG

**这是本报告中改动最大的候选，也是唯一一个改变 WoG 拓扑的候选。**

- **QFormer 输入**：变成 **student 只吃 frame 0**（DINOv2 patches + 当前帧 Wan VAE latent）；未来 4 帧移到一个 **EMA target encoder**（frozen momentum copy of QFormer）里。这就是你说的变体 (a)（完全去掉未来帧作为输入）的镜像版本 —— 严格来说是"去掉未来帧作为 **student** 输入"。
- **加损失**：`L_jepa = 1 - cos( student(frame_0), sg[target_enc(frames_1..4)] )`，加 EMA 更新（`m ≈ 0.99`）。
- **Stage I**：重训，且需要新增 EMA target encoder 分支。
- **Stage II**：**结构上被吸收掉了。** student 本身就是"只看当前观测预测未来 latent"的东西 —— 这正是 `oformer` 想干的事。所以 Stage II 的 `oformer` + `obs_loss` 可以整个删掉，两阶段变一阶段。
- **改动量**：**数百行**（EMA encoder、两路前向、Stage 结构重组）。这不是 surgical graft，是重构。

### 诊断预测

- **83%：这个指标失去意义。** student 的输入就只有当前帧，所以"从 current-only 重建"的可重建性必然接近 100% —— 但那不再是缺陷，因为 student 的**目标**已经是 Δ。诊断需要换成"student 输出与 target encoder 的 Δ 分量的对齐度"。
- **Stage II 的张力在这里被最优雅地化解 —— 因为 Stage II 不存在了。** 0.177 那个数没有对应物：不再有"先造一个只有未来才知道的东西，再让当前帧去猜它"的环节。**如果你愿意接受重构成本，这是对 Stage-II 张力唯一的结构性解法，而不是权衡。**
- **风险**：无公开代码；EMA target encoder 有表示塌缩风险（JEPA 系通病，Delta-JEPA 整篇就是在治这个）；且它偏离了 WoG 的 ControlNet 式 zero-init cross-attn 设计，"最小改动验证一个假设"的性质丧失。**建议把它当作候选 1/2 成功后的下一步，而不是第一步。**

---

## 候选 4（anchor 验证）：Delta-JEPA — 从**潜在位移**反解动作（LDAD）

| | |
|---|---|
| arXiv | **2606.31232**（2026-06-30） |
| URL | https://arxiv.org/abs/2606.31232 |
| Comments/venue | **(none)** |
| 主题 | cs.AI（**注意：不是 cs.RO**） |
| 代码 | **未找到公开链接** → **未验证有代码** |

### 精确机制（用户的 lead，已验证）

核心是 **LDAD（Latent Difference Action Decoder）**，原文：*"Unlike inverse decoders that infer actions from concatenated endpoint embeddings, LDAD reconstructs the executed action from the latent displacement between consecutive observations."*

也就是说，标准 inverse dynamics 是 `a_hat = IDM([z_t ; z_{t+1}])`（拼接两端点），LDAD 改成 `a_hat = IDM(z_{t+1} - z_t)`（只喂差）。作者给出的理由正中 WoG 的痛点：*"This displacement-level supervision directly regularizes transition geometry: adjacent embeddings cannot collapse without losing action information, and different actions are encouraged to induce distinguishable latent changes."* 消融结论：*"displacement-based action decoding is consistently more effective than endpoint concatenation."*

**关键差异要讲清楚**：拼接版 `IDM([z_t; z_{t+1}])` 允许网络自由地从 `z_t` 里读静态内容（正是 WoG 现在的病）；差分版把输入限制在一个减法之后的量上，**静态分量在输入端就被消掉了，网络没有捷径可走。** 这与 CoMo 的 Td 是同一个原理在不同位置（CoMo 在 encoder 输入端，Delta-JEPA 在 decoder 输入端）的两次应用。

### 如何移植进 WoG

WoG 有**真实动作标签**（rectified-flow 的 `act_loss` 就在监督 (16,7) 的动作），所以这里比原论文更容易 —— 不需要自监督地"发现"动作，可以直接监督。

- **QFormer 输入**：**完全不改**（这是它相对候选 1 的优势 —— 更 surgical）。
- **加损失（辅助头，约 20–30 行）**：
  ```
  d = flatten(O^c) - flatten(O^c_current_only)      # 或直接用 O^c 的 future-only 投影分量
  a_hat = LDAD_MLP(d)                               # 2~3 层 MLP → (16, 7)
  L_ldad = MSE(a_hat, a_gt)                         # a_gt 就是 WoG 已有的动作标签
  loss = act_loss + λ_ldad · L_ldad
  ```
  更省事的变体（不需要跑第二遍 QFormer 拿 current-only）：直接对 `O^c` 加一个 LDAD 头，但把梯度限制在 future-only 子空间上 —— 而"future-only 子空间"你**已经有了**，因为你已经做过 96%–108% 那个因果分解实验。
- **Stage I**：需重训（新增辅助头 + 新损失项）。
- **Stage II**：代码与目标**都不变**（O^c 本身的定义没改，只是加了一个正则让它的 Δ 分量必须能反解出动作）。这是候选 4 的最大优点。
- **改动量**：**20–40 行 + Stage I 重训**。

### 诊断预测

- **83%：可能只是小幅下降。** 这是必须诚实指出的局限：LDAD 是**加压**（要求 Δ 分量必须含动作信息），不是**减压**（不阻止静态分量继续存在）。它会让 17% 那部分变得更强、更结构化，但不直接惩罚 83% 那部分。**所以它是候选 2（正交化，减压）的天然互补，而不是替代。最强的组合大概是 2B（HSIC）+ 4（LDAD）：一个压掉静态、一个把腾出的容量填上动作信息。**
- **effective rank / 冗余：应改善。** 原文的动机就是防塌缩（*"adjacent embeddings cannot collapse"*，*"different actions are encouraged to induce distinguishable latent changes"*），而 rank 18 + 单 token R²=0.62 ≈ 全 16 token R²=0.645 正是塌缩的表现。**这可能是四个诊断里 LDAD 最直接命中的那一个。**
- **Stage II：几乎不变难，这是它相对候选 1/2/3 的核心优势。** 因为 O^c 的定义没被改，`O^c` 与当前帧的互信息没被主动削减，0.177 不会被结构性地压低。**如果你想要"提升 O^c 因果有用容量"但"不动 Stage II 分毫"，候选 4 是唯一的选项。**
- **风险**：无公开代码、非 cs.RO、只在 4 个 visual continuous-control 任务上验证（不是 manipulation / OXE 规模），且 Comments 为 (none)（无 venue）。**机制本身是 20 行的减法，可靠性不依赖于论文质量；但不要把它当成经过大规模验证的方法。**

---

## 候选 5：Mantis DVF — 用 residual connection 让 meta query 被迫只编码"当前帧解释不了的部分"

| | |
|---|---|
| arXiv | **2511.16175**（v1 2025-11-20，v2 2026-02-23） |
| URL | https://arxiv.org/abs/2511.16175 |
| Comments/venue（abs 页） | **(none)** —— 但仓库 README 标注 **CVPR 2026**（README 原文："[CVPR 2026] Mantis: ..."）。**abs 页 Comments 与 README 不一致，以 abs 页为准记作 (none)，CVPR 2026 来自 README。** |
| 代码 | **公开**，https://github.com/SJTU-DENG-Lab/Mantis （94 star，最后 push 2026-06-05，33 个 .py，含 checkpoints on HF） |

### 精确机制 —— 这是"残差/预测编码"这一类里最贴 WoG 的一个

原文关键句：*"With the current visual state provided to the DiT via a residual connection, a simple next-state prediction objective enables the meta queries to automatically capture the latent actions that delineate the visual trajectory."*

机制的精髓在于**信息经济学**，而不是任何显式的正交化损失：DiT 头要预测下一帧；当前帧通过 residual **免费**给到 DiT；因此 meta query 若去编码当前帧内容就是纯浪费带宽 —— **梯度会自动把 meta query 推向"当前帧无法解释的残差"，也就是 Δ。** 这是"predictive coding：一个分支只预测 current frame 无法解释的部分"这个类别在 VLA 上的落地实现，且不需要任何额外损失项。

作者的问题陈述几乎是 WoG 83% 诊断的复述：*"letting VLA directly predict high-dimensional visual states can distribute model capacity ... while compressing visual states into more compact supervisory signals inevitably incurs information bottlenecks"*，`num_metaqueries` 默认 **9**（`models/model.py:29`），与 WoG 的 16 queries 同一量级。

**代码验证说明（诚实标注）**：我下载并读了 `models/mantis.py`（606 行）、`models/model.py`（533 行）、`models/action_model/models.py`。我确认了：`num_metaqueries=9` 的 meta query 机制存在；图像分支是 Sana VAE + flow-matching（`noisy_latents = (1-sigmas)*latents + sigmas*noise`，`target = noise - latents`）；action 分支独立走 `policy_head.loss`。但**我没有在这三个文件里定位到那个 residual connection 的具体那一行** —— `image_loss` 分支里 DiT 的调用是 `self.model(x=noisy_latents, timestep=..., prompt_embeds=...)`，residual 大概在 `models/transformer_encoder.py` 或 DiT 内部。**"residual connection 提供 current visual state"这一点我依据的是论文原文陈述，未在代码层面逐行确认。** 移植前应先把这一行找出来。

### 如何移植进 WoG

**这是"WoG 已有结构里最小的一处改动"，因为 WoG 缺的恰恰就是这个 residual。**

WoG 现在：Stage I 的 `obs_loss` 是 **hard-zeroed** 的（`loss = act_loss only`）。也就是说 **O^c 从来没有被要求预测未来** —— 它只是通过 `act_loss` 间接被塑造。这意味着"用信息经济学逼迫 O^c 编码 Δ"这个机制在 WoG 里**完全缺失**。Mantis 补的正是这一块。

- **QFormer 输入**：不改。
- **新增一个轻量 future-prediction 头 + residual（约 60–100 行）**：
  ```
  # 一个小 DiT/MLP 头，预测未来帧的 Wan VAE latent（WoG 已经在算这个了，白拿）
  pred = FutureHead( concat[ O^c , WanVAE(frame_0) ] )    # ← frame_0 latent 走 residual，免费提供
  L_fut = MSE( pred, WanVAE(frames_1..4) )                 # 或 flow-matching，照 Mantis
  loss = act_loss + λ_fut · L_fut
  ```
  **关键是 residual 那一项必须存在。** 如果不给 `WanVAE(frame_0)`，`FutureHead` 就得从 O^c 里读静态内容，那就会**加剧** 83% 而不是缓解 —— 这个机制的正负号完全取决于 residual 在不在。
- **Stage I**：重训。注意这等价于**把 WoG 硬置零的 `obs_loss` 以"带 residual 的形式"重新打开** —— 这是一个漂亮的叙事点：WoG 作者关掉 obs_loss 大概是因为它没用（不带 residual 的 obs_loss 会鼓励 O^c 编码绝对未来，与 act_loss 争容量）；加上 residual 后它才变成有益的。
- **Stage II**：目标不变（仍是 cos(O^c_hat, O^c)），但 O^c 语义变了。`FutureHead` 训完即弃（和 `oformer` 一样）。
- **改动量**：**60–100 行 + Stage I 重训。**

### 诊断预测

- **83% → 应下降**，且是"被梯度自然推着降"而非被损失强行压降，所以副作用小。降幅不如候选 1/2 可控（没有 λ 直接对准可重建性）。
- **因果有用容量 → 上升**，且额外好处：`L_fut` 给 O^c 提供了一个**除 act_loss 之外的第二个训练信号**。现在 O^c 的全部梯度都来自 `act_loss` 经由 zero-init cross-attn 的回传，信号很弱 —— 这可能正是 rank 只有 18、16 query 高度冗余的原因。`L_fut` 直接给 O^c 施加一个高维、结构化的监督，**这可能比任何 Δ 相关的改动更能提升 effective rank。**
- **Stage II：中等难度上升。** O^c 会偏向 Δ，所以从 `obs_ins` 预测变难；但因为它同时被 `L_fut` 结构化了（不再是 act_loss 的弱梯度塑造出的杂乱 18 维），可能**更容易**被 `oformer` 拟合 —— 一个低维但结构清晰的目标可能比一个高维杂乱目标更好预测。**净效应方向不确定，需要实测。这是唯一一个我认为 Stage-II 的 0.177 有可能上升而不是下降的候选。**

---

## 候选 6：ACT-LAM — 直接针对"IDM 不区分动作与外观 + FDM 走当前帧捷径"的双重欠约束

| | |
|---|---|
| arXiv | **2609.15189**（2026-09-14，**极新，两天前**） |
| URL | https://arxiv.org/abs/2609.15189 |
| Comments/venue | **Tech report, 26 pages**（无 venue） |
| 代码 | **仓库存在但基本是空的**：https://github.com/DingjieFu/ACT-LAM （**1 star，最后 push 2026-09-15**）→ **代码"公开"但实质未发布，视为不可用** |

### 为什么值得列入：它把 WoG 的病症用两句话做了理论化

原文对 *reconstruction-action mismatch* 的两点归因，是我在整个搜索里看到的对 WoG 诊断最精确的重述：

1. *"the inverse dynamics model (IDM) is not explicitly encouraged to distinguish action-related transitions from nuisance appearance"* → 对应 WoG 的 **83%**（QFormer 就是那个 IDM，它没被要求区分动作与外观）
2. *"the forward dynamics model (FDM) can underutilize the inferred latent action by exploiting predictive shortcuts from the current state"* → 对应 WoG 的 **96%–108% 归于 future-only** 和 **"real z 下因果影响比 random z 低 36%"**（DiT 从 `z`/`c = t+z` 走捷径，绕过 O^c）

两个方案：
- **AQ-IDM**：learnable action queries + gated aggregation，*"without strong information bottlenecks"* —— **这里有一个与 WoG 直接冲突的论断**：WoG 用了极窄的瓶颈（16×32 = 512 个数），ACT-LAM 明确主张**不要**强瓶颈，而是靠 gating 做选择性提取。考虑到 WoG 的 effective rank 只有 18（512 个数里只用了 18 维），"瓶颈太窄"似乎不是 WoG 的问题；但"瓶颈迫使模型优先编码最省事的东西（静态内容）"可能正是 83% 的成因之一。**这是一个值得单独做消融的假设：把 16 queries × 32 dims 放宽（例如 32×64），配合 gating，会不会反而让 Δ 分量有地方待？**
- **AT-FDM**：把 latent action 投成 action token，*"progressively interact with evolving state representations, enabling continuous state-aware action conditioning"* —— 这是对 WoG 的 **zero-init `per_attn` 在 24 个 block 全部注入 K/V** 的一个替代设计。

### 如何移植进 WoG

- **AQ-IDM 部分**：在 QFormer 的 query 聚合处加 gating（约 20–30 行），并/或放宽 `v_proj` 的输出维度。QFormer 输入不改。Stage I 重训。若改了 O^c 的 shape，**Stage II 的 `oformer` 输出维度必须跟着改**（这是本报告中唯一会改 Stage II 结构的候选）。
- **AT-FDM 部分**：改 DiT 的注入方式。**不建议动** —— 这会破坏 WoG 的 ControlNet 式 zero-init 设计，而那正是你已经用因果分解精确度量过的东西
... [truncated 5262 chars]
