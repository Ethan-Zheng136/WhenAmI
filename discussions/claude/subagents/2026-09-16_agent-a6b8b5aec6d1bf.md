# Claude Code session agent-a6b8b5aec6d1bfe23

- Time (Beijing): 2026-09-16 17:05 → 2026-09-16 17:22
- cwd: `/home/zhengguantian/WoG_v2`
- Source: `~/.claude/projects/-home-zhengguantian/c976c298-d71c-4d38-9107-f692d869d083/subagents/agent-a6b8b5aec6d1bfe23.jsonl` (neolix_al_2)
- Messages (new, after de-dup): 175
- Note: tool outputs capped at 2000 chars; thinking blocks and images dropped; secrets/private IPs redacted.

---

### USER — 2026-09-16 17:05

You are doing a literature survey. Today is 2026-09-15. NETWORK: prefix every network command with `export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000`. IMPORTANT: the arXiv API only works over **https** (`https://export.arxiv.org/api/query?...`), plain http returns empty.

Context: LeJEPA (arXiv 2511.08544, Balestriero & LeCun, Nov 2025) proposed SIGReg (Sketched Isotropic Gaussian Regularization) — enforcing isotropic Gaussian embeddings via a sketched Epps-Pulley characteristic-function test. Its world-model successor is **LeWorldModel (LeWM), arXiv 2603.19312**, "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels", Maes/Le Lidec/Scieur/LeCun/Balestriero, v1 2026-03-13.

Your cluster: **ALL 2026 follow-ups that EXTEND or CRITIQUE LeJEPA / SIGReg / LeWorldModel**. For EACH paper, curl `https://arxiv.org/abs/<id>` and extract exact title, citation_date, citation_online_date, authors, "Comments:" raw field, Subjects, abstract. Then fetch `https://arxiv.org/html/<id>` (try v1/v2/v3/v4 suffixes) and grep for "limitation", "future work", "open problem", "remains open", "we leave", "caveat", "failure" and QUOTE VERBATIM.

Confirmed-existing IDs to cover (found via arXiv API listing):
- 2603.19312 (LeWorldModel) — READ CAREFULLY, this is the anchor
- 2605.09241 (Sub-JEPA: Subspace Gaussian Regularization for Stable End-to-End World Models)
- 2606.26217 (Fast LeWorldModel)
- 2607.12547 (Mind the Gap: Promises and Pitfalls of Hierarchical Planning in LeWorldModel)
- 2607.26924 (Temporally Centered SIGReg Improves LeWorldModel Representations for Robot Policy Learning)
- 2608.10145 (The Evaluation Protocol Determines the Result: An Independent Reproduction of LeWorldModel on TwoRoom)
- 2608.17542 (No Gaussian Required: Contrastive Inverse Dynamics for JEPA World Models)
- 2606.02572 (VISReg: Variance-Invariance-Sketching Regularization for JEPA training)
- 2606.01443 (UR-JEPA: Uniform Rectifiability as a Regularizer for Joint-Embedding Predictive Architectures)
- 2606.17603 (Expanding SPHERE-JEPA: A Family of Statistical Regularizers for the Hypersphere)
- 2607.28415 (QQWorld: Quantile-Quantile Matching for World Model Regularization)
- 2608.27395 (LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics)
- 2607.15295 (AV-JEPA: Extending LeJEPA to Audio-Visual Self-Supervised Learning)
- 2608.16287 (SCALE: State-Calibrated Latent Embeddings for JEPA Planning in the Right Geometry)

Pay SPECIAL attention to papers that CRITIQUE: 2608.10145 (independent reproduction — does it fail to reproduce?), 2608.17542 ("No Gaussian Required" — direct challenge to the isotropic-Gaussian premise), 2605.09241 (argues isotropic Gaussian in ambient space is too strong a bias), 2607.12547 (pitfalls). Extract their exact criticisms and any quantitative disagreement.

ALSO run additional searches for anything else citing/extending LeJEPA or SIGReg in 2026, e.g.:
  https://export.arxiv.org/api/query?search_query=all:%22SIGReg%22&start=0&max_results=100&sortBy=submittedDate&sortOrder=descending
  https://export.arxiv.org/api/query?search_query=all:%22LeJEPA%22&start=0&max_results=100&sortBy=submittedDate&sortOrder=descending
  https://export.arxiv.org/api/query?search_query=all:%22isotropic%20Gaussian%22+AND+all:%22JEPA%22&start=0&max_results=60&sortBy=submittedDate&sortOrder=descending

RULES: Never invent arXiv IDs. Label each claim VERIFIED (you curled it) / UNVERIFIED / COULD NOT CONFIRM. If a fetch fails, say so explicitly.

Return a dense structured report in your final message. Do NOT write any files. Per paper: id, URL, v1 date, institution, Comments raw, one-liner, relevance, and verbatim limitation/critique quotes.

### ASSISTANT — 2026-09-16 17:05

I'll start by verifying the anchor paper and running the search queries in parallel.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; curl -s \"https://export.arxiv.org/api/query?search_query=all:%22SIGReg%22&start=0&max_results=100&sortBy=submittedDate&sortOrder=descending\" | grep -E \"<id>|<title>|<published>\" | head -150", "description": "Search arXiv for SIGReg"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; curl -s \"https://export.arxiv.org/api/query?search_query=all:%22LeJEPA%22&start=0&max_results=100&sortBy=submittedDate&sortOrder=descending\" | grep -E \"<id>|<title>|<published>\" | head -200", "description": "Search arXiv for LeJEPA"}
```

<details><summary>tool result</summary>

```
  <id>https://arxiv.org/api/vsx3rRzn8j6/1SNCx9Vuj3y2PHU</id>
  <title>arXiv Query: search_query=all:"SIGReg"&amp;id_list=&amp;start=0&amp;max_results=100</title>
    <id>http://arxiv.org/abs/2609.04264v2</id>
    <title>Spectral-Target Physical Latent Structuring for JEPA-Style World Models</title>
    <published>2026-09-02T18:05:06Z</published>
    <id>http://arxiv.org/abs/2609.04261v1</id>
    <title>Self-Supervised Pretraining of Molecular Graph Encoders with LeJEPA</title>
    <published>2026-09-02T12:08:23Z</published>
    <id>http://arxiv.org/abs/2608.29426v1</id>
    <title>Polis: 3D Self-Supervision at City Scale</title>
    <published>2026-08-29T20:07:51Z</published>
    <id>http://arxiv.org/abs/2608.27395v1</id>
    <title>LeVJEPA: Efficient &amp; Scalable Video Pretraining without the Heuristics</title>
    <published>2026-08-27T17:26:24Z</published>
    <id>http://arxiv.org/abs/2608.27367v4</id>
    <title>Successive Capacity Growth: Task-Complexity-Driven Width and Depth Expansion for Vision Transformer Encoders in JEPA World Models</title>
    <published>2026-08-27T17:04:57Z</published>
    <id>http://arxiv.org/abs/2608.25244v1</id>
    <title>AllMusicCaps: Album Reviews as Complementary Supervision for Music CLAP</title>
    <published>2026-08-26T00:06:03Z</published>
    <id>http://arxiv.org/abs/2608.17542v1</id>
    <title>No Gaussian Required: Contrastive Inverse Dynamics for JEPA World Models</title>
    <published>2026-08-18T09:03:35Z</published>
    <id>http://arxiv.org/abs/2608.16287v2</id>
    <title>SCALE: State-Calibrated Latent Embeddings for JEPA Planning in the Right Geometry</title>
    <published>2026-08-17T08:58:55Z</published>
    <id>http://arxiv.org/abs/2608.11174v2</id>
    <title>VIScore: Diagnosing Planning-Relevant Quality in Latent World Models</title>
    <published>2026-08-11T17:34:10Z</published>
    <id>http://arxiv.org/abs/2608.28625v1</id>
    <title>Asymmetric Within-Document Predictive Learning for Scientific Document R
... [truncated 4879 chars]
```
</details>

<details><summary>tool result</summary>

```
  <id>https://arxiv.org/api/pdTJQ6VpzQMELowZB1s9u5YlpQI</id>
  <title>arXiv Query: search_query=all:"LeJEPA"&amp;id_list=&amp;start=0&amp;max_results=100</title>
    <id>http://arxiv.org/abs/2609.04325v1</id>
    <title>The microscope is the mask: privileged views and labels from a cryo-ET forward model</title>
    <published>2026-09-03T18:00:18Z</published>
    <id>http://arxiv.org/abs/2609.04261v1</id>
    <title>Self-Supervised Pretraining of Molecular Graph Encoders with LeJEPA</title>
    <published>2026-09-02T12:08:23Z</published>
    <id>http://arxiv.org/abs/2608.30975v1</id>
    <title>MR-JEPA: A General Purpose Video Foundation Model for Cardiac MRI</title>
    <published>2026-08-31T15:36:24Z</published>
    <id>http://arxiv.org/abs/2608.30594v1</id>
    <title>Learning Radio Astronomical Representations with LeJEPA and Very Small Models</title>
    <published>2026-08-31T11:08:40Z</published>
    <id>http://arxiv.org/abs/2608.27395v1</id>
    <title>LeVJEPA: Efficient &amp; Scalable Video Pretraining without the Heuristics</title>
    <published>2026-08-27T17:26:24Z</published>
    <id>http://arxiv.org/abs/2608.04213v1</id>
    <title>Attention-Only White-Box Transformer via LeJEPA-Based Self-Supervised Pretraining</title>
    <published>2026-08-04T20:26:12Z</published>
    <id>http://arxiv.org/abs/2607.29613v1</id>
    <title>WCM: A World Critic Model for Vision-Language-Action Reinforcement Learning</title>
    <published>2026-07-31T16:48:45Z</published>
    <id>http://arxiv.org/abs/2607.22430v2</id>
    <title>On the Identifiability of Controlled World Models</title>
    <published>2026-07-24T15:49:12Z</published>
    <id>http://arxiv.org/abs/2607.17019v1</id>
    <title>Regularize or Localize: When Training-Time KV-Cache Geometry Pays Under Quantization</title>
    <published>2026-07-19T01:13:09Z</published>
    <id>http://arxiv.org/abs/2607.14537v1</id>
    <title>MIDI-RAE-JEPA: Hierarchical Representation Learning and Generation for Symbolic Music</ti
... [truncated 4686 chars]
```
</details>

### ASSISTANT — 2026-09-16 17:06

Rich results. Let me run the remaining searches plus LeWorldModel-specific ones.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; curl -s \"https://export.arxiv.org/api/query?search_query=all:%22LeWorldModel%22&start=0&max_results=100&sortBy=submittedDate&sortOrder=descending\" | grep -E \"<id>|<title>|<published>\" | head -200", "description": "Search arXiv for LeWorldModel"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; curl -s \"https://export.arxiv.org/api/query?search_query=all:%22isotropic%20Gaussian%22+AND+all:%22JEPA%22&start=0&max_results=60&sortBy=submittedDate&sortOrder=descending\" | grep -E \"<id>|<title>|<published>\" | head -200", "description": "Search arXiv isotropic Gaussian AND JEPA"}
```

<details><summary>tool result</summary>

```
  <id>https://arxiv.org/api/C6C+ppOtvlF2EOFgS180Ios2p0k</id>
  <title>arXiv Query: search_query=all:"LeWorldModel"&amp;id_list=&amp;start=0&amp;max_results=100</title>
    <id>http://arxiv.org/abs/2609.10464v1</id>
    <title>Semigroup-JEPA: Latent Dynamics Consistency for Zero-Shot Physics Generalization</title>
    <published>2026-09-09T17:08:13Z</published>
    <id>http://arxiv.org/abs/2609.03565v1</id>
    <title>Toward Physically Grounded JEPA World Models for Goal-Conditioned Robotic Planning</title>
    <published>2026-09-03T09:11:13Z</published>
    <id>http://arxiv.org/abs/2609.03294v1</id>
    <title>Latent Energy Action Planning with World Models</title>
    <published>2026-09-03T02:41:36Z</published>
    <id>http://arxiv.org/abs/2609.04264v2</id>
    <title>Spectral-Target Physical Latent Structuring for JEPA-Style World Models</title>
    <published>2026-09-02T18:05:06Z</published>
    <id>http://arxiv.org/abs/2608.29998v1</id>
    <title>The Intervention Gap in Latent World Models</title>
    <published>2026-08-30T19:47:51Z</published>
    <id>http://arxiv.org/abs/2608.29029v1</id>
    <title>Flow-JEPA: Flow Matching for Robust Latent Dynamics in JEPA World Models</title>
    <published>2026-08-29T03:40:06Z</published>
    <id>http://arxiv.org/abs/2608.17542v1</id>
    <title>No Gaussian Required: Contrastive Inverse Dynamics for JEPA World Models</title>
    <published>2026-08-18T09:03:35Z</published>
    <id>http://arxiv.org/abs/2608.16287v2</id>
    <title>SCALE: State-Calibrated Latent Embeddings for JEPA Planning in the Right Geometry</title>
    <published>2026-08-17T08:58:55Z</published>
    <id>http://arxiv.org/abs/2608.12959v1</id>
    <title>The Objective Is the Bottleneck: Latent World Models Encode What Their Planners Cannot Use</title>
    <published>2026-08-13T08:43:54Z</published>
    <id>http://arxiv.org/abs/2608.10145v1</id>
    <title>The Evaluation Protocol Determines the Result: An Independent Reproduction of LeWorldModel on TwoRoom
... [truncated 1900 chars]
```
</details>

<details><summary>tool result</summary>

```
  <id>https://arxiv.org/api/5TEH6hYkXQLMxz9NDbHIy9tjRAU</id>
  <title>arXiv Query: search_query=all:"isotropic Gaussian" AND all:"JEPA"&amp;id_list=&amp;start=0&amp;max_results=60</title>
    <id>http://arxiv.org/abs/2608.27367v4</id>
    <title>Successive Capacity Growth: Task-Complexity-Driven Width and Depth Expansion for Vision Transformer Encoders in JEPA World Models</title>
    <published>2026-08-27T17:04:57Z</published>
    <id>http://arxiv.org/abs/2608.24093v1</id>
    <title>Joint-Embedding Prediction of Masked Point Tubes for Self-Supervised Learning on 4D Point Cloud Videos</title>
    <published>2026-08-25T05:38:43Z</published>
    <id>http://arxiv.org/abs/2608.22764v1</id>
    <title>LpWM: A Case for Sparse Representations in World Models</title>
    <published>2026-08-24T03:37:39Z</published>
    <id>http://arxiv.org/abs/2608.17542v1</id>
    <title>No Gaussian Required: Contrastive Inverse Dynamics for JEPA World Models</title>
    <published>2026-08-18T09:03:35Z</published>
    <id>http://arxiv.org/abs/2607.20202v1</id>
    <title>JEPA-CFM: A Joint Embedding Predictive Architecture-based Channel Foundation Model for Robust Fluid Antenna Systems</title>
    <published>2026-07-22T14:24:32Z</published>
    <id>http://arxiv.org/abs/2607.13612v1</id>
    <title>The SIGReg Objective as Variational Free Energy: A Theoretical Active-Inference Account of JEPA World Models</title>
    <published>2026-07-15T08:59:36Z</published>
    <id>http://arxiv.org/abs/2606.02572v1</id>
    <title>VISReg: Variance-Invariance-Sketching Regularization for JEPA training</title>
    <published>2026-06-01T17:58:01Z</published>
    <id>http://arxiv.org/abs/2606.01443v1</id>
    <title>UR-JEPA: Uniform Rectifiability as a Regularizer for Joint-Embedding Predictive Architectures</title>
    <published>2026-05-31T20:26:54Z</published>
    <id>http://arxiv.org/abs/2606.00706v2</id>
    <title>CR-JEPA: Cross-Modal Joint-Embedding Predictive Learning for Remote Sensing Image Retrieva
... [truncated 1578 chars]
```
</details>

### ASSISTANT — 2026-09-16 17:06

Now let me fetch the anchor paper's abs page in full detail.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; mkdir -p /tmp/wog_lit; curl -s \"https://arxiv.org/abs/2603.19312\" -o /tmp/wog_lit/2603.19312.html; echo \"SIZE: $(wc -c < /tmp/wog_lit/2603.19312.html)\"; python3 - <<'EOF'\nimport re,html\nh=open('/tmp/wog_lit/2603.19312.html',encoding='utf-8',errors='replace').read()\nfor name in ['citation_title','citation_date','citation_online_date','citation_author','citation_arxiv_id']:\n    for m in re.finditer(r'<meta[^>]*name=\"%s\"[^>]*content=\"([^\"]*)\"'%name,h):\n        print(name,'=',html.unescape(m.group(1)))\n# comments\nm=re.search(r'<td class=\"tablecell comments[^\"]*\">(.*?)</td>',h,re.S)\nprint('COMMENTS_RAW:',html.unescape(re.sub(r'<[^>]+>','',m.group(1))).strip() if m else 'NONE')\nm=re.search(r'<td class=\"tablecell subjects\">(.*?)</td>',h,re.S)\nprint('SUBJECTS:',html.unescape(re.sub(r'<[^>]+>','',m.group(1))).strip() if m else 'NONE')\nm=re.search(r'<blockquote class=\"abstract[^\"]*\">(.*?)</blockquote>',h,re.S)\nprint('ABSTRACT:',html.unescape(re.sub(r'<[^>]+>','',m.group(1))).strip() if m else 'NONE')\nfor m in re.finditer(r'\\[v(\\d)\\]\\s*</a>\\s*</strong>\\s*<a[^>]*>([^<]*)</a>',h):\n    print('VER',m.group(1),m.group(2))\nm=re.search(r'<div class=\"submission-history\">(.*?)</div>',h,re.S)\nprint('HIST:',re.sub(r'\\s+',' ',html.unescape(re.sub(r'<[^>]+>',' ',m.group(1)))).strip() if m else 'NONE')\nEOF", "description": "Fetch and parse LeWorldModel abs page"}
```

<details><summary>tool result</summary>

```
SIZE: 42448
citation_title = LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels
citation_date = 2026/03/13
citation_online_date = 2026/06/03
citation_author = Maes, Lucas
citation_author = Lidec, Quentin Le
citation_author = Scieur, Damien
citation_author = LeCun, Yann
citation_author = Balestriero, Randall
citation_arxiv_id = 2603.19312
COMMENTS_RAW: NONE
SUBJECTS: Machine Learning (cs.LG); Artificial Intelligence (cs.AI)
ABSTRACT: Abstract:Joint Embedding Predictive Architectures (JEPAs) offer a compelling framework for learning world models in compact latent spaces, yet existing methods remain fragile, relying on complex multi-term losses, exponential moving averages, pre-trained encoders, or auxiliary supervision to avoid representation collapse. In this work, we introduce LeWorldModel (LeWM), the first JEPA that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a regularizer enforcing Gaussian-distributed latent embeddings. This reduces tunable loss hyperparameters from six to one compared to the only existing end-to-end alternative. With ~15M parameters trainable on a single GPU in a few hours, LeWM plans up to 48x faster than foundation-model-based world models while remaining competitive across diverse 2D and 3D control tasks. Beyond control, we show that LeWM's latent space encodes meaningful physical structure through probing of physical quantities. Surprise evaluation confirms that the model reliably detects physically implausible events.
HIST: Submission history From: Lucas Maes [ view email ] [v1] Fri, 13 Mar 2026 19:48:14 UTC (4,454 KB) [v2] Tue, 24 Mar 2026 20:31:23 UTC (4,454 KB) [v3] Wed, 3 Jun 2026 18:50:40 UTC (4,478 KB)
```
</details>

### ASSISTANT — 2026-09-16 17:06

Now let me build a reusable extraction script and fetch the HTML for the anchor.

**tool call `Bash`**
```
{"command": "cat > /tmp/wog_lit/absx.py <<'PYEOF'\nimport re,html,sys,os\ndef clean(s):\n    return re.sub(r'\\s+',' ',html.unescape(re.sub(r'<[^>]+>',' ',s))).strip()\nfor aid in sys.argv[1:]:\n    p='/tmp/wog_lit/%s.html'%aid\n    if not os.path.exists(p) or os.path.getsize(p)<2000:\n        print('='*90); print('ID',aid,'FETCH FAILED / EMPTY'); continue\n    h=open(p,encoding='utf-8',errors='replace').read()\n    print('='*90)\n    print('ID:',aid)\n    def meta(n):\n        return [html.unescape(m.group(1)) for m in re.finditer(r'<meta[^>]*name=\"%s\"[^>]*content=\"([^\"]*)\"'%n,h)]\n    t=meta('citation_title'); print('TITLE:',t[0] if t else 'NONE')\n    d=meta('citation_date'); print('citation_date:',d[0] if d else 'NONE')\n    o=meta('citation_online_date'); print('citation_online_date:',o[0] if o else 'NONE')\n    print('AUTHORS:','; '.join(meta('citation_author')))\n    m=re.search(r'<td class=\"tablecell comments[^\"]*\">(.*?)</td>',h,re.S)\n    print('COMMENTS_RAW:',clean(m.group(1)) if m else '<<NO COMMENTS FIELD>>')\n    m=re.search(r'<td class=\"tablecell subjects\">(.*?)</td>',h,re.S)\n    print('SUBJECTS:',clean(m.group(1)) if m else 'NONE')\n    m=re.search(r'<blockquote class=\"abstract[^\"]*\">(.*?)</blockquote>',h,re.S)\n    print('ABSTRACT:',clean(m.group(1)) if m else 'NONE')\n    m=re.search(r'<div class=\"submission-history\">(.*?)</div>',h,re.S)\n    print('HIST:',clean(m.group(1)) if m else 'NONE')\nPYEOF\necho ok"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; cd /tmp/wog_lit; for v in v3 v2 v1 \"\"; do code=$(curl -s -o /tmp/wog_lit/h_2603.19312$v.html -w \"%{http_code}\" \"https://arxiv.org/html/2603.19312$v\"); echo \"$v -> $code size $(wc -c < /tmp/wog_lit/h_2603.19312$v.html)\"; done", "description": "Fetch LeWorldModel HTML versions"}
```

<details><summary>tool result</summary>

```
ok
```
</details>

<details><summary>tool result</summary>

```
v3 -> 200 size 442377
v2 -> 200 size 441320
v1 -> 200 size 440278
 -> 200 size 442377
```
</details>

### ASSISTANT — 2026-09-16 17:07

HTML available. Let me build a text extractor and grep the anchor paper.

**tool call `Bash`**
```
{"command": "cat > /tmp/wog_lit/txt.py <<'PYEOF'\nimport re,html,sys\np=sys.argv[1]\nh=open(p,encoding='utf-8',errors='replace').read()\nh=re.sub(r'<script.*?</script>','',h,flags=re.S)\nh=re.sub(r'<style.*?</style>','',h,flags=re.S)\nh=re.sub(r'<math[^>]*alttext=\"([^\"]*)\"[^>]*>.*?</math>',lambda m:html.unescape(m.group(1)),h,flags=re.S)\nh=re.sub(r'</(p|div|li|h1|h2|h3|h4|section|td|tr|blockquote)>','\\n',h)\nh=re.sub(r'<br[^>]*>','\\n',h)\nt=html.unescape(re.sub(r'<[^>]+>','',h))\nt=re.sub(r'[ \\t]+',' ',t)\nt=re.sub(r'\\n\\s*\\n+','\\n',t)\nprint(t)\nPYEOF\npython3 /tmp/wog_lit/txt.py /tmp/wog_lit/h_2603.19312v3.html > /tmp/wog_lit/t_2603.19312.txt; wc -l /tmp/wog_lit/t_2603.19312.txt\ngrep -n -i -E \"limitation|future work|open problem|remains open|we leave|caveat|failure|fail to|shortcoming|do not (yet|address)|does not (yet|address)\" /tmp/wog_lit/t_2603.19312.txt | head -60", "description": "Extract text and grep limitations in LeWorldModel"}
```

<details><summary>tool result</summary>

```
1074 /tmp/wog_lit/t_2603.19312.txt
43:Limitations & Future Work.
96:However, despite their conceptual simplicity, existing JEPA methods are highly prone to collapse. In this failure mode, the model maps all inputs to nearly identical representations to trivially satisfy the temporal prediction objective leading to unusable representations. Preventing collapse is therefore one of the central challenges in training JEPA models. Many influential works have proposed methods to address this issue. Yet, these approaches typically rely on heuristic regularization, multi-objective loss functions, external sources of information, or architectural simplifications such as pre-trained encoders. In practice, these strategies often introduce additional instability or significantly increase training complexity (see App. C).
97:To overcome these limitations, we propose LeWorldModel (LeWM), the first method to learn a stable JEPA end-to-end from raw pixels without heuristics, principled, and simple (cf. Fig 1). We evaluate LeWM across a diverse set of manipulation, navigation, and locomotion tasks in both 2D and 3D environments. In addition, we probe its intuitive physical understanding through targeted probing and surprise-quantification evaluations in latent space. Overall, our key findings and contributions are:
111:JEPA is a framework for learning world models that predict the dynamic evolution of a system in a compact, low-dimensional latent space. Since their introduction by LeCun [33], JEPA methods have evolved considerably, differing mainly in their target tasks and in the strategies used to learn non-collapsing representations. One prominent line of work applies JEPA to self-supervised representation learning by predicting the latent embeddings of masked input patches. Examples include I-JEPA [2] for images, V-JEPA [8, 3] for videos, and Echo-JEPA and Brain-JEPA [14, 38] for medical data. These approaches typically employ an exponential moving average (EMA) of the target en
... [truncated 3444 chars]
```
</details>

### ASSISTANT — 2026-09-16 17:08

Excellent — the anchor's limitations section is exactly on point. Let me get institutions and key numbers.

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; sed -n '1,60p' t_2603.19312.txt", "description": "Show header of LeWorldModel text"}
```

<details><summary>tool result</summary>

```
LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels
 Report GitHub Issue
 ×
 Title:
 Content selection saved. Describe the issue below:
 Description:
 Submit without GitHub
 Submit in GitHub
 arXiv is now an independent nonprofit!
 Learn more
 ×
 Back to arXiv
 Why HTML?
 Report Issue
 Back to Abstract
 Download PDF
Abstract
1 Introduction
2 Related Work
3 Method: LeWorldModel
3.1 Learning the Latent World Model
Offline Dataset.
Model Architecture.
Training Objective.
3.2 Latent Planning
4 Latent Planning Performance
4.1 Planning evaluation setup
Environments.
Baselines.
4.2 Towards Efficient Planning with WMs
4.3 Towards Stable Training of World Models
Ablations.
Training Curves.
5 Quantifying Physical Understanding in LeWM
5.1 Physical Structure of the Latent Space
Probing physical quantities.
Decoding Latent Space.
Visualizing Latent Space.
Temporal Latent Path Straightening.
5.2 Violation-of-expectation Framework
6 Conclusion
Limitations & Future Work.
References
A SIGReg
B Cross-Entropy Method
C Baselines
C.1 DINO-WM
C.2 PLDM
C.3 GC-RL
GCIQL
GCIVL
Policy extraction.
C.4 GCBC
D Implementation details
Encoder Architecture.
Predictor Architecture.
Decoder (Visualization Only).
Planning solver.
E Environment & Dataset
```
</details>

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; grep -n -i -E \"Two-Room|TwoRoom|success rate|PushT|Wall|success\" t_2603.19312.txt | head -40", "description": "Grep TwoRoom and success rates"}
```

<details><summary>tool result</summary>

```
65:TwoRoom.
66:PushT.
94:World Models (WMs) are a powerful family of methods [21] that learn to predict the consequences of actions in the environment. When successful, WMs allows agents to plan and to improve themselves solely form their model of the world, i.e., in imagination space. This is particularly valuable in the offline setting, where agents must learn from fixed datasets without environment interaction—leveraging the model to generate synthetic experience and evaluate counterfactual action sequences [36, 25].
109:Generative world models have been successfully applied to simulate existing game-like environments. For example, IRIS [36], DIAMOND [1], \Delta-IRIS [37], OASIS [13], and DreamerV4 [25] model environments such as Minecraft, Counter-Strike, and Crafter, improving policy sample efficiency in reinforcement learning. Other methods generate entirely new interactive simulators, e.g., Genie [11] and HunyuanWorld [29], while learned simulators have also been applied to robot policy evaluation [44].
160:Figure 5: Environments used for evaluation. Left: Push-T, a 2D manipulation task where the agent must push a block toward a target configuration, commonly used as a robotics benchmark. Center (1): OGBench-Cube, a visually richer 3D manipulation environment where a robotic arm interacts with a cube to reach a target position. Center (2): Two-Room, a simple 2D navigation environment where an agent moves between rooms to reach target positions. Right: Reacher, a task where a 2-joint arm needs to reach a target configuration in a 2D plane. All environments have a continuous action space. More details on environment and datasets are available in appendix E.
167:We report planning performance in Fig. 6. LeWM outperforms PLDM on the more challenging planning tasks, achieving an 18% higher success rate on PushT, while remaining competitive with DINO-WM. Notably, on PushT, LeWM (pixels-only) surpasses DINO-WM even when DINO-WM has access to additional proprioceptiv
... [truncated 14281 chars]
```
</details>

### ASSISTANT — 2026-09-16 17:08

The anchor is well-characterized. Now let me batch-fetch all the abs pages.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; cd /tmp/wog_lit\nIDS=\"2605.09241 2606.26217 2607.12547 2607.26924 2608.10145 2608.17542 2606.02572 2606.01443 2606.17603 2607.28415 2608.27395 2607.15295 2608.16287\"\nfor id in $IDS; do curl -s \"https://arxiv.org/abs/$id\" -o $id.html; printf \"%s %s\\n\" \"$id\" \"$(wc -c < $id.html)\"; sleep 1; done", "description": "Fetch assigned abs pages batch 1"}
```

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; python3 - <<'EOF'\nimport re,html\nh=open('h_2603.19312v3.html',encoding='utf-8',errors='replace').read()\nm=re.search(r'(?s)<div class=\"ltx_authors\">(.*?)</div>',h)\nif m: print(re.sub(r'\\s+',' ',html.unescape(re.sub(r'<[^>]+>',' ',m.group(1)))).strip())\nelse:\n    i=h.find('ltx_personname')\n    print(re.sub(r'\\s+',' ',html.unescape(re.sub(r'<[^>]+>',' ',h[i-200:i+2500]))))\nEOF", "description": "Extract LeWM author affiliations"}
```

<details><summary>tool result</summary>

```
2605.09241 44453
2606.26217 42324
2607.12547 42639
2607.26924 44331
2608.10145 44208
2608.17542 44218
2606.02572 41761
2606.01443 43992
2606.17603 44001
2607.28415 41931
2608.27395 44128
2607.15295 40113
2608.16287 43842
```
</details>

<details><summary>tool result</summary>

```
Lucas Maes* Quentin Le Lidec* Damien Scieur Yann LeCun Randall Balestriero Affiliation: Mila & Université de Montréal New York University Samsung SAIL Brown University
```
</details>

### ASSISTANT — 2026-09-16 17:09

All fetched. Now let me parse all abs pages.

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; python3 absx.py 2608.10145 2608.17542 2605.09241 2607.12547 2>&1", "description": "Parse the four critique papers"}
```

<details><summary>tool result</summary>

```
==========================================================================================
ID: 2608.10145
TITLE: The Evaluation Protocol Determines the Result: An Independent Reproduction of LeWorldModel on TwoRoom
citation_date: 2026/08/10
citation_online_date: 2026/08/10
AUTHORS: Singh, Joyjeet
COMMENTS_RAW: Independent reproduction of arXiv:2603.19312 - this https URL
SUBJECTS: Machine Learning (cs.LG)
ABSTRACT: Abstract: LeWorldModel trains a latent world model with a prediction loss and a single anti-collapse regulariser, and reports approximately 87% of goals reached on TwoRoom, its simplest diagnostic environment. We reproduce that result by independent reimplementation on roughly $25 of rented compute, with all evaluation on one laptop CPU. We reach 94.0% at the repository's evaluation goal offset, against 84.0% for the authors' own released checkpoint measured under our protocol on identical episodes, and we reproduce the reported representation result directly (position probe Pearson r = 0.9988 against a reported 0.996). Reaching that point required four conventions that determine the outcome and appear in no released configuration file: dense action gathering across a frameskip block, a programmatically-set action-encoder width, ImageNet pixel normalisation, and action z-scoring. A reproducer following the released configurations alone obtains a model whose predictor cannot converge. The evaluation protocol is itself contested by the released material. The paper's appendix and the repository's configuration specify different goal offsets and step budgets; on the authors' own weights these yield 14.0% and 84.0%, and only the configuration's values reproduce the reported figure. On fifty identical episodes, changing nothing but how the goal is constructed moves that checkpoint from 84.0% to 8.0%. Two findings generalise. One-step prediction accuracy does not predict long-horizon planning success: across three checkpoints spanning a sevenfold range in predic
... [truncated 6741 chars]
```
</details>

### ASSISTANT — 2026-09-16 17:09

These are the core critiques. Now let me parse the remaining papers and fetch the critique HTMLs in parallel.

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; python3 absx.py 2606.26217 2607.26924 2606.02572 2606.01443 2606.17603 2607.28415 2608.27395 2607.15295 2608.16287 2>&1", "description": "Parse remaining assigned papers"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; cd /tmp/wog_lit\nfor id in 2608.10145 2608.17542 2605.09241 2607.12547; do\n  ok=\"\"\n  for v in v3 v2 v1; do\n    code=$(curl -s -o hh_$id$v.html -w \"%{http_code}\" \"https://arxiv.org/html/$id$v\")\n    sz=$(wc -c < hh_$id$v.html)\n    echo \"$id$v -> $code $sz\"\n    if [ \"$code\" = \"200\" ] && [ \"$sz\" -gt 20000 ]; then cp hh_$id$v.html best_$id.html; ok=1; break; fi\n  done\n  [ -z \"$ok\" ] && echo \"!!! $id NO HTML\"\n  sleep 1\ndone; ls -la best_*.html", "description": "Fetch HTML for critique papers"}
```

<details><summary>tool result</summary>

```
==========================================================================================
ID: 2606.26217
TITLE: Fast LeWorldModel
citation_date: 2026/06/24
citation_online_date: 2026/06/24
AUTHORS: Gao, Yuntian; Xu, Xiangyu
COMMENTS_RAW: <<NO COMMENTS FIELD>>
SUBJECTS: Machine Learning (cs.LG) ; Computer Vision and Pattern Recognition (cs.CV); Robotics (cs.RO)
ABSTRACT: Abstract: Joint-Embedding Predictive Architectures (JEPAs), including recent LeWorldModel (LeWM), have become a promising foundation for reconstruction-free visual world models. For visual planning, however, LeWM evaluates candidate action sequences by repeatedly applying a local one-step latent transition model. This autoregressive rollout makes planning computationally expensive and exposes the predicted trajectory to accumulated latent errors as the horizon grows. We propose Fast LeWorldModel (Fast-LeWM), a fast latent world model that replaces repeated local rollout with action-prefix prediction. Given the current latent and a candidate action sequence, Fast-LeWM encodes its prefixes and predicts the future latents reached after executing those prefixes in parallel. By making action prefixes the basic prediction unit, Fast-LeWM directly models action effects accumulated to different extents over multiple horizons. This prefix-level supervision forces the model to learn how states continuously evolve under different action prefixes, rather than only fitting one-step state transitions. During planning, the predictor can use the last prefix token from the encoded action sequence to evaluate the corresponding future latent without explicitly rolling through each intermediate imagined state. Across multiple tasks, Fast-LeWM improves average success over LeWM while substantially reducing planning time, achieving lower open-loop latent loss whose growth becomes significantly slower as the rollout horizon increases.
HIST: Submission history From: Yuntian Gao [ view email ] [v1] Wed, 24 Jun 2026 17:59:59
... [truncated 16729 chars]
```
</details>

<details><summary>tool result</summary>

```
2608.10145v3 -> 404 7715
2608.10145v2 -> 404 7715
2608.10145v1 -> 200 372897
2608.17542v3 -> 404 7715
2608.17542v2 -> 404 7715
2608.17542v1 -> 200 245516
2605.09241v3 -> 404 7715
2605.09241v2 -> 404 7715
2605.09241v1 -> 200 316879
2607.12547v3 -> 404 7715
2607.12547v2 -> 200 312213
-rw-rw-r-- 1 zhengguantian zhengguantian 316879 Sep 16 09:09 best_2605.09241.html
-rw-rw-r-- 1 zhengguantian zhengguantian 312213 Sep 16 09:09 best_2607.12547.html
-rw-rw-r-- 1 zhengguantian zhengguantian 372897 Sep 16 09:09 best_2608.10145.html
-rw-rw-r-- 1 zhengguantian zhengguantian 245516 Sep 16 09:09 best_2608.17542.html
```
</details>

### ASSISTANT — 2026-09-16 17:09

All four critique HTMLs available. Let me deep-dive the reproduction paper first.

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; for id in 2608.10145 2608.17542 2605.09241 2607.12547; do python3 txt.py best_$id.html > T_$id.txt; done; wc -l T_*.txt\necho \"===== 2608.10145 GREP =====\"\ngrep -n -i -E \"limitation|future work|open problem|remains open|we leave|caveat|failure|fail(s|ed)? to|do(es)? not reproduce|discrepanc|cannot|contested\" T_2608.10145.txt | head -70", "description": "Grep reproduction paper for limitations"}
```

<details><summary>tool result</summary>

```
  1080 T_2605.09241.txt
   752 T_2607.12547.txt
  1813 T_2608.10145.txt
  1309 T_2608.17542.txt
  4954 total
===== 2608.10145 GREP =====
43:Why the long-horizon reading cannot be followed as written
45:An earlier failure, recorded
64:7 Limitations
88:configurations alone obtains a model whose predictor cannot converge.
89:The evaluation protocol is itself contested by the released material, and we
99:authors’ own — accuracy orders short-horizon success monotonically and fails to
126:The reproduction succeeded, but not on the first attempt, and the failures along
133:finding it. This paper reports the reproduction and the failures together,
134:because the failures are what a reproducer attempting the same work would most
151:show that the appendix’s protocol cannot be followed as written on the released
161:configurations alone obtains a model whose predictor cannot converge, and we
167:short-horizon planning success monotonically and fails to order long-horizon
171:We document two measurement failures that a probe would not catch (§4.3,
194:given in Table 1, our limitations in §7, and all code, configurations,
208:We report one methodological caveat that bears on any comparison of probe
690:are present in the clone and compile there. This catches the classic failure of
702:fail loudly are worth more than gates that are correct: of the gate failures we
770:accessible in its inputs. The failure to converge documented below is
771:therefore not an information-theoretic limitation of the representation. It is a
877:does not reproduce the paper’s reported result on the paper’s own released
888:The failure mode differs as well as the rate. Under the recorded-target
893:Why the long-horizon reading cannot be followed as written
905:This argument holds under the reading that the offset is exact. stable-worldmodel describes the offline protocol as constraining the maximum number of steps separating start and goal rather than fixing it, and under that reading all 10,000 epis
... [truncated 3173 chars]
```
</details>

### ASSISTANT — 2026-09-16 17:10

This is a substantive critique. Let me extract the key sections verbatim.

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; sed -n '1622,1720p' T_2608.10145.txt", "description": "Read limitations section of reproduction paper"}
```

<details><summary>tool result</summary>

```
7 Limitations
We list the constraints on our results in roughly descending order of how much
they should change a reader’s confidence.
A single seed throughout. Every training run, every planning evaluation and
every probe in this paper uses one seed. We make no estimate of seed variance,
and none of our differences should be read as robust to reinitialisation. This
matters most for the planning figures in §4.5, where the differences we report
between checkpoints (94.0% against 84.0% at goal offset 25, p = 0.0625; 94.0% against 78.0%,
p = 0.0574) are already not established at our sample size of fifty episodes.
Our evaluations are, however, deterministic: re-running a planning evaluation
from the committed commit reproduces its per-episode outcomes exactly, so the
variance we have not measured is between-seed and not within-run.
Ten epochs, not one hundred. We follow the paper’s appendix, which states
ten epochs; the released repository configuration specifies max_epochs: 100
(Table 1). Our convergence result (§4.4) is therefore convergence within the
budget the paper states, and says nothing about the asymptote. At four rented
GPU-runs of roughly six dollars each, a hundred-epoch run was outside our
budget, and we note that this constraint is itself a reproducibility finding: a
reader with our resources cannot test the repository’s own configuration.
Our results characterise our checkpoints, not the method. This is the
strongest lesson of §5.2. A same-room planning advantage that was pre-registered,
distance-matched, verified reachable by an oracle, and significant at
p = 3.4 \times 10-8 on one checkpoint fell to +12.7 points under a change of action
scaling and to -6.4 points on a different checkpoint. Effect sizes measured on a
single reproduction checkpoint should not be read as properties of the method,
and we no longer read our own that way. The latent-geometry measurement of §5.5 in particular was made on one
checkpoint only, before the pipeline correction, a
... [truncated 5949 chars]
```
</details>

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; sed -n '860,960p' T_2608.10145.txt", "description": "Read discrepancy section of reproduction paper"}
```

<details><summary>tool result</summary>

```
the episode’s recorded target, episodes the data policy solved
150
7/50 = 14.0%
the episode’s recorded target, all episodes
150
4/50 = 8.0%
Every figure in this table was read from a committed evaluation report; the
report path for each is listed in docs/paper/results_from_disk.csv. The fifth
row is discussed below: on this dataset that instruction has no effect at an
offset of 100, and the row is identical to the fourth for that reason.
Only the first reproduces the reported figure. Under it the checkpoint reaches
42 of 50 goals; a one-sample test against 0.87 gives p = 0.53, and the 95%
Wilson interval, [71.5%, 91.7%], contains the reported figure. A random-action
control over the same episodes reaches 9 of 50. Under the protocol the paper
describes, the same weights reach 14.0%.
The consequence is worth stating plainly, because it is the kind of thing a
reproduction exists to find: the evaluation protocol described in the paper
does not reproduce the paper’s reported result on the paper’s own released
weights, and the released repository’s evaluation defaults do. We do not know
which the authors used, and we do not assume; §6.4 records that we asked.
The goal construction, not the model, determines the number
The last row of the table draws the same fifty episodes as the first. Neither
imposes a minimum episode length, and both sample with the same seed, so the
two runs are a paired comparison on identical episodes, with identical weights,
an identical planner and an identical driving convention. They differ in how the goal is constructed and, because each protocol carries its own step budget, in budget: 50 steps for the first and 150 for the second. That difference runs against the comparison rather than for it — the recorded-target arm was given 3 times as many steps and still collapsed. The result is 84.0% against 8.0%: 39 episodes are
solved under the first and missed under the second, one is solved under the
second alone, and a matched-pair test gives \chi^{
... [truncated 6496 chars]
```
</details>

### ASSISTANT — 2026-09-16 17:10

Now the "No Gaussian Required" direct challenge.

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; grep -n -i -E \"limitation|future work|open problem|remains open|we leave|caveat|failure|bottleneck|prescribed geometry|too strong|mismatch|Gaussian.*(prior|assumption|premise)\" T_2608.17542.txt | head -60", "description": "Grep No Gaussian Required"}
```

<details><summary>tool result</summary>

```
48:7 Limitations
87:results are consistent with the prescribed geometry becoming a bottleneck:
159:geometry becoming a bottleneck. On OGBench Visual Scene, where a single arm controls a drawer,
175:weakly controlled object state, a limitation we analyze in Section 4.5.
303:Action-NCE keeps the same training-only inverse head but changes the failure
305:constant latents. Section 4.4 analyzes this failure mode.
343:drawer/window/button/cube scene. These tasks stress different failure modes: low
349:described above. Both MTM-MSE and AC-MTM inverse tasks prevent this failure while preserving the
532:family-level failure mode for MTM-MSE and AC-MTM
572:changes the failure geometry: even when absolute action regression is
649:prediction error, and does so on essentially every clip. Two caveats guide the
741:7 Limitations
775:exposes a real limitation of inverse-action signals when task-relevant state is
777:our results are consistent with prescribed geometry becoming a bottleneck: on
961:typo; issue #67 documents the same mismatch and low success under 100/150.
972:300/30/30, with the same CEM caveat as Reacher from issue #41.
1026:failure, not trivial collapse.
```
</details>

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; sed -n '741,800p' T_2608.17542.txt", "description": "Read Limitations of No Gaussian Required"}
```

<details><summary>tool result</summary>

```
7 Limitations
AC-MTM provides no unconditional geometric guarantee. It requires
variation in the action candidates and useful visual evidence about action
effects; unobserved actuators, stochastic dynamics, no-op-heavy datasets, or
many duplicate actions can weaken the contrastive task. We evaluate normalized
continuous controls only. Discrete, hybrid, structured, and very
high-dimensional action spaces may require different scores, learned action
embeddings, or hard-negative sampling. The loss also introduces training-time
dependence on batch composition, the temperature \tau, and one coefficient
\lambda; although one setting works across our suite, broader scaling
evidence is needed. PushT shows the flip side of the dynamics-native pressure:
a variable that actions barely move can be underweighted even when it matters
for the task, and exploratory single-seed runs on additional OGBench families
(Appendix D) suggest the same trade-off on
combinatorial puzzle tasks.
AC-MTM improves several goal-conditioned planning protocols and shows
strong latent surprise on counterfactual-action and state-discontinuity probes,
but these diagnostics remain internal to the trained latent dynamics. They do
not replace public violation-of-expectation suites with human-designed physical
events or prove robust out-of-distribution physical reasoning. OGBench Visual
Scene is also evaluated under our trajectory-goal protocol rather than the
official five-task fixed-goal OGBench leaderboard protocol, so its 80% success
rate should be read as a matched stress test, not a public benchmark score;
under the official 750-step fixed-goal protocol, neither method solves the
tasks at this model scale (Appendix B).
8 Conclusion
A world model’s representation has to be protected from collapse, but nothing
about the problem says that protection must take the form of a prescribed
latent distribution. AC-MTM replaces LeWM’s Gaussian matching with a
contrastive, dynamics-native signal: keep whatever inf
... [truncated 1633 chars]
```
</details>

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; sed -n '140,185p' T_2608.17542.txt; echo \"-----555-660-----\"; sed -n '555,660p' T_2608.17542.txt", "description": "Read intro/results of No Gaussian Required"}
```

<details><summary>tool result</summary>

```
A constant encoder maps every transition to the same query, so action
identification cannot beat chance (Section 3): collapse turns
the forward objective’s global optimum into a representation that cannot
outperform chance on the inverse task. The resulting
anti-collapse pressure is dynamics-native; the encoder is forced to keep
exactly the information that distinguishes the effects of actions. The
entire inverse branch is discarded after training, leaving the deployed model
identical to LeWM. We evaluate AC-MTM as a controlled LeWM
modification against SIGReg, a non-contrastive inverse-regression ablation
(MTM-MSE), and a contrastive forward-prediction alternative (AC-CPC). Three
findings emerge:
(1) Dynamics-derived anti-collapse works, with no distributional prior
and no test-time change. Trained from scratch under LeWM’s one-stage recipe,
AC-MTM never collapses across tasks and seeds and matches SIGReg on
the standard four-task suite (TwoRoom, Reacher, PushT, OGBench-Cube), winning
two tasks, tying one, and losing PushT. The planner never calls the inverse
head: evaluation uses the same encoder, autoregressive predictor, latent
distance, and CEM planner as LeWM.
(2) On the harder Scene task, results are consistent with the prescribed
geometry becoming a bottleneck. On OGBench Visual Scene, where a single arm controls a drawer,
window, buttons, and a movable cube, SIGReg drops to a three-seed mean of
58.0% under the matched trajectory-goal MPC protocol, while AC-MTM
remains at 80.0%, with gains of 24, 20, and 22 points across the three training
seeds. A single 50-episode random-policy run gives a 52% baseline estimate.
The inverse-regression ablation also
clears SIGReg decisively (75.3%), so the advantage belongs to the
dynamics-derived signal itself, not to one loss form. Both models avoid
collapse; the result may combine more useful transition geometry with more
accurate short-horizon latent dynamics. Appendix B
audits this result.
(3) The contrastive form of t
... [truncated 6975 chars]
```
</details>

### ASSISTANT — 2026-09-16 17:10

Now Sub-JEPA and Mind the Gap.

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; grep -n -i -E \"limitation|future work|open problem|remains open|we leave|caveat|failure|too strong|overly strong|manifold|bias-variance|ambient\" T_2605.09241.txt | head -50", "description": "Grep Sub-JEPA"}
```

<details><summary>tool result</summary>

```
58:However, JEPA training is subject to a bias-variance tradeoff.
63:However, latent representations inherently lie on low-dimensional manifolds
64:within a high-dimensional ambient space, and enforcing an isotropic Gaussian
65:prior directly in this ambient space introduces an overly strong bias.
67:point on the bias-variance frontier by applying Gaussian constraints in
102:subject to a bias-variance tradeoff Jing et al. (2022); Sobal et al. (2022):
116:typically lie on low-dimensional manifolds embedded within the
117:high-dimensional ambient space Bengio et al. (2013); Tenenbaum et al. (2000),
118:and enforcing an isotropic Gaussian prior directly in this ambient space
119:introduces an overly strong bias that mismatches the intrinsic geometry
122:point on the bias-variance frontier by moving Gaussian regularization
123:from the ambient space into low-dimensional subspaces.
135:from the ambient embedding space to multiple low-dimensional
167:Instead of enforcing a single global isotropic prior in the ambient space, the regularization is imposed independently on multiple low-dimensional projected views, yielding a more flexible inductive bias that better matches the intrinsic structure of the underlying dynamics.
283:in the ambient embedding space.
397:task-relevant features lie on a low-dimensional manifold, the full-space
405:and to understand the bias-variance tradeoff in subspace design.
502:signal grows unreliable. This reveals a bias-variance tradeoff in
725:from the ambient embedding space
730:reside on a low-dimensional manifold.
1024:Uniform manifold approximation and projection.
```
</details>

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; grep -n -i -E \"limitation|future work|open problem|remains open|we leave|caveat|failure|pitfall|bottleneck|mismatch|does not automatically\" T_2607.12547.txt | head -50", "description": "Grep Mind the Gap"}
```

<details><summary>tool result</summary>

```
2:Mind the Gap: Promises and Pitfalls of Hierarchical Planning in LeWorldModel
28:3.2 Failure Analysis: Subgoal Quality and Planner Exploitation
33:Limitations and Future Work.
59:Mind the Gap: Promises and Pitfalls of Hierarchical Planning in LeWorldModel
74:Effective long-horizon planning using latent world models remains a formidable challenge, and temporal abstraction is a promising direction for addressing it. We investigate whether this abstraction can improve LeWorldModel on long-horizon, goal-conditioned control tasks. To this end, we introduce Hi-LeWM, an extension that freezes the pretrained low-level LeWM and adds a high-level predictor that plans over latent subgoals. We evaluate Hi-LeWM on PushT and OGBCube across increasing goal offsets; our results suggest that hierarchy does not automatically improve performance. At short horizons, the best configuration benefits from a one-step high-level plan, while longer horizons reveal a mismatch between the learned macro-action space and the inference-time distribution, which we trace to the quality of CEM-selected subgoals as the main bottleneck. This is because unconstrained search can select latent macro-actions that appear favorable under the learned model but yield poor control targets. We find that constraining the search around macro-actions encoded from training trajectories, combined with appropriate subgoal execution timing, recovers effective hierarchical regimes, improving over flat LeWM by +11.3\% at medium-range horizons and +14.7\% at the longest PushT horizon. Overall, temporal abstraction can benefit long-horizon planning in LeWM, but only when paired with a constrained high-level search space.
77:Temporal hierarchies are a natural candidate for addressing this limitation (Gumbsch et al., 2024; Hafner et al., 2022). We therefore ask whether temporal abstraction can be added to LeWM as a minimal extension, without departing from its simple JEPA-style latent prediction and CEM-based planning fram
... [truncated 7657 chars]
```
</details>

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; sed -n '380,415p' T_2605.09241.txt; echo \"----490-520----\"; sed -n '490,515p' T_2605.09241.txt; echo \"----results----\"; grep -n -E \"outperform|success rate|[0-9]+\\.[0-9]% |Table 1\" T_2605.09241.txt | head -25", "description": "Read Sub-JEPA analysis and results"}
```

<details><summary>tool result</summary>

```
40
80
Effective RankSub-JEPALeWMTwo-RoomOGB-CubePushTReacher
60
80
100
Success Rate (%)
Figure 2: 
Effective rank and planning success rate across
four environments.
Top: effective rank of latent representations. Bottom: planning success
rate.
Larger effective-rank reductions from LeWM to Sub-JEPA broadly
coincide with larger planning gains, suggesting that Multi-Subspace
Gaussian regularization improves performance by reducing spurious
high-rank variation in the latent space.
This direct correspondence strongly supports our hypothesis: when
task-relevant features lie on a low-dimensional manifold, the full-space
Gaussian prior in LeWM forces the latent space into unnecessarily high
rank, imposing a bias that hinders task-aligned representation learning.
Subspace-wise regularization relaxes this bias and allows the latent
geometry to contract toward the task’s intrinsic dimensionality—the
greater this contraction, the larger the planning gain.
4.3 Ablation Studies
We conduct ablation studies to validate the design choices of Sub-JEPA
and to understand the bias-variance tradeoff in subspace design.
4.3.1 Number of subspaces K
We first study the effect of the number of subspaces K.
Specifically, we set D=192 and vary
K\in\{1,2,4,8,16,32,64\},
which corresponds to subspace dimensions d_{s}=\{192,96,48,24,12,6,3\}.
When K=1 and d_{s}=D, the regularizer reduces to a full-space
Gaussian regularization with an orthogonal transformation,
making it the closest counterpart to LeWM Maes et al. (2026) within our implementation,
whereas larger K values decompose the
same latent space into progressively smaller subspaces.
----490-520----
statistical reliability. Increasing K relaxes the global
Gaussian prior and enables more expressive latent structure,
while excessively small subspaces may weaken the effectiveness
of the regularization signal.
4.3.2 Joint effect of K and d_{s}
In this ablation,
we relax the constraint d_{s}=D/K
and independently vary both K and d_{s}
to study th
... [truncated 2737 chars]
```
</details>

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; sed -n '305,360p' T_2605.09241.txt", "description": "Read Sub-JEPA Table 1 numbers"}
```

<details><summary>tool result</summary>

```
4.2 Planning Performance
4.2.1 Success Rates
We first report the planning success rates.
Overall,
Sub-JEPA consistently outperforms LeWM Maes et al. (2026) across all four environments,
demonstrating that relocating the Gaussian regularization into subspaces
yields more powerful latent representations.
For a fair comparison,
LeWM and Sub-JEPA are evaluated across six random seeds (reporting mean \pm std).
Since official checkpoints for PLDM Sobal et al. (2025) and DINO-WM Zhou et al. (2025) are unavailable, we cite their success rates directly from LeWM under the exact same protocol.
Table 1: 
Planning success rate (%, higher is better) across four
continuous-control environments.
Sub-JEPA consistently improves upon LeWM across all tasks,
with the largest gain on
Two-Room Zhou et al. (2025),
where the full-space Gaussian prior of LeWM is most mismatched
to the low intrinsic dimensionality of the task.
Method
Two-Room Zhou et al. (2025)
Reacher Tassa et al. (2018)
PushT Chi et al. (2025)
OGB-Cube Park et al. (2025)
PLDM Sobal et al. (2025)
97.00
78.00
78.00
65.00
DINO-WM (w/o proprio.) Zhou et al. (2025)
100.00
79.00
74.00
86.00
DINO-WM (w/ proprio.) Zhou et al. (2025)
100.00
–
92.00
–
LeWM Maes et al. (2026)
84.33\pm4.23
82.67\pm4.42
84.67\pm6.53
67.33\pm5.01
Sub-JEPA (Ours)
95.00\pm2.76
84.00\pm4.00
89.00\pm5.33
76.33\pm5.99
As shown in Table 1,
the improvement is most prominent on Two-Room Zhou et al. (2025). LeWM is known to struggle here because enforcing isotropic Gaussianity over the entire high-dimensional embedding introduces an overly restrictive bias for this low-intrinsic-dimension task. By contrast, Sub-JEPA regularizes lower-dimensional orthogonal views. This relaxes the global constraint and better preserves task-relevant structure while maintaining the anti-collapse benefits.
On Reacher Tassa et al. (2018) and PushT Chi et al. (2025), Sub-JEPA achieves strong performance, consistently outperforming PLDM and surpassing DINO-WM without relying on a pret
... [truncated 777 chars]
```
</details>

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; sed -n '715,745p' T_2605.09241.txt", "description": "Read Sub-JEPA conclusion"}
```

<details><summary>tool result</summary>

```
Sub-JEPA maintains scene geometry more consistently, yielding lower
reconstruction error at distant steps.
This suggests that subspace-wise regularization improves recursive
latent stability: by replacing a single global Gaussian constraint
with structured subspace constraints, Sub-JEPA better supports
coherent long-term state propagation.
Conclusion
We presented Sub-JEPA,
a simple yet effective extension of
LeWorldModel that relocates Gaussian regularization
from the ambient embedding space
into random orthogonal subspaces.
Our key insight is that the isotropic Gaussian prior enforced by
LeWM operates in the full high-dimensional latent space, which
introduces an excessive bias when the underlying task dynamics
reside on a low-dimensional manifold.
Extensive experiments demonstrate that Sub-JEPA consistently
outperforms LeWM across four tasks, that physical state information
is better preserved and more nonlinearly decodable in the learned
representations, and that the latent trajectories are substantially
more temporally coherent.
References
[1]
D. Ha and J. Schmidhuber (2018)
Recurrent World Models Facilitate Policy Evolution.
In NeurIPS, S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (Eds.),
Vol. 31, pp. .
External Links: Link
Cited by: §1,
§2,
§4.6.
```
</details>

### ASSISTANT — 2026-09-16 17:10

Now let me fetch HTML for the remaining 9 papers to get their limitations sections.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; cd /tmp/wog_lit\nfor id in 2606.26217 2607.26924 2606.02572 2606.01443 2606.17603 2607.28415 2608.27395 2607.15295 2608.16287; do\n  ok=\"\"\n  for v in v3 v2 v1; do\n    code=$(curl -s -o tmp_h.html -w \"%{http_code}\" \"https://arxiv.org/html/$id$v\")\n    sz=$(wc -c < tmp_h.html)\n    if [ \"$code\" = \"200\" ] && [ \"$sz\" -gt 20000 ]; then cp tmp_h.html best_$id.html; echo \"$id -> $v OK $sz\"; ok=1; break; fi\n  done\n  [ -z \"$ok\" ] && echo \"!!! $id NO HTML AVAILABLE (last code $code size $sz)\"\n  sleep 1\ndone", "description": "Fetch HTML for remaining 9 papers"}
```

<details><summary>tool result</summary>

```
2606.26217 -> v1 OK 153046
2607.26924 -> v3 OK 271060
2606.02572 -> v1 OK 353362
2606.01443 -> v1 OK 788292
2606.17603 -> v1 OK 427139
2607.28415 -> v1 OK 245016
2608.27395 -> v1 OK 284355
2607.15295 -> v1 OK 294421
2608.16287 -> v2 OK 543718
```
</details>

### ASSISTANT — 2026-09-16 17:11

All 9 available. Let me extract and grep them.

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; for id in 2606.26217 2607.26924 2606.02572 2606.01443 2606.17603 2607.28415 2608.27395 2607.15295 2608.16287; do python3 txt.py best_$id.html > T_$id.txt; done\nfor id in 2606.26217 2607.28415 2607.26924; do echo \"########## $id ##########\"; grep -n -i -E \"limitation|future work|open problem|remains open|we leave|caveat|failure\" T_$id.txt | head -25; done", "description": "Grep limitations in Fast-LeWM, QQWorld, TC-SIGReg"}
```

<details><summary>tool result</summary>

```
########## 2606.26217 ##########
75:This creates two practical limitations.
156:Fast-LeWM addresses these limitations by reducing the sequential dependence among predicted future states.
########## 2607.28415 ##########
52:Latent world models enable efficient planning by predicting future states in a compact representation space, but their performance depends critically on the quality of the learned latent distribution. LeWorldModel (LeWM) regularizes its latents toward an isotropic Gaussian using the Epps–Pulley (EP) objective. We show that the corrective gradients of EP rapidly vanish for isolated tail samples, leaving heavy-tailed deviations insufficiently controlled. To address this limitation, we propose QQWorld, which replaces EP with a quantile–quantile matching objective that directly aligns projected latent samples with rank-matched Gaussian quantiles, thereby maintaining effective corrective gradients in the tails. We further develop cross-batch QQ, which enlarges the effective ranking pool using detached samples from previous batches, and characterize its bias–variance trade-off. Across four control environments, QQWorld effectively improves the average planning success rate of LeWM, while consistently yielding better Gaussian alignment and thinner latent tails.
56:Despite this regularization, we observe that the latents learned by LeWM have pronounced heavy tails, as shown in Figure 1 (Right). Such heavy-tailed behavior is undesirable in a latent world model: extreme latent values can push the learned dynamics into poorly represented regions, potentially amplifying errors during multi-step rollouts, while increasing the mismatch between the learned latent distribution and the target Gaussian prior. Crucially, this happens even though the model is explicitly penalized for non-normality, which suggests that the limitation lies in the optimization geometry of the particular test being used.
57:We analyze this limitation in two steps. First, we use the equiv
... [truncated 2622 chars]
```
</details>

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; echo \"=== 2607.26924 key sections ===\"; sed -n '95,110p' T_2607.26924.txt; echo \"---655-675---\"; sed -n '655,675p' T_2607.26924.txt; echo \"---1255-1275---\"; sed -n '1255,1275p' T_2607.26924.txt", "description": "Read TC-SIGReg critique sections"}
```

<details><summary>tool result</summary>

```
=== 2607.26924 key sections ===
to \mathbf{r}_{t} rather than directly to \mathbf{z}_{t}.
(b) Along each regularized projection direction, the Raw LeWM objective
couples the variance allocation of the persistent component and centered residual
under a shared near-unit variance budget, whereas TC-LeWM decouples their variance
allocation by regularizing the centered residual directly.
We trace this representation limitation to a variance-allocation bias in the objective of Raw LeWM.
We first decompose the latent representation into a
temporally persistent component and a temporally centered residual. Based on this
decomposition, we develop a controlled Monte Carlo analysis of the Raw LeWM objective
along individual SIGReg projection directions.
The analysis reveals a mechanism. Along each projection direction, SIGReg
encourages the projected latent variance toward the unit-variance target, while the
prediction objective favors allocating variance to more predictable temporally
persistent variation. Their interaction therefore couples the variance allocation
of the persistent component and centered residual under a shared near-unit variance
budget, biasing the allocation toward the persistent component and suppressing residual variance.
---655-675---
In this work, we identified a variance-allocation bias in the Raw LeWM objective:
the interaction between latent prediction and SIGReg favors temporally
persistent variation while suppressing the centered residual.
Based on this analysis, we introduced TC-LeWM, which applies SIGReg to
temporally centered residuals to decouple their variance allocation while
retaining an effective anti-collapse property.
Across single-task, suite-wise 10-task, and unified 40-task LIBERO settings,
TC-LeWM consistently improves the closed-loop performance of downstream
behavior-cloning policies and yields representations with more linearly
decodable robot state and dynamics, particularly gripper dynamics.
Future work will scale TC-LeWM to lar
... [truncated 1485 chars]
```
</details>

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; echo \"=== 2606.26217 ===\"; sed -n '70,85p' T_2606.26217.txt; echo \"--- results ---\"; grep -n -E \"success|speedup|faster|times|Table\" T_2606.26217.txt | head -20", "description": "Read Fast-LeWM limitations and results"}
```

<details><summary>tool result</summary>

```
=== 2606.26217 ===
Despite this progress, LeWM-style planning remains costly and susceptible to accumulated prediction error.
Its dynamics model is inherently local: it predicts only the next latent state \hat{z}_{t+1} from the current latent z_{t} and action a_{t}.
To evaluate a candidate action sequence, the planner must therefore apply this one-step model autoregressively, repeatedly feeding predicted latents back into the dynamics model G_{\mathrm{LeWM}}, and the terminal prediction \hat{z}_{t+k} is obtained by composing local transitions:
\hat{z}_{t+k}=G_{\mathrm{LeWM}}(G_{\mathrm{LeWM}}(\cdots G_{\mathrm{LeWM}}(z_{t},a_{t}),\ldots),a_{t+k-1}).
where each prediction depends on the intermediate imagined latents produced along the rollout.
This creates two practical limitations.
First, candidate evaluation becomes slow because autoregressive rollout must generate the entire imagined latent trajectory step by step, repeating action encoding and latent prediction.
Second, errors introduced at early or intermediate imagined states can propagate into later predictions, making rollout increasingly unreliable as the horizon grows.
We propose Fast LeWorldModel (Fast-LeWM), a fast latent world model built around action-prefix prediction. Instead of advancing the latent state one step at a time, Fast-LeWM predicts the latents reached after executing the prefixes of an encoded action sequence, allowing the model to evaluate these prefix outcomes directly and in parallel.
Given the current visual latent z_{t} and an action prefix
a_{t:t+k-1}=(a_{t},\ldots,a_{t+k-1}),
Fast-LeWM predicts the future latent reached after executing that prefix:
\hat{z}_{t+k}=G_{\mathrm{Fast-LeWM}}(z_{t},a_{t:t+k-1}),\quad k=1,\ldots,H.
Rather than estimating a long trajectory through repeated adjacent transitions, Fast-LeWM directly predicts the latent outcomes of different action prefixes from the observed anchor latent.
Different prefixes contain different degrees of accumulated action effects
... [truncated 4635 chars]
```
</details>

### ASSISTANT — 2026-09-16 17:11

Now the SIGReg-variant papers: VISReg, UR-JEPA, SPHERE-JEPA, LeVJEPA, AV-JEPA, SCALE.

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; for id in 2606.02572 2606.01443; do echo \"########## $id ##########\"; grep -n -i -E \"limitation|future work|open problem|remains open|we leave|caveat|failure|vanishing gradient|tension\" T_$id.txt | head -22; done", "description": "Grep VISReg and UR-JEPA"}
```

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; for id in 2606.17603 2608.27395 2607.15295 2608.16287; do echo \"########## $id ##########\"; grep -n -i -E \"limitation|future work|open problem|remains open|we leave|caveat|failure\" T_$id.txt | head -20; done", "description": "Grep SPHERE, LeVJEPA, AV-JEPA, SCALE"}
```

<details><summary>tool result</summary>

```
########## 2606.02572 ##########
53:Self-supervised learning methods prevent embedding collapse via modeling heuristics or explicit regularization of the embedding space. Among the latter, VICReg decomposes regularization into variance and covariance objectives, offering flexibility and interpretability. However, covariance captures only second-order statistics—encouraging decorrelation but failing to enforce the full distributional shape needed for stable training. Sketching-based methods such as SIGReg address this by aligning embeddings to an isotropic Gaussian, but lack flexibility and suffer from vanishing gradients under collapse. We propose Variance-Invariance-Sketching Regularization (VISReg), which replaces covariance with a Sliced-Wasserstein-based sketching objective that enforces full distributional shape, while retaining a variance term for scale control. By decoupling scale and shape, VISReg combines VICReg’s flexibility with the distributional rigor of sketching methods, providing robust gradients even under collapse. We show that VISReg scales linearly, outperforms existing regularization on low-quality datasets, and is resilient to long-tailed and low-rank regimes. Pre-trained on ImageNet-1K, VISReg achieves state-of-the-art performance on out-of-distribution datasets. Pre-trained on ImageNet-22K, it matches DINOv2’s OOD performance despite the latter using 10\times more data (LVD-142M). Project and code: https://haiyuwu.github.io/visreg.
59:However, both methods have clear limitations. VICReg regularizes covariance, which captures only second-order statistics. While this encourages decorrelation, it cannot enforce the full distributional shape of the embedding space—a distribution can match in mean and covariance yet remain far from Gaussian. This makes covariance regularization a comparatively weak proxy for the isotropy that stable, information-rich training requires. On the other hand, SIGReg addresses distributional shape directly through sketc
... [truncated 1924 chars]
```
</details>

<details><summary>tool result</summary>

```
########## 2606.17603 ##########
########## 2608.27395 ##########
122:The benefit of aggressive dropping is, however, not uniform across evaluations. On Something-Something-v2, which primarily probes motion understanding, accuracy declines for dropping ratios beyond 0.3, in contrast to the monotonic improvement observed on ImageNet. We find that this degradation is mitigated by extending the training schedule: with longer training, higher dropping ratios recover the accuracy of lower ones on Something-Something-v2 while retaining their lower per-iteration cost, such that aggressive dropping remains the more efficient configuration in terms of total pretraining compute. A plausible interpretation is that sparse random observations render motion cues, which depend on correspondences across frames, less frequently recoverable within a single view, and that additional iterations compensate for the reduced per-sample signal; a characterization of this interaction, and dropping schemes that preserve motion information at high sparsity, are left to future work.
########## 2607.15295 ##########
35:Limitations and future work
385:left to future work. The same audio-driven behaviour is visible during VGGSound
472:Limitations and future work
########## 2608.16287 ##########
419:The comparison with Aux further separates this effect from generic prediction accuracy. Aux achieves the lowest rollout error on Push-T and Cube, yet its gains in Kendall agreement are consistently smaller than SCALE’s and nearly vanish on the navigation tasks. DINO-WM illustrates the complementary limitation: on Two-Room and PointMaze it combines favorable rollout and ranking behavior, while on Push-T and Reacher its rollout errors are dramatically larger. On Cube, DINO-
```
</details>

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; echo \"=== UR-JEPA failure modes 820-870 ===\"; sed -n '820,870p' T_2606.01443.txt", "description": "Read UR-JEPA failure modes"}
```

<details><summary>tool result</summary>

```
=== UR-JEPA failure modes 820-870 ===
6 Practical considerations and failure modes
6.1 Degenerate optima
Every UR-style regularizer in this paper admits trivial minimizers,
just as LeJEPA’s SIGReg can be zeroed by any Gaussian linear
reparameterization of the projector outputs. The predictive term
\mathcal{L}_{\mathrm{pred}} of (1) provides the orthogonal
“invariance-plus-diversity” signal that picks among the trivial set;
without \mathcal{L}_{\mathrm{pred}}, a constant f_{\theta} would satisfy every
regularizer in the family. The interesting question is whether the
regularizer additionally admits a degenerate minimum that
\mathcal{L}_{\mathrm{pred}} alone cannot escape, because then training can be
trapped there. We address each loss in turn.
LeJEPA with \mathcal{L}^{\text{SIGReg}} [1].
Zeroed exactly on the desired full-D isotropic-Gaussian class. At
a point-mass cloud the 1-D marginals are \delta-distributions and
the Epps, Pulley characteristic-function statistic is bounded away
from zero, so \mathcal{L}_{\mathrm{pred}} + SIGReg has no trivial point-mass
minimum.
UR–JEPA with \mathcal{L}^{\text{CGLT}}, dyadic-difference variant (15).
Zeroed when \log\theta_{r}(x) is constant in r at every anchor,
i.e. on the n-AD-regular UR class (the desired target). At
collapse, the Gaussian-smoothed density satisfies
\theta_{r}(x)\propto r^{-n} near the cloud, so
\Delta_{r}\log\theta=-n\log 2 per dyadic scale; the loss takes
a fixed positive value at every point-mass configuration. No
trivial point-mass minimum.
UR–JEPA with \mathcal{L}^{\text{CGLT},\partial\!\log}, log-derivative variant (21).
Zeroed when t\partial_{t}\log\theta_{t}=0 at every anchor and
scale, again the AD-regular UR class. At collapse,
t\partial_{t}\log\theta_{t}=-n, so the loss is bounded away from
zero by exactly the same mechanism as the dyadic-difference variant.
UR–JEPA with \mathcal{L}^{\text{CGLT},\partial}, raw scale-derivative variant (22).
Zeroed on the smooth UR class characterized by
Theorem 1(
... [truncated 825 chars]
```
</details>

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; echo \"=== UR-JEPA future work 2122-2160 ===\"; sed -n '2122,2165p' T_2606.01443.txt; echo \"=== tension 270-300 ===\"; sed -n '270,300p' T_2606.01443.txt", "description": "Read UR-JEPA future work and tension"}
```

<details><summary>tool result</summary>

```
=== UR-JEPA future work 2122-2160 ===
10 Future work
Several directions follow naturally from the present study.
Remote sensing, multimodal, and multisource data.
Our experiments span Inet10 (\sim 13\mathrm{K} images),
Galaxy10 SDSS (\sim 21\mathrm{K}), and single-seed Stage-2
ImageNet-100 (\sim 130\mathrm{K}). The natural next domain for
UR–JEPA is remote sensing: satellite and aerial imagery are
structurally biased toward low-dimensional regularities (land-cover
textures, sensor noise, geographic priors) where the
rectifiable-support hypothesis underpinning UR–JEPA is plausibly
sharper than on ImageNet-style scenes. Standard benchmarks
(EuroSAT, BigEarthNet, fMoW, SpaceNet) provide a graded difficulty
ladder. Multimodal extensions are a logical next step: optical, SAR,
multispectral, hyperspectral, and elevation modalities each project
to a different rectifiable structure, so a joint embedding obtained
by applying \mathcal{L}^{\text{CGLT}} within each modality and an
invariance term across modalities is a principled formulation. The
multisource setting (sensor and platform heterogeneity, varying
acquisition geometry, mixed labeling regimes) is a further test of
whether a single geometric loss family can absorb the distributional
shifts that complicate distribution-matching objectives such as
SIGReg.
Theoretical analysis of UR–JEPA.
The present work motivates UR–JEPA from the CGLT and Jones-\beta
characterizations of uniform n-rectifiability, but does not
provide a finite-sample convergence guarantee. Open questions
include: (a) under what conditions does minimizing
\mathcal{L}^{\text{CGLT}} provably drive the projected distribution
toward an n-AD-regular, uniformly n-rectifiable target?
(b) how does the sample complexity scale with the ambient projector
dimension D, the target intrinsic dimension n, and the number
of dyadic scales K? (c) can the CGLT square function be
connected, via a gradient-flow or neural-tangent-kernel argument,
to the projected distribution’
... [truncated 2378 chars]
```
</details>

### ASSISTANT — 2026-09-16 17:12

UR-JEPA has a strong theoretical critique. Now let me check SPHERE-JEPA and the remaining papers.

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; echo \"=== 2606.17603 SPHERE grep2 ===\"; grep -n -i -E \"destabiliz|variance|projection noise|hinder|drawback|limit|stochastic|conclusion\" T_2606.17603.txt | head -30", "description": "Grep SPHERE-JEPA broader terms"}
```

<details><summary>tool result</summary>

```
=== 2606.17603 SPHERE grep2 ===
31:8 Conclusion
78:This stochasticity injects projection variance into the training gradients, destabilizing optimization, and hindering convergence.
79:In this work, we first show that analytically integrating out these random projections natively yields a deterministic Maximum Mean Discrepancy (MMD), bypassing the variance of sliced methods.
81:To prevent spatial bias, we equip these tests with rotationally invariant kernels constructed via spectral theory, systematically evaluating two canonical families: smooth exponential decay (Heat) and strict frequency cutoff (Bandlimited) filters.
82:Empirically, removing projection-induced noise results in more stable optimization, faster convergence, and consistent improvements over stochastic sliced regularizers on ImageNet and Galaxy10.
85:A central paradigm in modern Self-Supervised Learning (SSL) is the multi-view formulation, where an encoder learns representations by minimizing an invariance objective across augmented views of the same input [Chen et al., 2020, Grill et al., 2020].
90:However, relying on this Monte Carlo approximation naturally injects “projection variance” into the training gradients, destabilizing optimization and slowing convergence.
91:Concurrently, KerJEPA [Zimmermann et al., 2025] demonstrated that analogous sliced regularizers enforcing Gaussian embeddings can be analytically integrated to entirely bypass this variance, resulting in stable, closed-form Maximum Mean Discrepancy (MMD).
98:Any such kernel is characterized by the Laplacian’s eigenvalues. Leveraging this spectral foundation, we systematically evaluate three distinct kernels: the exact kernel implicitly induced by the analytical integration of SUSReg [Nicollier et al., 2026], alongside two canonical frequency filters adapted to the sphere: the Heat kernel (acting as a smooth exponential decay) and the Bandlimited kernel (enforcing a strict frequency cutoff).
109:Guided by the spectral characterizatio
... [truncated 4765 chars]
```
</details>

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; echo \"=== 2607.15295 AV-JEPA limitations ===\"; sed -n '472,510p' T_2607.15295.txt", "description": "Read AV-JEPA limitations"}
```

<details><summary>tool result</summary>

```
=== 2607.15295 AV-JEPA limitations ===
Limitations and future work
Modality-probe and per-modality mAP results show that AV-JEPA’s predictions are
largely carried by audio on both VGGSound and AudioSet, making the backbone most
useful as an audio representation with a visual side-channel from
pretraining. The gap with end-to-end fine-tuned MAE-based methods (63–67% on
VGGSound, 42–53 mAP on AudioSet) likely reflects their reconstruction and
contrastive objectives, the absence of a video-specific pretraining stage, and
(for MAViL) ImageNet initialisation. Next steps include longer AudioSet
pretraining, larger ViTs, initialising the visual stream from a video-only
stage, evaluating against single-modality SSL
backbones (Gong et al., 2021; Huang et al., 2022), and extending the
modality-dropout recipe to text or optical flow.
Impact Statement
This paper presents work whose goal is to advance the field of Machine Learning.
There are many potential societal consequences of our work, none which we feel
must be specifically highlighted here.
References
Araujo et al. (2025)
Araujo, E., Rouditchenko, A., Gong, Y., Bhati, S., Thomas, S., Kingsbury, B.,
Karlinsky, L., Feris, R., Glass, J. R., and Kuehne, H.
CAV-MAE Sync: Improving contrastive audio-visual mask autoencoders
via fine-grained alignment.
In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR), 2025.
Assran et al. (2023)
Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M.,
LeCun, Y., and Ballas, N.
Self-supervised learning from images with a joint-embedding
predictive architecture.
In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR), 2023.
Balestriero & LeCun (2025)
Balestriero, R. and LeCun, Y.
LeJEPA: Provable and scalable self-supervised learning without the
heuristics.
arXiv preprint arXiv:2511.08544, 2025.
Bardes et al. (2024)
Bardes, A., Garrido, Q., Ponce, J., Chen, X., Rabbat, M., LeCun, Y., Assran,
M., and Bal
... [truncated 7 chars]
```
</details>

### ASSISTANT — 2026-09-16 17:12

Now let me get SCALE's critique and check the additional papers the searches surfaced.

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; echo \"=== SCALE grep ===\"; grep -n -i -E \"LeWM|SIGReg|leading principal|geometry|limitation|conclusion|future\" T_2608.16287.txt | head -35", "description": "Grep SCALE for LeWM critique"}
```

<details><summary>tool result</summary>

```
=== SCALE grep ===
2:SCALE: State-Calibrated Latent Embeddings for JEPA Planning in the Right Geometry
22:Supervising latent geometry.
25:3.2 LeWM training objective
36:6.3 How SCALE reorganizes the latent geometry
39:The reorganized geometry survives rollout and improves candidate ranking.
40:7 Conclusion
56:SCALE: State-Calibrated Latent Embeddings for JEPA Planning in the Right Geometry
72:Joint-embedding predictive world models plan by scoring predicted terminal embeddings against a goal embedding using a cost defined on the representation itself. Two prominent strategies for obtaining non-collapsed representations are to inherit a pretrained feature space, as in DINO-WM, and to learn an embedding end to end with anti-collapse regularization, as in LeWorldModel (LeWM) with SIGReg. These strategies show complementary strengths across tasks. Although task-relevant state is decodable from the full embeddings of both models, DINO-WM’s leading principal components usually retain substantially more state information than LeWM’s. Because Euclidean planning costs are dominated by high-variance directions, this difference affects how strongly state can influence candidate selection. We propose SCALE (State-CAlibrated Latent Embeddings) to give the end-to-end LeWM representation the favorable geometric property observed in DINO-WM. SCALE induces this property by correlating sampled pairwise latent distances with distances in a standardized task-relevant state space, without replacing LeWM’s learned encoder. Across five tasks, three planning solvers, and five compute budgets, SCALE improves every task–solver average over LeWM. A latent-to-state regression control matches or exceeds SCALE’s full-embedding decodability yet leaves latent–state distance alignment essentially unchanged and yields less consistent planning gains. SCALE adds a single lightweight training-time regularizer and no planning-time overhead. These results show that planning depends not only on whether ta
... [truncated 10831 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; cd /tmp/wog_lit\nfor id in 2605.26379 2608.12959 2608.29998 2605.07278 2604.27743 2605.20107 2607.13612 2603.05924 2605.26900 2512.19605 2602.01456 2608.22764 2607.21622 2609.04264; do curl -s \"https://arxiv.org/abs/$id\" -o $id.html; sleep 0.5; done; ls -la 2605.26379.html 2608.12959.html", "description": "Fetch additional candidate papers"}
```

<details><summary>tool result</summary>

```
-rw-rw-r-- 1 zhengguantian zhengguantian 41215 Sep 16 09:12 2605.26379.html
-rw-rw-r-- 1 zhengguantian zhengguantian 43670 Sep 16 09:12 2608.12959.html
```
</details>

### ASSISTANT — 2026-09-16 17:12

Let me parse the additional candidate papers.

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; python3 absx.py 2605.26379 2608.12959 2608.29998 2605.07278 2604.27743 2605.20107 2>&1", "description": "Parse additional candidates batch A"}
```

<details><summary>tool result</summary>

```
==========================================================================================
ID: 2605.26379
TITLE: When Does LeJEPA Learn a World Model?
citation_date: 2026/05/25
citation_online_date: 2026/05/25
AUTHORS: Klindt, David; LeCun, Yann; Balestriero, Randall
COMMENTS_RAW: <<NO COMMENTS FIELD>>
SUBJECTS: Machine Learning (stat.ML) ; Machine Learning (cs.LG)
ABSTRACT: Abstract: A representation that scrambles the true degrees of freedom of the world cannot support reliable planning or compositional generalization. We prove that LeJEPA (alignment plus Gaussian regularization) linearly recovers the world's latent variables from nonlinear observations, a property known as linear identifiability, in a broad class of worlds where latents evolve under stationary, additive-noise transitions. Our main result is that among all such worlds, the Gaussian is the unique latent distribution for which this guarantee holds. The forward direction rests on a spectral decomposition in which each degree of nonlinearity is strictly penalized by alignment, making the linear map the optimum; the converse rules out every non-Gaussian alternative. We further prove an approximate identifiability result where the guarantee degrades gracefully, and show that linear, orthogonal identifiability enables optimal latent-space planning. We validate the theory with experiments ranging from 2D examples to 1024-dimensional latents, including distributional ablations and pixel-based robotic control. Our theory turns an empirically successful recipe into a mathematical guarantee, providing the foundation for building World Models that provably recover the structure of the world.
HIST: Submission history From: David Klindt [ view email ] [v1] Mon, 25 May 2026 22:56:26 UTC (6,493 KB)
==========================================================================================
ID: 2608.12959
TITLE: The Objective Is the Bottleneck: Latent World Models Encode What Their Planners Cannot Use
citation_date: 
... [truncated 10703 chars]
```
</details>

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; python3 absx.py 2607.13612 2603.05924 2605.26900 2512.19605 2602.01456 2608.22764 2607.21622 2609.04264 2>&1 | head -140", "description": "Parse additional candidates batch B"}
```

<details><summary>tool result</summary>

```
==========================================================================================
ID: 2607.13612
TITLE: The SIGReg Objective as Variational Free Energy: A Theoretical Active-Inference Account of JEPA World Models
citation_date: 2026/07/15
citation_online_date: 2026/07/15
AUTHORS: Arnez, Fabio; Gomez-Villa, Alexandra
COMMENTS_RAW: Theoretical paper; empirical validation of the stated predictions is left to separate work. 28 pages, 4 figures, 4 tables
SUBJECTS: Machine Learning (cs.LG) ; Artificial Intelligence (cs.AI)
ABSTRACT: Abstract: Joint-Embedding Predictive Architectures (JEPAs) are the dominant design for latent world models, yet they are usually justified by empirical performance rather than a normative principle. We show that the choice of anti-collapse regulariser determines whether a JEPA's training objective, a prediction loss plus a weighted embedding regulariser, is a valid Active Inference (AIF) variational free energy. We organise four non-contrastive regularisers (VICReg, LogDet, PairDist, and SIGReg) into an entropy-estimator hierarchy indexed by a prior-miscalibration gap, and show that the gap's sign, whether the estimator bounds the latent entropy from above or below, decides whether the AIF surprise bound survives: VICReg and LogDet are unsafe upper bounds, PairDist a safe lower bound, and SIGReg eliminates the gap. We then prove a correspondence theorem: under the standard constant-noise encoder model and successful SIGReg enforcement (isotropic-Gaussian embeddings), the gap vanishes, the objective becomes an exact information bottleneck, the surprise bound is preserved, and the latent goal cost becomes an exact proxy for AIF pragmatic value, whereas VICReg leaves an irreducible second-order anisotropy term. We extend the correspondence to multi-step expected free energy, ensemble epistemic value, and a learned-policy regime, and we identify the one AIF term no current JEPA world model computes: the state-epistemic value, a future-sta
... [truncated 13968 chars]
```
</details>

### ASSISTANT — 2026-09-16 17:13

Many more relevant papers found. Let me fetch HTMLs for the top additional critiques and abs pages for remaining candidates.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; cd /tmp/wog_lit\nfor id in 2608.12959 2609.04264 2608.22764 2605.26900 2605.20107 2605.26379; do\n  ok=\"\"\n  for v in v2 v1; do\n    code=$(curl -s -o tmp2.html -w \"%{http_code}\" \"https://arxiv.org/html/$id$v\"); sz=$(wc -c < tmp2.html)\n    if [ \"$code\" = \"200\" ] && [ \"$sz\" -gt 20000 ]; then cp tmp2.html best_$id.html; python3 txt.py best_$id.html > T_$id.txt; echo \"$id $v OK\"; ok=1; break; fi\n  done\n  [ -z \"$ok\" ] && echo \"!!! $id NO HTML ($code/$sz)\"\n  sleep 1\ndone", "description": "Fetch HTML for additional critique papers"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; cd /tmp/wog_lit\nfor id in 2608.11174 2605.31111 2608.29029 2609.03565 2609.10464 2607.16314 2607.02404 2608.04213 2602.07016 2607.22430 2607.29613 2609.04261 2608.27367; do curl -s \"https://arxiv.org/abs/$id\" -o $id.html; sleep 0.4; done; echo done", "description": "Fetch abs for remaining candidates"}
```

<details><summary>tool result</summary>

```
2608.12959 v1 OK
2609.04264 v2 OK
2608.22764 v1 OK
2605.26900 v1 OK
2605.20107 v1 OK
2605.26379 v1 OK
```
</details>

<details><summary>tool result</summary>

```
done
```
</details>

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; for id in 2608.12959 2609.04264 2608.22764 2605.26900 2605.20107 2605.26379; do echo \"########## $id ##########\"; grep -n -i -E \"limitation|future work|open problem|remains open|we leave|caveat|fundamental limitation|laziness|price of isotropy\" T_$id.txt | head -14; done", "description": "Grep limitations in additional critique papers"}
```

<details><summary>tool result</summary>

```
########## 2608.12959 ##########
37:9 Limitations
562:9 Limitations
########## 2609.04264 ##########
24:3 Beyond Representation Collapse: A Case Study of Encoder Laziness in JEPA World Models
57:Latent world models have become increasingly popular as a method to predict and plan in latent space rather than pixel space. Recent architectures, such as LeWorldModel (LeWM), jointly train the encoder and predictor using regularization techniques like SIGReg to prevent representation collapse. Even with such regularization preventing representation collapse, we identify a new world model failure mode of physical representation laziness, particularly noted in highly dynamic environments. For these lazy cases, the learned latent states do not collapse but nonetheless fail to represent key physical properties, causing ubiquitous downstream planning failure. To resolve this issue, we propose training-time auxiliary supervision with a lightweight ”Fourier auxiliary head”, which enforces physically-informed structuring of the latent space with no additional inference-time cost and can be generalized to any environment. Experimentally, we show that the auxiliary head substantially improves planning success rates in dynamic environments where the baseline LeWM exhibits physical representation laziness. It also leads to modest improvements in other environments, even when the baseline does not exhibit physical representation laziness. We further observe superior planning performance being accompanied by higher latent space correlations with key physical properties, indicating both the ability of our method to physically structure latent states and the potential planning-side benefit to the learned representation being physically structured. Finally, we ablate across various training data amounts, showing that in low-data regimes, auxiliary supervision is particularly impactful in increasing success rate. These findings support the use of our Fourier auxiliary head method to improve
... [truncated 12365 chars]
```
</details>

### ASSISTANT — 2026-09-16 17:14

Let me read the key verbatim quotes from these.

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; echo \"=== 2608.12959 Limitations 562-600 ===\"; sed -n '562,600p' T_2608.12959.txt", "description": "Read Objective Is Bottleneck limitations"}
```

<details><summary>tool result</summary>

```
=== 2608.12959 Limitations 562-600 ===
9 Limitations
One environment. TwoRoom is a two-dimensional diagnostic environment. That
the objective is the bottleneck there does not establish it elsewhere, and
the decoded-position cost is available only where a low-dimensional state can
be read out. The temporal-distance result (§7) is the part of this that is
designed to travel, and it is demonstrated on the same environment.
One seed per checkpoint. The four-checkpoint correspondence of §5.2 rests
on four models with one seed each. It is a correspondence, not a law, and we
make no estimate of seed variance.
Confounded conditions. History size, action width and pixel normalisation
differ together between the checkpoints with usable geometry and those
without. We cannot attribute the metric degradation to any one of them
without training runs we did not do.
No claim about the cause of saturation. We show that the embedding metric
saturates and inverts, and that it matters. We do not explain why the
representation is organised that way.
The repair is a planner change. It leaves the model’s own objective
untouched. Whether training against a reachability-aware criterion would
produce a representation whose plain L2 is usable is an open and more
interesting question.
10 Related work
The failure mode identified here is the one Li, Wang and Liu (2026) argue for
on general grounds: that Euclidean proximity in a latent space is a poor
proxy for reachability, and that latent world models need horizon-matched
metrics. Their argument is made against latent world models as a class; this
paper supplies a direct measurement of the phenomenon on a specific published
model — including the original authors’ own released weights — quantifies the
range over which the metric fails, and shows what repairing it is worth in
planning success.
The reproduction this work builds on (arXiv:2608.10145) established the
protocol-dependence of the published planning figure, the four undocumented
trainin
... [truncated 476 chars]
```
</details>

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; echo \"=== 2605.20107 Limitations 604-615 ===\"; sed -n '604,616p' T_2605.20107.txt; echo \"=== 2605.26379 Limitations 523-540 ===\"; sed -n '523,540p' T_2605.26379.txt", "description": "Read HamJEPA and Klindt limitations"}
```

<details><summary>tool result</summary>

```
=== 2605.20107 Limitations 604-615 ===
Limitations.
This is a controlled geometry study rather than a SOTA SSL recipe. The main
baseline is SIGReg, ImageNet-100 is single-seed, and we do not learn or recover
the downstream operator H; broader baselines, multi-seed ImageNet runs, and
learned structured geometries remain future work.
References
[1]
S. Amari (1985)
Differential-geometrical methods in statistics.
[2]
M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. Rabbat, Y. LeCun, and N. Ballas (2023)
Self-supervised learning from images with a joint-embedding predictive architecture.
External Links: 2301.08243
=== 2605.26379 Limitations 523-540 ===
7 Limitations
Are the latents Gaussian?
The Gaussian is the maximum-entropy distribution for a given mean and covariance [23], making it the least-assumption prior. Whether real-world latents are Gaussian is unknowable from observations alone, but the same applies to the non-Gaussianity assumptions of classical ICA [82] (maybe a structuralist perspective [83] is more agnostic). There is a scale-of-description argument in favor: individual micro-variables may be non-Gaussian, but task-relevant latent variables are often aggregates that tend toward Gaussianity by the central limit theorem.
What if the dimension is wrong?
Our theorem assumes the encoder output dimension matches the true latent dimension (m=n). When msuperposition [15, 84, 18]; when m>n, extra dimensions must collapse or encode redundancy. Understanding this interaction is an important open problem with direct consequences for JEPA design [18].
Finite samples and optimization.
Our result is a population-level statement about the global optimum. Thm. 5.3 shows the guarantee degrades continuously with respect to alignment gap and covariance deviation, but does not address how these scale with sample size [85] or training dynamics [86]. The few bound violations we observe empirically (Fig. 4a) are consistent with finite-sample estimation noise in \var
... [truncated 2256 chars]
```
</details>

### ASSISTANT — 2026-09-16 17:15

Let me parse the last batch of abs pages to complete the survey.

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; python3 absx.py 2608.11174 2605.31111 2608.29029 2609.03565 2608.27367 2>&1 | head -80", "description": "Parse remaining candidates batch C"}
```

<details><summary>tool result</summary>

```
==========================================================================================
ID: 2608.11174
TITLE: VIScore: Diagnosing Planning-Relevant Quality in Latent World Models
citation_date: 2026/08/11
citation_online_date: 2026/08/12
AUTHORS: Wu, Haiyu; Balestriero, Randall; Levine, Morgan
COMMENTS_RAW: arXiv admin comment: This version has been removed by arXiv administrators as the submitter did not have the rights to agree to the license at the time of submission
SUBJECTS: Robotics (cs.RO)
ABSTRACT: Abstract: Regulating the latent space to an isotropic Gaussian distribution provides a stable and information-maximized landscape for world model planning. However, the latent space property and successful planning remain disconnected. We first study this by comparing SIGReg and VISReg, two regularization loss functions with the same distribution target but different properties. Compared with SIGReg, VISReg has more flexibility in controlling the weights of center, scale, and shape regularization, and a larger batch size brings a finer distribution approximation. We find that the former, despite being beneficial in self-supervised learning (SSL), does not help the planning, whereas the latter improves the planning success on out-of-domain (OOD) datasets. This motivates a deep understanding of the factors that correlate with the success rate. Unlike the previous metrics focusing on the encoded latent only, we propose the Veracity-Influence-Sobriety score (VIScore), a metric that quantifies the reachability and capacity of a predictor given the encoded feature, and the hallucination of the searching-based planner. Compared with straightness, physical-state probing, and empowerment, we show that, with the measurement covering encoder, predictor, and planner, VIScore explains the success rate better than the others, as reflected by a strong Spearman correlation. Specifically, VIScore consistently achieves a Spearman correlation over 0.75 on both seen and unseen mod
... [truncated 9017 chars]
```
</details>

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; python3 absx.py 2609.10464 2607.02404 2602.07016 2609.04261 2>&1 | head -60", "description": "Parse last candidates"}
```

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; echo \"=== VISReg conclusion/limitations ===\"; grep -n -i -E \"conclusion|limitation of our|we do not|future\" T_2606.02572.txt | tail -12; echo \"=== SPHERE-family 2606.17603 conclusion ===\"; sed -n '/^8 Conclusion/,/^References/p' T_2606.17603.txt | head -25", "description": "Read VISReg and SPHERE-family conclusions"}
```

<details><summary>tool result</summary>

```
==========================================================================================
ID: 2609.10464
TITLE: Semigroup-JEPA: Latent Dynamics Consistency for Zero-Shot Physics Generalization
citation_date: 2026/09/09
citation_online_date: 2026/09/09
AUTHORS: Liu, Andy Zeyi; Sun, Haoran; Baker, Lucas; Balestriero, Randall; Sous, John
COMMENTS_RAW: <<NO COMMENTS FIELD>>
SUBJECTS: Machine Learning (cs.LG) ; Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV)
ABSTRACT: Abstract: Joint-Embedding Predictive Architecture (JEPA) world models learn a compact latent representation of the world that supports prediction and planning, but their capability to learn physics and generate physically realistic dynamics remains hitherto untested. In this work, we introduce SemiGroup-JEPA (SG-JEPA), which extends the LeWorldModel framework by supplying the parameter governing the physics to the temporal model via action-conditioning and jointly training an encoder and predictor through an autoregressive latent rollout. To evaluate the model's ability to generalize out of distribution, we design dynamical tasks under different gravitational fields that, despite obeying the same physical law, exhibit qualitatively different dynamics, ranging from floating motion in weak gravitational fields to rapid bouncing in strong ones. In contrast to DINO-WM, SG-JEPA reduces open-loop prediction error by up to 2 times on two-dimensional datasets, and increases control success rate up to 2.5 times for three-dimensional robotic datasets, for which we train independent diffusion policies. To explain this advantage, we develop a linear feature model that separates local law-conditioned error from its recursive amplification under rollout. Guided by this model, we find that back-propagating the multi-step rollout loss into the representation trains the encoder to keep the features that the predictor can carry forward, and that those are the features the dynamics depend on,
... [truncated 6260 chars]
```
</details>

<details><summary>tool result</summary>

```
=== VISReg conclusion/limitations ===
28:5 Conclusion
706:5 Conclusion
=== SPHERE-family 2606.17603 conclusion ===
8 Conclusion
References
8 Conclusion
In this work, we first demonstrated that sliced statistical regularizers for hyperspherical uniformity are strictly suboptimal compared to their analytically integrated Maximum Mean Discrepancy (MMD) equivalent.
Building on this insight, we introduced a broader family of exact, full-dimensional regularizers based on MMD, Kernel Stein Discrepancy (KSD), and the Kullback-Leibler (KL) divergence.
To avoid spatial bias, we grounded these tests in spectral theory, explicitly motivating the use of two canonical kernels: the Heat and Bandlimited filters.
Empirically, our exact formulations confirmed that completely bypassing Monte Carlo projection noise systematically improves representation quality across all evaluated datasets.
On object-centric domains, MMD and KSD equipped with our canonical spectral kernels achieved similar state-of-the-art performance, outperforming both the standard stochastic baseline and its exact SUSReg-induced kernel.
Finally, we revealed how the choice of statistical discrepancy fundamentally shapes the learned topology: while MMD and KSD inherently accommodate the local clustering necessary for distinct classes, the continuous KDE-based KL divergence optimally forces fine-grained instance repulsion, yielding vastly superior performance on unclustered procedural textures.
Despite these theoretical and empirical advantages, it is important to contextualize the computational trade-offs of exact manifold regularizers.
Because our full-dimensional tests rely on exact pairwise kernel evaluations, their computational complexity scales quadratically with the global batch size, \mathcal{O}(B^{2}).
In contrast, stochastic sliced methods like SUSReg compute 1D statistics and scale linearly with respect to the batch size, \mathcal{O}(B|\mathcal{A}|), where |\mathcal{A}| is the number of random projections.
... [truncated 300 chars]
```
</details>

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; echo \"=== VISReg conclusion 706+ ===\"; sed -n '706,725p' T_2606.02572.txt", "description": "Read VISReg conclusion"}
```

<details><summary>tool result</summary>

```
=== VISReg conclusion 706+ ===
5 Conclusion
This paper proposes VISReg, a self-supervised learning method that does not rely on heuristics for training stability. We present its effectiveness on model scaling, training stability, and training efficiency. In addition, we show that VISReg and its alike method is more robust to low-quality datasets than DINO, which is helpful in real-world applications. Last, we conduct extensive experiments to evaluate its performance in important aspects of a foundation model training method. It is intriguing that VISReg has a stronger performance on OOD data and transfer learning. With this potential, we hope this technical path can enhance the usefulness of the foundation models.
6 Acknowledgment
We appreciate Prof. Yann LeCun’s efforts in connecting resources and people to bring this project to fruition.
References
Assran et al. (2022)
Assran, M., Caron, M., Misra, I., Bojanowski, P., Bordes, F., Vincent, P., Joulin, A., Rabbat, M., and Ballas, N.
Masked siamese networks for label-efficient learning.
In ECCV, 2022.
Assran et al. (2023)
Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., and Ballas, N.
Self-supervised learning from images with a joint-embedding predictive architecture.
In CVPR, 2023.
Baevski et al. (2022)
Baevski, A., Hsu, W.-N., Xu, Q., Babu, A., Gu, J., and Auli, M.
Data2vec: A general framework for self-supervised learning in speech, vision and language.
In ICML, 2022.
Balestriero & LeCun (2025)
Balestriero, R. and LeCun, Y.
Lejepa: Provable and scalable self-supervised learning without the heuristics.
```
</details>

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; echo \"=== QQWorld results ===\"; grep -n -E \"Table 1|success rate|average|Two-Room|Reacher|PushT|Cube|conclusion|Conclusion\" T_2607.28415.txt | head -25", "description": "Read QQWorld results"}
```

<details><summary>tool result</summary>

```
=== QQWorld results ===
38:5 Conclusion
52:Latent world models enable efficient planning by predicting future states in a compact representation space, but their performance depends critically on the quality of the learned latent distribution. LeWorldModel (LeWM) regularizes its latents toward an isotropic Gaussian using the Epps–Pulley (EP) objective. We show that the corrective gradients of EP rapidly vanish for isolated tail samples, leaving heavy-tailed deviations insufficiently controlled. To address this limitation, we propose QQWorld, which replaces EP with a quantile–quantile matching objective that directly aligns projected latent samples with rank-matched Gaussian quantiles, thereby maintaining effective corrective gradients in the tails. We further develop cross-batch QQ, which enlarges the effective ranking pool using detached samples from previous batches, and characterize its bias–variance trade-off. Across four control environments, QQWorld effectively improves the average planning success rate of LeWM, while consistently yielding better Gaussian alignment and thinner latent tails.
178:Two-Room
179:Reacher
180:PushT
181:OGBench-Cube
231:Table 1: Planning success rate (%, higher is better) across four environments. Values are reported as mean \pm standard deviation across six random seeds.
234:We follow LeWM (Maes et al., 2026) for dataset preprocessing, model training, and CEM-based goal-conditioned evaluation. Experiments are conducted on the same offline datasets, including Two-Room (Sobal et al., 2025), PushT (Zhou et al., 2025), Reacher (Tassa et al., 2018), and OGBench-Cube (Park et al., 2025). QQWorld differs from LeWM only by replacing the EP regularizer with the proposed QQ objective. We set the QQ regularization weight to 3.5 for all environments. Unlike baseline methods (Thil et al., 2026; Zhao et al., 2026) which introduce additional hyperparameters that need to be tuned, QQWorld does not have new hyperparameters or require extra tuning.
23
... [truncated 5868 chars]
```
</details>

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; sed -n '170,235p' T_2607.28415.txt", "description": "Read QQWorld Table 1 numbers"}
```

<details><summary>tool result</summary>

```
(21)
where the expectation is over the pooled samples used for estimation, and \phi is the standard Gaussian density.
For comparison, the standard QQ in Eq. 6 corresponds to K=0 in Eq. 21 and has the approximate target-estimation error:
\frac{1}{\phi(q(x))^{2}}\frac{F_{t}(x)(1-F_{t}(x))}{N}.
(22)
From Eq. 21 and Eq. 22, increasing K reduces the variance term from order N^{-1} to order \{N(K+1)\}^{-1}, but may introduce a nonzero staleness bias.
Therefore, Cross-Batch QQ is most useful in the small-batch regime (small N), where the variance of current-batch ranks is substantial and a short queue can provide more reliable learning targets. When the current batch is already sufficiently large, the additional variance reduction becomes limited and may be outweighed by representation staleness. We consequently use Cross-Batch QQ as an optional memory-efficient training strategy: it enables a small backpropagation batch to recover much of the rank-estimation benefit of a larger batch, while our base QQ loss is still preferable when memory permits a sufficiently large batch.
Method
Two-Room
Reacher
PushT
OGBench-Cube
Avg.
PLDM (Sobal et al., 2025)
97.00
78.00
78.00
65.00
79.50
DINO-WM (w/o proprio.) (Zhou et al., 2025)
100.00
79.00
74.00
86.00
84.75
DINO-WM (w/ proprio.) (Zhou et al., 2025)
100.00
—
92.00
—
—
LeWM (Maes et al., 2026)
84.33{\scriptstyle\pm 4.23}
82.67{\scriptstyle\pm 4.42}
84.67{\scriptstyle\pm 6.53}
67.33{\scriptstyle\pm 5.01}
79.75
Sub-JEPA (Zhao et al., 2026)
\mathbf{93.67}{\scriptstyle\pm 4.27}
81.00{\scriptstyle\pm 2.10}
89.00{\scriptstyle\pm 5.33}
69.00{\scriptstyle\pm 8.69}
83.17
SD-JEPA (Thil et al., 2026)
86.33{\scriptstyle\pm 6.12}
\underline{85.00}{\scriptstyle\pm 5.02}
\underline{89.67}{\scriptstyle\pm 4.27}
69.67{\scriptstyle\pm 5.13}
82.67
SMWM (Ivashkov et al., 2026)
\underline{88.67}{\scriptstyle\pm 7.12}
73.00{\scriptstyle\pm 4.69}
86.00{\scriptstyle\pm 2.53}
\mathbf{84.33}{\scriptstyle\pm 4.97}
83.00
QQWorld (Ours)
\mathbf{93.67}{\scriptst
... [truncated 1027 chars]
```
</details>

### ASSISTANT — 2026-09-16 17:15

Found a reference to "SMWM (Ivashkov et al., 2026)". Let me search for it plus a few more coverage queries.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; curl -s \"https://export.arxiv.org/api/query?search_query=au:%22Ivashkov%22+AND+all:%22world%20model%22&start=0&max_results=20&sortBy=submittedDate&sortOrder=descending\" | grep -E \"<id>|<title>|<published>\"", "description": "Search for SMWM Ivashkov"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; curl -s \"https://export.arxiv.org/api/query?search_query=all:%22Epps-Pulley%22&start=0&max_results=60&sortBy=submittedDate&sortOrder=descending\" | grep -E \"<id>|<title>|<published>\" | head -80", "description": "Search Epps-Pulley"}
```

<details><summary>tool result</summary>

```
(Bash completed with no output)
```
</details>

<details><summary>tool result</summary>

```
  <id>https://arxiv.org/api/bX9H3n7LfQwXK/w4jR5nN0HJmF0</id>
  <title>arXiv Query: search_query=all:"Epps-Pulley"&amp;id_list=&amp;start=0&amp;max_results=60</title>
    <id>http://arxiv.org/abs/2607.28415v1</id>
    <title>QQWorld: Quantile-Quantile Matching for World Model Regularization</title>
    <published>2026-07-30T16:00:39Z</published>
    <id>http://arxiv.org/abs/2512.19605v1</id>
    <title>KerJEPA: Kernel Discrepancies for Euclidean Self-Supervised Learning</title>
    <published>2025-12-22T17:41:26Z</published>
    <id>http://arxiv.org/abs/2109.04897v1</id>
    <title>On the eigenvalues associated with the limit null distribution of the Epps-Pulley test of normality</title>
    <published>2021-09-10T14:08:54Z</published>
    <id>http://arxiv.org/abs/2106.13962v1</id>
    <title>Bahadur efficiencies of the Epps--Pulley test for normality</title>
    <published>2021-06-26T08:02:28Z</published>
    <id>http://arxiv.org/abs/1901.03227v2</id>
    <title>Closed-form Expressions for Maximum Mean Discrepancy with Applications to Wasserstein Auto-Encoders</title>
    <published>2019-01-10T15:43:58Z</published>
    <id>http://arxiv.org/abs/1605.06293v2</id>
    <title>The performance of univariate goodness-of-fit tests for normality based on the empirical characteristic function in large samples</title>
    <published>2016-05-20T11:25:15Z</published>
```
</details>

### ASSISTANT — 2026-09-16 17:16

SMWM (Ivashkov) not on arXiv. Let me run a couple final coverage searches.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; curl -s \"https://export.arxiv.org/api/query?search_query=abs:%22LeWM%22&start=0&max_results=100&sortBy=submittedDate&sortOrder=descending\" | grep -E \"<id>http|<title>\" | grep -v \"arXiv Query\" | head -60", "description": "Search abstract for LeWM"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; curl -s \"https://export.arxiv.org/api/query?search_query=all:%222603.19312%22+OR+all:%222511.08544%22&start=0&max_results=100&sortBy=submittedDate&sortOrder=descending\" | grep -E \"<id>http|<title>\" | grep -v \"arXiv Query\" | head -60", "description": "Search for citing arXiv IDs"}
```

<details><summary>tool result</summary>

```
  <id>https://arxiv.org/api/zjBY36Em/SYQ61blBUSd+0ztzNs</id>
    <id>http://arxiv.org/abs/2609.03294v1</id>
    <title>Latent Energy Action Planning with World Models</title>
    <id>http://arxiv.org/abs/2609.04264v2</id>
    <title>Spectral-Target Physical Latent Structuring for JEPA-Style World Models</title>
    <id>http://arxiv.org/abs/2608.29029v1</id>
    <title>Flow-JEPA: Flow Matching for Robust Latent Dynamics in JEPA World Models</title>
    <id>http://arxiv.org/abs/2608.22764v1</id>
    <title>LpWM: A Case for Sparse Representations in World Models</title>
    <id>http://arxiv.org/abs/2608.18746v1</id>
    <title>Decision-Metric Alignment in Latent World Models: Diagnostics and Action-Conditioned Objectives for MPC Planning</title>
    <id>http://arxiv.org/abs/2608.17542v1</id>
    <title>No Gaussian Required: Contrastive Inverse Dynamics for JEPA World Models</title>
    <id>http://arxiv.org/abs/2608.16287v2</id>
    <title>SCALE: State-Calibrated Latent Embeddings for JEPA Planning in the Right Geometry</title>
    <id>http://arxiv.org/abs/2608.14125v1</id>
    <title>Traj-LeWM: Path-Aware World-Model Planning via Latent Trajectory Cost</title>
    <id>http://arxiv.org/abs/2608.12939v1</id>
    <title>Diagnosing JEPA World Models with Action-Conditioned Predictive Consistency</title>
    <id>http://arxiv.org/abs/2608.12078v1</id>
    <title>Better Slots, Better Worlds: Representation Quality &amp; Robustness in Object-Centric World Models</title>
    <id>http://arxiv.org/abs/2608.01926v1</id>
    <title>ProWorld: Progress-Aware Hyperbolic World Models for Long-Horizon Visual Goal Reaching</title>
    <id>http://arxiv.org/abs/2607.28415v1</id>
    <title>QQWorld: Quantile-Quantile Matching for World Model Regularization</title>
    <id>http://arxiv.org/abs/2607.26924v3</id>
    <title>Temporally Centered SIGReg Improves LeWorldModel Representations for Robot Policy Learning</title>
    <id>http://arxiv.org/abs/2607.26056v1</id>
    <title>INTACT: Isomorp
... [truncated 1876 chars]
```
</details>

<details><summary>tool result</summary>

```
  <id>https://arxiv.org/api/K0fMP6/SVSll+xO4mWnu+i4dFtk</id>
    <id>http://arxiv.org/abs/2608.10145v1</id>
    <title>The Evaluation Protocol Determines the Result: An Independent Reproduction of LeWorldModel on TwoRoom</title>
```
</details>

### ASSISTANT — 2026-09-16 17:16

Several new LeWM-extending papers found. Let me fetch the most relevant new ones.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; cd /tmp/wog_lit\nfor id in 2608.18746 2608.14125 2608.12939 2607.25337 2607.26056 2606.21775 2608.01926 2609.03294 2607.04978; do curl -s \"https://arxiv.org/abs/$id\" -o $id.html; sleep 0.4; done; python3 absx.py 2608.18746 2608.14125 2608.12939 2607.25337 2>&1 | head -60", "description": "Fetch and parse new LeWM extensions"}
```

<details><summary>tool result</summary>

```
==========================================================================================
ID: 2608.18746
TITLE: Decision-Metric Alignment in Latent World Models: Diagnostics and Action-Conditioned Objectives for MPC Planning
citation_date: 2026/08/19
citation_online_date: 2026/08/19
AUTHORS: Wang, Jiawei; Rui, Ke; Zuo, Yushen; Feng, Yichun; Li, Minglei
COMMENTS_RAW: <<NO COMMENTS FIELD>>
SUBJECTS: Machine Learning (cs.LG) ; Computer Vision and Pattern Recognition (cs.CV)
ABSTRACT: Abstract: JEPA-style latent world models can use Euclidean distance to a goal latent as the cost for model-predictive control (MPC). Strong decoding of task variables, however, does not guarantee that this particular cost ranks candidate action sequences by real task progress. We call the latter property \emph{decision-metric alignment}. We introduce Plan-Real Spearman, which measures latent--real rank agreement on random plans, and CEM-stage Spearman, which measures the same agreement as cross-entropy-method (CEM) search concentrates its proposal. We analyze sufficient conditions under which latent distance preserves real-cost rankings, identifying encoder distortion, terminal rollout error, and candidate margins as the controlling quantities. Guided by the observed empirical alignment gap, DA-LeWM augments LeWM with inverse-dynamics and demonstration-conditioned goal-action heads. Across all our experiments, DA-LeWM accelerates convergence and achieves higher online success than LeWM, while probe scores remain similar. These results show that action-conditioned objectives improve the geometry used by Euclidean-cost, CEM-based latent MPC.
HIST: Submission history From: Jiawei Wang [ view email ] [v1] Wed, 19 Aug 2026 09:56:39 UTC (666 KB)
==========================================================================================
ID: 2608.14125
TITLE: Traj-LeWM: Path-Aware World-Model Planning via Latent Trajectory Cost
citation_date: 2026/08/14
citation_online_date: 2026/08/14
AUTHORS: Hu
... [truncated 6322 chars]
```
</details>

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; python3 absx.py 2607.26056 2606.21775 2608.01926 2609.03294 2607.04978 2>&1 | grep -E \"^ID:|^TITLE:|^citation_date|^AUTHORS|^COMMENTS_RAW|^SUBJECTS\"", "description": "Parse remaining new papers metadata"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000; cd /tmp/wog_lit\n# Get LeVJEPA and SCALE + TC-SIGReg key quantitative numbers\ngrep -n -E \"58\\.0|80\\.0|TwoRoom-long|Table 2|Table 3\" T_2608.17542.txt | head -20", "description": "Grep AC-MTM quantitative tables"}
```

<details><summary>tool result</summary>

```
ID: 2607.26056
TITLE: INTACT: Isomorphic Intent-to-Action Learning for Search-Free World Models
citation_date: 2026/07/28
AUTHORS: Sun, Junhan; Zhao, Hao; Zhang, Guofeng
COMMENTS_RAW: 28 pages, 11 figures, including appendices
SUBJECTS: Robotics (cs.RO)
ID: 2606.21775
TITLE: Beyond the Next Step: Variable-Length Latent World Models for Long-Horizon Planning
citation_date: 2026/06/19
AUTHORS: Du, Tianqi; Zhang, Qi; Wang, Yifei; Wang, Yisen
COMMENTS_RAW: <<NO COMMENTS FIELD>>
SUBJECTS: Machine Learning (cs.LG) ; Artificial Intelligence (cs.AI)
ID: 2608.01926
TITLE: ProWorld: Progress-Aware Hyperbolic World Models for Long-Horizon Visual Goal Reaching
citation_date: 2026/08/03
AUTHORS: Liu, Zihan; Zhuang, Yuzhe; Li, Yuanzu; Gou, Wanshuang; Liu, Jiahong; Zhou, Min; Yang, Menglin
COMMENTS_RAW: 24 pages, 14 figures, 15 tables
SUBJECTS: Artificial Intelligence (cs.AI)
ID: 2609.03294
TITLE: Latent Energy Action Planning with World Models
citation_date: 2026/09/03
AUTHORS: Pham, Phu; Bera, Aniket
COMMENTS_RAW: 9 pages, 5 figures, 8 tables
SUBJECTS: Machine Learning (cs.LG)
ID: 2607.04978
TITLE: Qantara: Bridge-Flow Training for Multi-Paradigm JEPA Control
citation_date: 2026/07/06
AUTHORS: Rakhimov, Ruslan; Bredis, George; Maksyuta, Yuriy; Gavrilov, Daniil
COMMENTS_RAW: 16 pages, 3 figures, 6 tables. Project page: this https URL
SUBJECTS: Machine Learning (cs.LG) ; Computer Vision and Pattern Recognition (cs.CV); Robotics (cs.RO)
```
</details>

<details><summary>tool result</summary>

```
88:AC-MTM reaches 80.0{\pm}2.0\% success versus 58.0{\pm}2.0\% for
161:58.0% under the matched trajectory-goal MPC protocol, while AC-MTM
162:remains at 80.0%, with gains of 24, 20, and 22 points across the three training
329:goal offset 25, and interaction budget 50. TwoRoom-long changes only the goal offset and
363:Table 2: TwoRoom anti-collapse ablation with 200 evaluation episodes
398:58.0{\pm}2.0
399:\mathbf{80.0{\pm}2.0}
401:Table 3: Planning success (%) with the shared autoregressive planner: 200
421:SIGReg falls to a three-seed mean of 58.0% success while AC-MTM
422:achieves 80.0%. The random-policy run helps calibrate these numbers: like
454:58.0
456:58.0{\pm}2.0
465:80.0
468:\mathbf{80.0{\pm}2.0}
489:On TwoRoom-long the non-contrastive inverse-regression ablation is stronger:
517:TwoRoom-long (100/150)
778:OGBench Visual Scene, SIGReg reaches 58.0% while AC-MTM reaches
779:80.0% across matched seeds, though the Scene gain may also reflect lower
781:TwoRoom-long AC-MTM retains most of the inverse-dynamics gain over
962:We keep 100/150 only as the separately labeled TwoRoom-long stress test.
```
</details>

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; sed -n '363,420p' T_2608.17542.txt", "description": "Read AC-MTM Table 2/3"}
```

<details><summary>tool result</summary>

```
Table 2: TwoRoom anti-collapse ablation with 200 evaluation episodes
per seed. Values are mean \pm standard deviation over three training seeds.
Plain next-latent prediction reaches a trivial low-loss solution and plans
substantially worse.
4.2 Planning results in the LeWM evaluation frame
Table 3 gives the controlled AC-MTM versus SIGReg
comparison, while Figure 3 restores the original evaluation
style by including the external paper-reported baselines. The figure is
contextual: PLDM, DINO-WM, GCBC, GCiQL, and GCIVL are taken from the LeWM
paper, while the statistical claims come from our matched SIGReg reruns.
AC-MTM uses the same test-time planner as LeWM and improves over
SIGReg on TwoRoom, Cube, and Scene, matches it on Reacher, and trails it on
PushT. The PushT deficit is analyzed in Section 4.5; the full
MTM-MSE ablation is reported in Appendix C.
Task
SIGReg (LeWM)
AC-MTM (ours)
\Delta
TwoRoom
85.5{\pm}0.4
\mathbf{90.7{\pm}0.6}
+5.2
Reacher
68.8{\pm}0.2
68.3{\pm}3.1
-0.5
PushT
\mathbf{93.2{\pm}0.2}
86.7{\pm}1.5
-6.5
OGB-Cube
66.2{\pm}0.2
\mathbf{78.8{\pm}1.7}
+12.6
OGB-Scene
58.0{\pm}2.0
\mathbf{80.0{\pm}2.0}
+22.0
Table 3: Planning success (%) with the shared autoregressive planner: 200
evaluation episodes per seed on the standard tasks and 50 on OGBench-Scene.
Values are mean \pm standard deviation over three training seeds; bold marks
differences larger than the cross-seed noise. Figure 3 places
these numbers next to external paper-reported baselines.
Figure 3: Planning success across the five environments. Salmon and blue
bars are paper-reported LeWM/external baselines from
Maes et al. (2026); purple, gold, and green bars are our controlled
SIGReg, MTM-MSE, and AC-MTM evaluations across three training
seeds. No external baseline has published OGBench-Scene results under this
protocol, so the Scene panel contains only our controlled models and the random
policy. The external baselines place AC-MTM in the original evaluation
frame; direct statistical claim
... [truncated 471 chars]
```
</details>

### ASSISTANT — 2026-09-16 17:16

I have comprehensive data. Let me get the final quantitative numbers for SCALE and TC-SIGReg.

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; echo \"=== SCALE Table1 alignment ===\"; sed -n '120,160p' T_2608.16287.txt; echo \"=== SCALE conclusion ===\"; sed -n '/^7 Conclusion/,+6p' T_2608.16287.txt | head -10", "description": "Read SCALE alignment table and conclusion"}
```

<details><summary>tool result</summary>

```
=== SCALE Table1 alignment ===
4 Where Task-Relevant State Lives
4.1 A motivating distance aligment contrast
The planner of Sec. 3.1 interacts with the representation only through the scalar cost of Eq. (3): whatever a representation encodes, the planner sees it only insofar as it registers in latent distance. We therefore quantify the contrast sketched in the introduction directly at this interface, computing the Spearman rank correlation between \lVert\mathbf{z}_{i}-\mathbf{z}_{j}\rVert_{2}^{2} and \lVert\mathbf{q}_{i}-\mathbf{q}_{j}\rVert_{2}^{2} over held-out trajectory frame pairs, where \mathbf{q} denotes the task-relevant state variables logged alongside each trajectory (formally specified in Sec. 5.1). For the sampled frame pairs, we rank the latent distances and state distances separately. Spearman \rho measures the agreement between these two rankings: a high value means that pairs farther apart in task state also tend to be farther apart in latent space. This is directly relevant because the planner selects actions by ranking candidates according to latent cost. Ties are assigned average ranks.
Table 1: Latent–state distance rank alignment (Spearman \rho) of the two reference world models over held-out frame pairs. Both models decode most of the same state accurately from their full embeddings (Appendix B); their metrics reflect it to very different degrees.
Task
LeWM
DINO-WM
Push-T
.13
.70
Reacher
.18
.46
Cube
.00
.46
Two-Room
.41
.85
PointMaze
.52
.50
Table 1 reveals a clear contrast. DINO-WM’s latent distances track state differences much more closely than LeWM’s on most tasks. The metric the planner will consume can be uninformative about task-relevant differences, even though probes still recover most of that state accurately from the full embedding (Appendix B). We next ask how such a dissociation between decodability and metric alignment can arise.
4.2 Metric leverage: which directions can influence the cost
Let \mu be the mean of the encoded distr
... [truncated 3087 chars]
```
</details>

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; grep -n -E \"63\\.6|83\\.8|1\\.66|Goal suite|LIBERO\" T_2607.26924.txt | head -15", "description": "Grep TC-LeWM quantitative results"}
```

<details><summary>tool result</summary>

```
61:On the LIBERO benchmark, our method improves downstream policy success on the Goal suite by 1.66× and raises the average success rate across all suites from 63.6% to 83.8%.
86:On the LIBERO-Long suite (Liu et al., 2023), behavior-cloning policies trained on Raw LeWM representations achieve only 26.1\% average success under independent single-task training.
87:Increasing the scale of world model training partially improves downstream policy performance, reaching 44.7\% under suite-wise 10-task joint training and 52.0\% when the world model is trained jointly on all 40 LIBERO tasks, but does not eliminate this performance limitation.
116:Across the four LIBERO suites, TC-LeWM raises the suite-wise 10-task average success rate of the downstream policy from 63.6\% to 83.8\%.
117:When the world model is jointly trained on all 40 LIBERO tasks, TC-LeWM improves the average downstream policy success rate from 72.6\% to 85.4\%.
136:of downstream behavior-cloning policies across three different training scales on LIBERO.
197:(c) Empirical variance allocation of trained LIBERO representations.
222:We evaluate Raw LeWM at three training scales of LIBERO-Long. When each of the ten tasks is trained
225:training the world model jointly on all 40 LIBERO tasks further raises it to
322:training on the ten LIBERO-Long tasks.
417:63.6
422:63.6\pm6.8
423:83.8
458:training. All four LIBERO suites share one encoder and latent predictor,
487:We evaluate TC-LeWM on the LIBERO benchmark through closed-loop manipulation
```
</details>

### ASSISTANT — 2026-09-16 17:17

Let me extract institutions for the key papers.

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; python3 - <<'EOF'\nimport re,html,os\ndef aff(f):\n    if not os.path.exists(f): return 'NO HTML'\n    h=open(f,encoding='utf-8',errors='replace').read()\n    m=re.search(r'(?s)<div class=\"ltx_authors\">(.*?)</div>\\s*(?:<div class=\"ltx_abstract\"|<blockquote)',h)\n    if not m:\n        m=re.search(r'(?s)<div class=\"ltx_authors\">(.*?)</div>',h)\n    if not m: return 'NOT FOUND'\n    t=re.sub(r'\\s+',' ',html.unescape(re.sub(r'<[^>]+>',' ',m.group(1)))).strip()\n    return t[:400]\nfor i in ['2608.10145','2608.17542','2605.09241','2607.12547','2606.26217','2607.26924','2606.02572','2606.01443','2606.17603','2607.28415','2608.27395','2607.15295','2608.16287','2608.12959','2609.04264','2608.22764','2605.26900','2605.20107','2605.26379']:\n    print(i,'::',aff('best_%s.html'%i))\nEOF", "description": "Extract author affiliations from HTMLs"}
```

<details><summary>tool result</summary>

```
2608.10145 :: Joyjeet Singh joyjeetsingh1@gmail.com Affiliation: Independent researcher Affiliation: ORCID 0009-0005-1512-7439
2608.17542 :: Jack Boylan Chris Hokamp Affiliation: Quantexa Email: {jackboylan,chrishokamp}@quantexa.com
2605.09241 :: Kai Zhao 1 Dongliang Nie 1 Yuchen Lin 1 Zhehan Luo 2 Yixiao Gu 1 Deng-Ping Fan 3 Dan Zeng 1 Affiliation: 1 Shanghai University 2 The University of Manchester 3 Nankai University Email: https://github.com/intcomp/sub-jepa
2607.12547 :: Niccoló Caselli * Affiliation: University of Amsterdam Francesco Massafra * Affiliation: University of Amsterdam Samuele Punzo * Affiliation: University of Amsterdam Salvatore Lo Sardo * Affiliation: University of Amsterdam Ippokratis Pantelidis * Affiliation: University of Amsterdam Sathya Kamesh Bhethanabhotla Affiliation: University of Amsterdam * Equal contribution
2606.26217 :: Yuntian Gao Affiliation: Xi’an Jiaotong University Xiangyu Xu Affiliation: Xi’an Jiaotong University Corresponding author
2607.26924 :: Chang Liu Fei Suo Yanzhou Jin Zeyu Ping Affiliation: Yusuke Iwasawa Yutaka Matsuo Yaonan Zhu † Affiliation: Graduate School of Engineering, The University of Tokyo, Tokyo, Japan Affiliation: † Corresponding author: yaonan.zhu@weblab.t.u-tokyo.ac.jp
2606.02572 :: Haiyu Wu Affiliation: Altos Labs Correspondence to: hwu@altoslabs.com Randall Balestriero Affiliation: Brown University Morgan Levine Affiliation: Altos Labs
2606.01443 :: Triet M. Le Ph.D † † thanks: Spatiolyx LLC, https://spatiolyx.ai, tml@spatiolyx.ai
2606.17603 :: Léo Nicollier leo.nicollier@gmail.com Affiliation: Université Paris-Saclay, CNRS, ENS Paris-Saclay, Centre Borelli, France Affiliation: Advanced Track and Trace Enric Meinhardt-Llopis Affiliation: Université Paris-Saclay, CNRS, ENS Paris-Saclay, Centre Borelli, France Max Dunitz Affiliation: Advanced Track and Trace Marc Pic Affiliation: Advanced Track and Trace Pablo Musé Affiliation: IIE, Facult
2607.28415 :: Zhoushun Yu Affiliation: Xi’an Jiaotong Universit
... [truncated 2550 chars]
```
</details>

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; python3 - <<'EOF'\nimport re,html\nh=open('best_2608.27395.html',encoding='utf-8',errors='replace').read()\ni=h.find('ltx_personname')\nprint(re.sub(r'\\s+',' ',html.unescape(re.sub(r'<[^>]+>',' ',h[i-300:i+1800]))))\nEOF", "description": "Extract LeVJEPA affiliations"}
```

<details><summary>tool result</summary>

```
(Bash completed with no output)
```
</details>

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; head -60 T_2608.27395.txt | tail -35", "description": "Read LeVJEPA header"}
```

<details><summary>tool result</summary>

```
3.1.1 Invariance.
3.1.2 Regularization.
3.2 Architecture
3.3 Pretraining Data and Evaluation Setup
Pretraining.
Evaluation.
4 What Matters for Efficient Video Pretraining?
4.1 Token Dropping Improves Representations
4.2 Local View Budget
4.3 Temporal Patch Aggregation Is Not Required
4.4 Causal Attention Incurs No Accuracy Penalty
5 Comparison with Prior Work
5.1 Comparison to Video Models
FLOP-matched comparison.
5.2 Comparison to Image Models
5.3 Pretraining on consumer hardware
5.4 Scaling the Pretraining Data
5.5 Emergent Token-Level Structure
6 Discussion
Acknowledgments
References
A SIGReg Implementation Details
B Architectural Details
C Evaluation Details
Attentive probing.
Linear probing on Kinetics-400.
 License: CC BY 4.0
arXiv:2608.27395v1 [cs.CV] 27 Aug 2026
LeVJEPA: Efficient & Scalable Video Pretraining
without the Heuristics
Lukas Kuhn1,2 Lucas Maes4,5 Giuseppe Serra1,2 Quentin Le Lidec8
Yann LeCun7,8 Randall Balestriero*,6,8 Florian Buettner*,1,2,3
1German Cancer Research Center 2German Cancer Consortium 3Goethe University Frankfurt 4Mila
5Université de Montréal 6Brown University 7Courant Institute, New York University
8Advanced Machine Intelligence (AMI Labs)
```
</details>

### ASSISTANT — 2026-09-16 17:17

I now have everything needed. Let me verify the LeVJEPA discussion section for limitations.

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; sed -n '/^6 Discussion/,/^Acknowledgments/p' T_2608.27395.txt | head -12", "description": "Read LeVJEPA discussion"}
```

<details><summary>tool result</summary>

```
6 Discussion
Acknowledgments
```
</details>

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; grep -n \"Discussion\" T_2608.27395.txt; sed -n '340,360p' T_2608.27395.txt", "description": "Find LeVJEPA discussion body"}
```

<details><summary>tool result</summary>

```
44:6 Discussion
194:6  Discussion
Leworldmodel: stable end-to-end joint-embedding predictive architecture from pixels.
arXiv preprint arXiv:2603.19312.
Cited by: §1,
§6.
[21]
L. Mur-Labadia, M. Muckley, A. Bar, M. Assran, K. Sinha, M. Rabbat, Y. LeCun, N. Ballas, and A. Bardes (2026)
V-jepa 2.1: unlocking dense features in video self-supervised learning.
External Links: 2603.14482,
Link
Cited by: Figure 3,
§5.5.
[22]
M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al. (2023)
Dinov2: learning robust visual features without supervision.
arXiv preprint arXiv:2304.07193.
Cited by: §5.2.
[23]
M. Oquab, T. Darcet, T. Moutakanni, H. V. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. HAZIZA, F. Massa, A. El-Nouby, M. Assran, N. Ballas, W. Galuba, R. Howes, P. Huang, S. Li, I. Misra, M. Rabbat, V. Sharma, G. Synnaeve, H. Xu, H. Jegou, J. Mairal, P. Labatut, A. Joulin, and P. Bojanowski (2024)
DINOv2: learning robust visual features without supervision.
Transactions on Machine Learning Research.
Note: Featured Certification
```
</details>

**tool call `Bash`**
```
{"command": "cd /tmp/wog_lit; sed -n '194,205p' T_2608.27395.txt", "description": "Read LeVJEPA discussion section body"}
```

<details><summary>tool result</summary>

```
6  Discussion
This work indicates that self-supervised learning from video can be both simpler and substantially more compute-efficient than previously assumed. A single encoder, trained with an invariance loss and a distributional regularizer whose single hyperparameter is not tuned, attains accuracy comparable to or exceeding that of established video joint-embedding methods at a fraction of their total pretraining compute; the target encoder, predictor network, and associated schedules that these methods employ for stability are not required under the present objective. The analysis of Section 4 extends this simplification to individual design choices: masking structure, temporal patch aggregation, and attention topology become free parameters that can be selected for efficiency or for downstream requirements, and the resulting configuration, sparse random observation of per-frame tokens under block-causal attention, is simultaneously the cheapest and among the most accurate that we evaluate.
The comparison to image-based pretraining suggests a broader implication. Image pretraining has constituted the standard source of general-purpose visual representations, with video regarded as a specialized complement for motion understanding, justified where its additional cost is warranted. Under the present objective, this division is no longer imposed by efficiency: at equal total compute on identical source data, the video-pretrained encoder approaches a DINOv2 baseline on appearance-centric evaluation while attaining nearly twice its accuracy on motion-centric evaluation. Since video strictly contains the appearance information of its constituent frames while additionally supplying temporal structure, these results indicate that video is a viable substrate for general-purpose visual pretraining rather than a specialized one, and that the choice between the two paradigms may increasingly be governed by data availability rather than by computational cost. The data-scali
... [truncated 3638 chars]
```
</details>

### ASSISTANT — 2026-09-16 17:22

I have complete data on all assigned papers plus substantial additional coverage. Here is the report.

---

# LeJEPA / SIGReg / LeWorldModel — 2026 Follow-Up Cluster

**Method**: All metadata below was extracted by `curl`-ing `https://arxiv.org/abs/<id>` and parsing `citation_*` meta tags + the Comments/Subjects table cells. Full texts were fetched from `https://arxiv.org/html/<id>v{1,2,3}` and grepped. Everything marked **VERIFIED** was directly curled by me. No fetch failed for any assigned ID — all 14 assigned abs pages and all 14 HTML full texts resolved (HTML at the version noted).

**Note on `citation_date` vs `citation_online_date`**: arXiv's `citation_date` = v1 submission date; `citation_online_date` = announcement/latest-version date. Where they differ materially I flag it (e.g., anchor: v1 2026-03-13, online 2026-06-03 = the v3 announcement).

---

## 1. ANCHOR — LeWorldModel (LeWM)

**arXiv 2603.19312** · https://arxiv.org/abs/2603.19312 · **VERIFIED**
- **Title**: *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels*
- **citation_date** 2026/03/13 · **citation_online_date** 2026/06/03 · versions v1 2026-03-13, v2 2026-03-24, v3 2026-06-03
- **Authors**: Lucas Maes\*, Quentin Le Lidec\*, Damien Scieur, Yann LeCun, Randall Balestriero
- **Institutions** (from HTML): Mila & Université de Montréal; New York University; Samsung SAIL; Brown University
- **Comments raw**: `<<NO COMMENTS FIELD>>` (the abs page has no Comments row)
- **Subjects**: Machine Learning (cs.LG); Artificial Intelligence (cs.AI)
- **One-liner**: First JEPA trained stably end-to-end from raw pixels with only two loss terms (next-embedding prediction + SIGReg), ~15M params, 1 GPU, plans up to 48× faster than foundation-model WMs.
- **Hyperparameter claim**: reduces tunable loss hyperparameters "from six to one" vs. the only existing end-to-end alternative (PLDM).

**Verbatim limitations (§"Limitations & Future Work")**:
> "Several limitations point to future directions. Planning remains restricted to short horizons, motivating hierarchical world modeling for long-horizon reasoning. Our method also relies on offline datasets with sufficient coverage; in particular, low data diversity weakens SIGReg in simple, low-dimensional environments where matching a high-dimensional Gaussian prior is harder. Pre-training on large, diverse video datasets could provide stronger priors and reduce domain-specific data needs. Finally, dependence on action labels could be alleviated by inverse dynamics modeling."

**The paper's own admission of the SIGReg weakness (§4.2, verbatim)** — this is the seed that four follow-ups exploit:
> "Interestingly, LeWM performs worse on the simplest environment, Two-Room. A possible explanation is that the low diversity and low intrinsic dimensionality of this dataset make it difficult for the encoder to match the isotropic Gaussian prior enforced by SIGReg in a high-dimensional latent space, which may lead to a less structured latent representation. This highlights a potential limitation of the SIGReg regularization in very low-complexity environments."

And (§4.3, verbatim) — the planning/representation dissociation that later papers formalize:
> "Although LeWM underperforms PLDM in downstream planning on this environment, it matches or outperforms PLDM across all probing metrics… This suggests that the learned latent space captures the underlying physical state equally well and that the planning gap is not due to a less informative representation but rather to other factors such as the dynamics model or the planning procedure itself."

Other anchor facts (VERIFIED): λ (SIGReg weight) is the only effective hyperparameter; success >80% for λ∈[0.01,0.2], peak ≈0.09, degrades sharply at λ=0.5. Reference numbers later reused as baselines: Two-Room 84.33±4.23, Reacher 82.67±4.42, PushT 84.67±6.53, OGB-Cube 67.33±5.01 (six seeds, as cited by Sub-JEPA/QQWorld).

---

## 2. THE CRITIQUES (highest priority)

### 2.1 Independent reproduction — does it reproduce? *Partly, and the failure is protocol-level.*

**arXiv 2608.10145** · https://arxiv.org/abs/2608.10145 · **VERIFIED** (HTML v1)
- **Title**: *The Evaluation Protocol Determines the Result: An Independent Reproduction of LeWorldModel on TwoRoom*
- **citation_date** 2026/08/10 · **citation_online_date** 2026/08/10 · v1 only
- **Author / institution**: Joyjeet Singh — **Independent researcher** (ORCID 0009-0005-1512-7439)
- **Comments raw**: `Independent reproduction of arXiv:2603.19312 - this https URL`
- **Subjects**: Machine Learning (cs.LG)
- **Relevance**: The single most damaging paper in the cluster. It does NOT claim LeWM is wrong; it claims the *published number is protocol-contingent* and the *released configs alone do not train*.

**Headline quantitative disagreement (VERIFIED, verbatim)**:
> "the evaluation protocol described in the paper does not reproduce the paper's reported result on the paper's own released weights, and the released repository's evaluation defaults do. We do not know which the authors used, and we do not assume"

> "The three claims we set out to test (§2) resolve as follows. The representation claim reproduces directly and easily. The planning claim reproduces under one of the two evaluation protocols the released material publishes — 94.0% at the repository's goal offset against a reported ~87%, and 84.0% for the authors' own weights under that same protocol — once four undocumented conventions are corrected; under the protocol the paper's appendix describes, the authors' own weights reach 14.0%. The training claim does not reproduce from the released configuration files alone, and does reproduce once those conventions are supplied — which we take to be a documentation gap rather than a defect in the method."

> "On fifty identical episodes, changing nothing but how the goal is constructed moves that checkpoint from 84.0% to 8.0%." (abstract) — with in-text: "The result is 84.0% against 8.0%: 39 episodes are solved under the first and missed under the second… \chi^2 = 34.2, p < 10^-8. No property of the model changed between those two numbers."

**Four undocumented conventions (verbatim)**: "dense action gathering across a frameskip block, a programmatically-set action-encoder width, ImageNet pixel normalisation, and action z-scoring. A reproducer following the released configurations alone obtains a model whose predictor cannot converge."

**Twelvefold preprocessing sensitivity (verbatim)**: "Encoding frames under raw [0,1] pixel values, their checkpoint predicts the next latent state with an error 5.1 times a frozen-world baseline — five times worse than assuming nothing moves. Under ImageNet normalisation the same weights score 0.410. A twelvefold difference makes the convention effectively mandatory".

**Dataset-level objection to the paper's own appendix protocol (verbatim)**: "The dataset contains 10,000 episodes of mean length 92.1 and maximum length exactly 101. An episode can supply a goal 100 frames after its start only if it runs longer than 100 frames, so the eligible set is precisely the 6,056 episodes that reached the length cap. That cap is a timeout… Of the 6,056 eligible episodes, only 91 — 1.5% — were solved by the policy that generated them. Evaluating at an offset of 100 therefore asks the planner to reach the point at which a failing policy ran out of time." Also: "the maximum sampling span across all 6,056 eligible episodes is zero… the description is implementable at the offset the configuration uses, and not at the offset the appendix states."

**The generalizing negative result (verbatim)**: "A world model that predicts one step ahead seven times more accurately than another was not detectably better at short-horizon planning and was decisively worse at long-horizon planning — and this holds for the authors' released checkpoint as well as for ours. Selecting a world model by held-out prediction error is not a reliable way to select a world model for planning".

**Its own self-declared limitations (verbatim, §7)** — important for calibrating how much weight to give it:
> "A single seed throughout. Every training run, every planning evaluation and every probe in this paper uses one seed. We make no estimate of seed variance, and none of our differences should be read as robust to reinitialisation… the differences we report between checkpoints (94.0% against 84.0% at goal offset 25, p = 0.0625; 94.0% against 78.0%, p = 0.0574) are already not established at our sample size of fifty episodes."
> "Ten epochs, not one hundred. We follow the paper's appendix, which states ten epochs; the released repository configuration specifies max_epochs: 100… a reader with our resources cannot test the repository's own configuration."
> "Our results characterise our checkpoints, not the method… A same-room planning advantage that was pre-registered, distance-matched, verified reachable by an oracle, and significant at p = 3.4×10⁻⁸ on one checkpoint fell to +12.7 points under a change of action scaling and to −6.4 points on a different checkpoint."
> "The mechanism of the long-horizon reversal is untested… Three checkpoints cannot separate prediction accuracy from context length."
> "Scope. All results concern the TwoRoom diagnostic environment. We make no claim about the original's embodied or zero-shot results, about its other environments, or about the method's behaviour at scales other than the 18.03M-parameter configuration studied here."
> "a batch normalisation layer inflated our reported validation loss by up to a factor of 300, concealing a training loss that was flat throughout." (abstract)

### 2.2 Direct challenge to the isotropic-Gaussian premise

**arXiv 2608.17542** · https://arxiv.org/abs/2608.17542 · **VERIFIED** (HTML v1)
- **Title**: *No Gaussian Required: Contrastive Inverse Dynamics for JEPA World Models*
- **citation_date** 2026/08/18 · **citation_online_date** 2026/08/18 · v1 only
- **Authors / institution**: Jack Boylan, Chris Hokamp — **Quantexa**
- **Comments raw**: `17 pages, 5 figures. Code: this https URL`
- **Subjects**: Machine Learning (cs.LG); Artificial Intelligence (cs.AI)
- **One-liner**: Replaces SIGReg with Action-NCE (inverse-dynamics contrastive head, discarded after training) — a *distribution-free* anti-collapse signal; test-time model is byte-identical to LeWM.

**The core philosophical critique (verbatim, abstract + Conclusion)**:
> "LeWorldModel (LeWM) prevents collapse with SIGReg, a regularizer that forces the latent distribution to match an isotropic Gaussian: the representation is stabilized by prescribing what it must look like, independently of the environment it models. We argue that the anti-collapse pressure can instead come from the transition data itself."
> "A world model's representation has to be protected from collapse, but nothing about the problem says that protection must take the form of a prescribed latent distribution."

**Quantitative disagreement (VERIFIED, Table 3 — 3 seeds, 200 eval episodes/seed; 50 on Scene)**:

| Task | SIGReg (LeWM) | AC-MTM | Δ |
|---|---|---|---|
| TwoRoom | 85.5±0.4 | **90.7±0.6** | +5.2 |
| Reacher | 68.8±0.2 | 68.3±3.1 | −0.5 |
| PushT | **93.2±0.2** | 86.7±1.5 | **−6.5** |
| OGB-Cube | 66.2±0.2 | **78.8±1.7** | +12.6 |
| OGB-Scene | 58.0±2.0 | **80.0±2.0** | **+22.0** |

> "On OGBench Visual Scene, where a single arm controls a drawer, window, buttons, and a movable cube, SIGReg drops to a three-seed mean of 58.0%… while AC-MTM remains at 80.0%, with gains of 24, 20, and 22 points across the three training seeds. A single 50-episode random-policy run gives a 52% baseline estimate."

**Where it concedes SIGReg is better (verbatim, §5 "What Action-NCE does not solve")** — critical for a balanced read:
> "The contrastive loss still emphasizes state that separates actions. It does not unconditionally preserve every task-relevant variable, unlike SIGReg's global variance pressure. PushT requires accurate block pose and orientation even when the same commanded action has little visible effect before contact. This is a plausible reason both inverse variants remain below SIGReg."

**Self-declared limitations (verbatim, §7)**:
> "AC-MTM provides no unconditional geometric guarantee. It requires variation in the action candidates and useful visual evidence about action effects; unobserved actuators, stochastic dynamics, no-op-heavy datasets, or many duplicate actions can weaken the contrastive task. We evaluate normalized continuous controls only. Discrete, hybrid, structured, and very high-dimensional action spaces may require different scores…"
> "PushT shows the flip side of the dynamics-native pressure: a variable that actions barely move can be underweighted even when it matters for the task"
> "under the official 750-step fixed-goal protocol, neither method solves the tasks at this model scale (Appendix B)."
> "the Scene gain may also reflect lower one-step forward error." (i.e., it explicitly hedges its own headline result)

Also notable: SIGReg *wins decisively* on the surprise/violation-of-expectation diagnostic (PushT action-counterfactual 151.7±4.7× vs AC-MTM 9.6±0.3×; state-discontinuity 1246.0±34.1× vs 40.0±1.4×) — though the paper cautions ratios are within-model normalized and "should be compared within a model, not across models".

### 2.3 "Isotropic Gaussian in ambient space is too strong a bias"

**arXiv 2605.09241** · https://arxiv.org/abs/2605.09241 · **VERIFIED** (HTML v1)
- **Title**: *Sub-JEPA: Subspace Gaussian Regularization for Stable End-to-End World Models*
- **citation_date** 2026/05/10 · **citation_online_date** 2026/05/10 · v1 only
- **Authors / institutions**: Kai Zhao, Dongliang Nie, Yuchen Lin, Zhehan Luo, Yixiao Gu, Deng-Ping Fan, Dan Zeng — **Shanghai University**; University of Manchester; Nankai University
- **Comments raw**: `this https URL` (bare code link)
- **Subjects**: Machine Learning (cs.LG); Artificial Intelligence (cs.AI)
- **Note**: the abs-page abstract is mangled — "this http URL" tokens replaced sentence-ending words, and the method name renders as "ame" (a broken LaTeX macro). The HTML full text is clean.

**The critique (verbatim)**:
> "latent representations typically lie on low-dimensional manifolds embedded within the high-dimensional ambient space… and enforcing an isotropic Gaussian prior directly in this ambient space introduces an overly strong bias that mismatches the intrinsic geometry"
> "Our key insight is that the isotropic Gaussian prior enforced by LeWM operates in the full high-dimensional latent space, which introduces an excessive bias when the underlying task dynamics reside on a low-dimensional manifold."

**Mechanistic evidence (verbatim, §4.2.2)**:
> "This direct correspondence strongly supports our hypothesis: when task-relevant features lie on a low-dimensional manifold, the full-space Gaussian prior in LeWM forces the latent space into unnecessarily high rank, imposing a bias that hinders task-aligned representation learning. Subspace-wise regularization relaxes this bias and allows the latent geometry to contract toward the task's intrinsic dimensionality—the greater this contraction, the larger the planning gain."

**Quantitative disagreement (VERIFIED, Table 1 — 6 seeds)**: LeWM 84.33±4.23 / 82.67±4.42 / 84.67±6.53 / 67.33±5.01 → Sub-JEPA **95.00±2.76 / 84.00±4.00 / 89.00±5.33 / 76.33±5.99** (Two-Room / Reacher / PushT / OGB-Cube). Largest gain (+10.7 pts) is exactly on Two-Room — the environment LeWM itself flagged.
> "the improvement is most prominent on Two-Room. LeWM is known to struggle here because enforcing isotropic Gaussianity over the entire high-dimensional embedding introduces an overly restrictive bias for this low-intrinsic-dimension task."

**Its own bias-variance caveat (verbatim, §4.3.2)**:
> "performance is stable across a broad mid-range of (K,d_s) configurations, but degrades sharply when subspaces become too narrow (small d_s), where the normality signal grows unreliable. This reveals a bias-variance tradeoff in subspace design: larger K relaxes the global prior but requires sufficient d_s to maintain regularization stability."
> "Increasing K relaxes the global Gaussian prior and enables more expressive latent structure, while excessively small subspaces may weaken the effectiveness of the regularization signal."

**No dedicated Limitations section** — the paper has none (COULD NOT CONFIRM any beyond the ablation caveats above).

### 2.4 Pitfalls of hierarchical planning

**arXiv 2607.12547** · https://arxiv.org/abs/2607.12547 · **VERIFIED** (HTML v2)
- **Title**: *Mind the Gap: Promises and Pitfalls of Hierarchical Planning in LeWorldModel*
- **citation_date** 2026/07/14 · **citation_online_date** 2026/07/15 · v1 2026-07-14, v2 2026-07-15
- **Authors / institution**: Niccolò Caselli\*, Francesco Massafra\*, Samuele Punzo\*, Salvatore Lo Sardo\*, Ippokratis Pantelidis\*, Sathya Kamesh Bhethanabhotla — all **University of Amsterdam** (\*equal contribution)
- **Comments raw**: `Accepted at WM@Booth 2026`
- **Subjects**: Robotics (cs.RO); Artificial Intelligence (cs.AI); Machine Learning (cs.LG)
- **Relevance**: Directly attacks the anchor's own top-listed future direction ("motivating hierarchical world modeling for long-horizon reasoning") and reports that naive hierarchy *fails*.

**Verbatim pitfalls**:
> "hierarchy does not automatically improve performance: at short horizons, the best configuration uses a one-step high-level horizon, while longer horizons reveal a mismatch between the learned high-level action space and the inference-time search distribution."
> "The main failure mode is therefore not low-level executability alone, nor temporal abstraction alone. Unconstrained high-level CEM can search macro-actions outside the support induced by training trajectories, producing subgoals that appear favorable under the learned terminal objective but are poor targets for control."
> "On PushT at d=50, oracle-subgoal acting reaches 73.3±4.2% success… whereas generated targets are diffuse, poorly localized, and inconsistent with reachable intermediate configurations. This suggests that the main failure is not low-level executability alone, but the quality of high-level subgoal generation."

**Positive result when repaired**: +11.3 pp at medium horizons, +14.7 pp at longest PushT horizon over flat LeWM.

**Verbatim Limitations and Future Work**:
> "Our results should be interpreted as an analysis of hierarchy in LeWM rather than a complete evaluation of hierarchical world-model design. We primarily use PushT because varying the goal offset d gives a controlled way to study longer-horizon planning; broader environments are needed to test how far the conclusions generalize. We also leave retraining-based representation choices, especially VQ macro-actions, to future work… the cost of retraining and tuning these variants led us to prioritize inference-time interventions"
> "Interpretation should also be conditioned on model scale. Zhang et al. (2026) report substantially stronger PushT performance with a higher-capacity DINO-WM setup, so our findings should not be read as a general limitation of hierarchy."
> Appendix: "We report Cube results in the appendix because OGBCube-v0 is less diagnostic for the long-horizon planning failure that motivates the main text. Unlike PushT, the flat LeWM baseline remains comparatively stable as the goal offset increases: it obtains 65.3%, 52.0%, and 53.3% at d∈{25,50,75}".

---

## 3. LeWM EXTENSIONS (assigned)

### 3.1 Fast LeWorldModel
**arXiv 2606.26217** · https://arxiv.org/abs/2606.26217 · **VERIFIED** (HTML v1)
- **citation_date/online** 2026/06/24 · v1 only · **Yuntian Gao, Xiangyu Xu — Xi'an Jiaotong University**
- **Comments raw**: `<<NO COMMENTS FIELD>>` · **Subjects**: cs.LG; cs.CV; cs.RO
- **One-liner**: Replaces autoregressive one-step rollout with *action-prefix predictio
... [truncated 36190 chars]
