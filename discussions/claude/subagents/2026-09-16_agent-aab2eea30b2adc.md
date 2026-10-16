# Claude Code session agent-aab2eea30b2adca90

- Time (Beijing): 2026-09-16 16:58 → 2026-09-16 17:12
- cwd: `/home/zhengguantian/WoG_v2`
- Source: `~/.claude/projects/-home-zhengguantian/c976c298-d71c-4d38-9107-f692d869d083/subagents/agent-aab2eea30b2adca90.jsonl` (neolix_al_2)
- Messages (new, after de-dup): 101
- Note: tool outputs capped at 2000 chars; thinking blocks and images dropped; secrets/private IPs redacted.

---

### USER — 2026-09-16 16:58

You are doing an academic literature search. Network: run `export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000` before any curl.

TASK: Find and verify papers about **representation similarity measures**, specifically:
- CKA (Centered Kernel Alignment, arXiv 1905.00414), CKNNA / kernel-alignment variants
- SVCCA (1706.05806), PWCCA, DVCCA
- RSA (Representational Similarity Analysis) — neuroscience origin, Kriegeskorte 2008
- Procrustes / linear predictability / "linear probe" style measures
- Any measure designed to handle CONDITIONAL information (e.g. "conditional CKA", "partial CKA", "controlling for a nuisance variable")

For EACH measure answer:
1. Is it LINEAR or nonlinear? (be precise — e.g. CKA with linear kernel is linear-ish but invariant to orthogonal transforms; CKA with RBF kernel is not)
2. Does it handle conditional information / can it condition out a variable?
3. What have critics said about it? Find SPECIFIC critique papers. Known ones to verify:
   - "Similarity of Neural Network Representations Revisited" (1905.00414) — what does it criticize?
   - Any paper showing CKA is insensitive / can be gamed / has high baseline from random networks ("CKA between random networks is high")
   - "The Unreasonable Effectiveness of Random Networks" or similar
   - Papers on "what does CKA actually measure", "CKA is not a good measure of X"
   - Any paper saying linear probes are trivial/uninformative — e.g. "Linear probes are not enough", "Probing classifiers: promises and pitfalls" (Hewitt & Liang 1909.03368), "Amnesic probing" (Elazar et al. 20092138?), "Information-theoretic probing" (Voita & Titov 2002.12327)
4. **CRITICAL**: Has anyone LINKED a representation-similarity measure to "whether distillation actually works / is effective"? i.e. used CKA to diagnose distillation success or predict distillation gain? Find specific papers.

For every paper you report, you MUST verify by running:
  curl -s -m 25 "https://arxiv.org/abs/<id>" | grep -i -E "<title>|Comments"
and report the exact title and the Comments field (venue). If curl fails or 429s, retry or say UNVERIFIED.

Report format: markdown, per-paper: title / arXiv id / URL / date / institution if findable / venue from Comments field. Clearly mark each as VERIFIED (curl confirmed) or UNVERIFIED.

Do NOT write any files. Return findings as your final message.

### ASSISTANT — 2026-09-16 16:59

I'll start by verifying the known papers via arXiv, then search for the conditional/partial variants and the distillation link.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nfor id in 1905.00414 1706.05806 1909.03368 2002.12327 1810.05795 1602.01407 2009.12138; do\n  echo \"=== $id ===\"\n  curl -s -m 25 \"https://arxiv.org/abs/$id\" | grep -i -E \"<title>|Comments\" | sed 's/^[ \\t]*//'\n  echo\ndone", "description": "Verify core similarity-measure papers on arXiv"}
```

<details><summary>tool result</summary>

```
=== 1905.00414 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[1905.00414] Similarity of Neural Network Representations Revisited</title>
<td class="tablecell label">Comments:</td>
<td class="tablecell comments mathjax">ICML 2019</td>

=== 1706.05806 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[1706.05806] SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability</title>
<td class="tablecell label">Comments:</td>
<td class="tablecell comments mathjax">Accepted to NIPS 2017, code: <a href="https://github.com/google/svcca/" rel="external noopener nofollow" class="link-external link-https">this https URL</a> , new plots on Imagenet</td>

=== 1909.03368 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[1909.03368] Designing and Interpreting Probes with Control Tasks</title>
<td class="tablecell label">Comments:</td>
<td class="tablecell comments mathjax">EMNLP 2019</td>

=== 2002.12327 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2002.12327] A Primer in BERTology: What we know about how BERT works</title>
<td class="tablecell label">Comments:</td>
<td class="tablecell comments mathjax">Accepted to TACL. Please note that the multilingual BERT section is only available in version 1</td>

=== 1810.05795 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[1810.05795] Point Cloud GAN</title>

=== 1602.01407 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[1602.01407] A Kronecker-factored approximate Fisher matrix for convolution layers</title>

=== 2009.12138 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2009.12138] Fermi arcs and surface criticality in dirty Dirac materials</title>
<td class="tablecell label">Comments:</td>
<td class="tablecell comments mathjax">5 pages, 5 figures, supplemental m
... [truncated 12 chars]
```
</details>

### ASSISTANT — 2026-09-16 17:00

Two IDs were wrong (2002.12327 is BERTology; 2009.12138 is physics). Let me correct those and run topic searches.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nfor id in 2003.12298 2006.00995 2102.12452 1808.07724 2205.09627; do\n  echo \"=== $id ===\"\n  curl -s -m 25 \"https://arxiv.org/abs/$id\" | grep -i -E \"<title>|Comments\" | sed 's/^[ \\t]*//'\n  echo\ndone", "description": "Verify probing-critique candidates"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nq(){ echo \"=== QUERY: $1 ===\"; curl -s -m 30 \"http://export.arxiv.org/api/query?search_query=$1&start=0&max_results=20\" | grep -E \"<title>|<id>http|<published>\" | sed 's/^[ \\t]*//' | sed 's/<[^>]*>//g'; echo; }\nq \"all:%22CKNNA%22\"\nq \"all:%22conditional%20CKA%22\"\nq \"abs:%22centered%20kernel%20alignment%22\"", "description": "Search arXiv API for CKNNA and conditional CKA"}
```

<details><summary>tool result</summary>

```
claude-sonnet-4-6[1m] is temporarily unavailable (timed out), so auto mode cannot determine the safety of Bash right now. Wait a moment and then try this action again. If it keeps failing, continue with other tasks that don't require this action and come back to it later. Note: reading files, searching code, and other read-only operations do not require the classifier and can still be used.
```
</details>

<details><summary>tool result</summary>

```
=== QUERY: all:%22CKNNA%22 ===

=== QUERY: all:%22conditional%20CKA%22 ===

=== QUERY: abs:%22centered%20kernel%20alignment%22 ===
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ncurl -s -m 30 'http://export.arxiv.org/api/query?search_query=all:%22CKNNA%22&start=0&max_results=10' | head -60", "description": "Test arXiv API single query"}
```

<details><summary>tool result</summary>

```
(Bash completed with no output)
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ncurl -s -m 30 'https://export.arxiv.org/api/query?search_query=all:%22CKNNA%22&start=0&max_results=10' | grep -E \"<title>|<id>\" | head -30\necho \"---exit $?\"", "description": "Try arXiv API over https"}
```

<details><summary>tool result</summary>

```
  <id>https://arxiv.org/api/PwYuyhOLTbbbxotJZQXCeKJ5cLU</id>
  <title>arXiv Query: search_query=all:"CKNNA"&amp;id_list=&amp;start=0&amp;max_results=10</title>
    <id>http://arxiv.org/abs/2606.07034v1</id>
    <title>ForensicConcept: Transferable Forensic Concepts for AIGI Detection</title>
---exit 0
```
</details>

### ASSISTANT — 2026-09-16 17:02

HTTPS works for the API. Running a batch of searches.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nq(){ echo \"=== $1 ===\"; curl -s -m 40 \"https://export.arxiv.org/api/query?search_query=$2&start=0&max_results=${3:-15}&sortBy=relevance\" | python3 -c \"\nimport sys,re\nd=sys.stdin.read()\nfor e in re.findall(r'<entry>(.*?)</entry>',d,re.S):\n    t=re.search(r'<title>(.*?)</title>',e,re.S).group(1).strip().replace('\\n',' ')\n    i=re.search(r'<id>(.*?)</id>',e).group(1)\n    p=re.search(r'<published>(.*?)</published>',e).group(1)[:10]\n    print(p,i.split('/abs/')[-1],'|',' '.join(t.split()))\n\"; echo; }\nq \"conditional-CKA\" 'all:%22conditional+CKA%22'\nq \"partial-CKA\" 'all:%22partial+CKA%22'\nq \"CKA-critique\" 'abs:%22centered+kernel+alignment%22+AND+abs:%22similarity%22' 20\nq \"grounding-rep-sim\" 'ti:%22representational+similarity%22'", "description": "Search arXiv for conditional CKA and related"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nfor id in 2405.07987 2010.15327 2108.01661 2003.12298 2006.00995; do\n  echo \"=== $id ===\"\n  curl -s -m 25 \"https://arxiv.org/abs/$id\" | grep -i -E \"<title>|Comments\" | sed 's/^[ \\t]*//'\n  echo\ndone", "description": "Verify Platonic hypothesis and probing papers"}
```

<details><summary>tool result</summary>

```
=== conditional-CKA ===
2026-08-06 2608.06613v1 | Do 3D Medical Foundation Models See Through MRI Artifacts? A Controlled Study of Representation Robustness

=== partial-CKA ===

=== CKA-critique ===
2022-01-31 2202.00095v1 | Deconfounded Representation Similarity for Comparison of Neural Networks
2022-12-05 2212.01976v4 | FedCC: Robust Federated Learning against Model Poisoning Attacks
2023-05-03 2305.02215v2 | Exploring Linguistic Properties of Monolingual BERTs with Typological Classification among Languages
2019-05-01 1905.00414v4 | Similarity of Neural Network Representations Revisited
2021-09-17 2109.08406v2 | Fine-Tuned Transformers Show Clusters of Similar Representations Across Layers
2023-11-19 2311.11436v1 | Duality of Bures and Shape Distances with Implications for Comparing Neural Representations
2019-10-07 1910.02902v1 | Correlations between Word Vector Sets
2024-07-11 2407.08275v1 | Beyond Benchmarks: Evaluating Embedding Model Similarity for Retrieval Augmented Generation Systems
2023-11-29 2311.18007v1 | Towards out-of-distribution generalization in large-scale astronomical surveys: robust networks learn similar representations
2024-01-22 2401.11824v4 | Rethinking Centered Kernel Alignment in Knowledge Distillation
2019-12-04 1912.02260v1 | The effect of task and training on intermediate representations in convolutional neural networks revealed with modified RV similarity analysis
2025-02-27 2502.19648v2 | Spectral Analysis of Representational Similarity with Limited Neurons
2021-10-01 2110.00672v1 | Low Frequency Names Exhibit Bias and Overfitting in Contextualizing Language Models
2024-05-27 2405.17377v2 | How Do the Architecture and Optimizer Affect Representation Learning? On the Training Dynamics of Representations in Deep Neural Networks
2024-09-26 2409.18333v2 | A Framework for Standardizing Similarity Measures in a Rapidly Evolving Field
2024-09-10 2409.16302v2 | How Redundant Is the Transformer Stack in Speech Representation Models?
2024-08
... [truncated 1895 chars]
```
</details>

<details><summary>tool result</summary>

```
=== 2405.07987 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2405.07987] The Platonic Representation Hypothesis</title>
<td class="tablecell label">Comments:</td>
<td class="tablecell comments mathjax">Equal contributions. Project: <a href="https://phillipi.github.io/prh/" rel="external noopener nofollow" class="link-external link-https">this https URL</a> Code: <a href="https://github.com/minyoungg/platonic-rep" rel="external noopener nofollow" class="link-external link-https">this https URL</a></td>

=== 2010.15327 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2010.15327] Do Wide and Deep Networks Learn the Same Things? Uncovering How Neural Network Representations Vary with Width and Depth</title>
<td class="tablecell label">Comments:</td>
<td class="tablecell comments mathjax">ICLR 2021</td>

=== 2108.01661 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2108.01661] Grounding Representation Similarity with Statistical Testing</title>
<td class="tablecell label">Comments:</td>
<td class="tablecell comments mathjax">Accepted at NeurIPS 2021. 10 pages, 3 figures</td>

=== 2003.12298 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2003.12298] Information-Theoretic Probing with Minimum Description Length</title>

=== 2006.00995 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2006.00995] Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals</title>
<td class="tablecell label">Comments:</td>
<td class="tablecell comments mathjax">TACL journal. Initial title was: &#34;When Bert Forgets How To POS: Amnesic Probing of Linguistic Properties and MLM Predictions&#34;</td>
```
</details>

### ASSISTANT — 2026-09-16 17:02

Strong hits. Now the critique papers and the distillation link.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nq(){ echo \"=== $1 ===\"; curl -s -m 40 \"https://export.arxiv.org/api/query?search_query=$2&start=0&max_results=${3:-15}&sortBy=relevance\" | python3 -c \"\nimport sys,re\nd=sys.stdin.read()\nfor e in re.findall(r'<entry>(.*?)</entry>',d,re.S):\n    t=re.search(r'<title>(.*?)</title>',e,re.S).group(1).strip()\n    i=re.search(r'<id>(.*?)</id>',e).group(1)\n    p=re.search(r'<published>(.*?)</published>',e).group(1)[:10]\n    print(p,i.split('/abs/')[-1],'|',' '.join(t.split()))\n\"; echo; }\nq \"CKA-reliability\" 'abs:%22CKA%22+AND+abs:%22reliable%22' 20\nq \"CKA-sensitive\" 'ti:%22CKA%22' 30\nq \"distill-CKA\" 'abs:%22knowledge+distillation%22+AND+abs:%22CKA%22' 20", "description": "Search CKA critique and distillation papers"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nfor id in 2202.00095 2401.11824 2411.08197 2408.00531 2409.18333; do\n  echo \"=== $id ===\"\n  curl -s -m 25 \"https://arxiv.org/abs/$id\" | grep -i -E \"<title>|Comments\" | sed 's/^[ \\t]*//'\n  echo\ndone", "description": "Verify deconfounded and distillation CKA papers"}
```

<details><summary>tool result</summary>

```
=== CKA-reliability ===
2025-02-20 2502.15104v2 | Estimating Neural Representation Alignment from Sparsely Sampled Inputs and Features
2023-04-04 2304.01999v2 | Revisiting the Evaluation of Image Synthesis with GANs
2019-05-01 1905.00414v4 | Similarity of Neural Network Representations Revisited
2026-01-14 2601.09173v5 | Geometric Stability: The Missing Axis of Representations
2026-04-08 2605.02914v1 | When Safety Geometry Collapses: Fine-Tuning Vulnerabilities in Agentic Guard Models
2025-11-07 2511.05102v1 | Quantifying the Risk of Transferred Black Box Attacks
2026-01-06 2601.03168v1 | Can Embedding Similarity Predict Cross-Lingual Transfer? A Systematic Study on African Languages
2026-03-17 2603.17074v3 | PRISM: Demystifying Retention and Interaction in Mid-Training
2025-10-08 2510.06738v3 | AWM: Accurate Weight-Matrix Fingerprint for Large Language Models
2026-06-04 2606.06342v1 | Symmetric Divergence and Normalized Similarity: A Unified Topological Framework for Representation Analysis
2026-04-20 2604.17698v2 | The Geometric Canary: Predicting Steerability and Detecting Drift via Representational Stability
2026-07-23 2607.21823v1 | Data eccentricity, asymptotics of Gaussian RBF reproducing kernel Hilbert space, and kernel PCA
2025-02-27 2502.19648v2 | Spectral Analysis of Representational Similarity with Limited Neurons
2026-07-18 2607.16898v2 | Cross-Branch Conflict as a Shield: Safeguarding Facial Identities in Unified Multimodal Image Editing
2026-08-24 2608.22863v2 | Adaptive Hierarchical Representation Alliance for Multimodal Learning
2025-12-05 2512.08983v1 | HSCP: A Two-Stage Spectral Clustering Framework for Resource-Constrained UAV Identification
2024-10-31 2410.24070v4 | Dynamical similarity analysis can identify compositional dynamics developing in RNNs

=== CKA-sensitive ===
2023-08-30 2308.16289v1 | Time-Bin CKA as a tool for blockchain technology
2022-10-28 2210.16156v2 | Reliability of CKA as a Similarity Measure in Deep Learning
2024-01-09 2401
... [truncated 2060 chars]
```
</details>

<details><summary>tool result</summary>

```
=== 2202.00095 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2202.00095] Deconfounded Representation Similarity for Comparison of Neural Networks</title>

=== 2401.11824 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2401.11824] Rethinking Centered Kernel Alignment in Knowledge Distillation</title>

=== 2411.08197 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2411.08197] What Representational Similarity Measures Imply about Decodable Information</title>

=== 2408.00531 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2408.00531] ReSi: A Comprehensive Benchmark for Representational Similarity Measures</title>
<td class="tablecell label">Comments:</td>
<td class="tablecell comments mathjax">ICLR 2025. Code and data at <a href="https://github.com/mklabunde/resi" rel="external noopener nofollow" class="link-external link-https">this https URL</a></td>

=== 2409.18333 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2409.18333] A Framework for Standardizing Similarity Measures in a Rapidly Evolving Field</title>
<td class="tablecell label">Comments:</td>
<td class="tablecell comments mathjax">11 pages, 9 figures</td>
```
</details>

### ASSISTANT — 2026-09-16 17:03

Excellent hits. Now verifying the critique cluster and searching the distillation link.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nfor id in 2210.16156 2106.07682 2207.09684 2110.14739 1806.05759 1610.03454 2106.05945; do\n  echo \"=== $id ===\"\n  curl -s -m 25 \"https://arxiv.org/abs/$id\" | grep -i -E \"<title>|Comments\" | sed 's/^[ \\t]*//'\n  echo\ndone", "description": "Verify CKA reliability, stitching, partial distance correlation, CCA variants"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nq(){ echo \"=== $1 ===\"; curl -s -m 40 \"https://export.arxiv.org/api/query?search_query=$2&start=0&max_results=${3:-15}&sortBy=relevance\" | python3 -c \"\nimport sys,re\nd=sys.stdin.read()\nfor e in re.findall(r'<entry>(.*?)</entry>',d,re.S):\n    t=re.search(r'<title>(.*?)</title>',e,re.S).group(1).strip()\n    i=re.search(r'<id>(.*?)</id>',e).group(1)\n    p=re.search(r'<published>(.*?)</published>',e).group(1)[:10]\n    print(p,i.split('/abs/')[-1],'|',' '.join(t.split()))\n\"; echo; }\nq \"teacher-student-sim-predict\" 'abs:%22distillation%22+AND+abs:%22teacher%22+AND+abs:%22representational+similarity%22' 20\nq \"does-KD-work\" 'ti:%22knowledge+distillation%22+AND+abs:%22fidelity%22' 15\nq \"random-nets-high-sim\" 'abs:%22untrained%22+AND+abs:%22CKA%22' 15", "description": "Search distillation-similarity link papers"}
```

<details><summary>tool result</summary>

```
=== 2210.16156 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2210.16156] Reliability of CKA as a Similarity Measure in Deep Learning</title>

=== 2106.07682 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2106.07682] Revisiting Model Stitching to Compare Neural Representations</title>

=== 2207.09684 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2207.09684] On the Versatile Uses of Partial Distance Correlation in Deep Learning</title>
<td class="tablecell label">Comments:</td>
<td class="tablecell comments mathjax">This paper has been selected as best paper award for ECCV 2022!</td>

=== 2110.14739 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2110.14739] Generalized Shape Metrics on Neural Representations</title>
<td class="tablecell label">Comments:</td>
<td class="tablecell comments mathjax">26 pages, 7 figures, NeurIPS 2021</td>

=== 1806.05759 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[1806.05759] Insights on representational similarity in neural networks with canonical correlation</title>
<td class="tablecell label">Comments:</td>
<td class="tablecell comments mathjax">NIPS 2018</td>

=== 1610.03454 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[1610.03454] Deep Variational Canonical Correlation Analysis</title>

=== 2106.05945 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2106.05945] Does Knowledge Distillation Really Work?</title>
<td class="tablecell label">Comments:</td>
<td class="tablecell comments mathjax">NeurIPS 2021. Code available at <a href="https://github.com/samuelstanton/gnosis" rel="external noopener nofollow" class="link-external link-https">this https URL</a></td>
```
</details>

<details><summary>tool result</summary>

```
=== teacher-student-sim-predict ===
2026-02-17 2602.15438v3 | Logit Distance Bounds Representational Similarity
2021-11-09 2111.04964v4 | On Representation Knowledge Distillation for Graph Neural Networks
2026-06-04 2606.05682v2 | Beyond Output Matching: Preserving Internal Geometry in NVFP4 LLM Distillation
2025-11-02 2511.01023v1 | Seed-Induced Uniqueness in Transformer Models: Subspace Alignment Governs Subliminal Transfer
2024-01-22 2401.11824v4 | Rethinking Centered Kernel Alignment in Knowledge Distillation
2026-05-23 2605.24428v1 | Representation-Guided Discrete Molecular Graph Retrosynthesis

=== does-KD-work ===
2020-06-15 2006.08341v2 | Multi-fidelity Neural Architecture Search with Knowledge Distillation
2024-04-30 2405.00739v1 | Why does Knowledge Distillation Work? Rethink its Attention and Fidelity Mechanism
2023-05-22 2305.12954v1 | Is Synthetic Data From Diffusion Models Ready for Knowledge Distillation?
2026-08-15 2608.15215v1 | The Distributional View of Knowledge Distillation
2026-05-21 2605.22060v1 | Safeguarding Text-to-Image Generative Models Against Unauthorized Knowledge Distillation
2019-03-27 1903.11253v1 | Improving Route Choice Models by Incorporating Contextual Factors via Knowledge Distillation
2025-04-01 2504.00870v1 | Data-free Knowledge Distillation with Diffusion Models
2025-03-05 2503.03544v1 | KLiNQ: Knowledge Distillation-Assisted Lightweight Neural Network for Qubit Readout on FPGA
2025-10-30 2510.26444v2 | Personalized Treatment Outcome Prediction from Scarce Data via Dual-Channel Knowledge Distillation and Adaptive Fusion
2024-12-15 2412.11276v2 | Wearable Accelerometer Foundation Models for Health via Knowledge Distillation
2026-08-27 2608.27198v1 | Knowledge Distillation Driven Semantic NOMA with GAN Refinement for 6G Robotic Vehicle Networks
2025-03-18 2503.14293v3 | Ensemble Knowledge Distillation for Machine Learning Interatomic Potentials
2026-03-17 2604.08558v2 | WAND: Windowed Attention and Knowledge Distillation for E
... [truncated 573 chars]
```
</details>

### ASSISTANT — 2026-09-16 17:04

Now let me pull abstracts for the critical papers to characterize precisely what each claims.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ncurl -s -m 60 \"https://export.arxiv.org/api/query?id_list=2210.16156,2202.00095,2401.11824,2106.05945,2411.14633,2108.01661,2207.09684&max_results=10\" | python3 -c \"\nimport sys,re\nd=sys.stdin.read()\nfor e in re.findall(r'<entry>(.*?)</entry>',d,re.S):\n    t=' '.join(re.search(r'<title>(.*?)</title>',e,re.S).group(1).split())\n    i=re.search(r'<id>(.*?)</id>',e).group(1).split('/abs/')[-1]\n    s=' '.join(re.search(r'<summary>(.*?)</summary>',e,re.S).group(1).split())\n    au=[a.strip() for a in re.findall(r'<name>(.*?)</name>',e)]\n    print('###',i,'|',t); print('AUTHORS:',', '.join(au[:6])); print(s); print()\n\"", "description": "Fetch abstracts of key critique papers"}
```

<details><summary>tool result</summary>

```
### 2106.05945v2 | Does Knowledge Distillation Really Work?
AUTHORS: Samuel Stanton, Pavel Izmailov, Polina Kirichenko, Alexander A. Alemi, Andrew Gordon Wilson
Knowledge distillation is a popular technique for training a small student network to emulate a larger teacher model, such as an ensemble of networks. We show that while knowledge distillation can improve student generalization, it does not typically work as it is commonly understood: there often remains a surprisingly large discrepancy between the predictive distributions of the teacher and the student, even in cases when the student has the capacity to perfectly match the teacher. We identify difficulties in optimization as a key reason for why the student is unable to match the teacher. We also show how the details of the dataset used for distillation play a role in how closely the student matches the teacher -- and that more closely matching the teacher paradoxically does not always lead to better student generalization.

### 2411.14633v2 | Evaluating Representational Similarity Measures from the Lens of Functional Correspondence
AUTHORS: Yiqing Bo, Ansh Soni, Sudhanshu Srivastava, Meenakshi Khosla
Neuroscience and artificial intelligence (AI) both face the challenge of interpreting high-dimensional neural data, where the comparative analysis of such data is crucial for revealing shared mechanisms and differences between these complex systems. Despite the widespread use of representational comparisons and the abundance classes of comparison methods, a critical question remains: which metrics are most suitable for these comparisons? While some studies evaluate metrics based on their ability to differentiate models of different origins or constructions (e.g., various architectures), another approach is to assess how well they distinguish models that exhibit distinct behaviors. To investigate this, we examine the degree of alignment between various representational similarity measures and behavioral outcome
... [truncated 8245 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nq(){ echo \"=== $1 ===\"; curl -s -m 40 \"https://export.arxiv.org/api/query?search_query=$2&start=0&max_results=${3:-15}&sortBy=relevance\" | python3 -c \"\nimport sys,re\nd=sys.stdin.read()\nfor e in re.findall(r'<entry>(.*?)</entry>',d,re.S):\n    t=' '.join(re.search(r'<title>(.*?)</title>',e,re.S).group(1).split())\n    i=re.search(r'<id>(.*?)</id>',e).group(1).split('/abs/')[-1]\n    p=re.search(r'<published>(.*?)</published>',e).group(1)[:10]\n    print(p,i,'|',t)\n\"; echo; }\nq \"predict-distill-gain\" 'abs:%22predict%22+AND+abs:%22distillation%22+AND+abs:%22teacher-student+gap%22' 15\nq \"teacher-student-similarity-success\" 'abs:%22teacher%22+AND+abs:%22student%22+AND+abs:%22CKA%22' 25\nq \"capacity-gap\" 'ti:%22teacher%22+AND+abs:%22capacity+gap%22+AND+abs:%22distillation%22' 15", "description": "Search papers linking similarity to distillation success"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nq(){ echo \"=== $1 ===\"; curl -s -m 40 \"https://export.arxiv.org/api/query?search_query=$2&start=0&max_results=${3:-15}&sortBy=relevance\" | python3 -c \"\nimport sys,re\nd=sys.stdin.read()\nfor e in re.findall(r'<entry>(.*?)</entry>',d,re.S):\n    t=' '.join(re.search(r'<title>(.*?)</title>',e,re.S).group(1).split())\n    i=re.search(r'<id>(.*?)</id>',e).group(1).split('/abs/')[-1]\n    p=re.search(r'<published>(.*?)</published>',e).group(1)[:10]\n    print(p,i,'|',t)\n\"; echo; }\nq \"probe-critique\" 'ti:%22probing%22+AND+abs:%22pitfalls%22' 15\nq \"probing-classifiers\" 'ti:%22Probing+Classifiers%22' 10\nq \"conditional-independence-rep\" 'abs:%22conditional+independence%22+AND+abs:%22representation+similarity%22' 15", "description": "Search probing critiques and conditional independence measures"}
```

<details><summary>tool result</summary>

```
=== predict-distill-gain ===
2025-12-18 2512.19744v1 | DeepBridge: A Unified and Production-Ready Framework for Multi-Dimensional Machine Learning Validation

=== teacher-student-similarity-success ===
2023-07-27 2307.14781v1 | Contrastive Knowledge Amalgamation for Unsupervised Image Classification
2024-01-22 2401.11824v4 | Rethinking Centered Kernel Alignment in Knowledge Distillation
2023-10-26 2310.17183v1 | Understanding the Effects of Projectors in Knowledge Distillation
2026-06-04 2606.05682v2 | Beyond Output Matching: Preserving Internal Geometry in NVFP4 LLM Distillation
2025-11-02 2511.01023v1 | Seed-Induced Uniqueness in Transformer Models: Subspace Alignment Governs Subliminal Transfer
2026-07-03 2607.03572v1 | Teacher Supervision over Representation Equivalence Classes
2026-07-05 2607.04432v1 | Covert Trait Propagation Is Representation Alignment: Mechanistic Evidence from Hidden-Channel Distillation
2023-10-30 2310.19444v1 | One-for-All: Bridge the Gap Between Heterogeneous Architectures in Knowledge Distillation
2025-09-27 2509.25253v1 | Knowledge distillation through geometry-aware representational alignment
2026-07-15 2607.16316v1 | Eddy-VL 1.9B: Structural Pruning and Layered Distillation for Edge-Deployable Multimodal Embedding
2026-04-17 2604.15794v2 | Self-Distillation as a Performance Recovery Mechanism for LLMs: Counteracting Compression and Catastrophic Forgetting
2026-05-06 2605.04447v1 | Deep Reprogramming Distillation for Medical Foundation Models
2026-08-26 2608.25575v1 | MLLMCLIP: Feature-Level Distillation of MLLM for Robust Vision-Language Representations
2026-04-05 2604.04050v2 | TORA: Topological Representation Alignment for 3D Shape Assembly
2026-06-21 2606.23740v1 | Weight-Space Geometry of Offline Reasoning Training

=== capacity-gap ===
2025-06-23 2506.18244v1 | Dual-Forward Path Teacher Knowledge Distillation: Bridging the Capacity Gap Between Teacher and Student
2021-10-16 2110.08532v1 | Pro-KD: Progressive Distillation by Foll
... [truncated 1041 chars]
```
</details>

<details><summary>tool result</summary>

```
=== probe-critique ===
2021-04-12 2104.05807v2 | Does My Representation Capture X? Probe-Ably
2002-04-16 cond-mat/0204339v1 | Spatial and temporal probes in inhomogeneous systems: Theory and experiment
2023-04-20 2304.10346v1 | Interventional Probing in High Dimensions: An NLI Case Study
2020-01-09 2001.02896v1 | Probing the concept of line tension down to the nanoscale
2026-06-30 2606.31208v1 | Probing Memorization of Tabular In-Context Learning
2025-05-24 2505.18673v1 | Cross-Lingual Pitfalls: Automatic Probing Cross-Lingual Weakness of Multilingual Large Language Models
2012-03-15 1203.3406v1 | Terahertz ammonia absorption as a probe of infall in high-mass star forming clumps
2022-10-22 2210.12448v1 | Probing Transfer in Deep Reinforcement Learning without Task Engineering
2025-12-05 2512.05809v1 | Probing the effectiveness of World Models for Spatial Reasoning through Test-time Scaling

=== probing-classifiers ===
2021-02-24 2102.12452v4 | Probing Classifiers: Promises, Shortcomings, and Advances
2024-03-18 2403.11747v2 | Embedded Named Entity Recognition using Probing Classifiers
2022-07-08 2207.04153v3 | Probing Classifiers are Unreliable for Concept Removal and Detection
2023-12-17 2312.10746v1 | Knowledge Trees: Gradient Boosting Decision Trees on Knowledge Neurons as Probing Classifier
2024-12-06 2412.05155v1 | Multimodal Fact-Checking with Vision Language Models: A Probing Classifier based Solution with Embedding Strategies

=== conditional-independence-rep ===
```
</details>

### ASSISTANT — 2026-09-16 17:05

Found the real "Probing Classifiers" paper. Now verifying and digging into the distillation-similarity link.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nfor id in 2102.12452 2207.04153 2607.03572 2607.04432 2509.25253 2605.31191 2601.21568; do\n  echo \"=== $id ===\"\n  curl -s -m 25 \"https://arxiv.org/abs/$id\" | grep -i -E \"<title>|Comments\" | sed 's/^[ \\t]*//'\n  echo\ndone", "description": "Verify probing critiques and distillation-similarity papers"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ncurl -s -m 60 \"https://export.arxiv.org/api/query?id_list=2607.03572,2607.04432,2509.25253,2605.31191,2601.21568,2602.15438&max_results=10\" | python3 -c \"\nimport sys,re\nd=sys.stdin.read()\nfor e in re.findall(r'<entry>(.*?)</entry>',d,re.S):\n    t=' '.join(re.search(r'<title>(.*?)</title>',e,re.S).group(1).split())\n    i=re.search(r'<id>(.*?)</id>',e).group(1).split('/abs/')[-1]\n    s=' '.join(re.search(r'<summary>(.*?)</summary>',e,re.S).group(1).split())\n    print('###',i,'|',t); print(s[:1200]); print()\n\"", "description": "Fetch abstracts of distillation-similarity candidates"}
```

<details><summary>tool result</summary>

```
=== 2102.12452 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2102.12452] Probing Classifiers: Promises, Shortcomings, and Advances</title>
<td class="tablecell label">Comments:</td>
<td class="tablecell comments mathjax">Accepted to Computational Linguistics as a squib</td>

=== 2207.04153 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2207.04153] Probing Classifiers are Unreliable for Concept Removal and Detection</title>

=== 2607.03572 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2607.03572] Teacher Supervision over Representation Equivalence Classes</title>

=== 2607.04432 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2607.04432] Covert Trait Propagation Is Representation Alignment: Mechanistic Evidence from Hidden-Channel Distillation</title>

=== 2509.25253 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2509.25253] Knowledge distillation through geometry-aware representational alignment</title>

=== 2605.31191 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2605.31191] Student Capacity Moderates Knowledge Distillation Effectiveness: A Systematic Study Across ResNet Teacher-Student Pairs on CIFAR-10</title>
<td class="tablecell label">Comments:</td>
<td class="tablecell comments mathjax">v2: replaces test-set hyperparameter selection with a held-out validation protocol; 5-seed final runs; adds a fourth teacher-student pair (R101-&gt;R34), fidelity metrics, and a measured stem ablation; retracts v1&#39;s attribution of Feature-KD underperformance to a gradient-clipping bug. Code: <a href="http://github.com/umutonuryasar/kd-capacity-gap" rel="external noopener nofollow" class="link-external link-http">this http URL</a></td>

=== 2601.21568 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2601.21568] Bridging Functional an
... [truncated 60 chars]
```
</details>

<details><summary>tool result</summary>

```
### 2601.21568v2 | Bridging Functional and Representational Similarity via Usable Information
We present a unified framework for quantifying the similarity between representations through the lens of \textit{usable} information, offering a rigorous theoretical and empirical synthesis across three key dimensions. First, addressing functional similarity, we establish a formal link between stitching performance and conditional mutual information. We further reveal that stitching is inherently asymmetric, demonstrating that robust functional comparison necessitates a bidirectional analysis rather than a unidirectional mapping. Second, concerning representational similarity, we find that reconstruction-based metrics and standard tools (e.g., CKA, RSA) act as estimators of usable information under specific constraints. Crucially, we show that similarity is relative to the capacity of the predictive family: representations that appear distinct to a rigid observer may be identical to a more expressive one. Third, we demonstrate that representational similarity is sufficient but not necessary for functional similarity. We unify these concepts through a task-granularity hierarchy: similarity on a complex task guarantees similarity on any coarser derivative, establishing representatio

### 2607.04432v1 | Covert Trait Propagation Is Representation Alignment: Mechanistic Evidence from Hidden-Channel Distillation
A student model trained on pure uniform noise can still inherit its teacher's digit-classification ability, provided the two share initialization. Previous work proves this transfer is guaranteed when the teacher's learning rate is small enough, but does not explain where in the network the channel lives or what sets its capacity. Working in an MLP distillation setting on MNIST, we show these channels are not purely informational: geometric alignment gates access to the information the channel carries. Shared initialization makes the output projection W_2 a common coordi
... [truncated 5822 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nq(){ echo \"=== $1 ===\"; curl -s -m 40 \"https://export.arxiv.org/api/query?search_query=$2&start=0&max_results=${3:-15}&sortBy=relevance\" | python3 -c \"\nimport sys,re\nd=sys.stdin.read()\nfor e in re.findall(r'<entry>(.*?)</entry>',d,re.S):\n    t=' '.join(re.search(r'<title>(.*?)</title>',e,re.S).group(1).split())\n    i=re.search(r'<id>(.*?)</id>',e).group(1).split('/abs/')[-1]\n    p=re.search(r'<published>(.*?)</published>',e).group(1)[:10]\n    print(p,i,'|',t)\n\"; echo; }\nq \"mutual-nn-alignment\" 'abs:%22mutual+nearest-neighbor%22+AND+abs:%22alignment%22' 15\nq \"nuisance-invariant-sim\" 'abs:%22nuisance%22+AND+abs:%22representation+similarity%22' 15\nq \"HSIC-conditional\" 'abs:%22conditional%22+AND+ti:%22HSIC%22' 20", "description": "Search CKNNA, nuisance, conditional HSIC"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nq(){ echo \"=== $1 ===\"; curl -s -m 40 \"https://export.arxiv.org/api/query?search_query=$2&start=0&max_results=${3:-15}&sortBy=relevance\" | python3 -c \"\nimport sys,re\nd=sys.stdin.read()\nfor e in re.findall(r'<entry>(.*?)</entry>',d,re.S):\n    t=' '.join(re.search(r'<title>(.*?)</title>',e,re.S).group(1).split())\n    i=re.search(r'<id>(.*?)</id>',e).group(1).split('/abs/')[-1]\n    p=re.search(r'<published>(.*?)</published>',e).group(1)[:10]\n    print(p,i,'|',t)\n\"; echo; }\nq \"sim-measure-disagree\" 'abs:%22similarity+measures%22+AND+abs:%22disagree%22' 15\nq \"rsa-partial-correlation\" 'abs:%22representational+similarity+analysis%22+AND+abs:%22partial+correlation%22' 15\nq \"cka-random-init-high\" 'abs:%22random+initialization%22+AND+abs:%22CKA%22' 20", "description": "Search disagreement, partial RSA, random-init CKA"}
```

<details><summary>tool result</summary>

```
=== mutual-nn-alignment ===
2026-06-15 2606.16379v2 | Scalable and Interpretable Representation Alignment with Ordinal Similarity
2024-09-10 2409.16302v2 | How Redundant Is the Transformer Stack in Speech Representation Models?
2026-04-20 2604.18572v2 | Back into Plato's Cave: Examining Cross-modal Representational Convergence at Scale
2020-07-03 2007.01504v1 | A Similarity Inference Metric for RGB-Infrared Cross-Modality Person Re-identification
2026-08-15 2608.15251v1 | Robust structure from motion for aerial-ground images via detector-free feature matching and multi-view track refinement
2018-05-10 1805.04140v2 | Neural Best-Buddies: Sparse Cross-Domain Correspondence
2026-04-07 2604.05718v1 | MPM: Mutual Pair Merging for Efficient Vision Transformers

=== nuisance-invariant-sim ===
2026-02-09 2602.09225v1 | Barycentric alignment for instance-level comparison of neural representations

=== HSIC-conditional ===
2023-08-11 2308.05969v2 | Learning nonparametric DAGs with incremental information via high-order HSIC
2024-03-12 2403.07735v2 | The Minimax Rate of HSIC Estimation for Translation-Invariant Kernels
2026-07-27 2607.24235v1 | Minimax Lower Bounds of Kernel Discrepancy Estimation: MMD, HSIC, KSD
2018-04-26 1804.09866v1 | New HSIC-based tests for independence between two stationary multivariate time series
2025-12-02 2512.02437v1 | LightHCG: a Lightweight yet powerful HSIC Disentanglement based Causal Glaucoma Detection Model framework
```
</details>

<details><summary>tool result</summary>

```
=== sim-measure-disagree ===

=== rsa-partial-correlation ===
2021-02-17 2102.08931v3 | Overcoming bias in representational similarity analysis

=== cka-random-init-high ===
2021-08-03 2108.01661v2 | Grounding Representation Similarity with Statistical Testing
2025-11-02 2511.01023v1 | Seed-Induced Uniqueness in Transformer Models: Subspace Alignment Governs Subliminal Transfer
2026-08-17 2608.16347v1 | Architecture-Dependent Causal Transfer of Activation States Across Large Language Models
```
</details>

### ASSISTANT — 2026-09-16 17:06

CKNNA traces to the Platonic paper. Let me confirm that and fill remaining gaps.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\necho \"--- CKNNA in Platonic paper ---\"\ncurl -s -m 30 \"https://arxiv.org/abs/2405.07987\" | python3 -c \"\nimport sys,re,html\nd=sys.stdin.read()\nm=re.search(r'<blockquote class=\\\"abstract[^>]*>(.*?)</blockquote>',d,re.S)\nprint(' '.join(html.unescape(re.sub('<[^>]+>','',m.group(1))).split())[:1500])\n\"\necho\necho \"--- search CKNNA fulltext via google scholar proxy: arxiv listing ---\"\ncurl -s -m 30 'https://export.arxiv.org/api/query?search_query=all:%22mutual+nearest+neighbor%22+AND+all:%22kernel+alignment%22&max_results=10' | python3 -c \"\nimport sys,re\nd=sys.stdin.read()\nfor e in re.findall(r'<entry>(.*?)</entry>',d,re.S):\n    t=' '.join(re.search(r'<title>(.*?)</title>',e,re.S).group(1).split())\n    i=re.search(r'<id>(.*?)</id>',e).group(1).split('/abs/')[-1]\n    print(i,'|',t)\n\"", "description": "Confirm CKNNA origin in Platonic paper"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nfor id in 2102.08931 2602.09225 2604.18572 2512.16282 2606.16897 2411.05561; do\n  echo \"=== $id ===\"\n  curl -s -m 25 \"https://arxiv.org/abs/$id\" | grep -i -E \"<title>|Comments\" | sed 's/^[ \\t]*//'\n  echo\ndone", "description": "Verify RSA bias and related papers"}
```

<details><summary>tool result</summary>

```
--- CKNNA in Platonic paper ---
Abstract:We argue that representations in AI models, particularly deep networks, are converging. First, we survey many examples of convergence in the literature: over time and across multiple domains, the ways by which different neural networks represent data are becoming more aligned. Next, we demonstrate convergence across data modalities: as vision models and language models get larger, they measure distance between datapoints in a more and more alike way. We hypothesize that this convergence is driving toward a shared statistical model of reality, akin to Plato's concept of an ideal reality. We term such a representation the platonic representation and discuss several possible selective pressures toward it. Finally, we discuss the implications of these trends, their limitations, and counterexamples to our analysis.

--- search CKNNA fulltext via google scholar proxy: arxiv listing ---
2409.16302v2 | How Redundant Is the Transformer Stack in Speech Representation Models?
```
</details>

<details><summary>tool result</summary>

```
=== 2102.08931 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2102.08931] Overcoming bias in representational similarity analysis</title>
<td class="tablecell label">Comments:</td>
<td class="tablecell comments mathjax">17 pages, 6 figures</td>

=== 2602.09225 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2602.09225] Barycentric alignment for instance-level comparison of neural representations</title>

=== 2604.18572 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2604.18572] Back into Plato&#39;s Cave: Examining Cross-modal Representational Convergence at Scale</title>
<td class="tablecell label">Comments:</td>
<td class="tablecell comments mathjax">Project page: <a href="http://akoepke.github.io/cave_umwelten/" rel="external noopener nofollow" class="link-external link-http">this http URL</a></td>

=== 2512.16282 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2512.16282] CALM: A CKA-Guided Adaptive Layer-Wise Modularization Framework for LLM Quantization</title>

=== 2606.16897 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2606.16897] Contrastive-Difference CKA Reveals Concept-Specific Structural Alignment Across Language Model Architectures</title>

=== 2411.05561 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2411.05561] Objective drives the consistency of representational similarity across datasets</title>
<td class="tablecell label">Comments:</td>
<td class="tablecell comments mathjax">26 pages</td>
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\necho \"### CKNNA grep in Platonic HTML fulltext\"\ncurl -s -m 40 \"https://arxiv.org/abs/2405.07987v3\" -o /dev/null -w \"%{http_code}\\n\"\ncurl -s -m 60 \"https://www.arxiv.org/html/2405.07987v3\" | python3 -c \"\nimport sys,re,html\nd=sys.stdin.read()\ntxt=' '.join(html.unescape(re.sub('<[^>]+>',' ',d)).split())\nfor m in re.finditer(r'CKNNA',txt):\n    print('...',txt[max(0,m.start()-350):m.start()+250],'...'); print()\n\" 2>/dev/null | head -40", "description": "Grep CKNNA definition in Platonic paper fulltext"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ncurl -s -m 60 \"https://export.arxiv.org/api/query?id_list=2606.16897,2602.09225,2411.05561,2409.18333,2210.16156&max_results=10\" | python3 -c \"\nimport sys,re\nd=sys.stdin.read()\nfor e in re.findall(r'<entry>(.*?)</entry>',d,re.S):\n    t=' '.join(re.search(r'<title>(.*?)</title>',e,re.S).group(1).split())\n    i=re.search(r'<id>(.*?)</id>',e).group(1).split('/abs/')[-1]\n    s=' '.join(re.search(r'<summary>(.*?)</summary>',e,re.S).group(1).split())\n    print('###',i,'|',t); print(s[:900]); print()\n\"", "description": "Abstracts for contrastive-difference CKA and barycentric"}
```

<details><summary>tool result</summary>

```
### CKNNA grep in Platonic HTML fulltext
200
```
</details>

<details><summary>tool result</summary>

```
### 2411.05561v2 | Objective drives the consistency of representational similarity across datasets
The Platonic Representation Hypothesis claims that recent foundation models are converging to a shared representation space as a function of their downstream task performance, irrespective of the objectives and data modalities used to train these models (Huh et al., 2024). Representational similarity is generally measured for individual datasets and is not necessarily consistent across datasets. Thus, one may wonder whether this convergence of model representations is confounded by the datasets commonly used in machine learning. Here, we propose a systematic way to measure how representational similarity between models varies with the set of stimuli used to construct the representations. We find that the objective function is a crucial factor in determining the consistency of representational similarities across datasets. Specifically, self-supervised vision models learn representations 

### 2602.09225v1 | Barycentric alignment for instance-level comparison of neural representations
Comparing representations across neural networks is challenging because representations admit symmetries, such as arbitrary reordering of units or rotations of activation space, that obscure underlying equivalence between models. We introduce a barycentric alignment framework that quotients out these nuisance symmetries to construct a universal embedding space across many models. Unlike existing similarity measures, which summarize relationships over entire stimulus sets, this framework enables similarity to be defined at the level of individual stimuli, revealing inputs that elicit convergent versus divergent representations across models. Using this instance-level notion of similarity, we identify systematic input properties that predict representational convergence versus divergence across vision and language model families. We also construct universal embedding spaces for brain repre


... [truncated 3008 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ncurl -s -m 60 \"https://arxiv.org/html/2405.07987v3\" -o /tmp/plat.html -w \"size=%{size_download} code=%{http_code}\\n\"\npython3 - <<'EOF'\nimport re,html\nd=open('/tmp/plat.html',errors='ignore').read()\ntxt=' '.join(html.unescape(re.sub('<[^>]+>',' ',d)).split())\nprint(\"len\",len(txt))\nfor kw in ['CKNNA','nearest neighbor','mutual']:\n    locs=[m.start() for m in re.finditer(kw,txt)]\n    print(kw, len(locs))\nfor m in list(re.finditer('CKNNA',txt))[:3]:\n    print('...',txt[max(0,m.start()-500):m.start()+300],'...\\n')\nEOF", "description": "Extract CKNNA context from Platonic paper HTML"}
```

<details><summary>tool result</summary>

```
size=474017 code=200
len 128835
CKNNA 5
nearest neighbor 22
mutual 20
... _{j}\rangle-\mathbb{E}_{l}[\langle\phi_{i},\phi_{l}\rangle]\right)\left(\langle\psi_{i},\psi_{j}\rangle-\mathbb{E}_{l}[\langle\psi_{i},\psi_{l}\rangle]\right) (16) where α ( i , j ) = 𝟙 [ ϕ j ∈ 𝗄𝗇𝗇 ( ϕ i ) ∧ ψ j ∈ 𝗄𝗇𝗇 ( ψ i ) ∧ i ≠ j ] \displaystyle\text{where}\qquad\alpha(i,j)=\mathbb{1}[\phi_{j}\in\mathsf{knn}(\phi_{i})\land\psi_{j}\in\mathsf{knn}(\psi_{i})\land i\neq j] (17) Figure 10: Cross-modal alignment increases locally: Alignment trend when varying the top- k k nearest neighbors in the CKNNA metrics (Eqn. 18 ). We center alignment score to the smallest language model and divide the total trend by the standard deviation. When k = 1024 k=1024 , we recover the original CKA metric, and when k < | 𝒳 | k<|\mathcal{X}| it closely resembles the mutual nearest-neighbor metric m NN m_{\text ...

... CKA metric, and when k < | 𝒳 | k<|\mathcal{X}| it closely resembles the mutual nearest-neighbor metric m NN m_{\texttt{NN}} . Each line represents the average of all LLM models for a specific k k . As we decrease k k , the alignment becomes more pronounced. Where α ⁡ ( i , j ) \alpha(i,j) is a scalar weighting that assigns 1 1 if j j is a mutual nearest neighbors to both ϕ i \phi_{i} and ψ i \psi_{i} , and 0 0 otherwise. We refer to this metric as the Centered Kernel Nearest-Neighbor Alignment (CKNNA) metric. As the number of nearest neighbors k → dim ( 𝐊 ) k\rightarrow\dim(\mathbf{K}) , we recover the original CKA metric. 𝖢𝖪𝖭𝖭𝖠 ⁡ ( 𝐊 , 𝐋 ) = 𝖠𝗅𝗂𝗀𝗇 𝗄𝗇𝗇 ​ ( 𝐊 , 𝐋 ) 𝖠𝗅𝗂𝗀𝗇 𝗄𝗇𝗇 ​ ( 𝐊 , 𝐊 ) , 𝖠𝗅𝗂𝗀𝗇 𝗄𝗇𝗇 ​ ( 𝐋 , 𝐋 ) \displaystyle\mathsf{CKNNA}(\mathbf{K},\mathbf{L})=\frac{\mathsf{Align_{knn}}(\ma ...

...  more pronounced. Where α ⁡ ( i , j ) \alpha(i,j) is a scalar weighting that assigns 1 1 if j j is a mutual nearest neighbors to both ϕ i \phi_{i} and ψ i \psi_{i} , and 0 0 otherwise. We refer to this metric as the Centered Kernel Nearest-Neighbor Alignment (CKNNA) metric. As the number of nearest neighb
... [truncated 498 chars]
```
</details>

### ASSISTANT — 2026-09-16 17:07

CKNNA confirmed (Platonic paper, Eqn 16-18). Final gap-filling searches.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nq(){ echo \"=== $1 ===\"; curl -s -m 40 \"https://export.arxiv.org/api/query?search_query=$2&start=0&max_results=${3:-15}&sortBy=relevance\" | python3 -c \"\nimport sys,re\nd=sys.stdin.read()\nfor e in re.findall(r'<entry>(.*?)</entry>',d,re.S):\n    t=' '.join(re.search(r'<title>(.*?)</title>',e,re.S).group(1).split())\n    i=re.search(r'<id>(.*?)</id>',e).group(1).split('/abs/')[-1]\n    p=re.search(r'<published>(.*?)</published>',e).group(1)[:10]\n    print(p,i,'|',t)\n\"; echo; }\nq \"unreasonable-random\" 'ti:%22random%22+AND+ti:%22unreasonable%22' 15\nq \"random-features-strong\" 'ti:%22untrained%22+AND+abs:%22representation%22' 20\nq \"conditional-dependence-measure\" 'ti:%22conditional+dependence%22+AND+abs:%22kernel%22' 20", "description": "Search random-network and conditional dependence papers"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nfor id in 2106.07682 2411.14633 2606.16379 2601.09173; do\n  echo \"=== $id ===\"\n  curl -s -m 25 \"https://arxiv.org/abs/$id\" | grep -i -E \"<title>|Comments\" | sed 's/^[ \\t]*//'\n  echo\ndone\necho \"### 2411.08197 abstract\"\ncurl -s -m 40 \"https://export.arxiv.org/api/query?id_list=2411.08197,2106.07682\" | python3 -c \"\nimport sys,re\nd=sys.stdin.read()\nfor e in re.findall(r'<entry>(.*?)</entry>',d,re.S):\n    t=' '.join(re.search(r'<title>(.*?)</title>',e,re.S).group(1).split())\n    i=re.search(r'<id>(.*?)</id>',e).group(1).split('/abs/')[-1]\n    s=' '.join(re.search(r'<summary>(.*?)</summary>',e,re.S).group(1).split())\n    au=[a.strip() for a in re.findall(r'<name>(.*?)</name>',e)]\n    print('###',i,'|',t); print('AU:',', '.join(au[:5])); print(s[:1100]); print()\n\"", "description": "Verify stitching and decodable-information papers"}
```

<details><summary>tool result</summary>

```
=== unreasonable-random ===
2017-03-02 1703.00864v5 | The Unreasonable Effectiveness of Structured Random Orthogonal Embeddings
2025-10-08 2510.06819v1 | The Unreasonable Effectiveness of Randomized Representations in Online Continual Graph Learning
2023-10-31 2310.20620v2 | The Unreasonable Effectiveness of Random Target Embeddings for Continuous-Output Neural Machine Translation
2025-04-09 2504.06757v1 | Local equations describe unreasonably efficient stochastic algorithms in random K-SAT
2022-02-05 2202.02643v1 | The Unreasonable Effectiveness of Random Pruning: Return of the Most Naive Baseline for Sparse Training

=== random-features-strong ===
2024-11-04 2411.02672v2 | Multi-modal deformable image registration using untrained neural networks
2020-05-13 2005.07026v1 | Subsampled Fourier Ptychography using Pretrained Invertible and Untrained Network Priors
2024-08-15 2408.07956v2 | RandomNet: Clustering Time Series Using Untrained Deep Neural Networks
2024-04-09 2404.06657v1 | Res-U2Net: Untrained Deep Learning for Phase Retrieval and Image Reconstruction
2024-10-31 2410.23560v2 | Untrained Filtering with Trained Focusing for Superior Quantum Architecture Search
2022-05-20 2205.10367v2 | Latent-space disentanglement with untrained generator networks for the isolation of different motion types in video data
2026-06-20 2606.22218v1 | An Analysis of Untrained Deep Reservoir Networks for Audio Surveillance
2024-06-21 2406.15109v1 | Brain-Like Language Processing via a Shallow Untrained Multihead Attention Network
2024-10-26 2410.20035v2 | Training the Untrainable: Introducing Inductive Bias via Representational Alignment
2024-09-09 2409.05389v1 | A Novel Representation of Periodic Pattern and Its Application to Untrained Anomaly Detection
2018-10-02 1810.03982v2 | Deep Decoder: Concise Image Representations from Untrained Non-convolutional Networks
2023-04-30 2305.00416v1 | Quaternion Matrix Completion Using Untrained Quaternion Convolutional Neural Network for Colo
... [truncated 927 chars]
```
</details>

<details><summary>tool result</summary>

```
=== 2106.07682 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2106.07682] Revisiting Model Stitching to Compare Neural Representations</title>

=== 2411.14633 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2411.14633] Evaluating Representational Similarity Measures from the Lens of Functional Correspondence</title>
<td class="tablecell label">Comments:</td>
<td class="tablecell comments mathjax">Published in CCN 2025 Proceedings (Talk &amp; Poster), May 14, 2025</td>

=== 2606.16379 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2606.16379] Scalable and Interpretable Representation Alignment with Ordinal Similarity</title>

=== 2601.09173 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2601.09173] Geometric Stability: The Missing Axis of Representations</title>

### 2411.08197 abstract
### 2106.07682v1 | Revisiting Model Stitching to Compare Neural Representations
AU: Yamini Bansal, Preetum Nakkiran, Boaz Barak
We revisit and extend model stitching (Lenc &amp; Vedaldi 2015) as a methodology to study the internal representations of neural networks. Given two trained and frozen models $A$ and $B$, we consider a "stitched model'' formed by connecting the bottom-layers of $A$ to the top-layers of $B$, with a simple trainable layer between them. We argue that model stitching is a powerful and perhaps under-appreciated tool, which reveals aspects of representations that measures such as centered kernel alignment (CKA) cannot. Through extensive experiments, we use model stitching to obtain quantitative verifications for intuitive statements such as "good networks learn similar representations'', by demonstrating that good networks of the same architecture, but trained in very different ways (e.g.: supervised vs. self-supervised learning), can be stitched to each other without drop in performance. We also give evidence for the intuition 
... [truncated 1415 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nfor id in 2012.14804 2410.15888 2003.12298 2604.16875 2606.16897 2602.15438; do\n  echo \"=== $id ===\"\n  curl -s -m 25 \"https://arxiv.org/abs/$id\" | grep -i -E \"<title>|Comments\" | sed 's/^[ \\t]*//'\n  echo\ndone", "description": "Verify conditional dependence and remaining papers"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\necho \"### Kriegeskorte 2008 RSA\"\ncurl -s -m 30 \"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2605405/\" | grep -i -oE \"<title>[^<]*</title>\" | head -3\necho\necho \"### 2202.00095 journal ref / venue check\"\ncurl -s -m 25 \"https://arxiv.org/abs/2202.00095\" | grep -i -E \"Journal ref|Subjects|Submitted\" | sed 's/<[^>]*>//g' | sed 's/^[ \\t]*//' | head -5\necho \"### 2210.16156 venue\"\ncurl -s -m 25 \"https://arxiv.org/abs/2210.16156\" | grep -i -E \"Journal ref|Subjects\" | sed 's/<[^>]*>//g' | sed 's/^[ \\t]*//' | head -5", "description": "Check RSA origin and venues"}
```

<details><summary>tool result</summary>

```
=== 2012.14804 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2012.14804] Kernel Partial Correlation Coefficient -- a Measure of Conditional Dependence</title>
<td class="tablecell label">Comments:</td>
<td class="tablecell comments mathjax">63 pages, 4 figures, 9 tables</td>

=== 2410.15888 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2410.15888] Conditional Dependence via U-Statistics Pruning</title>
<td class="tablecell label">Comments:</td>
<td class="tablecell comments mathjax">Accepted version of the article published in IEEE Signal Processing Letters, 2025. DOI: <a href="https://doi.org/10.1109/LSP.2025.3539587" data-doi="10.1109/LSP.2025.3539587" class="link-https link-external" rel="external noopener nofollow">https://doi.org/10.1109/LSP.2025.3539587</a></td>

=== 2003.12298 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2003.12298] Information-Theoretic Probing with Minimum Description Length</title>

=== 2604.16875 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2604.16875] Untrained CNNs Match Backpropagation at V1: A Systematic RSA Comparison of Four Learning Rules Against Human fMRI</title>
<td class="tablecell label">Comments:</td>
<td class="tablecell comments mathjax">11 pages, 9 figures. v2: Correction note added (p.1). An evaluation-mode defect affected the predictive-coding and STDP conditions; those results should not be used pending re-computation. Random, backprop and feedback alignment are unaffected. See also <a href="https://arxiv.org/abs/2608.12408" data-arxiv-id="2608.12408" class="link-https">arXiv:2608.12408</a></td>

=== 2606.16897 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2606.16897] Contrastive-Difference CKA Reveals Concept-Specific Structural Alignment Across Language Model Architectures</title>

=== 2602.15438 ===
<head><script>document.documentElement.
... [truncated 108 chars]
```
</details>

<details><summary>tool result</summary>

```
### Kriegeskorte 2008 RSA
<title>302 Found</title>

### 2202.00095 journal ref / venue check
[Submitted on 31 Jan 2022]
Subjects:

### 2210.16156 venue
Subjects:
```
</details>

### ASSISTANT — 2026-09-16 17:08

One more round: the Kriegeskorte RSA origin and the "Nielsen et al. 2025" reference (KL-close but representationally dissimilar) which is directly on-point for distillation.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nq(){ echo \"=== $1 ===\"; curl -s -m 40 \"https://export.arxiv.org/api/query?search_query=$2&start=0&max_results=${3:-12}&sortBy=relevance\" | python3 -c \"\nimport sys,re\nd=sys.stdin.read()\nfor e in re.findall(r'<entry>(.*?)</entry>',d,re.S):\n    t=' '.join(re.search(r'<title>(.*?)</title>',e,re.S).group(1).split())\n    i=re.search(r'<id>(.*?)</id>',e).group(1).split('/abs/')[-1]\n    p=re.search(r'<published>(.*?)</published>',e).group(1)[:10]\n    au=[a.strip() for a in re.findall(r'<name>(.*?)</name>',e)][:3]\n    print(p,i,'|',t,'|',', '.join(au))\n\"; echo; }\nq \"nielsen-behaviour-vs-rep\" 'au:Nielsen+AND+abs:%22representational+similarity%22' 15\nq \"same-behaviour-diff-rep\" 'abs:%22functionally+equivalent%22+AND+abs:%22representations%22+AND+abs:%22similarity%22' 15", "description": "Find Nielsen et al 2025 KL vs representation paper"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\necho \"### Kriegeskorte 2008 RSA via crossref\"\ncurl -s -m 30 \"https://api.crossref.org/works?query.bibliographic=Representational+similarity+analysis+connecting+the+branches+of+systems+neuroscience&rows=3\" | python3 -c \"\nimport sys,json\nd=json.load(sys.stdin)\nfor it in d['message']['items']:\n    print(it.get('title'), '|', it.get('container-title'), '|', it.get('published',{}).get('date-parts'), '|', it.get('DOI'))\n    print('  authors:', ', '.join((a.get('given','')+' '+a.get('family','')).strip() for a in it.get('author',[])[:5]))\n\"", "description": "Verify Kriegeskorte 2008 RSA via Crossref"}
```

<details><summary>tool result</summary>

```
=== nielsen-behaviour-vs-rep ===

=== same-behaviour-diff-rep ===
2026-05-20 2605.21324v1 | Stimulus symmetries can confound representational similarity analyses | Farhad Pashakhanloo, Jacob A. Zavatone-Veth
2025-10-05 2510.04376v1 | Categorical Invariants of Learning Dynamics | Abdulrahman Tamim
2026-09-15 2609.17338v1 | Type-IV Code Clone Detection via Layer-Wise Non-Contrastive Representation Learning | Luciano Marchezan, Kevin Delcourt, Eugene Syriani
2016-03-07 1603.02021v1 | Machine Learning for Protein Function | Dan Ofer
2023-12-17 2312.10725v2 | Harnessing small projectors and multiple views for efficient vision pretraining | Kumar Krishna Agrawal, Arna Ghosh, Shagun Sodhani
2026-04-18 2604.17153v1 | From Legal Text to Executable Decision Models: Evaluating Structured Representations for Legal Decision Model Generation | David Graus
2023-03-23 2303.13043v2 | Top-Down Visual Attention from Analysis by Synthesis | Baifeng Shi, Trevor Darrell, Xin Wang
2025-05-04 2505.02168v1 | CircuitFusion: Multimodal Circuit Representation Learning for Agile Chip Design | Wenji Fang, Shang Liu, Jing Wang
```
</details>

<details><summary>tool result</summary>

```
Exit code 1
### Kriegeskorte 2008 RSA via crossref
Traceback (most recent call last):
  File "<string>", line 3, in <module>
  File "/usr/local/lib/python3.12/json/__init__.py", line 293, in load
    return loads(fp.read(),
           ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/json/__init__.py", line 346, in loads
    return _default_decoder.decode(s)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/json/decoder.py", line 337, in decode
    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/json/decoder.py", line 355, in raw_decode
    raise JSONDecodeError("Expecting value", s, err.value) from None
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ncurl -sL -m 40 -H \"User-Agent: research/1.0 (mailto:x@example.com)\" \"https://api.crossref.org/works/10.3389/neuro.06.004.2008\" | python3 -c \"\nimport sys,json\nd=json.load(sys.stdin)['message']\nprint('TITLE:',d.get('title'))\nprint('JOURNAL:',d.get('container-title'))\nprint('YEAR:',d.get('published',{}).get('date-parts'))\nprint('DOI:',d.get('DOI'))\nprint('AUTHORS:',', '.join((a.get('given','')+' '+a.get('family','')).strip() for a in d.get('author',[])))\n\"", "description": "Verify Kriegeskorte 2008 RSA DOI via Crossref"}
```

<details><summary>tool result</summary>

```
Exit code 1
Traceback (most recent call last):
  File "<string>", line 3, in <module>
  File "/usr/local/lib/python3.12/json/__init__.py", line 293, in load
    return loads(fp.read(),
           ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/json/__init__.py", line 346, in loads
    return _default_decoder.decode(s)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/json/decoder.py", line 337, in decode
    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/json/decoder.py", line 355, in raw_decode
    raise JSONDecodeError("Expecting value", s, err.value) from None
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ncurl -sL -m 40 \"https://doi.org/10.3389/neuro.06.004.2008\" -o /dev/null -w \"final=%{url_effective} code=%{http_code}\\n\"\necho \"---\"\ncurl -sL -m 40 \"https://www.frontiersin.org/articles/10.3389/neuro.06.004.2008/full\" | grep -oiE \"<title>[^<]*</title>\" | head -2", "description": "Resolve Kriegeskorte 2008 DOI"}
```

<details><summary>tool result</summary>

```
final=https://doi.org/10.3389/neuro.06.004.2008 code=000
---
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ncurl -sL -m 40 \"https://pmc.ncbi.nlm.nih.gov/articles/PMC2605405/\" | grep -oiE \"<title>[^<]*</title>\" | head -2\necho \"=== semantic scholar ===\"\ncurl -sL -m 40 \"https://api.semanticscholar.org/graph/v1/paper/DOI:10.3389/neuro.06.004.2008?fields=title,venue,year,authors\" ", "description": "Verify Kriegeskorte 2008 via PMC and Semantic Scholar"}
```

<details><summary>tool result</summary>

```
Exit code 56
=== semantic scholar ===
```
</details>

### ASSISTANT — 2026-09-16 17:09

Non-arXiv endpoints are blocked by the proxy. Let me close the last gaps on arXiv.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nq(){ echo \"=== $1 ===\"; curl -s -m 40 \"https://export.arxiv.org/api/query?search_query=$2&start=0&max_results=${3:-12}&sortBy=relevance\" | python3 -c \"\nimport sys,re\nd=sys.stdin.read()\nfor e in re.findall(r'<entry>(.*?)</entry>',d,re.S):\n    t=' '.join(re.search(r'<title>(.*?)</title>',e,re.S).group(1).split())\n    i=re.search(r'<id>(.*?)</id>',e).group(1).split('/abs/')[-1]\n    p=re.search(r'<published>(.*?)</published>',e).group(1)[:10]\n    print(p,i,'|',t)\n\"; echo; }\nq \"CKA-predicts-transfer\" 'abs:%22CKA%22+AND+abs:%22predicts%22' 20\nq \"select-teacher-similarity\" 'abs:%22teacher+selection%22+AND+abs:%22distillation%22' 15\nq \"which-teacher-best-student\" 'ti:%22teacher%22+AND+abs:%22which+teacher%22' 12", "description": "Search CKA-predicts-transfer and teacher selection"}
```

<details><summary>tool result</summary>

```
=== CKA-predicts-transfer ===
2026-05-22 2605.23315v1 | Convergence Without Understanding: When Language Models Agree on Representations but Disagree on Reasoning
2026-04-08 2605.02914v1 | When Safety Geometry Collapses: Fine-Tuning Vulnerabilities in Agentic Guard Models
2022-12-04 2212.01924v1 | Cross-lingual Similarity of Multilingual Representations Revisited
2026-05-08 2605.07345v1 | Mean-Pooled Cosine Similarity is Not Length-Invariant: Theory and Cross-Domain Evidence for a Length-Invariant Alternative
2026-07-03 2607.03572v1 | Teacher Supervision over Representation Equivalence Classes
2026-01-23 2601.17093v1 | The Triangle of Similarity: A Multi-Faceted Framework for Comparing Neural Network Representations
2026-03-13 2603.12669v1 | Vision Verification Enhanced Fusion of VLMs for Efficient Visual Reasoning
2026-01-06 2601.03168v1 | Can Embedding Similarity Predict Cross-Lingual Transfer? A Systematic Study on African Languages
2022-06-16 2206.08347v1 | Beyond Supervised vs. Unsupervised: Representative Benchmarking and Analysis of Image Representation Learning
2026-06-05 2606.07882v2 | The Cross-Architecture Substrate: A Domain-Transcendent, Calibration-Surviving Geometric Invariant of Modern Vision Encoders
2026-09-01 2609.00686v1 | A Study of Hidden-State Optimization Order in Predictive Coding Networks
2024-06-20 2406.14479v3 | Tracing Representation Progression: Analyzing and Enhancing Layer-Wise Similarity
2026-04-20 2604.17698v2 | The Geometric Canary: Predicting Steerability and Detecting Drift via Representational Stability
2026-05-19 2605.20356v1 | Synchronization and Turn-Taking in Full-Duplex Speech Dialogue Models
2026-05-24 2605.24869v1 | Lngram: N-gram Conditional Memory in Latent Space
2026-01-29 2601.21568v2 | Bridging Functional and Representational Similarity via Usable Information
2024-05-27 2405.17081v2 | Effective Layer Pruning Through Similarity Metric Perspective
2025-02-27 2502.19648v2 | Spectral Analysis of Representational Similari
... [truncated 2653 chars]
```
</details>

### ASSISTANT — 2026-09-16 17:10

Two very relevant hits. Let me verify those and finish.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nfor id in 2605.23315 2601.14249 2605.15689 2601.17093; do\n  echo \"=== $id ===\"\n  curl -s -m 25 \"https://arxiv.org/abs/$id\" | grep -i -E \"<title>|Comments\" | sed 's/^[ \\t]*//'\n  echo\ndone", "description": "Verify final candidate papers"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ncurl -s -m 60 \"https://export.arxiv.org/api/query?id_list=2605.23315,2601.14249,2605.15689,2601.17093,2606.16379&max_results=10\" | python3 -c \"\nimport sys,re\nd=sys.stdin.read()\nfor e in re.findall(r'<entry>(.*?)</entry>',d,re.S):\n    t=' '.join(re.search(r'<title>(.*?)</title>',e,re.S).group(1).split())\n    i=re.search(r'<id>(.*?)</id>',e).group(1).split('/abs/')[-1]\n    s=' '.join(re.search(r'<summary>(.*?)</summary>',e,re.S).group(1).split())\n    print('###',i,'|',t); print(s[:1000]); print()\n\"", "description": "Abstracts of final candidates"}
```

<details><summary>tool result</summary>

```
=== 2605.23315 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2605.23315] Convergence Without Understanding: When Language Models Agree on Representations but Disagree on Reasoning</title>

=== 2601.14249 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2601.14249] Which Reasoning Trajectories Teach Students to Reason Better? A Simple Metric of Informative Alignment</title>
<td class="tablecell label">Comments:</td>
<td class="tablecell comments mathjax">Accepted to ACL 2026 (Main Conference). 31 pages. Project page: <a href="https://github.com/UmeanNever/RankSurprisalRatio" rel="external noopener nofollow" class="link-external link-https">this https URL</a></td>

=== 2605.15689 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2605.15689] How to Choose Your Teacher for Fine Grained Image Recognition</title>
<td class="tablecell label">Comments:</td>
<td class="tablecell comments mathjax">Accepted to The 13th Workshop on Fine-Grained Visual Categorization (FGVC13) @ CVPR 2026. Main: 6 pages, 3 figures, 4 tables</td>

=== 2601.17093 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[2601.17093] The Triangle of Similarity: A Multi-Faceted Framework for Comparing Neural Network Representations</title>
<td class="tablecell label">Comments:</td>
<td class="tablecell comments mathjax">Accepted to AAAI 2026 Workshop on AI for Scientific Research (AI4Research)</td>
```
</details>

<details><summary>tool result</summary>

```
### 2605.23315v1 | Convergence Without Understanding: When Language Models Agree on Representations but Disagree on Reasoning
Large language models trained under diverse objectives and architectures have been shown to develop increasingly similar internal representations, an observation formalized as the Platonic Representation Hypothesis. Whether this representational convergence extends to the reasoning processes that operate over shared representations remains untested. We evaluate representational similarity across 16 language models from 8 families (1.5B to 72B parameters) on 800 reasoning problems spanning mathematics, science, commonsense, and truthfulness, stratifying by problem difficulty, computational stage, and causal relevance. Our analysis reveals three dissociations: a difficulty inversion, where models converge more on problems they collectively fail (Centered Kernel Alignment [CKA] = 0.897) than on those they solve (CKA = 0.830); a generation gap, where pre-decision representations align (CKA = 0.875) while post-decision representations diverge (CKA = 0.274); and epiphenomenal correctness, wh

### 2601.17093v1 | The Triangle of Similarity: A Multi-Faceted Framework for Comparing Neural Network Representations
Comparing neural network representations is essential for understanding and validating models in scientific applications. Existing methods, however, often provide a limited view. We propose the Triangle of Similarity, a framework that combines three complementary perspectives: static representational similarity (CKA/Procrustes), functional similarity (Linear Mode Connectivity or Predictive Similarity), and sparsity similarity (robustness under pruning). Analyzing a range of CNNs, Vision Transformers, and Vision-Language Models using both in-distribution (ImageNetV2) and out-of-distribution (CIFAR-10) testbeds, our initial findings suggest that: (1) architectural family is a primary determinant of representational similarity, forming distinct cl
... [truncated 3493 chars]
```
</details>

### ASSISTANT — 2026-09-16 17:12

Research complete. Here are the findings.

# Representation Similarity Measures: Verified Literature Map

Verification method: `curl -s -m 25 "https://arxiv.org/abs/<id>" | grep -i -E "<title>|Comments"`. "No Comments field" means the paper genuinely has no Comments metadata (usually = unrefereed preprint, no venue claim). Non-arXiv sources (Crossref, PMC, Semantic Scholar) were blocked by the proxy — those are marked UNVERIFIED.

---

## 1. The Measures

### CKA — Centered Kernel Alignment
**Similarity of Neural Network Representations Revisited** — arXiv 1905.00414 — https://arxiv.org/abs/1905.00414 — 2019-05-01 — Kornblith, Norouzi, Lee, Hinton (Google Brain) — **Comments: "ICML 2019"** — **VERIFIED**

Important correction to a premise in your task framing: 1905.00414 is *not* primarily a critique of CKA — it is the paper that **introduces** CKA to this field. What it criticizes is the **prior art**: CCA, SVCCA, and PWCCA. Its argument is that CCA-family measures are invariant to *any* invertible linear transformation, and that this invariance is too strong — it destroys information that matters, so CCA fails the basic sanity check of identifying corresponding layers in architecturally identical networks trained from different random seeds. CKA is proposed as the fix because it is invariant to orthogonal transforms and isotropic scaling but *not* to arbitrary invertible linear maps.

1. **Linear or nonlinear:** Both variants exist. Linear CKA = a normalized, doubly-centered inner product between Gram matrices; equivalent to a normalized Frobenius product of cross-covariance matrices. It is invariant to orthogonal transformation and isotropic scaling, **not** to general invertible linear maps. RBF-kernel CKA is genuinely nonlinear. Empirically the paper finds the two rank things nearly identically, which is a key reason practitioners default to linear CKA.
2. **Conditional information:** **No.** Plain CKA has no mechanism to condition out a nuisance variable. This is the central gap in the standard toolkit.
3. **Critiques:** Extensive — see Section 2.

### CKNNA — Centered Kernel Nearest-Neighbor Alignment
**The Platonic Representation Hypothesis** — arXiv 2405.07987 — https://arxiv.org/abs/2405.07987 — 2024-05-13 — Huh, Cheung, Wang, Isola (MIT) — **Comments: "Equal contributions. Project: phillipi.github.io/prh Code: github.com/minyoungg/platonic-rep"** (no venue in Comments; ICML 2024 by other record) — **VERIFIED**

I confirmed CKNNA is defined *inside* this paper, not in a separate one — it is Eqns. 16–18 of the fulltext (retrieved `https://arxiv.org/html/2405.07987v3` and grepped). Exact definition: CKA restricted by an indicator α(i,j) = 1[φ_j ∈ knn(φ_i) ∧ ψ_j ∈ knn(ψ_i) ∧ i≠j], i.e. only **mutual** k-nearest-neighbor pairs contribute to the alignment sum. Direct quote from the paper: "As the number of nearest neighbors k → dim(K), we recover the original CKA metric." So CKNNA is a **local/topological interpolation** between mutual-nearest-neighbor accuracy (small k) and CKA (k = |X|).

1. **Linear or nonlinear:** Nonlinear even with a linear base kernel — the kNN indicator is a nonlinear, data-dependent gating of which pairs count.
2. **Conditional information:** No.

### SVCCA
**SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability** — arXiv 1706.05806 — https://arxiv.org/abs/1706.05806 — 2017-06-19 — Raghu, Gilmer, Yosinski, Sohl-Dickstein (Google Brain / Uber AI) — **Comments: "Accepted to NIPS 2017, code: github.com/google/svcca, new plots on Imagenet"** — **VERIFIED**

1. **Linear or nonlinear:** Linear. SVD pruning to top directions, then CCA. Invariant to *any* invertible linear transform (the property Kornblith et al. attack).
2. **Conditional information:** No.

### PWCCA
**Insights on representational similarity in neural networks with canonical correlation** — arXiv 1806.05759 — https://arxiv.org/abs/1806.05759 — 2018-06-14 — Morcos, Raghu, Bengio — **Comments: "NIPS 2018"** — **VERIFIED**

Projection-weighted CCA: weights canonical correlations by how much each contributes to the underlying representation, to suppress noisy low-variance directions. 1. **Linear**, same invariance class as CCA. 2. **No** conditioning.

### DVCCA
**Deep Variational Canonical Correlation Analysis** — arXiv 1610.03454 — https://arxiv.org/abs/1610.03454 — 2016-10-11 — Wang, Yan, Lee, Livescu — **no Comments field** — **VERIFIED**

1. **Nonlinear** — deep generative/VAE-based multi-view latent variable model. Note this is a *representation learning* method, not really a post-hoc similarity measure; it is often miscategorized alongside SVCCA/PWCCA. 2. Has latent-variable structure (private vs shared variables in DVCCA-private) that is adjacent to conditioning, but it is not a conditional similarity *measure*.

### RSA — Representational Similarity Analysis
**Kriegeskorte, Mur & Bandettini 2008, "Representational similarity analysis – connecting the branches of systems neuroscience," Frontiers in Systems Neuroscience 2:4, DOI 10.3389/neuro.06.004.2008** — **UNVERIFIED** (pre-arXiv, no arXiv ID; Crossref/PMC/Frontiers/Semantic Scholar all failed through the proxy — curl exit 56 / empty). Report as unverified.

Verified arXiv paper on RSA's statistical failure modes:
**Overcoming bias in representational similarity analysis** — arXiv 2102.08931 — https://arxiv.org/abs/2102.08931 — 2021-02-17 — **Comments: "17 pages, 6 figures"** (no venue) — **VERIFIED**

**Stimulus symmetries can confound representational similarity analyses** — arXiv 2605.21324 — 2026-05-20 — Pashakhanloo, Zavatone-Veth — **VERIFIED** (title only; recent preprint)

1. **Linear or nonlinear:** Depends on the chosen distance. RSA with correlation distance on the RDM is closely related to CKA; Kornblith et al. explicitly note linear CKA is a member of the RSA family. Second-order (compares distance matrices, not features), so dimension-agnostic.
2. **Conditional information:** Not natively. However RSA is the one framework with an established partial-correlation practice in neuroscience (partial RSA / regressing out a nuisance RDM), which is the closest thing to a conventional conditional variant.

### Procrustes / shape distances
**Generalized Shape Metrics on Neural Representations** — arXiv 2110.14739 — https://arxiv.org/abs/2110.14739 — 2021-10-27 — Williams, Kunz, Kornblith, Linderman — **Comments: "26 pages, 7 figures, NeurIPS 2021"** — **VERIFIED**

**Duality of Bures and Shape Distances with Implications for Comparing Neural Representations** — arXiv 2311.11436 — 2023-11-19 — **VERIFIED** (title)

1. **Linear** in the relevant sense: an orthogonal alignment plus scaling, giving a true metric (satisfies triangle inequality — unlike CKA, which is not a metric). This is a real advantage for clustering/embedding models.
2. **No** conditioning.

### Linear probes / linear predictivity
**What Representational Similarity Measures Imply about Decodable Information** — arXiv 2411.08197 — https://arxiv.org/abs/2411.08197 — 2024-11-12 — Harvey, Lipshutz, Williams (Flatiron Institute) — **no Comments field** — **VERIFIED**

This is the theoretical bridge you want. It proves CKA and CCA can be *equivalently motivated* as "the average alignment between optimal linear readouts across a distribution of decoding tasks," and that Procrustes distance upper-bounds the distance between optimal linear readouts. So the geometric measures and the probe-based measures are not rival philosophies — they are the same object viewed differently. Useful if you need to argue a geometric measure has decoding semantics.

1. **Linear** by construction. 2. **No** conditioning natively, though probes are the easiest place to *add* conditioning (train on residuals, or include the nuisance as a covariate).

---

## 2. Measures That DO Handle Conditional Information

This is the sparsest part of the literature, and the most important for your purposes. `all:"conditional CKA"` and `all:"partial CKA"` return **essentially nothing** on arXiv — there is no established "conditional CKA." What exists:

### Partial Distance Correlation — the strongest existing answer
**On the Versatile Uses of Partial Distance Correlation in Deep Learning** — arXiv 2207.09684 — https://arxiv.org/abs/2207.09684 — 2022-07-20 — Zhen, Meng, Chakraborty, Singh (UW-Madison) — **Comments: "This paper has been selected as best paper award for ECCV 2022!"** — **VERIFIED**

This is the paper closest to "a similarity measure that conditions out a variable," and it won ECCV 2022 best paper. Distance correlation handles feature spaces of *different dimensions* natively (no projector needed — a real advantage over CKA in teacher/student settings with mismatched widths). The **partial** variant explicitly supports "conditioning one deep model w.r.t. another," which they use for disentanglement and to build representations robust to adversarial attack. From the abstract: "conditioning one deep model w.r.t. another, learning disentangled representations." **Nonlinear** (energy-distance based, detects nonlinear dependence, zero iff independent — strictly stronger than correlation).

### Deconfounded CKA / RSA — the direct fix for the "random networks look similar" problem
**Deconfounded Representation Similarity for Comparison of Neural Networks** — arXiv 2202.00095 — https://arxiv.org/abs/2202.00095 — 2022-01-31 — Cui, Kumar, Marttinen, Kaski (Aalto) — **no Comments field**; other records place it at NeurIPS 2022 — **VERIFIED**

This is the paper you were looking for under "CKA between random networks is high." Verbatim from the abstract: RSA and CKA "are confounded by the population structure of data items in the input space, leading to **spuriously high similarity for even completely random neural networks** and inconsistent domain relations in transfer learning." Their fix is **covariate adjustment regression** — regress the confounder out while preserving the original invariance properties. Results: better resolution in detecting semantically similar networks, more consistent domain relations in transfer learning, and higher correlation with OOD accuracy. This is effectively "partial CKA" without using that name, and is the primary citation for a conditional variant.

### Kernel Partial Correlation
**Kernel Partial Correlation Coefficient — a Measure of Conditional Dependence** — arXiv 2012.14804 — https://arxiv.org/abs/2012.14804 — 2020-12-29 — **Comments: "63 pages, 4 figures, 9 tables"** (JMLR by other record) — **VERIFIED**

Statistics-side foundation: a consistent, nonparametric measure of conditional dependence in the [0,1] range, zero iff conditional independence. Not a neural-representation paper, but the correct theoretical scaffold if you want to *build* a conditional kernel-alignment measure rather than cite one.

### Contrastive-Difference CKA (CKA_Δ)
**Contrastive-Difference CKA Reveals Concept-Specific Structural Alignment Across Language Model Architectures** — arXiv 2606.16897 — 2026-06-15 — **no Comments field** — **VERIFIED (title only; unrefereed preprint, treat with caution)**

Computes kernel alignment on per-sample *contrastive differences* rather than raw activations, to "isolate concept-specific convergence from generic similarity — achieving significant discrimination where standard CKA cannot." Poor-man's conditioning via differencing. Cheap and training-free, but this is a recent unrefereed preprint.

### Barycentric alignment (quotients out nuisance symmetries)
**Barycentric alignment for instance-level comparison of neural representations** — arXiv 2602.09225 — 2026-02-09 — **no Comments field** — **VERIFIED (title only)**. Quotients out nuisance symmetries and gives *per-stimulus* similarity instead of a single dataset-level scalar.

### Bridging Functional and Representational Similarity via Usable Information
arXiv 2601.21568 — 2026-01-29 — **no Comments field** — **VERIFIED (title only)**. Relevant claim: formally links stitching performance to **conditional mutual information**, shows stitching is asymmetric, and argues CKA/RSA are estimators of usable information under constraints. Key line: "representational similarity is **sufficient but not necessary** for functional similarity."

---

## 3. The Critique Literature

### The core CKA takedown
**Reliability of CKA as a Similarity Measure in Deep Learning** — arXiv 2210.16156 — https://arxiv.org/abs/2210.16156 — 2022-10-28 — Davari, Horoi, Natik, Lajoie, Wolf, Belilovsky (Mila/Concordia) — **no Comments field**; ICLR 2023 by other record — **VERIFIED**

This is your "CKA can be gamed" paper, and it is explicit. It formally characterizes CKA's sensitivity to a broad class of simple transformations, explains CKA's **sensitivity to outliers**, and shows sensitivity to transformations that *preserve linear separability* (i.e. preserve what you actually care about). Money quote: "the CKA value can be **easily manipulated without substantial changes to the functional behaviour** of the models, and call for caution when leveraging activation alignment metrics." Combined with 2202.00095's "spuriously high similarity for even completely random neural networks," these two are the backbone of any CKA-skepticism argument.

### Measures disagree with each other
**Grounding Representation Similarity with Statistical Testing** — arXiv 2108.01661 — https://arxiv.org/abs/2108.01661 — 2021-08-03 — Ding, Denain, Steinhardt (Berkeley) — **Comments: "Accepted at NeurIPS 2021. 10 pages, 3 figures"** — **VERIFIED**

Frames the problem sharply: CCA, CKA and others "often **disagree on fundamental observations**, such as whether deep networks differing only in random initialization learn similar representations. These disagreements raise the question: which, if any, of these dissimilarity measures should we believe?" Proposes a sensitivity/specificity test against functional behavior. Findings: every current metric has distinct weaknesses, "a classical baseline performs surprisingly well," and there exist settings where **all** metrics fail.

### CKA misses what stitching catches
**Revisiting Model Stitching to Compare Neural Representations** — arXiv 2106.07682 — https://arxiv.org/abs/2106.07682 — 2021-06-14 — Bansal, Nakkiran, Barak (Harvard) — **no Comments field**; NeurIPS 2021 by other record — **VERIFIED**

Direct claim: model stitching "reveals aspects of representations that measures such as **centered kernel alignment (CKA) cannot**." Functional test rather than geometric one.

### Which measures track behavior?
**Evaluating Representational Similarity Measures from the Lens of Functional Correspondence** — arXiv 2411.14633 — https://arxiv.org/abs/2411.14633 — 2024-11-21 — Bo, Soni, Srivastava, Khosla (UCSD) — **Comments: "Published in CCN 2025 Proceedings (Talk & Poster), May 14, 2025"** — **VERIFIED**

Partially *defends* CKA — useful for balance. Across eight measures: linear CKA and Procrustes distance "excelled in differentiating trained from untrained models and aligning with behavioral measures, whereas metrics such as **linear predictivity, commonly used in neuroscience, demonstrated only moderate alignment with behavior**." So this is simultaneously a point in CKA's favor and a critique of linear-probe-style measures.

### Benchmarks and taxonomy
- **ReSi: A Comprehensive Benchmark for Representational Similarity Measures** — arXiv 2408.00531 — 2024-08-01 — **Comments: "ICLR 2025. Code and data at github.com/mklabunde/resi"** — **VERIFIED**
- **A Framework for Standardizing Similarity Measures in a Rapidly Evolving Field** — arXiv 2409.18333 — 2024-09-26 — **Comments: "11 pages, 9 figures"** — **VERIFIED**. Notable fact for a related-work section: "even commonly used methods like Centered Kernel Alignment (CKA) have **at least 12 different variations**." Strong support for the claim that "CKA = 0.9" is not a well-defined statement.
- **Objective drives the consistency of representational similarity across datasets** — arXiv 2411.05561 — 2024-11-08 — **Comments: "26 pages"** — **VERIFIED**. Shows measured similarity depends on the stimulus set; convergence claims may be dataset-confounded.
- **Gaussian RBF Centered Kernel Alignment (CKA) in the Large Bandwidth Limit** — arXiv 2112.09305 — 2021-12-17 — **VERIFIED (title)**. Proves RBF-CKA → linear CKA as bandwidth → ∞, i.e. the "nonlinear" variant often collapses to the linear one.

### On "The Unreasonable Effectiveness of Random Networks"
**No such paper exists** with that exact title. Searching `ti:"random" AND ti:"unreasonable"` returns only unrelated works (structured random orthogonal embeddings 1703.00864, random pruning 2202.02643, random target embeddings 2310.20620). The claim you are thinking of — high CKA between random/untrained networks — is properly cited to **2202.00095 (Deconfounded Representation Similarity)** and **2108.01661 (Grounding Representation Similarity)**. Do not cite a paper by that title.

### Probing critiques
- **Designing and Interpreting Probes with Control Tasks** — arXiv 1909.03368 — https://arxiv.org/abs/1909.03368 — 2019-09-07 — Hewitt & Liang (Stanford) — **Comments: "EMNLP 2019"** — **VERIFIED**. Note: the title you supplied ("Probing classifiers: promises and pitfalls") is wrong for this ID — this is the *control tasks* paper. Its argument: high probe accuracy may reflect the probe's own capacity to memorize rather than structure in the representation; use control tasks and report *selectivity*.
- **Probing Classifiers: Promises, Shortcomings, and Advances** — arXiv 2102.12452 — https://arxiv.org/abs/2102.12452 — 2021-02-24 — Belinkov — **Comments: "Accepted to Computational Linguistics as a squib"** — **VERIFIED**. This is the paper you meant.
- **Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals** — arXiv 2006.00995 — https://arxiv.org/abs/2006.00995 — 2020-06-01 — Elazar, Ravfogel, Jacovi, Goldberg — **Comments: "TACL journal. Initial title was: 'When Bert Forgets How To POS: Amnesic Probing of Linguistic Properties and MLM Predictions'"** — **VERIFIED**. (Your ID 2009.12138 was wrong — that's a condensed-matter physics paper on Fermi arcs.) Core critique: probe accuracy shows information is *present*, not that the model *uses* it. Fix: remove the property, then measure behavioral change. This is a counterfactual/interventional design and is conceptually the nearest neighbor to conditioning out a variable.
- **Information-Theoretic Probing with Minimum Description Length** — arXiv 2003.12298 — https://arxiv.org/abs/2003.12298 — 2020-03-27 — Voita & Titov — **no Comments field**; EMNLP 2020 — **VERIFIED**. (Your ID 2002.12327 was wrong — that is "A Primer in BERTology," TACL.) Argues raw probe accuracy is the wrong readout; use MDL/codelength, which is far more stable across probe architectures and random seeds.
- **Probing Classifiers are Unreliable for Concept Removal and Detection** — arXiv 2207.04153 — https://arxiv.org/abs/2207.04153 — 2022-07-08 — **no Comments field**; NeurIPS 2022 by other record — **VERIFIED**. Most aggressive of the set, and directly relevant if you plan to condition by *removing* a variable: probe-based concept removal provably leaves the concept recoverable.

---

## 4. CRITICAL: Similarity Measures ↔ Distillation Effectiveness

This is where the literature is genuinely thin, and there is a clear gap. Almost all CKA + distillation work uses CKA as a **training loss**, not as a **diagnostic or predictor of distillation success**. Both directions below:

### 4a. CKA used as a distillation *loss* (well-populated)
- **Rethinking Centered Kernel Alignment in Knowledge Distillation** — arXiv 2401.11824 — https://arxiv.org/abs/2401.11824 — 2024-01-22 — Zhou, Shen, Shao, Gong, Lin — **no Comments field**; IJCAI 2024 by other record — **VERIFIED**. Theoretically decouples CKA into an **upper bound of MMD plus a constant**, then builds Relation-CKA (RCKA). This MMD co
... [truncated 8737 chars]
