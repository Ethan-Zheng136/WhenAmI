# Claude Code session agent-ac5adfeb7d3aab0ba

- Time (Beijing): 2026-09-16 17:02 → 2026-09-16 17:22
- cwd: `/home/zhengguantian/WoG_v2`
- Source: `~/.claude/projects/-home-zhengguantian/c976c298-d71c-4d38-9107-f692d869d083/subagents/agent-ac5adfeb7d3aab0ba.jsonl` (neolix_al_2)
- Messages (new, after de-dup): 123
- Note: tool outputs capped at 2000 chars; thinking blocks and images dropped; secrets/private IPs redacted.

---

### USER — 2026-09-16 17:02

You are doing academic literature search. Before ANY curl, run: export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000

TOPIC: **Redundancy / usefulness diagnostics of AUXILIARY OBJECTIVES in multi-task and auxiliary-task learning.**

Find and VERIFY papers on:
- "auxiliary task redundancy", "which auxiliary tasks help", "auxiliary task selection", "negative transfer in auxiliary tasks"
- measures of "task relatedness" / "task affinity" / "task similarity" — ESPECIALLY any measure computed via **representation predictability** (i.e., can task B's target be predicted from the representation learned for task A / by a linear probe). Taskonomy (1804.08328) and its affinity matrix is relevant — check how they compute transferability.
- Gradient-based task-interference measures: "Gradient Surgery / PCGrad" (2001.06782), "Adapting Auxiliary Losses Using Gradient Similarity" (1812.02224), "Auxiliary Task Update Decomposition", "AuxiLearn / Auxiliary Learning by Implicit Differentiation" (2007.02693), "Meta Learning via Learned Loss", "Task Affinity Grouping / TAG" (2109.04617 - already verified NeurIPS 2021 spotlight, skip re-verify but describe its measure).
- "Which Tasks Should Be Learned Together in Multi-task Learning?" (1905.07553)
- MMoE "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts" — this is a KDD 2018 paper, find if it has an arXiv id (the id 1803.09820 given to me was WRONG, that's Leslie Smith's LR paper). Report correct venue/DOI.
- Also: information-theoretic views — "conditional mutual information between auxiliary target and main target given input", "V-usable information" (2110.08420 Ethayarajh et al.), "Understanding Dataset Difficulty with V-Usable Information" — this defines predictability of a target given a representation under a model family; highly relevant.
- Also "Probing / linear probe" based redundancy: "Information-Theoretic Probing with Minimum Description Length" (2003.12298), "Amnesic Probing", "Conditional probing: measuring usable information beyond a baseline" (2109.09234) — THIS LAST ONE measures usable information BEYOND a baseline representation, which is exactly a redundancy diagnostic.

VERIFY every paper by: curl -s -m 25 "https://arxiv.org/abs/<id>" | grep -i -E "<title>|Comments"
Report exact title + Comments (venue). Mark VERIFIED / UNVERIFIED. If a curl fails, retry once with sleep 3.

Report format: markdown, per paper: exact title / arXiv id / URL / date / venue / 1-2 sentence description of the MEASURE it defines.
**Emphasize loudly any paper that measures "predictability of a target from a (frozen) representation" as a diagnostic of redundancy or usefulness.**

Do NOT write any files. Return findings as your final message. Be thorough but efficient — batch curls in loops with sleep 1 between.

### ASSISTANT — 2026-09-16 17:02

I'll start by verifying the arXiv papers in batches, then investigate the non-arXiv ones.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nfor id in 1804.08328 2001.06782 1812.02224 2007.02693 1905.07553 2110.08420 2003.12298 2109.09234; do\n  echo \"=== $id ===\"\n  curl -s -m 25 \"https://arxiv.org/abs/$id\" | grep -i -E \"<title>|Comments|citation_date|citation_journal\" | head -20\n  sleep 1\ndone", "description": "Verify first batch of arXiv papers"}
```

<details><summary>tool result</summary>

```
=== 1804.08328 ===
<head><script>document.documentElement.classList.add('js');</script>  <title>[1804.08328] Taskonomy: Disentangling Task Transfer Learning</title>
  <script src="/static/browse/0.3.4/js/cite.js" type="text/javascript"></script><meta name="citation_title" content="Taskonomy: Disentangling Task Transfer Learning" /><meta name="citation_author" content="Zamir, Amir" /><meta name="citation_author" content="Sax, Alexander" /><meta name="citation_author" content="Shen, William" /><meta name="citation_author" content="Guibas, Leonidas" /><meta name="citation_author" content="Malik, Jitendra" /><meta name="citation_author" content="Savarese, Silvio" /><meta name="citation_date" content="2018/04/23" /><meta name="citation_online_date" content="2018/04/23" /><meta name="citation_pdf_url" content="https://arxiv.org/pdf/1804.08328" /><meta name="citation_arxiv_id" content="1804.08328" /><meta name="citation_abstract" content="Do visual tasks have a relationship, or are they unrelated? For instance, could having surface normals simplify estimating the depth of an image? Intuition answers these questions positively, implying existence of a structure among visual tasks. Knowing this structure has notable values; it is the concept underlying transfer learning and provides a principled way for identifying redundancies across tasks, e.g., to seamlessly reuse supervision among related tasks or solve many tasks in one system without piling up the complexity. We proposes a fully computational approach for modeling the structure of space of visual tasks. This is done via finding (first and higher-order) transfer learning dependencies across a dictionary of twenty six 2D, 2.5D, 3D, and semantic tasks in a latent space. The product is a computational taxonomic map for task transfer learning. We study the consequences of this structure, e.g. nontrivial emerged relationships, and exploit them to reduce the demand for labeled data. For example, we show that the total number 
... [truncated 15811 chars]
```
</details>

### ASSISTANT — 2026-09-16 17:03

Now let me check the venue/journal fields for the two without Comments, and search for the remaining papers.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nfor id in 1812.02224 2003.12298; do\n  echo \"=== $id ===\"\n  curl -s -m 25 \"https://arxiv.org/abs/$id\" | grep -i -E \"journal|doi|tablecell subjects|Subjects\" | head -10\n  sleep 1\ndone\necho \"###### SEARCH MMoE ######\"\ncurl -s -m 25 \"http://export.arxiv.org/api/query?search_query=all:%22Multi-gate%20Mixture-of-Experts%22&max_results=10\" | grep -E \"<title>|<id>|<published>\" | head -40", "description": "Check venue fields and search MMoE on arXiv API"}
```

<details><summary>tool result</summary>

```
=== 1812.02224 ===
          <td class="tablecell label">Subjects:</td>
          <td class="tablecell subjects">
          <td class="tablecell arxivdoi">              <a href="https://doi.org/10.48550/arXiv.1812.02224"  id="arxiv-doi-link">https://doi.org/10.48550/arXiv.1812.02224</a><div class="button-and-tooltip">
                <span class="left-corner"></span>                  arXiv-issued DOI via DataCite</div>
=== 2003.12298 ===
          <td class="tablecell label">Subjects:</td>
          <td class="tablecell subjects">
          <td class="tablecell arxivdoi">              <a href="https://doi.org/10.48550/arXiv.2003.12298"  id="arxiv-doi-link">https://doi.org/10.48550/arXiv.2003.12298</a><div class="button-and-tooltip">
                <span class="left-corner"></span>                  arXiv-issued DOI via DataCite</div>
      <a href="https://dblp.uni-trier.de/db/journals/corr/corr2003.html#abs-2003-12298" title="listing on DBLP">listing</a> | <a href="https://dblp.uni-trier.de/rec/bibtex/journals/corr/abs-2003-12298" title="DBLP bibtex record">bibtex</a>    </div>
###### SEARCH MMoE ######
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\necho \"=== MMoE via semanticscholar search ===\"\ncurl -s -m 25 \"https://api.semanticscholar.org/graph/v1/paper/search?query=Modeling+Task+Relationships+in+Multi-task+Learning+with+Multi-gate+Mixture-of-Experts&fields=title,year,venue,externalIds,publicationVenue&limit=5\"\necho\nsleep 2\necho \"=== 1812.02224 venue ===\"\ncurl -s -m 25 \"https://api.semanticscholar.org/graph/v1/paper/arXiv:1812.02224?fields=title,year,venue,externalIds,publicationTypes,journal\"\necho\nsleep 2\necho \"=== 2003.12298 venue ===\"\ncurl -s -m 25 \"https://api.semanticscholar.org/graph/v1/paper/arXiv:2003.12298?fields=title,year,venue,externalIds,journal\"\necho", "description": "Query Semantic Scholar for MMoE and venues"}
```

<details><summary>tool result</summary>

```
=== MMoE via semanticscholar search ===

=== 1812.02224 venue ===

=== 2003.12298 venue ===
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ncurl -s -m 25 -w \"\\nHTTP:%{http_code}\\n\" \"https://api.semanticscholar.org/graph/v1/paper/arXiv:1812.02224?fields=title,venue,year\" | head -5\nsleep 3\necho \"=== DBLP MMoE ===\"\ncurl -s -m 25 \"https://dblp.org/search/publ/api?q=Multi-gate+Mixture-of-Experts&format=json&h=10\" | python3 -c \"\nimport json,sys\nd=json.load(sys.stdin)\nfor h in d['result']['hits'].get('hit',[]):\n    i=h['info']\n    print(i.get('title'),'|',i.get('venue'),'|',i.get('year'),'|',i.get('doi'),'|',i.get('ee'))\n\"", "description": "Test S2 status and query DBLP for MMoE"}
```

<details><summary>tool result</summary>

```
Exit code 1

HTTP:000
=== DBLP MMoE ===
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
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nfor u in \"https://dblp.org/search/publ/api?q=Multi-gate+Mixture-of-Experts&format=json&h=5\" \"https://api.crossref.org/works?query.bibliographic=Modeling+Task+Relationships+Multi-gate+Mixture-of-Experts&rows=3\" \"https://api.openalex.org/works?search=Multi-gate%20Mixture-of-Experts&per-page=3\"; do\n echo \"=== $u ===\"\n curl -s -m 25 -w \"\\nHTTP:%{http_code}\\n\" \"$u\" | head -c 600\n echo\n sleep 1\ndone", "description": "Test reachability of DBLP, Crossref, OpenAlex"}
```

<details><summary>tool result</summary>

```
=== https://dblp.org/search/publ/api?q=Multi-gate+Mixture-of-Experts&format=json&h=5 ===

HTTP:000

=== https://api.crossref.org/works?query.bibliographic=Modeling+Task+Relationships+Multi-gate+Mixture-of-Experts&rows=3 ===

HTTP:000

=== https://api.openalex.org/works?search=Multi-gate%20Mixture-of-Experts&per-page=3 ===

HTTP:000
```
</details>

### ASSISTANT — 2026-09-16 17:04

Only arxiv.org seems reachable via curl. Let me use WebSearch/WebFetch for the non-arXiv items.

**tool call `WebSearch`**
```
{"query": "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts KDD 2018 DOI Ma Zhao Yi Chen Hong Chi"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\necho \"=== arXiv API: auxiliary task update decomposition ===\"\ncurl -s -m 25 \"http://export.arxiv.org/api/query?search_query=ti:%22Auxiliary+Task+Update+Decomposition%22&max_results=5\" | grep -E \"<title>|<id>|<published>|<summary>\" | head -20\nsleep 1\necho \"=== arXiv API: Meta-Learning via Learned Loss ===\"\ncurl -s -m 25 \"http://export.arxiv.org/api/query?search_query=ti:%22Meta-Learning+via+Learned+Loss%22&max_results=5\" | grep -E \"<title>|<id>|<published>\" | head -20", "description": "Search arXiv API for two more papers"}
```

<details><summary>tool result</summary>

```
Web search results for query: "Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts KDD 2018 DOI Ma Zhao Yi Chen Hong Chi"

Links: [{"title":"KDD 2018 | Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts","url":"https://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-"},{"title":"Modeling Task Relationships in Multi-task Learning with ...","url":"https://dl.acm.org/doi/pdf/10.1145/3219819.3220007"},{"title":"Modeling task relationships in multi-task learning with multi-gate mixture-of-experts - Illinois Experts","url":"https://experts.illinois.edu/en/publications/modeling-task-relationships-in-multi-task-learning-with-multi-gat/"},{"title":"Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts | Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining","url":"https://dl.acm.org/doi/10.1145/3219819.3220007"},{"title":"Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts","url":"https://dl.acm.org/doi/epdf/10.1145/3219819.3220007"},{"title":"Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts | Request PDF","url":"https://www.researchgate.net/publication/326495135_Modeling_Task_Relationships_in_Multi-task_Learning_with_Multi-gate_Mixture-of-Experts"},{"title":"GitHub - drawbridge/keras-mmoe: A TensorFlow Keras implementation of \"Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts\" (KDD 2018) · GitHub","url":"https://github.com/drawbridge/keras-mmoe"},{"title":"Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts - researchr publication","url":"https://researchr.org/publication/MaZYCHC18"},{"title":"Multi-gate Mixture-of-Experts for Multi-task Learning | PDF | Deep Learning | Learning","url":"https://www.scribd.com/document/9694773
... [truncated 2319 chars]
```
</details>

<details><summary>tool result</summary>

```
=== arXiv API: auxiliary task update decomposition ===
=== arXiv API: Meta-Learning via Learned Loss ===
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\necho \"=== test export api ===\"\ncurl -s -m 25 -w \"\\nHTTP:%{http_code}\\n\" \"http://export.arxiv.org/api/query?search_query=all:electron&max_results=1\" | head -c 300\necho\necho \"=== arxiv web search ===\"\ncurl -s -m 25 -w \"\\nHTTP:%{http_code}\\n\" \"https://arxiv.org/a/test\" -o /dev/null\ncurl -s -m 25 \"https://arxiv.org/list/cs.LG/recent\" -o /dev/null -w \"HTTP:%{http_code}\\n\"", "description": "Test arXiv export API reachability"}
```

<details><summary>tool result</summary>

```
=== test export api ===

HTTP:301

=== arxiv web search ===

HTTP:303
HTTP:200
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nq() {\necho \"=== $1 ===\"\ncurl -s -m 25 \"https://export.arxiv.org/api/query?search_query=$1&max_results=6\" | python3 -c \"\nimport sys,re,xml.etree.ElementTree as ET\nns={'a':'http://www.w3.org/2005/Atom'}\ntry:\n  r=ET.fromstring(sys.stdin.read())\nexcept Exception as e:\n  print('parse err',e); raise SystemExit\nfor e in r.findall('a:entry',ns):\n  print(e.find('a:id',ns).text,'|',' '.join(e.find('a:title',ns).text.split()),'|',e.find('a:published',ns).text[:10])\n  c=e.find('{http://arxiv.org/schemas/atom}comment')\n  if c is not None: print('   COMMENT:',' '.join(c.text.split()))\n\"\nsleep 1\n}\nq 'ti:%22Auxiliary+Task+Update+Decomposition%22'\nq 'ti:%22Meta-Learning+via+Learned+Loss%22'\nq 'ti:%22Multi-gate+Mixture-of-Experts%22'\nq 'all:%22auxiliary+task+selection%22'", "description": "Search arXiv API over https for remaining papers"}
```

<details><summary>tool result</summary>

```
=== ti:%22Auxiliary+Task+Update+Decomposition%22 ===
http://arxiv.org/abs/2108.11346v1 | Auxiliary Task Update Decomposition: The Good, The Bad and The Neutral | 2021-08-25
   COMMENT: 15 pages, 3 figures, Accepted to International Conference on Learning Representations (ICLR) 2021 See https://github.com/ldery/ATTITTUD}{https://github.com/ldery/ATTITTUD for associated code
=== ti:%22Meta-Learning+via+Learned+Loss%22 ===
http://arxiv.org/abs/1906.05374v4 | Meta-Learning via Learned Loss | 2019-06-12
   COMMENT: Project website with code and video at https://sites.google.com/view/mlthree
=== ti:%22Multi-gate+Mixture-of-Experts%22 ===
http://arxiv.org/abs/2401.14426v1 | M$^3$TN: Multi-gate Mixture-of-Experts based Multi-valued Treatment Network for Uplift Modeling | 2024-01-24
   COMMENT: ICASSP 2024
http://arxiv.org/abs/2308.00533v1 | A Novel Temporal Multi-Gate Mixture-of-Experts Approach for Vehicle Trajectory and Driving Intention Prediction | 2023-08-01
http://arxiv.org/abs/2411.02787v1 | Advancing Robust Underwater Acoustic Target Recognition through Multi-task Learning and Multi-Gate Mixture-of-Experts | 2024-11-05
http://arxiv.org/abs/2407.17267v1 | M4: Multi-Proxy Multi-Gate Mixture of Experts Network for Multiple Instance Learning in Histopathology Image Analysis | 2024-07-24
   COMMENT: 25pages,5figures
http://arxiv.org/abs/2406.08742v1 | DeepUnifiedMom: Unified Time-series Momentum Portfolio Construction via Multi-Task Learning with Multi-Gate Mixture of Experts | 2024-06-13
   COMMENT: 21 pages, 5 figures
=== all:%22auxiliary+task+selection%22 ===
http://arxiv.org/abs/2109.05748v1 | GradTS: A Gradient-Based Automatic Auxiliary Task Selection Method Based on Transformer Networks | 2021-09-13
   COMMENT: In EMNLP 2021
```
</details>

### ASSISTANT — 2026-09-16 17:06

MMoE confirmed to have no arXiv version. Now let me search broadly for additional relevant work, especially representation-predictability measures.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nq() {\necho \"=== QUERY: $2 ===\"\ncurl -s -m 25 \"https://export.arxiv.org/api/query?search_query=$1&max_results=${3:-8}&sortBy=relevance\" | python3 -c \"\nimport sys,xml.etree.ElementTree as ET\nns={'a':'http://www.w3.org/2005/Atom'}\ntry: r=ET.fromstring(sys.stdin.read())\nexcept Exception as e: print('parse err',e); raise SystemExit\nfor e in r.findall('a:entry',ns):\n  print(e.find('a:id',ns).text,'|',' '.join(e.find('a:title',ns).text.split()),'|',e.find('a:published',ns).text[:10])\n  c=e.find('{http://arxiv.org/schemas/atom}comment')\n  if c is not None: print('   C:',' '.join(c.text.split())[:160])\n\"\nsleep 1\n}\nq 'ti:%22Amnesic+Probing%22' 'amnesic probing'\nq 'all:%22task+affinity%22+AND+cat:cs.LG' 'task affinity'\nq 'all:%22negative+transfer%22+AND+all:%22multi-task%22' 'negative transfer MTL'\nq 'ti:%22auxiliary+task%22+AND+abs:%22main+task%22+AND+abs:%22helpful%22' 'aux task helpful'\nq 'abs:%22task+grouping%22+AND+cat:cs.LG' 'task grouping'", "description": "Search arXiv for related measures"}
```

<details><summary>tool result</summary>

```
=== QUERY: amnesic probing ===
http://arxiv.org/abs/2006.00995v3 | Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals | 2020-06-01
   C: TACL journal. Initial title was: "When Bert Forgets How To POS: Amnesic Probing of Linguistic Properties and MLM Predictions"
http://arxiv.org/abs/2506.11673v1 | Improving Causal Interventions in Amnesic Probing with Mean Projection or LEACE | 2025-06-13
=== QUERY: task affinity ===
http://arxiv.org/abs/2409.06091v2 | Scalable Multitask Learning Using Gradient-based Estimation of Task Affinity | 2024-09-09
   C: 16 pages. Appeared in KDD 2024
http://arxiv.org/abs/2301.02873v1 | "It's a Match!" -- A Benchmark of Task Affinity Scores for Joint Learning | 2023-01-07
   C: 7 pages. AAAI'23 - 2nd International Workshop on Practical Deep Learning in the Wild
http://arxiv.org/abs/2401.11124v2 | Cross-Task Affinity Learning for Multitask Dense Scene Predictions | 2024-01-20
   C: Accepted for publication at the IEEE Winter Conference on Applications of Computer Vision (WACV) 2025
http://arxiv.org/abs/2602.18591v1 | Ensemble Prediction of Task Affinity for Efficient Multi-Task Learning | 2026-02-20
http://arxiv.org/abs/2306.14009v4 | Boosting Multitask Learning on Graphs through Higher-Order Task Affinities | 2023-06-24
   C: 16 pages. Appeared in KDD 2023
http://arxiv.org/abs/2110.02399v2 | Task Affinity with Maximum Bipartite Matching in Few-Shot Learning | 2021-10-05
   C: Accepted as a conference paper at ICLR 2022
http://arxiv.org/abs/2310.16241v1 | Task Grouping for Automated Multi-Task Machine Learning via Task Affinity Prediction | 2023-10-24
http://arxiv.org/abs/2604.07848v2 | Information-Theoretic Requirements for Gradient-Based Task Affinity Estimation in Multi-Task Learning | 2026-04-09
   C: 8 pages, 4 figures. ACM BCB 2026 Short Paper. Accepted at workshop on AI for Accelerated Materials Design, Foundation Models for Science: Real-World Impact and 
=== QUERY: negative transfer MTL ===
http://arxiv.org/abs/230
... [truncated 1944 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nq() {\necho \"=== QUERY: $2 ===\"\ncurl -s -m 25 \"https://export.arxiv.org/api/query?search_query=$1&max_results=${3:-8}&sortBy=relevance\" | python3 -c \"\nimport sys,xml.etree.ElementTree as ET\nns={'a':'http://www.w3.org/2005/Atom'}\ntry: r=ET.fromstring(sys.stdin.read())\nexcept Exception as e: print('parse err',e); raise SystemExit\nfor e in r.findall('a:entry',ns):\n  print(e.find('a:id',ns).text,'|',' '.join(e.find('a:title',ns).text.split()),'|',e.find('a:published',ns).text[:10])\n  c=e.find('{http://arxiv.org/schemas/atom}comment')\n  if c is not None: print('   C:',' '.join(c.text.split())[:170])\n\"\nsleep 1\n}\nq 'all:%22conditional+entropy%22+AND+all:%22transferability%22+AND+cat:cs.LG' 'CE transferability'\nq 'ti:%22LEEP%22+OR+ti:%22Log+Expected+Empirical+Prediction%22' 'LEEP'\nq 'ti:%22transferability+estimation%22' 'transferability estimation'\nq 'ti:%22which+tasks%22+AND+cat:cs' 'which tasks'\nq 'abs:%22auxiliary+tasks%22+AND+abs:%22redundan%22' 'aux redundancy'", "description": "Search for transferability estimation measures"}
```

<details><summary>tool result</summary>

```
=== QUERY: CE transferability ===
http://arxiv.org/abs/2106.10479v3 | Practical Transferability Estimation for Image Classification Tasks | 2021-06-19
   C: This paper is not the latest version. Please refer to Transferability-Guided Cross-Domain Cross-Task Transfer Learning (IEEE TNNLS'24) for more details.https://ieeexplore
http://arxiv.org/abs/2207.05510v2 | Transferability-Guided Cross-Domain Cross-Task Transfer Learning | 2022-07-12
   C: This work is accepted by IEEE TNNLS. Please see the official version https://ieeexplore.ieee.org/abstract/document/10420486.Copyright may be transferred without notice, a
http://arxiv.org/abs/2301.06928v1 | Towards Estimating Transferability using Hard Subsets | 2023-01-17
   C: First three authors contributed equally
http://arxiv.org/abs/1908.08142v1 | Transferability and Hardness of Supervised Classification Tasks | 2019-08-21
   C: This paper is published at the International Conference on Computer Vision (ICCV) 2019
http://arxiv.org/abs/2103.13843v1 | OTCE: A Transferability Metric for Cross-Domain Cross-Task Representations | 2021-03-25
   C: 13 pages, accepted by CVPR2021
http://arxiv.org/abs/2002.12462v2 | LEEP: A New Measure to Evaluate Transferability of Learned Representations | 2020-02-27
   C: Published at the International Conference on Machine Learning (ICML) 2020
http://arxiv.org/abs/2506.10433v1 | Measuring Semantic Information Production in Generative Diffusion Models | 2025-06-12
   C: 4 pages, 3 figures, an appendix with derivations and implementation details, accepted at ICLR DeLTa 2025
http://arxiv.org/abs/2006.00223v2 | Self-adaptive Re-weighted Adversarial Domain Adaptation | 2020-05-30
   C: to appear in IJCAI2020
=== QUERY: LEEP ===
http://arxiv.org/abs/2002.12462v2 | LEEP: A New Measure to Evaluate Transferability of Learned Representations | 2020-02-27
   C: Published at the International Conference on Machine Learning (ICML) 2020
http://arxiv.org/abs/2607.28077v1 | LEEPS: Latent-Guided Explore-Explo
... [truncated 1695 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nq() {\necho \"=== QUERY: $2 ===\"\ncurl -s -m 25 \"https://export.arxiv.org/api/query?search_query=$1&max_results=${3:-6}&sortBy=relevance\" | python3 -c \"\nimport sys,xml.etree.ElementTree as ET\nns={'a':'http://www.w3.org/2005/Atom'}\ntry: r=ET.fromstring(sys.stdin.read())\nexcept Exception as e: print('parse err',e); raise SystemExit\nfor e in r.findall('a:entry',ns):\n  print(e.find('a:id',ns).text,'|',' '.join(e.find('a:title',ns).text.split()),'|',e.find('a:published',ns).text[:10])\n  c=e.find('{http://arxiv.org/schemas/atom}comment')\n  if c is not None: print('   C:',' '.join(c.text.split())[:170])\n\"\nsleep 1\n}\nq 'ti:%22Usable+Information+Under+Computational+Constraints%22' 'V-information original'\nq 'ti:%22Auxiliary+Task+Reweighting%22' 'aux task reweighting'\nq 'ti:%22Exploring+and+Predicting+Transferability+across+NLP+Tasks%22' 'NLP transferability'\nq 'ti:%22Identifying+beneficial+task+relations%22' 'beneficial task relations'\nq 'ti:%22Intermediate-Task+Transfer%22+OR+ti:%22intermediate+task+fine-tuning%22' 'intermediate task'\nq 'ti:%22Efficiently+Identifying+Task+Groupings%22' 'TAG'", "description": "Search for more task-relatedness papers"}
```

<details><summary>tool result</summary>

```
=== QUERY: V-information original ===
http://arxiv.org/abs/2002.10689v1 | A Theory of Usable Information Under Computational Constraints | 2020-02-25
   C: ICLR 2020 (Talk)
=== QUERY: aux task reweighting ===
parse err no element found: line 1, column 0
=== QUERY: NLP transferability ===
http://arxiv.org/abs/2005.00770v2 | Exploring and Predicting Transferability across NLP Tasks | 2020-05-02
   C: Accepted as a conference paper at EMNLP 2020, 45 pages, 3 figures, 34 tables
=== QUERY: beneficial task relations ===
http://arxiv.org/abs/1702.08303v1 | Identifying beneficial task relations for multi-task learning in deep neural networks | 2017-02-27
   C: Accepted for publication at EACL 2017
=== QUERY: intermediate task ===
http://arxiv.org/abs/2108.11696v2 | Rethinking Why Intermediate-Task Fine-Tuning Works | 2021-08-26
   C: Findings of EMNLP 2021
http://arxiv.org/abs/2306.01382v2 | Leveraging Auxiliary Domain Parallel Data in Intermediate Task Fine-tuning for Low-resource Translation | 2023-06-02
   C: Accepted for poster presentation at the Practical Machine Learning for Developing Countries (PML4DC) workshop, ICLR 2023
http://arxiv.org/abs/2503.03172v1 | Intermediate-Task Transfer Learning: Leveraging Sarcasm Detection for Stance Detection | 2025-03-05
   C: 8 pages, 2 figures, published in The Sixteenth International Conference on Information (eKNOW 2024)
http://arxiv.org/abs/2407.16245v1 | Exploring the Effectiveness and Consistency of Task Selection in Intermediate-Task Transfer Learning | 2024-07-23
   C: Accepted to ACL SRW 2024
http://arxiv.org/abs/2005.00628v2 | Intermediate-Task Transfer Learning with Pretrained Models for Natural Language Understanding: When and Why Does It Work? | 2020-05-01
   C: ACL 2020
=== QUERY: TAG ===
http://arxiv.org/abs/2109.04617v2 | Efficiently Identifying Task Groupings for Multi-Task Learning | 2021-09-10
   C: In NeurIPS 2021 (spotlight). Code is available at https://github.com/google-research/google-research/tree/master/
... [truncated 3 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nq() {\necho \"=== QUERY: $2 ===\"\ncurl -s -m 25 \"https://export.arxiv.org/api/query?search_query=$1&max_results=${3:-6}&sortBy=relevance\" | python3 -c \"\nimport sys,xml.etree.ElementTree as ET\nns={'a':'http://www.w3.org/2005/Atom'}\ntry: r=ET.fromstring(sys.stdin.read())\nexcept Exception as e: print('parse err',e); raise SystemExit\nfor e in r.findall('a:entry',ns):\n  print(e.find('a:id',ns).text,'|',' '.join(e.find('a:title',ns).text.split()),'|',e.find('a:published',ns).text[:10])\n  c=e.find('{http://arxiv.org/schemas/atom}comment')\n  if c is not None: print('   C:',' '.join(c.text.split())[:170])\n\"\nsleep 1\n}\nq 'ti:%22Auxiliary+Task+Reweighting+for+Minimum-data+Learning%22' 'aux reweighting'\nq 'all:%22linear+probe%22+AND+all:%22auxiliary+task%22' 'linear probe aux'\nq 'abs:%22frozen+representation%22+AND+abs:%22probing%22+AND+abs:%22redundant%22' 'frozen rep redundant'\nq 'ti:%22Adaptive+Auxiliary+Task+Weighting%22' 'adaptive aux weighting'\nq 'ti:%22Auxiliary+Tasks+Benefit%22+OR+ti:%22auxiliary+tasks+in+reinforcement+learning%22' 'aux tasks RL'\nq 'abs:%22conditional+mutual+information%22+AND+abs:%22auxiliary+task%22' 'CMI aux task'", "description": "Search for probe-based and info-theoretic aux task papers"}
```

<details><summary>tool result</summary>

```
=== QUERY: aux reweighting ===
http://arxiv.org/abs/2010.08244v1 | Auxiliary Task Reweighting for Minimum-data Learning | 2020-10-16
   C: NeurIPS 2020. Project page: https://sites.google.com/view/auxiliary-task-reweighting/home
=== QUERY: linear probe aux ===
=== QUERY: frozen rep redundant ===
=== QUERY: adaptive aux weighting ===
=== QUERY: aux tasks RL ===
parse err no element found: line 1, column 0
=== QUERY: CMI aux task ===
http://arxiv.org/abs/2305.14133v2 | Conditional Mutual Information for Disentangled Representations in Reinforcement Learning | 2023-05-23
   C: Conference on Neural Information Processing Systems (NeurIPS), 2023
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nq() {\necho \"=== QUERY: $2 ===\"\ncurl -s -m 25 \"https://export.arxiv.org/api/query?search_query=$1&max_results=${3:-6}&sortBy=relevance\" | python3 -c \"\nimport sys,xml.etree.ElementTree as ET\nns={'a':'http://www.w3.org/2005/Atom'}\ntry: r=ET.fromstring(sys.stdin.read())\nexcept Exception as e: print('parse err',e); raise SystemExit\nfor e in r.findall('a:entry',ns):\n  print(e.find('a:id',ns).text,'|',' '.join(e.find('a:title',ns).text.split()),'|',e.find('a:published',ns).text[:10])\n  c=e.find('{http://arxiv.org/schemas/atom}comment')\n  if c is not None: print('   C:',' '.join(c.text.split())[:170])\n\"\nsleep 1\n}\nq 'ti:%22Task2Vec%22' 'Task2Vec'\nq 'ti:%22Representation+Similarity%22+AND+abs:%22transfer%22' 'rep similarity transfer'\nq 'ti:%22Adaptive+Auxiliary+Task%22' 'adaptive aux task'\nq 'ti:%22Auxiliary+Learning%22+AND+cat:cs.LG' 'auxiliary learning' 10\nq 'abs:%22probing+classifier%22+AND+abs:%22multi-task%22' 'probing MTL'", "description": "Search for Task2Vec and auxiliary learning surveys"}
```

<details><summary>tool result</summary>

```
=== QUERY: Task2Vec ===
http://arxiv.org/abs/1902.03545v1 | Task2Vec: Task Embedding for Meta-Learning | 2019-02-10
http://arxiv.org/abs/2604.10849v1 | Task2vec Readiness: Diagnostics for Federated Learning from Pre-Training Embeddings | 2026-04-12
=== QUERY: rep similarity transfer ===
parse err no element found: line 1, column 0
=== QUERY: adaptive aux task ===
=== QUERY: auxiliary learning ===
parse err no element found: line 1, column 0
=== QUERY: probing MTL ===
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nfor id in 2108.11346 1906.05374 2006.00995 2002.10689 2002.12462 1908.08142 2005.00770 1702.08303 2010.08244 2109.05748 1902.03545 2106.09362 2010.15413; do\n  echo \"=== $id ===\"\n  curl -s -m 25 \"https://arxiv.org/abs/$id\" | python3 -c \"\nimport sys,re,html\nt=sys.stdin.read()\nm=re.search(r'<title>(.*?)</title>',t,re.S); print('TITLE:',html.unescape(m.group(1)).strip() if m else 'NONE')\nm=re.search(r'citation_date\\\" content=\\\"(.*?)\\\"',t); print('DATE:',m.group(1) if m else '?')\nm=re.search(r'comments mathjax\\\">(.*?)</td>',t,re.S)\nprint('COMMENTS:',html.unescape(re.sub(r'<[^>]+>','',m.group(1))).strip() if m else 'NONE')\nm=re.search(r'jref\\\">(.*?)</td>',t,re.S)\nif m: print('JREF:',re.sub(r'<[^>]+>','',m.group(1)).strip())\n\"\n  sleep 1\ndone", "description": "Verify batch of candidate papers"}
```

<details><summary>tool result</summary>

```
=== 2108.11346 ===
TITLE: [2108.11346] Auxiliary Task Update Decomposition: The Good, The Bad and The Neutral
DATE: 2021/08/25
COMMENTS: 15 pages, 3 figures, Accepted to International Conference on Learning Representations (ICLR) 2021 See this https URL}{this https URL for associated code
=== 1906.05374 ===
TITLE: [1906.05374] Meta-Learning via Learned Loss
DATE: 2019/06/12
COMMENTS: Project website with code and video at this https URL
=== 2006.00995 ===
TITLE: [2006.00995] Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals
DATE: 2020/06/01
COMMENTS: TACL journal. Initial title was: "When Bert Forgets How To POS: Amnesic Probing of Linguistic Properties and MLM Predictions"
=== 2002.10689 ===
TITLE: [2002.10689] A Theory of Usable Information Under Computational Constraints
DATE: 2020/02/25
COMMENTS: ICLR 2020 (Talk)
=== 2002.12462 ===
TITLE: [2002.12462] LEEP: A New Measure to Evaluate Transferability of Learned Representations
DATE: 2020/02/27
COMMENTS: Published at the International Conference on Machine Learning (ICML) 2020
=== 1908.08142 ===
TITLE: [1908.08142] Transferability and Hardness of Supervised Classification Tasks
DATE: 2019/08/21
COMMENTS: This paper is published at the International Conference on Computer Vision (ICCV) 2019
=== 2005.00770 ===
TITLE: [2005.00770] Exploring and Predicting Transferability across NLP Tasks
DATE: 2020/05/02
COMMENTS: Accepted as a conference paper at EMNLP 2020, 45 pages, 3 figures, 34 tables
=== 1702.08303 ===
TITLE: [1702.08303] Identifying beneficial task relations for multi-task learning in deep neural networks
DATE: 2017/02/27
COMMENTS: Accepted for publication at EACL 2017
=== 2010.08244 ===
TITLE: [2010.08244] Auxiliary Task Reweighting for Minimum-data Learning
DATE: 2020/10/16
COMMENTS: NeurIPS 2020. Project page: this https URL
=== 2109.05748 ===
TITLE: [2109.05748] GradTS: A Gradient-Based Automatic Auxiliary Task Selection Method Based on Transformer Networks
DATE: 2021/09/13
COMMENTS: In E
... [truncated 389 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nq() {\necho \"=== QUERY: $2 ===\"\ncurl -s -m 25 \"https://export.arxiv.org/api/query?search_query=$1&max_results=${3:-6}&sortBy=relevance\" | python3 -c \"\nimport sys,xml.etree.ElementTree as ET\nns={'a':'http://www.w3.org/2005/Atom'}\ntry: r=ET.fromstring(sys.stdin.read())\nexcept Exception as e: print('parse err',e); raise SystemExit\nfor e in r.findall('a:entry',ns):\n  print(e.find('a:id',ns).text,'|',' '.join(e.find('a:title',ns).text.split()),'|',e.find('a:published',ns).text[:10])\n  c=e.find('{http://arxiv.org/schemas/atom}comment')\n  if c is not None: print('   C:',' '.join(c.text.split())[:170])\n\"\nsleep 1\n}\nq 'ti:%22Representation+Similarity+Analysis+for+Efficient+Task+Taxonomy%22' 'RSA taxonomy'\nq 'ti:%22Auxiliary+Task+Discovery%22' 'aux task discovery'\nq 'ti:%22task+redundancy%22' 'task redundancy'\nq 'ti:%22LogME%22+OR+ti:%22Ranking+and+Tuning+Pre-trained+Models%22' 'LogME'\nq 'all:%22information-theoretic+task+selection%22' 'info-theoretic task selection'\nq 'ti:%22Rethinking+Hard-Parameter+Sharing%22' 'rethinking hps'", "description": "Search for RSA taxonomy and related"}
```

<details><summary>tool result</summary>

```
=== QUERY: RSA taxonomy ===
http://arxiv.org/abs/1904.11740v1 | Representation Similarity Analysis for Efficient Task taxonomy & Transfer Learning | 2019-04-26
   C: Accepted at CVPR 2019. Code available at https://github.com/kshitijd20/RSA-CVPR19-release
=== QUERY: aux task discovery ===
http://arxiv.org/abs/2210.14361v2 | Auxiliary task discovery through generate-and-test | 2022-10-25
http://arxiv.org/abs/2608.15841v1 | Self-Supervised Auxiliary Task Discovery for Stable Reinforcement Learning in Stock Trading | 2026-08-16
=== QUERY: task redundancy ===
=== QUERY: LogME ===
http://arxiv.org/abs/2110.10545v4 | Ranking and Tuning Pre-trained Models: A New Paradigm for Exploiting Model Hubs | 2021-10-20
   C: 47 pages, camera-ready version for JMLR 2022
http://arxiv.org/abs/2102.11005v3 | LogME: Practical Assessment of Pre-trained Models for Transfer Learning | 2021-02-22
   C: 13 pages (ICML 2021 camera ready version)
=== QUERY: info-theoretic task selection ===
http://arxiv.org/abs/2011.01054v2 | Information-theoretic Task Selection for Meta-Reinforcement Learning | 2020-11-02
   C: Published at NeurIPS 2020
http://arxiv.org/abs/2107.02603v1 | Meta-Reinforcement Learning for Heuristic Planning | 2021-07-06
   C: ICAPS 2021
http://arxiv.org/abs/2605.05703v2 | Active Learning for Communication Structure Optimization in LLM-Based Multi-Agent Systems | 2026-05-07
=== QUERY: rethinking hps ===
parse err no element found: line 1, column 0
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nfor id in 1904.11740 2102.11005 2210.14361 2011.01054 2108.11696 2204.00565 2409.06091; do\n  echo \"=== $id ===\"\n  curl -s -m 25 \"https://arxiv.org/abs/$id\" | python3 -c \"\nimport sys,re,html\nt=sys.stdin.read()\nm=re.search(r'<title>(.*?)</title>',t,re.S); print('TITLE:',html.unescape(m.group(1)).strip() if m else 'NONE')\nm=re.search(r'citation_date\\\" content=\\\"(.*?)\\\"',t); print('DATE:',m.group(1) if m else '?')\nm=re.search(r'comments mathjax\\\">(.*?)</td>',t,re.S)\nprint('COMMENTS:',html.unescape(re.sub(r'<[^>]+>','',m.group(1))).strip()[:200] if m else 'NONE')\n\"\n  sleep 1\ndone", "description": "Verify remaining candidate papers"}
```

<details><summary>tool result</summary>

```
=== 1904.11740 ===
TITLE: [1904.11740] Representation Similarity Analysis for Efficient Task taxonomy & Transfer Learning
DATE: 2019/04/26
COMMENTS: Accepted at CVPR 2019. Code available at this https URL
=== 2102.11005 ===
TITLE: [2102.11005] LogME: Practical Assessment of Pre-trained Models for Transfer Learning
DATE: 2021/02/22
COMMENTS: 13 pages (ICML 2021 camera ready version)
=== 2210.14361 ===
TITLE: [2210.14361] Auxiliary task discovery through generate-and-test
DATE: 2022/10/25
COMMENTS: NONE
=== 2011.01054 ===
TITLE: [2011.01054] Information-theoretic Task Selection for Meta-Reinforcement Learning
DATE: 2020/11/02
COMMENTS: Published at NeurIPS 2020
=== 2108.11696 ===
TITLE: [2108.11696] Rethinking Why Intermediate-Task Fine-Tuning Works
DATE: 2021/08/26
COMMENTS: Findings of EMNLP 2021
=== 2204.00565 ===
TITLE: [2204.00565] What makes useful auxiliary tasks in reinforcement learning: investigating the effect of the target policy
DATE: 2022/04/01
COMMENTS: NONE
=== 2409.06091 ===
TITLE: [2409.06091] Scalable Multitask Learning Using Gradient-based Estimation of Task Affinity
DATE: 2024/09/09
COMMENTS: 16 pages. Appeared in KDD 2024
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nq() {\necho \"=== QUERY: $2 ===\"\ncurl -s -m 25 \"https://export.arxiv.org/api/query?search_query=$1&max_results=${3:-6}&sortBy=relevance\" | python3 -c \"\nimport sys,xml.etree.ElementTree as ET\nns={'a':'http://www.w3.org/2005/Atom'}\ntry: r=ET.fromstring(sys.stdin.read())\nexcept Exception as e: print('parse err',e); raise SystemExit\nfor e in r.findall('a:entry',ns):\n  print(e.find('a:id',ns).text,'|',' '.join(e.find('a:title',ns).text.split()),'|',e.find('a:published',ns).text[:10])\n  c=e.find('{http://arxiv.org/schemas/atom}comment')\n  if c is not None: print('   C:',' '.join(c.text.split())[:170])\n\"\nsleep 1\n}\nq 'abs:%22readout%22+AND+abs:%22frozen+encoder%22+AND+abs:%22task%22' 'readout frozen encoder'\nq 'ti:%22Which+Model+to+Transfer%22' 'which model to transfer'\nq 'ti:%22Auxiliary+Task+Weighting%22' 'aux task weighting'\nq 'abs:%22linear+probing%22+AND+abs:%22task+affinity%22' 'linear probing affinity'\nq 'ti:%22Rissanen%22+AND+ti:%22Data+Analysis%22' 'RDA'\nq 'ti:%22Probing+Classifiers%22+AND+ti:%22Promises%22' 'probing classifiers survey'", "description": "More targeted searches on probe-based measures"}
```

<details><summary>tool result</summary>

```
=== QUERY: readout frozen encoder ===
http://arxiv.org/abs/2606.31495v1 | Surprise as a Signal for Plasticity and Metacognition | 2026-06-30
=== QUERY: which model to transfer ===
http://arxiv.org/abs/2402.15231v1 | Which Model to Transfer? A Survey on Transferability Estimation | 2024-02-23
http://arxiv.org/abs/2010.06402v2 | Which Model to Transfer? Finding the Needle in the Growing Haystack | 2020-10-13
=== QUERY: aux task weighting ===
http://arxiv.org/abs/2107.07805v1 | Multiple Instance Learning with Auxiliary Task Weighting for Multiple Myeloma Classification | 2021-07-16
   C: Accepted at MICCAI 2021
=== QUERY: linear probing affinity ===
parse err no element found: line 1, column 0
=== QUERY: RDA ===
http://arxiv.org/abs/2103.03872v1 | Rissanen Data Analysis: Examining Dataset Characteristics via Description Length | 2021-03-05
   C: Code at https://github.com/ethanjperez/rda along with a script to run RDA on your own dataset
=== QUERY: probing classifiers survey ===
http://arxiv.org/abs/2102.12452v4 | Probing Classifiers: Promises, Shortcomings, and Advances | 2021-02-24
   C: Accepted to Computational Linguistics as a squib
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nfor id in 2103.03872 2102.12452 2402.15231 2010.06402; do\n  echo \"=== $id ===\"\n  curl -s -m 25 \"https://arxiv.org/abs/$id\" | python3 -c \"\nimport sys,re,html\nt=sys.stdin.read()\nm=re.search(r'<title>(.*?)</title>',t,re.S); print('TITLE:',html.unescape(m.group(1)).strip() if m else 'NONE')\nm=re.search(r'citation_date\\\" content=\\\"(.*?)\\\"',t); print('DATE:',m.group(1) if m else '?')\nm=re.search(r'comments mathjax\\\">(.*?)</td>',t,re.S)\nprint('COMMENTS:',html.unescape(re.sub(r'<[^>]+>','',m.group(1))).strip()[:200] if m else 'NONE')\n\"\n  sleep 1\ndone", "description": "Verify final batch"}
```

<details><summary>tool result</summary>

```
=== 2103.03872 ===
TITLE: [2103.03872] Rissanen Data Analysis: Examining Dataset Characteristics via Description Length
DATE: 2021/03/05
COMMENTS: Code at this https URL along with a script to run RDA on your own dataset
=== 2102.12452 ===
TITLE: [2102.12452] Probing Classifiers: Promises, Shortcomings, and Advances
DATE: 2021/02/24
COMMENTS: Accepted to Computational Linguistics as a squib
=== 2402.15231 ===
TITLE: [2402.15231] Which Model to Transfer? A Survey on Transferability Estimation
DATE: 2024/02/23
COMMENTS: NONE
=== 2010.06402 ===
TITLE: [2010.06402] Which Model to Transfer? Finding the Needle in the Growing Haystack
DATE: 2020/10/13
COMMENTS: NONE
```
</details>

**tool call `WebSearch`**
```
{"query": "Rissanen Data Analysis Perez Kiela Cho ICML 2021 published venue"}
```

**tool call `WebSearch`**
```
{"query": "\"Meta-Learning via Learned Loss\" Bechtle Molchanov ML3 published ICPR 2021 venue"}
```

<details><summary>tool result</summary>

```
Web search results for query: "Rissanen Data Analysis Perez Kiela Cho ICML 2021 published venue"

Links: [{"title":"Rissanen Data Analysis: Examining Dataset Characteristics","url":"https://icml.cc/media/icml-2021/Slides/10437_b1d2n3V.pdf"},{"title":"Rissanen Data Analysis: Examining Dataset Characteristics via Description Length | ML Anthology","url":"https://mlanthology.org/icml/2021/perez2021icml-rissanen/"},{"title":"GitHub - ethanjperez/rda: Code for \"Rissanen Data Analysis: Examining Dataset Characteristics via Description Length\" by Ethan Perez, Douwe Kiela, and Kyungyhun Cho · GitHub","url":"https://github.com/ethanjperez/rda"},{"title":"Rissanen Data Analysis: Examining Dataset Characteristics via Description Length [Quick Review]","url":"https://liner.com/review/rissanen-data-analysis-examining-dataset-characteristics-via-description-length"},{"title":"Rissanen Data Analysis: Examining Dataset Characteristics via Description Length","url":"https://proceedings.mlr.press/v139/perez21a.html"},{"title":"[PDF] Rissanen Data Analysis: Examining Dataset Characteristics via Description Length | Semantic Scholar","url":"https://www.semanticscholar.org/paper/Rissanen-Data-Analysis:-Examining-Dataset-via-Perez-Kiela/8a7df164c4687e3c402b1cbf0ab404de49cfb75e"},{"title":"Track: Applications (CV and NLP)","url":"https://icml.cc/virtual/2021/session/12097"},{"title":"Kiela, Douwe | ML Anthology","url":"https://mlanthology.org/authors/k/kiela-douwe/"},{"title":"rda/README.md at main · ethanjperez/rda","url":"https://github.com/ethanjperez/rda/blob/main/README.md"},{"title":"Perez, Ethan | ML Anthology","url":"https://mlanthology.org/authors/p/perez-ethan/"}]

**Publication venue confirmed:** ICML 2021 (38th International Conference on Machine Learning), published in PMLR.

**Full citation details:**
- **Title:** Rissanen Data Analysis: Examining Dataset Characteristics via Description Length
- **Authors:** Ethan Perez, Douwe Kiela, Kyunghyun Cho
- **Venue:** Proceedings 
... [truncated 1528 chars]
```
</details>

<details><summary>tool result</summary>

```
Web search results for query: ""Meta-Learning via Learned Loss" Bechtle Molchanov ML3 published ICPR 2021 venue"

Links: [{"title":"Meta Learning via Learned Loss Sarah Bechtle*1 sbechtle@tuebingen.mpg.de","url":"https://arxiv.org/pdf/1906.05374"},{"title":"Meta Learning via Learned Loss","url":"https://www.computer.org/csdl/proceedings-article/icpr/2021/09412010/1tmiQxztZwA"},{"title":"Home","url":"https://sites.google.com/view/mlthree"},{"title":"GitHub - facebookresearch/LearningToLearn: Collection of algorithms to learn loss and reward functions via gradient-based bi-level optimization.","url":"https://github.com/facebookresearch/LearningToLearn"},{"title":"[1906.05374] Meta-Learning via Learned Loss","url":"https://arxiv.org/abs/1906.05374"},{"title":"Meta-Learning via Learned Loss","url":"https://amolchanov86.github.io/pdf/bechtle19icml.pdf"},{"title":"Meta Learning via Learned Loss | Request PDF","url":"https://www.researchgate.net/publication/352832624_Meta_Learning_via_Learned_Loss"},{"title":"Meta Learning Via Learned Loss: Sarah Bechtle Artem Molchanov Yevgen Chebotar | PDF | Mathematical Optimization | Loss Function","url":"https://www.scribd.com/document/547483427/ML3"},{"title":"What Is Meta-Learning via Learned Losses (with Python Code) | AIM","url":"https://analyticsindiamag.com/ai-mysteries/what-is-meta-learning-via-learned-losses-with-python-code/"},{"title":"[1906.05374] Meta Learning via Learned Loss *Equal contributions. 1The authors are with the Max Planck Institute for Intelligent Systems, Tübingen, Germany 2The authors are with the Viterbi School of Engineering, University of Southern California, Los Angeles, CA 90089 3The authors are with Facebook AI Research 4The authors are with the Tandon School of Engineering, New York University, Brooklyn, NY 11201","url":"https://ar5iv.labs.arxiv.org/html/1906.05374"}]

Yes — the paper was published at ICPR 2021. Here are the venue details:

**Publication venue:** *Meta Learning via Learned Loss*, by S
... [truncated 1975 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nq() {\necho \"=== QUERY: $2 ===\"\ncurl -s -m 25 \"https://export.arxiv.org/api/query?search_query=$1&max_results=${3:-6}&sortBy=relevance\" | python3 -c \"\nimport sys,xml.etree.ElementTree as ET\nns={'a':'http://www.w3.org/2005/Atom'}\ntry: r=ET.fromstring(sys.stdin.read())\nexcept Exception as e: print('parse err',e); raise SystemExit\nfor e in r.findall('a:entry',ns):\n  print(e.find('a:id',ns).text,'|',' '.join(e.find('a:title',ns).text.split()),'|',e.find('a:published',ns).text[:10])\n  c=e.find('{http://arxiv.org/schemas/atom}comment')\n  if c is not None: print('   C:',' '.join(c.text.split())[:170])\n\"\nsleep 1\n}\nq 'abs:%22auxiliary%22+AND+abs:%22probing%22+AND+abs:%22representation%22+AND+cat:cs.LG' 'aux probing rep'\nq 'ti:%22Measuring+and+Harnessing+Transference%22' 'transference'\nq 'abs:%22information+gain%22+AND+abs:%22auxiliary+label%22' 'info gain aux label'\nq 'ti:%22Task+Discovery%22+OR+ti:%22Finding+the+Tasks%22' 'task discovery'\nq 'abs:%22already+encoded%22+AND+abs:%22auxiliary+task%22' 'already encoded'\nq 'ti:%22Auxiliary+Losses%22' 'auxiliary losses'", "description": "Final round of arXiv searches"}
```

<details><summary>tool result</summary>

```
=== QUERY: aux probing rep ===
http://arxiv.org/abs/2509.21565v1 | No Alignment Needed for Generation: Learning Linearly Separable Representations in Diffusion Models | 2025-09-25
http://arxiv.org/abs/2105.06849v1 | Sparsity-Probe: Analysis tool for Deep Learning Models | 2021-05-14
http://arxiv.org/abs/2607.24553v1 | EchoBridge: Long-Tail-Aware ECG-Echocardiography Text Alignment for Echocardiography-Derived Cardiac Findings | 2026-07-27
http://arxiv.org/abs/2310.15905v1 | Is Probing All You Need? Indicator Tasks as an Alternative to Probing Embedding Spaces | 2023-10-24
   C: Findings of EMNLP 2023
http://arxiv.org/abs/2602.17050v3 | Multi-Probe Zero Collision Hash (MPZCH): Mitigating Embedding Collisions and Enhancing Model Freshness in Large-Scale Recommenders | 2026-02-19
   C: 9 pages, 6 figures
http://arxiv.org/abs/2607.08839v1 | Mixture of Probes: Learning from Privileged Modalities in Multimodal LLMs Through Probing | 2026-07-09
   C: Preprint (16 pages)
=== QUERY: transference ===
http://arxiv.org/abs/2010.15413v3 | Measuring and Harnessing Transference in Multi-Task Learning | 2020-10-29
=== QUERY: info gain aux label ===
=== QUERY: task discovery ===
http://arxiv.org/abs/2212.00261v1 | Task Discovery: Finding the Tasks that Neural Networks Generalize on | 2022-12-01
   C: NeurIPS 2022, Project page at https://taskdiscovery.epfl.ch
http://arxiv.org/abs/2404.05729v2 | Finding Visual Task Vectors | 2024-04-08
   C: https://github.com/alhojel/visual_task_vectors
http://arxiv.org/abs/2605.07572v1 | Open-Ended Task Discovery via Bayesian Optimization | 2026-05-08
   C: 60 pages, 11 figures
http://arxiv.org/abs/2210.14361v2 | Auxiliary task discovery through generate-and-test | 2022-10-25
http://arxiv.org/abs/2603.19415v2 | Scalable Prompt Routing via Fine-Grained Latent Task Discovery | 2026-03-19
http://arxiv.org/abs/1810.02358v2 | Transfer Learning via Unsupervised Task Discovery for Visual Question Answering | 2018-10-03
   C: CVPR 2019
=== QUERY: already e
... [truncated 758 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nfor id in 2202.03250 2212.00261; do\n  echo \"=== $id ===\"\n  curl -s -m 25 \"https://arxiv.org/abs/$id\" | python3 -c \"\nimport sys,re,html\nt=sys.stdin.read()\nm=re.search(r'<title>(.*?)</title>',t,re.S); print('TITLE:',html.unescape(m.group(1)).strip() if m else 'NONE')\nm=re.search(r'citation_date\\\" content=\\\"(.*?)\\\"',t); print('DATE:',m.group(1) if m else '?')\nm=re.search(r'comments mathjax\\\">(.*?)</td>',t,re.S)\nprint('COMMENTS:',html.unescape(re.sub(r'<[^>]+>','',m.group(1))).strip()[:200] if m else 'NONE')\n\"\n  sleep 1\ndone", "description": "Verify last two"}
```

**tool call `WebFetch`**
```
{"url": "https://arxiv.org/abs/2010.15413", "prompt": "What is the abstract, and what is the venue/comments field? What measure of inter-task transference does this paper define?"}
```

<details><summary>tool result</summary>

```
=== 2202.03250 ===
TITLE: [2202.03250] Adaptive Mixing of Auxiliary Losses in Supervised Learning
DATE: 2022/02/07
COMMENTS: NONE
=== 2212.00261 ===
TITLE: [2212.00261] Task Discovery: Finding the Tasks that Neural Networks Generalize on
DATE: 2022/12/01
COMMENTS: NeurIPS 2022, Project page at this https URL
```
</details>

<details><summary>tool result</summary>

```
Unable to verify if domain arxiv.org is safe to fetch. This may be due to network restrictions or enterprise security policies blocking claude.ai.
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nfor id in 2010.15413 2108.11346 2010.08244 1904.11740 2109.05748 2005.00770 1702.08303 2202.03250 2402.15231; do\n  echo \"===== $id =====\"\n  curl -s -m 25 \"https://arxiv.org/abs/$id\" | python3 -c \"\nimport sys,re,html\nt=sys.stdin.read()\nm=re.search(r'citation_abstract\\\" content=\\\"(.*?)\\\" />',t,re.S)\nprint(html.unescape(m.group(1))[:1600] if m else 'NO ABS')\n\"\n  sleep 1\ndone", "description": "Fetch abstracts for measure descriptions"}
```

<details><summary>tool result</summary>

```
===== 2010.15413 =====
Multi-task learning can leverage information learned by one task to benefit the training of other tasks. Despite this capacity, naive formulations often degrade performance and in particular, identifying the tasks that would benefit from co-training remains a challenging design question. In this paper, we analyze the dynamics of information transfer, or transference, across tasks throughout training. Specifically, we develop a similarity measure that can quantify transference among tasks and use this quantity to both better understand the optimization dynamics of multi-task learning as well as improve overall learning performance. In the latter case, we propose two methods to leverage our transference metric. The first operates at a macro-level by selecting which tasks should train together while the second functions at a micro-level by determining how to combine task gradients at each training step. We find these methods can lead to significant improvement over prior work on three supervised multi-task learning benchmarks and one multi-task reinforcement learning paradigm.
===== 2108.11346 =====
While deep learning has been very beneficial in data-rich settings, tasks with smaller training set often resort to pre-training or multitask learning to leverage data from other tasks. In this case, careful consideration is needed to select tasks and model parameterizations such that updates from the auxiliary tasks actually help the primary task. We seek to alleviate this burden by formulating a model-agnostic framework that performs fine-grained manipulation of the auxiliary task gradients. We propose to decompose auxiliary updates into directions which help, damage or leave the primary task loss unchanged. This allows weighting the update directions differently depending on their impact on the problem of interest. We present a novel and efficient algorithm for that purpose and show its advantage in practice. Our method leverages efficient automati
... [truncated 8458 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nq() {\necho \"=== QUERY: $2 ===\"\ncurl -s -m 25 \"https://export.arxiv.org/api/query?search_query=$1&max_results=${3:-5}&sortBy=relevance\" | python3 -c \"\nimport sys,xml.etree.ElementTree as ET\nns={'a':'http://www.w3.org/2005/Atom'}\ntry: r=ET.fromstring(sys.stdin.read())\nexcept Exception as e: print('parse err',e); raise SystemExit\nfor e in r.findall('a:entry',ns):\n  print(e.find('a:id',ns).text,'|',' '.join(e.find('a:title',ns).text.split()),'|',e.find('a:published',ns).text[:10])\n  c=e.find('{http://arxiv.org/schemas/atom}comment')\n  if c is not None: print('   C:',' '.join(c.text.split())[:170])\n\"\nsleep 1\n}\nq 'ti:%22AUTOSEM%22' 'AUTOSEM'\nq 'ti:%22H-score%22' 'H-score'\necho \"==== Taskonomy method (grep pdf-free: use ar5iv? try arxiv abs page full text is not avail). Try listing v1 comments only ====\"\ncurl -s -m 25 \"https://arxiv.org/abs/2109.04617\" | python3 -c \"\nimport sys,re,html\nt=sys.stdin.read()\nm=re.search(r'citation_abstract\\\" content=\\\"(.*?)\\\" />',t,re.S)\nprint(html.unescape(m.group(1))[:1500] if m else 'NO ABS')\n\"", "description": "Search AUTOSEM, H-score, get TAG abstract"}
```

<details><summary>tool result</summary>

```
=== QUERY: AUTOSEM ===
http://arxiv.org/abs/1904.04153v1 | AutoSeM: Automatic Task Selection and Mixing in Multi-Task Learning | 2019-04-08
   C: NAACL 2019 (12 pages)
=== QUERY: H-score ===
http://arxiv.org/abs/1907.11142v1 | On the bias of H-scores for comparing biclusters, and how to correct it | 2019-07-24
   C: 12 pages, 3 figures
http://arxiv.org/abs/2308.11562v1 | EndoNet: model for automatic calculation of H-score on histological slides | 2023-08-22
==== Taskonomy method (grep pdf-free: use ar5iv? try arxiv abs page full text is not avail). Try listing v1 comments only ====
Multi-task learning can leverage information learned by one task to benefit the training of other tasks. Despite this capacity, naively training all tasks together in one model often degrades performance, and exhaustively searching through combinations of task groupings can be prohibitively expensive. As a result, efficiently identifying the tasks that would benefit from training together remains a challenging design question without a clear solution. In this paper, we suggest an approach to select which tasks should train together in multi-task learning models. Our method determines task groupings in a single run by training all tasks together and quantifying the effect to which one task's gradient would affect another task's loss. On the large-scale Taskonomy computer vision dataset, we find this method can decrease test loss by 10.0% compared to simply training all tasks together while operating 11.6 times faster than a state-of-the-art task grouping method.
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\necho \"=== AutoSeM verify ===\"\ncurl -s -m 25 \"https://arxiv.org/abs/1904.04153\" | python3 -c \"\nimport sys,re,html\nt=sys.stdin.read()\nm=re.search(r'<title>(.*?)</title>',t,re.S); print('TITLE:',html.unescape(m.group(1)).strip())\nm=re.search(r'citation_date\\\" content=\\\"(.*?)\\\"',t); print('DATE:',m.group(1))\nm=re.search(r'comments mathjax\\\">(.*?)</td>',t,re.S); print('COMMENTS:',re.sub(r'<[^>]+>','',m.group(1)).strip() if m else 'NONE')\nm=re.search(r'citation_abstract\\\" content=\\\"(.*?)\\\" />',t,re.S); print('ABS:',html.unescape(m.group(1))[:900])\n\"\nsleep 2\necho \"=== Taskonomy transfer function text ===\"\ncurl -s -m 40 \"https://ar5iv.labs.arxiv.org/html/1804.08328\" -o /tmp/tk.html -w \"HTTP:%{http_code}\\n\"\npython3 - <<'EOF'\nimport re,html\nt=open('/tmp/tk.html',errors='ignore').read()\nt=re.sub(r'<[^>]+>',' ',t); t=html.unescape(t); t=re.sub(r'\\s+',' ',t)\nfor kw in ['frozen','transfer function','read-out','readout','affinity','normalize']:\n    for m in re.finditer(kw,t,re.I):\n        print(f'[{kw}]',t[max(0,m.start()-250):m.start()+300].strip()[:550])\n        print('---')\n        break\nEOF", "description": "Verify AutoSeM and extract Taskonomy method details"}
```

<details><summary>tool result</summary>

```
=== AutoSeM verify ===
TITLE: [1904.04153] AutoSeM: Automatic Task Selection and Mixing in Multi-Task Learning
DATE: 2019/04/08
COMMENTS: NAACL 2019 (12 pages)
ABS: Multi-task learning (MTL) has achieved success over a wide range of problems, where the goal is to improve the performance of a primary task using a set of relevant auxiliary tasks. However, when the usefulness of the auxiliary tasks w.r.t. the primary task is not known a priori, the success of MTL models depends on the correct choice of these auxiliary tasks and also a balanced mixing ratio of these tasks during alternate training. These two problems could be resolved via manual intuition or hyper-parameter tuning over all combinatorial task choices, but this introduces inductive bias or is not scalable when the number of candidate auxiliary tasks is very large. To address these issues, we present AutoSeM, a two-stage MTL pipeline, where the first stage automatically selects the most useful auxiliary tasks via a Beta-Bernoulli multi-armed bandit with Thompson Sampling, and the second st
=== Taskonomy transfer function text ===
HTTP:200
[frozen] Sec. 4.2 with supportive results. For evaluation of the robustness of results w.r.t the choice of dictionary, see the supplementary material . Figure 4 : Transfer Function . A small readout function is trained to map representations of source task’s frozen encoder to target task’s labels. If order > 1 >1 , transfer function receives representations from multiple sources. Dataset: We need a dataset that has annotations for every task on every image . Training all of our tasks on exactly the same pixels eliminates the possibility that the observe
---
[transfer function] d our outcome can be seen as a computational meta-structure of the space of tasks. Figure 2 : Computational modeling of task relations and creating the taxonomy. From left to right: I. Train task-specific networks. II. Train (first order and higher) transfer functions among tasks in a latent space. 
... [truncated 1951 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nq() {\necho \"=== QUERY: $2 ===\"\ncurl -s -m 25 \"https://export.arxiv.org/api/query?search_query=$1&max_results=${3:-6}&sortBy=relevance\" | python3 -c \"\nimport sys,xml.etree.ElementTree as ET\nns={'a':'http://www.w3.org/2005/Atom'}\ntry: r=ET.fromstring(sys.stdin.read())\nexcept Exception as e: print('parse err',e); raise SystemExit\nfor e in r.findall('a:entry',ns):\n  print(e.find('a:id',ns).text,'|',' '.join(e.find('a:title',ns).text.split()),'|',e.find('a:published',ns).text[:10])\n  c=e.find('{http://arxiv.org/schemas/atom}comment')\n  if c is not None: print('   C:',' '.join(c.text.split())[:170])\n\"\nsleep 1\n}\nq 'abs:%22auxiliary%22+AND+abs:%22already+predictable%22' 'already predictable'\nq 'ti:%22Do+we+need%22+AND+abs:%22auxiliary%22' 'do we need aux'\nq 'abs:%22probing%22+AND+abs:%22visuomotor%22+AND+abs:%22representation%22' 'probing visuomotor'\nq 'abs:%22linear+probe%22+AND+abs:%22policy%22+AND+abs:%22spatial%22+AND+cat:cs.RO' 'linear probe policy spatial'\nq 'ti:%22auxiliary%22+AND+ti:%22redundant%22' 'aux redundant title'", "description": "Search for embodied/probe redundancy angle"}
```

<details><summary>tool result</summary>

```
=== QUERY: already predictable ===
=== QUERY: do we need aux ===
=== QUERY: probing visuomotor ===
http://arxiv.org/abs/2608.13474v1 | Decoding Task Progress from VLA Representations | 2026-08-13
=== QUERY: linear probe policy spatial ===
=== QUERY: aux redundant title ===
http://arxiv.org/abs/2510.13677v2 | APRIL: Auxiliary Physically-Redundant Information in Loss -- A physics-informed framework for parameter estimation with a gravitational-wave case study | 2025-10-15
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nfor id in 2002.10689 2006.00995 2002.12462 1908.08142 2608.13474; do\n  echo \"===== $id =====\"\n  curl -s -m 25 \"https://arxiv.org/abs/$id\" | python3 -c \"\nimport sys,re,html\nt=sys.stdin.read()\nm=re.search(r'<title>(.*?)</title>',t,re.S); print('TITLE:',html.unescape(m.group(1)).strip())\nm=re.search(r'comments mathjax\\\">(.*?)</td>',t,re.S); print('COMMENTS:',html.unescape(re.sub(r'<[^>]+>','',m.group(1))).strip()[:180] if m else 'NONE')\nm=re.search(r'citation_date\\\" content=\\\"(.*?)\\\"',t); print('DATE:',m.group(1))\nm=re.search(r'citation_abstract\\\" content=\\\"(.*?)\\\" />',t,re.S); print('ABS:',html.unescape(m.group(1))[:1000] if m else 'NO')\n\"\n  sleep 1\ndone", "description": "Get abstracts for V-info and amnesic probing"}
```

<details><summary>tool result</summary>

```
===== 2002.10689 =====
TITLE: [2002.10689] A Theory of Usable Information Under Computational Constraints
COMMENTS: ICLR 2020 (Talk)
DATE: 2020/02/25
ABS: We propose a new framework for reasoning about information in complex systems. Our foundation is based on a variational extension of Shannon's information theory that takes into account the modeling power and computational constraints of the observer. The resulting \emph{predictive $\mathcal{V}$-information} encompasses mutual information and other notions of informativeness such as the coefficient of determination. Unlike Shannon's mutual information and in violation of the data processing inequality, $\mathcal{V}$-information can be created through computation. This is consistent with deep neural networks extracting hierarchies of progressively more informative features in representation learning. Additionally, we show that by incorporating computational constraints, $\mathcal{V}$-information can be reliably estimated from data even in high dimensions with PAC-style guarantees. Empirically, we demonstrate predictive $\mathcal{V}$-information is more effective than mutual information 
===== 2006.00995 =====
TITLE: [2006.00995] Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals
COMMENTS: TACL journal. Initial title was: "When Bert Forgets How To POS: Amnesic Probing of Linguistic Properties and MLM Predictions"
DATE: 2020/06/01
ABS: A growing body of work makes use of probing to investigate the working of neural models, often considered black boxes. Recently, an ongoing debate emerged surrounding the limitations of the probing paradigm. In this work, we point out the inability to infer behavioral conclusions from probing results and offer an alternative method that focuses on how the information is being used, rather than on what information is encoded. Our method, Amnesic Probing, follows the intuition that the utility of a property for a given task can be assessed by measuring the influence of 
... [truncated 3901 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nq() {\necho \"=== QUERY: $2 ===\"\ncurl -s -m 25 \"https://export.arxiv.org/api/query?search_query=$1&max_results=${3:-6}&sortBy=relevance\" | python3 -c \"\nimport sys,xml.etree.ElementTree as ET\nns={'a':'http://www.w3.org/2005/Atom'}\ntry: r=ET.fromstring(sys.stdin.read())\nexcept Exception as e: print('parse err',e); raise SystemExit\nfor e in r.findall('a:entry',ns):\n  print(e.find('a:id',ns).text,'|',' '.join(e.find('a:title',ns).text.split()),'|',e.find('a:published',ns).text[:10])\n  c=e.find('{http://arxiv.org/schemas/atom}comment')\n  if c is not None: print('   C:',' '.join(c.text.split())[:170])\n\"\nsleep 1\n}\nq 'ti:%22Auxiliary+Learning%22+AND+ti:%22Survey%22' 'aux learning survey'\nq 'ti:%22Multi-Task+Learning%22+AND+ti:%22Survey%22+AND+cat:cs.LG' 'MTL survey'\nq 'abs:%22task+relatedness%22+AND+cat:cs.LG' 'task relatedness'", "description": "Search for surveys"}
```

**tool call `WebSearch`**
```
{"query": "\"Adapting Auxiliary Losses Using Gradient Similarity\" Du Czarnecki 2018 published venue ICLR rejected workshop"}
```

<details><summary>tool result</summary>

```
=== QUERY: aux learning survey ===
=== QUERY: MTL survey ===
http://arxiv.org/abs/1707.08114v3 | A Survey on Multi-Task Learning | 2017-07-25
   C: Accepted by IEEE Transactions on Knowledge and Data Engineering
http://arxiv.org/abs/2009.09796v1 | Multi-Task Learning with Deep Neural Networks: A Survey | 2020-09-10
http://arxiv.org/abs/2508.00917v2 | A Survey on Deep Multi-Task Learning in Connected Autonomous Vehicles | 2025-07-29
http://arxiv.org/abs/1806.03903v1 | Multi-task learning of daily work and study round-trips from survey data | 2018-06-11
http://arxiv.org/abs/2510.26392v1 | Multi-Task Learning Based on Support Vector Machines and Twin Support Vector Machines: A Comprehensive Survey | 2025-10-30
http://arxiv.org/abs/2308.08234v1 | Challenges and Opportunities of Using Transformer-Based Multi-Task Learning in NLP Through ML Lifecycle: A Survey | 2023-08-16
=== QUERY: task relatedness ===
http://arxiv.org/abs/2505.21438v1 | Measuring Fine-Grained Relatedness in Multitask Learning via Data Attribution | 2025-05-27
http://arxiv.org/abs/2307.00823v2 | Understanding the Transferability of Representations via Task-Relatedness | 2023-07-03
   C: NeurIPS 2024
http://arxiv.org/abs/2111.05323v1 | Variational Multi-Task Learning with Gumbel-Softmax Priors | 2021-11-09
   C: 19 pages, 6 figures, accepted by NeurIPS 2021
http://arxiv.org/abs/2406.18311v2 | Online Learning of Multiple Tasks and Their Relationships : Testing on Spam Email Data and EEG Signals Recorded in Construction Fields | 2024-06-26
http://arxiv.org/abs/2403.03535v1 | Task Attribute Distance for Few-Shot Learning: Theoretical Analysis and Applications | 2024-03-06
http://arxiv.org/abs/2309.09517v3 | FedGKD: Unleashing the Power of Collaboration in Federated Graph Neural Networks | 2023-09-18
```
</details>

<details><summary>tool result</summary>

```
Web search results for query: ""Adapting Auxiliary Losses Using Gradient Similarity" Du Czarnecki 2018 published venue ICLR rejected workshop"

Links: [{"title":"[PDF] Adapting Auxiliary Losses Using Gradient Similarity | Semantic Scholar","url":"https://www.semanticscholar.org/paper/Adapting-Auxiliary-Losses-Using-Gradient-Similarity-Du-Czarnecki/f6b956d5ea11b18cd59a42d2bf2791fc0ab8671f"},{"title":"[1812.02224] Adapting Auxiliary Losses Using Gradient Similarity","url":"https://arxiv.org/abs/1812.02224"},{"title":"Adapting Auxiliary Losses Using Gradient Similarity","url":"http://www.gatsby.ucl.ac.uk/~balaji/CL-NeurIPS2018-adapt.pdf"},{"title":"ADAPTING AUXILIARY LOSSES USING GRADIENT SIMILARITY","url":"https://arxiv.org/pdf/1812.02224"},{"title":"Adapting Auxiliary Losses Using Gradient Similarity","url":"https://www.deepmind.com/publications/adapting-auxiliary-losses-using-gradient-similarity"},{"title":"Adapting Auxiliary Losses Using Gradient Similarity — Lacuna","url":"https://lacuna.tiptreesystems.com/work/adapting-auxiliary-losses-using-gradient-similarity/wrk_db9fbde011cbc620c6e1895a8d766859"},{"title":"Adapting Auxiliary Losses Using Gradient Similarity | OpenReview","url":"https://openreview.net/forum?id=r1gl7hC5Km"},{"title":"Adapting Auxiliary Losses Using Gradient Similarity. - Google Search","url":"https://google.com/search?q=Adapting+Auxiliary+Losses+Using+Gradient+Similarity."},{"title":"(PDF) Adapting Auxiliary Losses Using Gradient Similarity","url":"https://www.researchgate.net/publication/329464688_Adapting_Auxiliary_Losses_Using_Gradient_Similarity"},{"title":"Adapting Auxiliary Losses Using Gradient Similarity | alphaXiv","url":"https://www.alphaxiv.org/abs/1812.02224"}]

Here's what I found about the paper and its publication status:

**Paper:** "Adapting Auxiliary Losses Using Gradient Similarity"

**Authors:** Yunshu Du, Wojciech M. Czarnecki, Siddhant M. Jayakumar, Mehrdad Farajtabar, Razvan Pascanu, Balaji Lakshminarayanan (DeepMind / Was
... [truncated 1500 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nfor id in 2307.00823 2505.21438; do\n  echo \"===== $id =====\"\n  curl -s -m 25 \"https://arxiv.org/abs/$id\" | python3 -c \"\nimport sys,re,html\nt=sys.stdin.read()\nm=re.search(r'<title>(.*?)</title>',t,re.S); print('TITLE:',html.unescape(m.group(1)).strip())\nm=re.search(r'comments mathjax\\\">(.*?)</td>',t,re.S); print('COMMENTS:',html.unescape(re.sub(r'<[^>]+>','',m.group(1))).strip()[:180] if m else 'NONE')\nm=re.search(r'citation_date\\\" content=\\\"(.*?)\\\"',t); print('DATE:',m.group(1))\nm=re.search(r'citation_abstract\\\" content=\\\"(.*?)\\\" />',t,re.S); print('ABS:',html.unescape(m.group(1))[:900] if m else 'NO')\n\"\n  sleep 1\ndone", "description": "Verify task-relatedness papers"}
```

<details><summary>tool result</summary>

```
===== 2307.00823 =====
TITLE: [2307.00823] Understanding the Transferability of Representations via Task-Relatedness
COMMENTS: NeurIPS 2024
DATE: 2023/07/03
ABS: The growing popularity of transfer learning, due to the availability of models pre-trained on vast amounts of data, makes it imperative to understand when the knowledge of these pre-trained models can be transferred to obtain high-performing models on downstream target tasks. However, the exact conditions under which transfer learning succeeds in a cross-domain cross-task setting are still poorly understood. To bridge this gap, we propose a novel analysis that analyzes the transferability of the representations of pre-trained models to downstream tasks in terms of their relatedness to a given reference task. Our analysis leads to an upper bound on transferability in terms of task-relatedness, quantified using the difference between the class priors, label sets, and features of the two tasks. Our experiments using state-of-the-art pre-trained models show the effectiveness of task-relatedn
===== 2505.21438 =====
TITLE: [2505.21438] Measuring Fine-Grained Relatedness in Multitask Learning via Data Attribution
COMMENTS: NONE
DATE: 2025/05/27
ABS: Measuring task relatedness and mitigating negative transfer remain a critical open challenge in Multitask Learning (MTL). This work extends data attribution -- which quantifies the influence of individual training data points on model predictions -- to MTL setting for measuring task relatedness. We propose the MultiTask Influence Function (MTIF), a method that adapts influence functions to MTL models with hard or soft parameter sharing. Compared to conventional task relatedness measurements, MTIF provides a fine-grained, instance-level relatedness measure beyond the entire-task level. This fine-grained relatedness measure enables a data selection strategy to effectively mitigate negative transfer in MTL. Through extensive experiments, we demonstrate that the proposed M
... [truncated 121 chars]
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ncurl -s -m 25 \"https://arxiv.org/abs/2010.15413\" | python3 -c \"\nimport sys,re,html\nt=sys.stdin.read()\nfor lbl in ['comments','jref','doi']:\n    m=re.search(r'tablecell '+lbl+r'[^>]*>(.*?)</td>',t,re.S)\n    print(lbl.upper()+':', html.unescape(re.sub(r'<[^>]+>','',m.group(1))).strip()[:200] if m else 'NONE')\nm=re.search(r'citation_date\\\" content=\\\"(.*?)\\\"',t); print('DATE:',m.group(1))\n\"\nsleep 1\necho \"=== confirm MMoE not on arxiv (author search) ===\"\ncurl -s -m 25 \"https://export.arxiv.org/api/query?search_query=au:%22Zhe+Zhao%22+AND+abs:%22mixture-of-experts%22+AND+abs:%22multi-task%22&max_results=10\" | python3 -c \"\nimport sys,xml.etree.ElementTree as ET\nns={'a':'http://www.w3.org/2005/Atom'}\nr=ET.fromstring(sys.stdin.read())\nes=r.findall('a:entry',ns)\nprint('n=',len(es))\nfor e in es: print(e.find('a:id',ns).text,'|',' '.join(e.find('a:title',ns).text.split()))\n\"", "description": "Check transference venue and confirm MMoE absence from arXiv"}
```

<details><summary>tool result</summary>

```
COMMENTS: NONE
JREF: NONE
DOI: NONE
DATE: 2020/10/29
=== confirm MMoE not on arxiv (author search) ===
n= 1
http://arxiv.org/abs/2106.03760v3 | DSelect-k: Differentiable Selection in the Mixture of Experts with Applications to Multi-Task Learning
```
</details>

**tool call `WebSearch`**
```
{"query": "\"Measuring and Harnessing Transference in Multi-Task Learning\" Fifty Amid arXiv 2010.15413 published venue"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nq() {\necho \"=== QUERY: $2 ===\"\ncurl -s -m 25 \"https://export.arxiv.org/api/query?search_query=$1&max_results=${3:-6}&sortBy=relevance\" | python3 -c \"\nimport sys,xml.etree.ElementTree as ET\nns={'a':'http://www.w3.org/2005/Atom'}\ntry: r=ET.fromstring(sys.stdin.read())\nexcept Exception as e: print('parse err',e); raise SystemExit\nfor e in r.findall('a:entry',ns):\n  print(e.find('a:id',ns).text,'|',' '.join(e.find('a:title',ns).text.split()),'|',e.find('a:published',ns).text[:10])\n  c=e.find('{http://arxiv.org/schemas/atom}comment')\n  if c is not None: print('   C:',' '.join(c.text.split())[:170])\n\"\nsleep 1\n}\nq 'abs:%22can+be+decoded+from%22+AND+abs:%22auxiliary%22' 'decoded from aux'\nq 'abs:%22V-information%22+AND+abs:%22task%22+AND+cat:cs.LG' 'V-info task'\nq 'ti:%22Conditional+Mutual+Information%22+AND+abs:%22feature+selection%22+AND+abs:%22redundancy%22' 'CMI feature selection'\nq 'ti:%22Auxiliary+Task%22+AND+abs:%22linear+probe%22' 'aux task linear probe'", "description": "Final targeted searches"}
```

<details><summary>tool result</summary>

```
Web search results for query: ""Measuring and Harnessing Transference in Multi-Task Learning" Fifty Amid arXiv 2010.15413 published venue"

Links: [{"title":"[2010.15413] Measuring and Harnessing Transference in Multi-Task Learning","url":"https://arxiv.org/abs/2010.15413"},{"title":"Measuring and Harnessing Transference in Multi-Task Learning — Machine Learning","url":"https://awesomepapers.io/machine-learning/papers/2010.15413"},{"title":"Measuring and Harnessing Transference in Multi-Task Learning | OpenReview","url":"https://openreview.net/forum?id=HowQIZwD_42"},{"title":"Measuring and Harnessing Transference in Multi-Task Learning — Reinforcement Learning","url":"https://awesomepapers.io/reinforcement-learning/papers/2010.15413"},{"title":"[PDF] Measuring and Harnessing Transference in Multi-Task Learning | Semantic Scholar","url":"https://www.semanticscholar.org/paper/Measuring-and-Harnessing-Transference-in-Multi-Task-Fifty-Amid/7ed5b171bb468c11028f09999270c858830d89ba"},{"title":"Measuring and Harnessing Transference in Multi-Task Learning — Lacuna","url":"https://lacuna.tiptreesystems.com/work/measuring-and-harnessing-transference-in-multi-task-learning/wrk_bc35c60619ed19c161b03bb4d7ad03b8"},{"title":"Robust Analysis of Multi-Task Learning Efficiency: New Benchmarks on Light-Weighed Backbones and Effective Measurement of Multi-Task Learning Challenges by Feature Disentanglement","url":"https://arxiv.org/html/2402.03557v2"},{"title":"Hacking task confounder in meta-learning | Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence","url":"https://dl.acm.org/doi/10.24963/ijcai.2024/560"},{"title":"Multi-objective meta learning | Proceedings of the 35th International Conference on Neural Information Processing Systems","url":"https://dl.acm.org/doi/10.5555/3540261.3541893"},{"title":"Machine Learning Oct 2020","url":"https://www.arxiv.org/list/cs.LG/2020-10?skip=1275&show=50"}]

Here's what I found on the publication venue and
... [truncated 1866 chars]
```
</details>

<details><summary>tool result</summary>

```
=== QUERY: decoded from aux ===
http://arxiv.org/abs/1811.04708v1 | Analyzing deep CNN-based utterance embeddings for acoustic model adaptation | 2018-11-12
   C: accepted to SLT 2018
http://arxiv.org/abs/2212.03857v2 | Phase2vec: Dynamical systems embedding with a physics-informed convolutional network | 2022-12-07
=== QUERY: V-info task ===
http://arxiv.org/abs/2501.09631v1 | Empowering Large Language Models in Wireless Communication: A Novel Dataset and Fine-Tuning Framework | 2025-01-16
   C: 13 pages, 13 figure, journal
http://arxiv.org/abs/2304.10703v2 | ReCEval: Evaluating Reasoning Chains via Correctness and Informativeness | 2023-04-21
   C: EMNLP 2023 camera-ready (21 pages)
http://arxiv.org/abs/2507.00038v3 | Quality over Quantity: An Effective Large-Scale Data Reduction Strategy Based on Pointwise V-Information | 2025-06-19
=== QUERY: CMI feature selection ===
parse err syntax error: line 1, column 0
=== QUERY: aux task linear probe ===
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nq() {\necho \"=== QUERY: $2 ===\"\ncurl -s -m 25 \"https://export.arxiv.org/api/query?search_query=$1&max_results=${3:-6}&sortBy=relevance\" | python3 -c \"\nimport sys,xml.etree.ElementTree as ET\nns={'a':'http://www.w3.org/2005/Atom'}\ntry: r=ET.fromstring(sys.stdin.read())\nexcept Exception as e: print('parse err',e); raise SystemExit\nfor e in r.findall('a:entry',ns):\n  print(e.find('a:id',ns).text,'|',' '.join(e.find('a:title',ns).text.split()),'|',e.find('a:published',ns).text[:10])\n  c=e.find('{http://arxiv.org/schemas/atom}comment')\n  if c is not None: print('   C:',' '.join(c.text.split())[:170])\n\"\nsleep 1\n}\nq 'abs:%22auxiliary+task%22+AND+abs:%22world+model%22+AND+abs:%22probe%22' 'aux task world model probe'\nq 'ti:%22Does%22+AND+ti:%22auxiliary%22+AND+cat:cs.LG' 'does aux'\nq 'abs:%22probing%22+AND+abs:%22implicitly+encode%22+AND+abs:%22policy%22' 'implicitly encode policy'\nq 'abs:%22readout+probe%22+AND+abs:%22world+model%22' 'readout probe world model'", "description": "Search for probe-based aux/world-model diagnostics"}
```

<details><summary>tool result</summary>

```
=== QUERY: aux task world model probe ===
=== QUERY: does aux ===
http://arxiv.org/abs/2406.17718v1 | When does Self-Prediction help? Understanding Auxiliary Tasks in Reinforcement Learning | 2024-06-25
=== QUERY: implicitly encode policy ===
=== QUERY: readout probe world model ===
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ncurl -s -m 25 \"https://arxiv.org/abs/2406.17718\" | python3 -c \"\nimport sys,re,html\nt=sys.stdin.read()\nm=re.search(r'<title>(.*?)</title>',t,re.S); print('TITLE:',html.unescape(m.group(1)).strip())\nm=re.search(r'comments mathjax\\\">(.*?)</td>',t,re.S); print('COMMENTS:',html.unescape(re.sub(r'<[^>]+>','',m.group(1))).strip()[:200] if m else 'NONE')\nm=re.search(r'citation_date\\\" content=\\\"(.*?)\\\"',t); print('DATE:',m.group(1))\nm=re.search(r'citation_abstract\\\" content=\\\"(.*?)\\\" />',t,re.S); print('ABS:',html.unescape(m.group(1))[:900])\n\"\nsleep 1\necho \"=== 2010.08244 / 2109.05748 done. Check 2204.00565 venue ===\"\ncurl -s -m 25 \"https://arxiv.org/abs/2204.00565\" | python3 -c \"\nimport sys,re,html\nt=sys.stdin.read()\nm=re.search(r'citation_abstract\\\" content=\\\"(.*?)\\\" />',t,re.S); print('ABS:',html.unescape(m.group(1))[:700])\n\"", "description": "Verify self-prediction aux tasks RL paper"}
```

<details><summary>tool result</summary>

```
TITLE: [2406.17718] When does Self-Prediction help? Understanding Auxiliary Tasks in Reinforcement Learning
COMMENTS: NONE
DATE: 2024/06/25
ABS: We investigate the impact of auxiliary learning tasks such as observation reconstruction and latent self-prediction on the representation learning problem in reinforcement learning. We also study how they interact with distractions and observation functions in the MDP. We provide a theoretical analysis of the learning dynamics of observation reconstruction, latent self-prediction, and TD learning in the presence of distractions and observation functions under linear model assumptions. With this formalization, we are able to explain why latent-self prediction is a helpful \emph{auxiliary task}, while observation reconstruction can provide more useful features when used in isolation. Our empirical analysis shows that the insights obtained from our learning dynamics framework predicts the behavior of these loss functions beyond the linear model assumption in non-linear neural networks. This
=== 2010.08244 / 2109.05748 done. Check 2204.00565 venue ===
ABS: Auxiliary tasks have been argued to be useful for representation learning in reinforcement learning. Although many auxiliary tasks have been empirically shown to be effective for accelerating learning on the main task, it is not yet clear what makes useful auxiliary tasks. Some of the most promising results are on the pixel control, reward prediction, and the next state prediction auxiliary tasks; however, the empirical results are mixed, showing substantial improvements in some cases and marginal improvements in others. Careful investigations of how auxiliary tasks help the learning of the main task is necessary. In this paper, we take a step studying the effect of the target policies on th
```
</details>

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\ncurl -s -m 25 \"https://arxiv.org/abs/2106.03760\" | python3 -c \"\nimport sys,re,html\nt=sys.stdin.read()\nm=re.search(r'<title>(.*?)</title>',t,re.S); print('TITLE:',html.unescape(m.group(1)).strip())\nm=re.search(r'comments mathjax\\\">(.*?)</td>',t,re.S); print('COMMENTS:',html.unescape(re.sub(r'<[^>]+>','',m.group(1))).strip()[:200] if m else 'NONE')\nm=re.search(r'citation_date\\\" content=\\\"(.*?)\\\"',t); print('DATE:',m.group(1))\n\"", "description": "Verify DSelect-k"}
```

**tool call `WebSearch`**
```
{"query": "\"Information-Theoretic Probing with Minimum Description Length\" Voita Titov EMNLP 2020 venue proceedings"}
```

<details><summary>tool result</summary>

```
TITLE: [2106.03760] DSelect-k: Differentiable Selection in the Mixture of Experts with Applications to Multi-Task Learning
COMMENTS: Appeared in NeurIPS 2021
DATE: 2021/06/07
```
</details>

<details><summary>tool result</summary>

```
Web search results for query: ""Information-Theoretic Probing with Minimum Description Length" Voita Titov EMNLP 2020 venue proceedings"

Links: [{"title":"Information-Theoretic Probing with Minimum Description Length - ACL Anthology","url":"https://aclanthology.org/2020.emnlp-main.14/"},{"title":"Information-Theoretic Probing with Minimum Description Length | Request PDF","url":"https://www.researchgate.net/publication/347232984_Information-Theoretic_Probing_with_Minimum_Description_Length"},{"title":"GitHub - lena-voita/description-length-probing: This is a repository with the code for the EMNLP 2020 paper \"Information-Theoretic Probing with Minimum Description Length\"","url":"https://github.com/lena-voita/description-length-probing"},{"title":"[PDF] Information-Theoretic Probing with Minimum Description Length | Semantic Scholar","url":"https://www.semanticscholar.org/paper/Information-Theoretic-Probing-with-Minimum-Length-Voita-Titov/f4b585c9a79dfce0807b445a09036ea0f9cbcdce"},{"title":"Information-Theoretic Probing with MDL","url":"https://lena-voita.github.io/posts/mdl_probes.html"},{"title":"[Quick Review] Information-Theoretic Probing with Minimum Description Length","url":"https://liner.com/review/informationtheoretic-probing-with-minimum-description-length"},{"title":"Notes on: Information-Theoretic Probing with Minimum Description Length by Voita, E., & Titov, I. (2020) - Hugo Cisneros","url":"https://hugocisneros.com/notes/voitainformationtheoreticprobingminimum2020/"},{"title":"Information-Theoretic Probing with Minimum Description ...","url":"https://aclanthology.org/2020.emnlp-main.14.pdf"},{"title":"[2003.12298] Information-Theoretic Probing with Minimum Description Length","url":"https://arxiv.org/abs/2003.12298"},{"title":"Information-Theoretic Probing with Minimum Description Length · Pith","url":"https://pith.science/citations/791aae27-880c-4990-9794-7b4a4fb3b74a"}]

Here are the venue/proceedings details:

**Citation:** Elena Voita and Ivan Tit
... [truncated 1906 chars]
```
</details>

**tool call `WebSearch`**
```
{"query": "Task2Vec Task Embedding for Meta-Learning Achille ICCV 2019 published proceedings"}
```

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\necho \"=== 2109.09234 conditional probing detail from ar5iv ===\"\ncurl -s -m 40 \"https://ar5iv.labs.arxiv.org/html/2109.09234\" -o /tmp/cp.html -w \"HTTP:%{http_code}\\n\"\npython3 - <<'EOF'\nimport re,html\nt=open('/tmp/cp.html',errors='ignore').read()\nt=re.sub(r'<[^>]+>',' ',t); t=html.unescape(t); t=re.sub(r'\\s+',' ',t)\nfor kw in ['conditional V-information','beyond','we define','baseline B']:\n    for m in re.finditer(re.escape(kw),t,re.I):\n        print(f'[{kw}]',t[max(0,m.start()-300):m.start()+400].strip()[:700]); print('---'); break\nEOF", "description": "Get conditional probing measure detail"}
```

<details><summary>tool result</summary>

```
Web search results for query: "Task2Vec Task Embedding for Meta-Learning Achille ICCV 2019 published proceedings"

Links: [{"title":"TASK2VEC: Task Embedding for Meta-Learning Alessandro Achille1,2 Michael Lam1","url":"https://openaccess.thecvf.com/content_ICCV_2019/papers/Achille_Task2Vec_Task_Embedding_for_Meta-Learning_ICCV_2019_paper.pdf"},{"title":"[1902.03545] Task2Vec: Task Embedding for Meta-Learning","url":"https://arxiv.org/abs/1902.03545"},{"title":"[PDF] Task2Vec: Task Embedding for Meta-Learning | Semantic Scholar","url":"https://www.semanticscholar.org/paper/Task2Vec:-Task-Embedding-for-Meta-Learning-Achille-Lam/3f0e82d56d18787feac8ddbc7c3c8490eb76a4c7"},{"title":"GitHub - awslabs/aws-cv-task2vec: Official code for the paper \"Task2Vec: Task Embedding for Meta-Learning\" (https://arxiv.org/abs/1902.03545, ICCV 2019) · GitHub","url":"https://github.com/awslabs/aws-cv-task2vec"},{"title":"Task2Vec: Task Embedding for Meta-Learning","url":"https://www.computer.org/csdl/proceedings-article/iccv/2019/480300g429/1hVlCGy5GM0"},{"title":"TASK2VEC: Task Embedding for Meta-Learning - Amazon Science","url":"https://www.amazon.science/publications/task2vec-task-embedding-for-meta-learning"},{"title":"Task2Vec: Task Embedding for Meta-Learning | ML Anthology","url":"https://mlanthology.org/iccv/2019/achille2019iccv-task2vec/"},{"title":"Alessandro Achille homepage","url":"https://alexachi.github.io/"},{"title":"Task2Vec: Task Embedding for Meta-Learning","url":"https://authors.library.caltech.edu/records/rd06a-axa63"},{"title":"Task2Vec: Task Embedding for Meta-Learning | Request PDF","url":"https://www.researchgate.net/publication/339558384_Task2Vec_Task_Embedding_for_Meta-Learning"}]

Here's what I found on the published proceedings version:

## Publication Details

**Title:** Task2Vec: Task Embedding for Meta-Learning

**Authors:** Alessandro Achille, Michael Lam, Rahul Tewari, Avinash Ravichandran, Subhransu Maji, Charless C. Fowlkes, Stefano Soatto, and Pietro
... [truncated 1709 chars]
```
</details>

<details><summary>tool result</summary>

```
=== 2109.09234 conditional probing detail from ar5iv ===
HTTP:200
[beyond] [2109.09234] Conditional probing: measuring usable information beyond a baseline function detectColorScheme(){ var theme="light"; var current_theme = localStorage.getItem("ar5iv_theme"); if(current_theme){ if(current_theme == "dark"){ theme = "dark"; } } else if(!window.matchMedia) { return false; } else if(window.matchMedia("(prefers-color-scheme: dark)").matches) { theme = "dark"; } if (theme=="dark") { document.documentElement.setAttribute("data-theme", "d
---
[we define] =\text{softmax}(W_{2}\ \sigma(W_{1}[x_{1};\cdots;x_{n}]+b)+b)\} , the set of 1-layer multi-layer perceptrons, is a predictive family. Ignorance of some x i x_{i} can be achieved by setting the corresponding rows of W 1 W_{1} to zero. Given the predictive family of functions the agent has access to, we define the multivariable 𝒱 \mathcal{V} -information analogue of entropy: Definition 2 (Multivariable Predictive 𝒱 \mathcal{V} -entropy) . Let X 1 , … , X n ∈ 𝒳 1 , … , 𝒳 n X_{1},\dots,X_{n}\in\mathcal{X}_{1},\dots,\mathcal{X}_{n} . Let C ∈ 𝒞 C\in\mathcal{C} and C ¯ ∈ 𝒞 ¯ \bar{C}\in\bar{\mathcal{C}} form a binary partition of X 1 , … , X n X_{1},\dots,X_{n} . Let a ¯ ∈ 𝒞 ¯ \bar{a}\in\bar{\mathca
---
[baseline B] resentation and on the baseline Zhang and Bowman 2018 ; we call this baselined probing performance : Perf ​ ( ϕ ​ ( X ) ) − Perf ​ ( B ) . \displaystyle\text{Perf}(\phi(X))-\text{Perf}(B). (1) This difference in performances estimates how much more accessible Y Y is in ϕ ⁡ ( X ) \phi(X) than in the baseline B B , under probe family 𝒱 \mathcal{V} . But what if B B and ϕ ⁡ ( X ) \phi(X) capture distinct aspects of Y Y ? For example, consider if ϕ ⁡ ( X ) \phi(X) captures parts-of-speech that aren’t the most common label for a given word identity, while B B captures parts-of-speech that are the most common for the word identity. Baselined probing will indicate that ϕ ⁡ ( X ) \phi(X) explains le
---
```
</details>

### ASSISTANT — 2026-09-16 17:19

I have enough verified data. Let me do a final confirmation pass on two remaining details.

**tool call `Bash`**
```
{"command": "export https_proxy=http://[REDACTED]:18000 http_proxy=http://[REDACTED]:18000\nfor id in 2010.15413 2402.15231 2202.03250 2406.17718 2204.00565 1902.03545 2210.14361; do\n  echo -n \"$id -> \"\n  curl -s -m 25 \"https://arxiv.org/abs/$id\" | python3 -c \"\nimport sys,re,html\nt=sys.stdin.read()\nm=re.search(r'comments mathjax\\\">(.*?)</td>',t,re.S)\nj=re.search(r'tablecell jref\\\">(.*?)</td>',t,re.S)\nprint('C=',(html.unescape(re.sub(r'<[^>]+>','',m.group(1))).strip()[:120] if m else 'NONE'),'| JREF=',(re.sub(r'<[^>]+>','',j.group(1)).strip() if j else 'NONE'))\n\"\n  sleep 1\ndone", "description": "Final comments/journal-ref check"}
```

<details><summary>tool result</summary>

```
2010.15413 -> C= NONE | JREF= NONE
2402.15231 -> C= NONE | JREF= NONE
2202.03250 -> C= NONE | JREF= NONE
2406.17718 -> C= NONE | JREF= NONE
2204.00565 -> C= NONE | JREF= NONE
1902.03545 -> C= NONE | JREF= NONE
2210.14361 -> C= NONE | JREF= NONE
```
</details>

### ASSISTANT — 2026-09-16 17:22

All verification complete. Here are the findings.

---

# Redundancy / Usefulness Diagnostics for Auxiliary Objectives — Verified Literature

**Method note:** every arXiv item below was verified by direct `curl` to `https://arxiv.org/abs/<id>`, extracting `<title>`, `citation_date`, and the `Comments` field. Non-arXiv venue facts were cross-checked via web search. Two corrections to your input list are flagged in §0.

---

## 0. Corrections to the input list

| Item | Status |
|---|---|
| MMoE at `1803.09820` | **CONFIRMED WRONG.** `1803.09820` is Leslie Smith's *"A disciplined approach to neural network hyper-parameters: Part 1"*. **MMoE has NO arXiv version at all** — I searched `ti:"Multi-gate Mixture-of-Experts"` (only unrelated later papers) and `au:"Zhe Zhao" AND abs:"mixture-of-experts" AND abs:"multi-task"` (returns only DSelect-k, 2106.03760). Correct citation in §3. |
| "Auxiliary Task Update Decomposition" | Found: **arXiv 2108.11346**. Note the oddity — posted Aug 2021 but its own Comments field says **ICLR 2021** (i.e., posted after acceptance). |

---

## 1. ⭐⭐⭐ PREDICTABILITY-OF-TARGET-FROM-FROZEN-REPRESENTATION MEASURES

**This is the cluster that matters most for your question. Read this section first.**

### ⭐⭐⭐ Taskonomy: Disentangling Task Transfer Learning
- **arXiv:** 1804.08328 · https://arxiv.org/abs/1804.08328
- **Date:** 2018/04/23 · **Venue (VERIFIED):** `CVPR 2018 (Oral)`
- **Status:** VERIFIED
- **THE MEASURE — this is exactly a frozen-representation-predictability diagnostic.** I pulled the full text to confirm the mechanics. The paper's own words: *"we compute an affinity matrix among tasks based on whether the solution for one task can be **sufficiently easily read out of the representation trained for another task**."* Concretely (their Fig. 4, "Transfer Function"): *"A small **readout function** is trained to map representations of source task's **frozen encoder** to target task's labels."* Pipeline: (I) train task-specific nets → (II) train first- and higher-order transfer/readout functions between them → (III) normalize raw transfer losses into comparable **affinities via AHP (Analytic Hierarchy Process)** → (IV) extract a globally optimal transfer policy via a **Binary Integer Program**. The AHP step matters: raw readout losses across heterogeneous tasks are not comparable, so they convert to pairwise-ordinal-then-normalized affinities. Their framing explicitly calls this *"a principled way for identifying **redundancies** across tasks."*

### ⭐⭐⭐ Conditional probing: measuring usable information beyond a baseline
- **arXiv:** 2109.09234 · https://arxiv.org/abs/2109.09234
- **Date:** 2021/09/19 · **Venue (VERIFIED):** `EMNLP 2021 + typo fixes`
- **Status:** VERIFIED
- **THE MEASURE — the single closest formal match to "redundancy diagnostic."** Hewitt, Ethayarajh, Liang, Manning. They point out the standard practice of **baselined probing**, `Perf(φ(X)) − Perf(B)`, is broken: if baseline `B` and representation `φ(X)` capture *distinct* aspects of `Y`, the subtraction understates `φ(X)`. Their fix defines a **multivariable predictive V-entropy** and a *conditional* V-information `I_V(φ(X) → Y | B)` — usable information in the representation **that is not already in the baseline**, by giving the probe family access to both and requiring it to be able to ignore either (e.g., zeroing rows of `W₁` in an MLP family). Verbatim from the abstract: *"they cannot detect when the representation is predictive of just the aspects of part-of-speech **not explainable by the word identity**."* **Direct transplant to your setting:** set `B` = representation/features already learned for the main task, `φ(X)` = the auxiliary-task representation; conditional V-information ≈ 0 ⟹ **the auxiliary objective is redundant.**

### ⭐⭐⭐ Understanding Dataset Difficulty with V-Usable Information
- **arXiv:** 2110.08420 · https://arxiv.org/abs/2110.08420
- **Date:** 2021/10/16 (v-latest 2025/04/27) · **Venue (VERIFIED):** `ICML 2022 (Outstanding Paper)`
- **Status:** VERIFIED
- **THE MEASURE:** Ethayarajh, Choi, Swayamdipta. Frames difficulty as **lack of V-usable information** — how predictable target `Y` is from input `X` under a *bounded model family* `V`. Adds **pointwise V-information (PVI)** for per-instance attribution. Crucially it supports **input transformations**: ablate/mask an attribute of the input and watch V-information drop, which is the usefulness-of-a-feature diagnostic. They used this to find annotation artefacts in NLP benchmarks.

### ⭐⭐⭐ A Theory of Usable Information Under Computational Constraints
- **arXiv:** 2002.10689 · https://arxiv.org/abs/2002.10689
- **Date:** 2020/02/25 · **Venue (VERIFIED):** `ICLR 2020 (Talk)`
- **Status:** VERIFIED
- **THE MEASURE — the theoretical foundation for the whole cluster.** Xu et al. define **predictive V-information**: a variational extension of Shannon information restricted to a computationally bounded observer family `V`. Two properties are load-bearing for you: (a) it **subsumes mutual information AND the coefficient of determination R²** as special cases — so an R²-from-linear-probe *is* a V-information estimate, which formally legitimizes probe-based diagnostics; (b) it **violates the data processing inequality**, i.e. V-information *can be created by computation*. That is precisely why a *learned* representation can make a target predictable when the raw input did not, and why probing a representation is not vacuous.

### ⭐⭐ Information-Theoretic Probing with Minimum Description Length
- **arXiv:** 2003.12298 · https://arxiv.org/abs/2003.12298
- **Date:** 2020/03/27 · **Comments field: EMPTY.** **Venue (verified externally):** EMNLP 2020 main conference, pp. 183–196, DOI `10.18653/v1/2020.emnlp-main.14`
- **Status:** arXiv VERIFIED / venue from ACL Anthology
- **THE MEASURE:** Voita & Titov. Replaces probe *accuracy* with the **description length of labels given representations** (via variational coding and online coding). Key motivating failure they document: plain probe accuracy *"[does] not substantially favour pretrained representations over randomly initialized ones"* and is similar for real labels vs. random synthetic tasks. MDL additionally charges for "the amount of effort" (probe size / data needed) to reach a given quality — **more stable than accuracy for the exact use you want.**

### ⭐⭐ Representation Similarity Analysis for Efficient Task Taxonomy & Transfer Learning
- **arXiv:** 1904.11740 · https://arxiv.org/abs/1904.11740
- **Date:** 2019/04/26 · **Venue (VERIFIED):** `Accepted at CVPR 2019. Code available at [github]`
- **Status:** VERIFIED
- **THE MEASURE:** The **cheap alternative to Taskonomy's readout-training.** Borrows Representation Similarity Analysis from neuroscience: build per-task representational dissimilarity matrices over a shared small image set, then correlate RDMs across tasks. Needs **only pre-trained models and a few images, zero further training** — vs. Taskonomy's O(n²) readout trainings. They validate that higher RSA score ⟹ higher actual transfer performance.

### ⭐⭐ Rissanen Data Analysis: Examining Dataset Characteristics via Description Length
- **arXiv:** 2103.03872 · https://arxiv.org/abs/2103.03872
- **Date:** 2021/03/05 · **Comments field:** code link only. **Venue (verified externally):** ICML 2021, PMLR 139:8500–8513
- **Status:** VERIFIED
- **THE MEASURE:** MDL of labels via **online/prequential coding**, used to ask *"does capability/feature X help achieve an accurate model of this data?"* — you compute codelength with and without the capability. This is a **usefulness test with a built-in complexity penalty**, framed exactly as an ablation.

### ⭐⭐ Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals
- **arXiv:** 2006.00995 · https://arxiv.org/abs/2006.00995
- **Date:** 2020/06/01 · **Venue (VERIFIED):** `TACL journal. Initial title was: "When Bert Forgets How To POS..."`
- **Status:** VERIFIED
- **THE MEASURE — and the important caveat against naive probing.** Instead of asking *what* is encoded, it **causally removes** a property from the representation (INLP-style linear projection) and measures the downstream behavioral drop. **The finding you should quote:** *"conventional probing performance is **not correlated to task importance**."* This is the strongest published warning that high probe accuracy ≠ the information is used. Any probe-based redundancy claim needs a causal-removal control alongside it.
- Follow-up: **arXiv 2506.11673**, *"Improving Causal Interventions in Amnesic Probing with Mean Projection or LEACE"* (2025/06/13) — fixes over-removal in the original projection.

### ⭐ Probing Classifiers: Promises, Shortcomings, and Advances
- **arXiv:** 2102.12452 · https://arxiv.org/abs/2102.12452
- **Date:** 2021/02/24 · **Venue (VERIFIED):** `Accepted to Computational Linguistics as a squib`
- **Status:** VERIFIED
- Belinkov's critical survey of the probing paradigm. **Read this before designing a probe-based diagnostic** — it enumerates the confounds (probe capacity, control tasks, selectivity, correlation-vs-causation) that referees will raise.

### ⭐ Decoding Task Progress from VLA Representations
- **arXiv:** 2608.13474 · https://arxiv.org/abs/2608.13474 · **Date:** 2026/08/13 · **Comments: NONE (preprint)**
- **Status:** VERIFIED (arXiv record exists; unpublished)
- Included because it is the embodied-AI instance of the pattern and is directly adjacent to your project: linear probes on π₀.₅'s residual stream show **task progress is linearly readable**, and — importantly — **the signal is already present in the pretrained PaliGemma backbone before any robot data**, i.e. an auxiliary progress-prediction objective would have been *redundant*. They also report the probe **does not enable meaningful steering**, echoing the Amnesic Probing caveat (readable ≠ used).

---

## 2. GRADIENT-BASED TASK-INTERFERENCE MEASURES

### Gradient Surgery for Multi-Task Learning (PCGrad)
- **arXiv:** 2001.06782 · https://arxiv.org/abs/2001.06782
- **Date:** 2020/01/19 · **Venue (VERIFIED):** `NeurIPS 2020. Code at github.com/tianheyu927/PCGrad`
- **Status:** VERIFIED
- **MEASURE:** Conflict detected by **negative cosine similarity between task gradients**; identifies three landscape conditions causing detrimental interference (conflicting directions + high curvature + large magnitude differences). Mitigation: project a task's gradient onto the **normal plane** of any conflicting task's gradient. Diagnostic, not selection — it fixes conflict rather than dropping the task.

### Adapting Auxiliary Losses Using Gradient Similarity
- **arXiv:** 1812.02224 · https://arxiv.org/abs/1812.02224
- **Date:** 2018/12/05 (v2 2020/11/25) · **Comments field: EMPTY. Journal-ref: NONE.**
- **Venue:** **arXiv preprint / workshop only.** Externally: submitted to ICLR 2019 (OpenReview `r1gl7hC5Km`), not accepted to main track; a version appeared at the **NeurIPS 2018 Continual Learning Workshop**. **Cite as arXiv preprint.**
- **Status:** VERIFIED (title/date) — **UNVERIFIED as a peer-reviewed main-conference paper**
- **MEASURE:** The canonical one-liner for your topic, straight from the abstract: *"it is not always trivial to know if an auxiliary task will be helpful for the main task and **when it could start hurting**."* Uses **cosine similarity between main and auxiliary gradients as an adaptive weight** (gate the auxiliary loss to zero when cosine < 0), with a convergence guarantee to critical points of the *main* task.

### Auxiliary Task Update Decomposition: The Good, The Bad and The Neutral
- **arXiv:** 2108.11346 · https://arxiv.org/abs/2108.11346
- **Date:** 2021/08/25 · **Venue (VERIFIED):** `15 pages, 3 figures, Accepted to ICLR 2021. Code: github.com/ldery/ATTITTUD`
- **Status:** VERIFIED
- **MEASURE — finer-grained than PCGrad.** Decomposes the auxiliary gradient into subspaces that **help / damage / leave unchanged** the primary loss (the "good, bad, neutral" of the title), then reweights each direction by its impact. Uses **randomized SVD** for scalability. Explicitly notes it *"encompasses some prior work as particular cases"* — so PCGrad and gradient-cosine are special cases. This is arguably the **most complete gradient-side redundancy diagnostic** (the "neutral" subspace is literally the redundant component).

### Efficiently Identifying Task Groupings for Multi-Task Learning (TAG)
- **arXiv:** 2109.04617 · **Venue:** NeurIPS 2021 spotlight (per your note, not re-verified)
- **MEASURE (as requested):** **Inter-task affinity** = *lookahead loss ratio*. Take a gradient step on task `i`'s loss only, then measure the relative change in task `j`'s loss: `Z_{i→j} = 1 − L_j(θ − α∇L_i) / L_j(θ)`. Positive ⟹ `i` helps `j`. Averaged over training to get an asymmetric affinity matrix, then network-selection is solved over it. **Its key efficiency claim: computed in a SINGLE multi-task training run** (all tasks trained together, affinities read off during that run), vs. 1905.07553's exhaustive-pairs approach — 11.6× faster with 10.0% lower test loss than train-all-together on Taskonomy.
- **Its predecessor, needed for the full story:** **Measuring and Harnessing Transference in Multi-Task Learning**, arXiv **2010.15413**, 2020/10/29, **Comments/journal-ref: BOTH EMPTY → arXiv preprint only** (ICLR 2021 submission `HowQIZwD_42`, not accepted). Same authors (Fifty, Amid, Zhao, Yu, Anil, Finn). Introduces the "transference" measure that TAG is built on, and applies it at two granularities: **macro** (which tasks train together) and **micro** (how to combine gradients per step). **Status: VERIFIED as preprint, UNVERIFIED as published.**

### GradTS: A Gradient-Based Automatic Auxiliary Task Selection Method Based on Transformer Networks
- **arXiv:** 2109.05748 · https://arxiv.org/abs/2109.05748
- **Date:** 2021/09/13 · **Venue (VERIFIED):** `In EMNLP 2021`
- **Status:** VERIFIED
- **MEASURE:** Auxiliary-task selection from **attention-head gradient statistics** in a Transformer. Its practical advantage: gradients come from **single-task experiments only** and are **re-used when the candidate task set changes** — no re-run per candidate subset. Beats AutoSeM by 0.33–17.93% on 8 GLUE NLU tasks at 21.32% less time.

### Scalable Multitask Learning Using Gradient-based Estimation of Task Affinity
- **arXiv:** 2409.06091 · https://arxiv.org/abs/2409.06091 · **Date:** 2024/09/09
- **Venue (VERIFIED):** `16 pages. Appeared in KDD 2024`
- **Status:** VERIFIED — the modern scalable successor to TAG-style affinity estimation.

---

## 3. MMoE — CORRECTED CITATION (no arXiv version exists)

### Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts
- **arXiv id:** **NONE.** (`1803.09820` is *not* this paper — it is Leslie N. Smith's LR/hyper-parameter paper.)
- **Authors:** Jiaqi Ma, Zhe Zhao, Xinyang Yi, Jilin Chen, Lichan Hong, Ed H. Chi
- **Venue:** **KDD 2018** — Proc. 24th ACM SIGKDD Int'l Conf. on Knowledge Discovery & Data Mining, London, UK, Aug 19–23 2018, **pp. 1930–1939**
- **DOI:** **`10.1145/3219819.3220007`** · https://dl.acm.org/doi/10.1145/3219819.3220007
- **Status:** VERIFIED (ACM DL + KDD 2018 accepted-papers page)
- **THE MEASURE — note it's an architecture, not a diagnostic.** MMoE **implicitly models** task relationships via per-task softmax **gating networks over shared experts**; the learned gate distributions can be *inspected post hoc* as a soft relatedness signal, but the paper defines **no explicit affinity metric**. Its most-cited empirical contribution is the controlled synthetic setup where **task correlation is a tunable knob**, showing MMoE's advantage grows as tasks become **less related**, plus a trainability benefit under varying initialization/data randomness.
- **Related, and it does have an arXiv id:** **DSelect-k: Differentiable Selection in the Mixture of Experts with Applications to Multi-Task Learning**, arXiv **2106.03760**, 2021/06/07, Comments: `Appeared in NeurIPS 2021` (VERIFIED) — same Zhe Zhao line of work, makes expert selection differentiable and explicitly sparse.

---

## 4. TASK GROUPING / SELECTION

### Which Tasks Should Be Learned Together in Multi-task Learning?
- **arXiv:** 1905.07553 · https://arxiv.org/abs/1905.07553
- **Date:** 2019/05/18 (v-latest 2020/09/03) · **Venue (VERIFIED):** `Presented to ICML 2020. Project: taskgrouping.stanford.edu`
- **Status:** VERIFIED
- **MEASURE:** The **brute-force empirical ground truth** everyone else's cheap proxy is validated against. Trains many task *subsets* and measures actual cooperation vs. competition, then solves an assignment of tasks to a **budgeted set of networks** so cooperating tasks share a net and competing tasks are separated. Offers an explicit **time-accuracy trade-off** curve. Same Stanford group as Taskonomy (Zamir, Guibas, Malik, Savarese).

### AutoSeM: Automatic Task Selection and Mixing in Multi-Task Learning
- **arXiv:** 1904.04153 · https://arxiv.org/abs/1904.04153
- **Date:** 2019/04/08 · **Venue (VERIFIED):** `NAACL 2019 (12 pages)`
- **Status:** VERIFIED
- **MEASURE:** Two-stage pipeline: (1) select useful auxiliaries with a **Beta-Bernoulli multi-armed bandit + Thompson Sampling** (usefulness = bandit reward on primary-task dev performance); (2) learn the **mixing ratio** via Gaussian-Process Bayesian optimization. The standard baseline that GradTS is compared against.

### Identifying beneficial task relations for multi-task learning in deep neural networks
- **arXiv:** 1702.08303 · https://arxiv.org/abs/1702.08303
- **Date:** 2017/02/27 · **Venue (VERIFIED):** `Accepted for publication at EACL 2017`
- **Status:** VERIFIED
- Bingel & Søgaard. **Early and still-cited:** runs 90 main/auxiliary NLP task pairs, then **regresses MTL gain on dataset- and single-task-learning-curve features** to find *which features of a task pair predict benefit*. Finding: **learning-curve gradients of the main task** are the strongest predictors — a cheap, meta-level usefulness proxy.

### Auxiliary Task Reweighting for Minimum-data Learning
- **arXiv:** 2010.08244 · https://arxiv.org/abs/2010.08244
- **Date:** 2020/10/16 · **Venue (VERIFIED):** `NeurIPS 2020. Project page: sites.google.com/view/auxiliary-task-reweighting`
- **Status:** VERIFIED
- **MEASURE — a genuinely different, non-gradient-cosine criterion.** Treats the **weighted auxiliary likelihood as a surrogate prior** for the main task, and sets weights to **minimize the divergence between that surrogate prior and the main task's true prior**. Objective is explicitly *minimizing the main task's data requirement*, which makes it a usefulness measure in **sample-complexity units** — avoids grid search entirely.

### Auxiliary Learning by Implicit Differentiation (AuxiLearn)
- **arXiv:** 2007.02693 · https://arxiv.org/abs/2007.02693
- **Date:** 2020/06/22 (v-latest 2021/05/11) · **Venue (VERIFIED):** `Published at ICLR 2021`
- **Status:** VERIFIED
- **MEASURE:** Bi-level / implicit differentiation. Two capabilities: (i) when auxiliaries are known, learn a **combination network** capturing *non-linear* interactions between losses (strictly more expressive than scalar weights); (ii) when no useful auxiliary is known, **generate a new auxiliary task**. Signal comes from hypergradients w.r.t. main-task validation loss.

### Adaptive Mixing of Auxiliary Losses in Supervised Learning (AMAL)
- **arXiv:** 2202.03250 · https://arxiv.org/abs/2202.03250 · **Date:** 2022/02/07 · **Comments: NONE**
- **Status:** VERIFIED as preprint; **venue UNVERIFIED**
- **MEASURE:** Bi-level optimization on validation data to learn **instance-level** mixing weights over auxiliary losses (not just per-task scalars). Applied to knowledge distillation and rule-denoising.

### Task Discovery: Finding the Tasks that Neural Networks Generalize on
- **arXiv:** 2212.00261 · https://arxiv.org/abs/2212.00261 · **Date:** 2022/12/01
- **Venue
... [truncated 14754 chars]
