# Upstream / provenance

- Snapshot of /home/zhengguantian/WoG_fresh (WhenAmI research code) taken 2026-10-11.
- WoG/ and WoG_16rank_probe/ are vendored copies of https://github.com/Selen-Suyue/WoG (upstream commit 872d1df6055c0dc5d2504d4d6a5fb71c2bae3e6c) with local modifications (see `git -C WoG status` in the original tree; nested .git not included).
- Not included: model checkpoints / pretrained weights (symlinks to cluster storage), datasets, outputs, logs, caches, .hf_token files, and the Llama-2 tokenizer files under prismatic/models/backbones/llm/llama2_config/ (also git-ignored upstream). Checkpoints will be published on HuggingFace separately.
