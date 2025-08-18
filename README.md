---
title: Fire App V2
emoji: 🦀
colorFrom: purple
colorTo: yellow
sdk: docker
pinned: false
license: apache-2.0
app_port: 7860
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

Environment switches:

- USE_HYDE: set to 0 to disable HyDE for faster retrieval.
- ENABLE_SENTENCE_LINKS: set to 1 to compute per-sentence source links (slower).
- ENABLE_CURATED_KB: set to 1 to enable curated JSON fast path.
- FAISS_TYPE: HNSW (default) for fast approximate search or FLAT for exact L2.

Curated KB file: `kb/curated_kb.json` with optional manual image mappings such as `images/Tesla_Model_3/designated_lift_areas.png`.