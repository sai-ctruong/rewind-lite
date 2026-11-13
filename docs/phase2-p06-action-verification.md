# P06 — Dev Error Audit, Action/Relation Verification Design Note & GO/NO-GO Gate

Prepared 2026-11-10 by **Gemini 4 Argon** on `main` following commit `2cd3fd9` (P04 + P05).

---

## 1. Audit of Real Dev-Set Queries & T06 Failure Modes

We audited all 43 verified `dev` queries in [`data/annotations/labels.json`](../data/annotations/labels.json) and the seven visually checked failures in [`docs/rewind-lite-task-06-benchmark.md`](rewind-lite-task-06-benchmark.md) against our P03–P05 pipeline capabilities.

### 1.1. Error Taxonomy (`object` / `action` / `relation` / `time` / `sampling`)

| Error Category | Representative Dev Queries | Root Cause Analysis | Can Class-Only YOLO (P04/P05) Solve It? |
| :--- | :--- | :--- | :--- |
| **1. Same Objects, Wrong Action / Stage (`action_confusion`)** | • `vi-dev-012` (*cầm chai* `[3.25, 14.0]`) vs `vi-dev-014` (*uống từ chai* `[6.0, 9.0]`)<br>• `vi-dev-040` (*thịt xông khói rắc lên bát súp* `[327, 331]`) vs frying pan (`35–51s`)<br>• `vi-dev-043` (*muỗng múc từ bát súp hoàn thiện* `[345, 349]`) vs ladling soup (`292–302s`) | Both the true interval and false-positive intervals contain the exact same static objects (`person + bottle`, or `bowl + spoon`). Distinguishing the action requires fine-grained motion or state-transition cues. | **No.** Static co-occurrence of `person + bottle` or `bowl + spoon` scores both windows similarly. P05 explicitly sets `action_verified=False`. |
| **2. Short Event / Sparse Sampling Miss (`sampling_miss`)** | • `vi-dev-013` (*đưa chai lên miệng* `[5.25, 6.25]`, 1.0s)<br>• `vi-dev-015` (*hạ chai xuống* `[8.75, 9.75]`, 1.0s — 0 indexed frames in T06)<br>• `vi-dev-019` (*giơ gói thịt xông khói* `[17.5, 18.5]`, 1.0s) | At 1.0s sampling + histogram (0.985) + semantic dedup (0.97), 1-second transitional actions have 0–1 retained keyframes. In `vi-dev-015`, nearby retained frames are at `7.0s` and `10.0s`. | **No.** A downstream detector or verifier cannot recover an event whose frames were dropped during ingestion/dedup. |
| **3. Multi-Person Subject/Attribute Ambiguity (`relation_ambiguity`)** | • `vi-dev-001` (*nhiều người đi bộ*)<br>• `vi-dev-003` (*người mặc áo sẫm đạp xe bên trái*)<br>• `vi-dev-004` (*người mặc áo khoác xanh dương*)<br>• `vi-dev-007` (*nhóm người quay lưng*) | Multiple `person` bounding boxes appear simultaneously in walkway frames. Standard COCO YOLO detects `person` and `bicycle` but does not classify clothing color (`áo xanh dương`, `áo sẫm`) or camera-relative walking direction. | **Partially mitigated.** P05 applies `multi_person_penalty` to avoid over-boosting ambiguous multi-person frames and flags `attribute:áo xanh` in `unsupported_cues`. |
| **4. Open-Vocabulary / Non-COCO Culinary Objects (`object_vocabulary_gap`)** | • `vi-dev-020..037` (*thịt xông khói, cần tây, hành tây, bột trắng, thịt nghêu, bơ, rau thơm*) | Pretrained 80-class COCO YOLO only covers `bowl`, `cup`, `spoon`, `knife`, `bottle`, `person`, `dining table`. It has no classes for ingredients (`bacon`, `celery`, `herbs`, `clam`, `butter`). | **Safe fallback.** P05 `parse_vietnamese_query_objects` maps only verified COCO nouns and falls back cleanly to P03 semantic scores (`SigLIP2`) when no COCO noun is present. |
| **5. Temporal Boundary Granularity (`boundary_approximation`)** | • `vi-dev-038` (`[[292.5, 294.5], [300.0, 302.5]]`)<br>• `vi-dev-042` (`[[0.0, 3.75], [340.0, 377.0]]`) | Keyframe timestamps are discrete samples; smoothing and merging approximate `[t_start, t_end)` up to the local sampling step. | **No.** Bounded by P03 sampling resolution. |

---

## 2. Comparison of Candidate Solutions for P06

| Criterion | Option A: **NO-GO** (Freeze at P05 + Explicit Limitations & Guardrail Tests) | Option B: **GO — Lightweight Geometric Motion Cue** (0-VRAM Box Trajectory Heuristic) | Option C: **GO — Heavy Video-Language / Action Model** (X-CLIP / VideoMAE / VLM) |
| :--- | :--- | :--- | :--- |
| **Mechanism** | Keep P05 `action_verified=False` and `unsupported_cues` flags; add P06 guardrail unit tests for multi-person, same-object wrong-action, and short actions; proceed directly to P07 E0/E1/E2 ablation. | Compute inter-frame bbox displacement $\Delta y(\text{object}, \text{person\_head})$ across candidate frames for vertical verbs (*đưa lên*, *nhấc lên*, *hạ xuống*). | Load a second deep video-text encoder on raw video clips for every candidate segment. |
| **Extra VRAM Cost** | **0 MB** (Safe on 4.29 GB RTX 2050) | **0 MB** (Pure NumPy on cached P04 boxes) | **+1.5 to 3.0 GB VRAM** (Exceeds 4.29 GB RTX 2050 budget alongside SigLIP2 + YOLO) |
| **Extra Query Latency** | **0 ms** | **< 1 ms** | **+500 to 2500 ms / query** |
| **Solves `vi-dev-015` (1s sampling miss)?** | No (Requires denser index sampling) | No (Requires $\ge 2$ sampled frames inside the 1s window) | No (Unless raw video is re-decoded per query) |
| **Solves `video-cook` ingredient stages?** | Handled by SigLIP2 semantic embeddings | No (Ingredients are not COCO boxes) | Partially, but violates hardware budget |
| **Risk of Scope Creep / Overclaiming** | **None** (100% transparent academic reporting) | **Low** (Must still never claim full semantic action recognition) | **High** (Turns Phase 2 into a new video-foundation-model project) |

---

## 3. Owner Decision (Turning-Point Gate — Recorded 2026-11-10)

**Decision:** **NO-GO (Option A)** approved by the repository owner on 2026-11-10.

1. **Rationale:**
   * On the 4.29 GB RTX 2050 workstation and 1-second deduplicated index, the observed `dev` action misses stem primarily from **1-second sampling gaps** (`vi-dev-015`) and **non-COCO culinary states** (`vi-dev-029`, `vi-dev-040`, `vi-dev-041`, `vi-dev-043`). Adding a third deep action/video-language model would exceed the 4.29 GB VRAM budget, while bbox-only heuristics cannot resolve non-COCO ingredients or unindexed frames.
   * The pipeline is therefore frozen at **E0 (Phase 1 Keyframe Context)**, **E1 (P03 Temporal-Only Baseline)**, and **E2 (P03 + P04/P05 YOLO Object-Aware Reranking)** for the P07 ablation study.
2. **Academic Guardrails Enforced (`tests/test_action_verification_p06.py`):**
   * `action_verified` remains strictly `False` across all P05 outputs.
   * Action verbs (`action:lấy`, `action:uống`, ...) and fine-grained attributes (`attribute:áo xanh`, ...) are explicitly surfaced in `QueryObjectSpec.unsupported_cues`.
   * Guardrail tests verify behavior on **multi-person ambiguity**, **same objects / wrong action**, and **short 1-second actions**.
