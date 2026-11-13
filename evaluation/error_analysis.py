"""P07 paired ablation comparison and temporal error taxonomy analysis.

Taxonomy categories (per `REWIND_LITE_IMPROVEMENT_PLAN.md` §3 P06/P07 & §5):
- ``hit``: Rank-1 predicted interval reaches ``best_tiou >= tiou_threshold``.
- ``sampling_or_coverage_miss``: Zero searchable frames exist inside the ground-truth
  interval(s) for that query due to frame sampling or deduplication.
- ``no_prediction``: Index has coverage, but no candidate interval passed the
  temporal proposal floor.
- ``boundary_error``: Rank-1 interval overlaps ground truth (`0 < best_tiou < tiou_threshold`),
  indicating coarse/shifted temporal boundaries rather than a completely wrong event.
- ``wrong_object``: Disjoint prediction (`best_tiou == 0`) where one or more
  mapped COCO target classes from the Vietnamese query are absent.
- ``wrong_relation``: Disjoint prediction where multiple people cause subject
  ambiguity or target objects do not co-occur in the same frame.
- ``wrong_action_or_stage``: Disjoint prediction where target objects are present
  (or query targets non-COCO culinary/action states) but the action or stage is wrong.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from retrieval.object_reranker import (
    QueryObjectSpec,
    RerankProvenance,
    parse_vietnamese_query_objects,
)

TEMPORAL_ERROR_TAXONOMY: tuple[str, ...] = (
    "hit",
    "sampling_or_coverage_miss",
    "no_prediction",
    "boundary_error",
    "wrong_object",
    "wrong_relation",
    "wrong_action_or_stage",
)


def classify_temporal_query_error(
    query_row: Mapping[str, Any],
    eval_row: Mapping[str, Any],
    *,
    top1_provenance: RerankProvenance | Mapping[str, Any] | None = None,
    detected_classes: Sequence[str] | None = None,
    covered_in_index: bool = True,
    tiou_threshold: float = 0.3,
) -> dict[str, Any]:
    """Classify a single query's Rank-1 temporal prediction into the P07 taxonomy."""
    query_vi = str(query_row.get("query_vi") or query_row.get("query") or "").strip()
    query_spec: QueryObjectSpec = (
        parse_vietnamese_query_objects(query_vi)
        if query_vi
        else QueryObjectSpec("", (), None, (), (), (), False)
    )

    preds = eval_row.get("predictions", [])
    if not preds:
        category = "sampling_or_coverage_miss" if not covered_in_index else "no_prediction"
        return {
            "query_id": query_row["query_id"],
            "video_id": query_row["video_id"],
            "split": query_row.get("split", "dev"),
            "query_vi": query_vi,
            "gt_windows": list(query_row.get("time_windows", [])),
            "predicted_interval": None,
            "best_tiou": 0.0,
            "detected_objects": [],
            "target_objects": list(query_spec.target_classes),
            "unsupported_cues": list(query_spec.unsupported_cues),
            "failure_category": category,
        }

    top1 = preds[0]
    best_tiou = float(top1.get("best_tiou", 0.0))
    pred_interval = [float(top1["t_start"]), float(top1["t_end"])]

    prov_dict: dict[str, Any] = {}
    if isinstance(top1_provenance, RerankProvenance):
        prov_dict = top1_provenance.to_dict()
    elif isinstance(top1_provenance, Mapping):
        prov_dict = dict(top1_provenance)

    if detected_classes is not None:
        det_objs = sorted({str(c).strip().lower() for c in detected_classes if str(c).strip()})
    elif "matched_classes" in prov_dict:
        det_objs = list(prov_dict.get("matched_classes") or [])
    else:
        det_objs = []

    if best_tiou >= tiou_threshold - 1e-12:
        category = "hit"
    elif not covered_in_index:
        category = "sampling_or_coverage_miss"
    elif best_tiou > 0.0:
        category = "boundary_error"
    else:
        # Disjoint interval (best_tiou == 0.0)
        target_set = set(query_spec.target_classes)
        det_set = set(det_objs)
        multi_person = bool(prov_dict.get("multi_person_ambiguous", False))
        coocc_frames = prov_dict.get("cooccurring_frame_ids")

        if target_set and not target_set.issubset(det_set):
            category = "wrong_object"
        elif multi_person or (
            len(target_set) >= 2
            and coocc_frames is not None
            and len(coocc_frames) == 0
        ):
            category = "wrong_relation"
        else:
            category = "wrong_action_or_stage"

    return {
        "query_id": query_row["query_id"],
        "video_id": query_row["video_id"],
        "split": query_row.get("split", "dev"),
        "query_vi": query_vi,
        "gt_windows": list(query_row.get("time_windows", [])),
        "predicted_interval": pred_interval,
        "best_tiou": best_tiou,
        "detected_objects": det_objs,
        "target_objects": list(query_spec.target_classes),
        "unsupported_cues": list(query_spec.unsupported_cues),
        "failure_category": category,
    }


def compare_paired_ablation(
    queries: Sequence[Mapping[str, Any]],
    e1_eval: Mapping[str, Any],
    e2_eval: Mapping[str, Any],
    *,
    e2_provenance_by_qid: Mapping[str, Sequence[RerankProvenance | Mapping[str, Any]]] | None = None,
    e2_detected_by_qid: Mapping[str, Sequence[str]] | None = None,
    index_coverage_by_qid: Mapping[str, bool] | None = None,
    tiou_threshold: float = 0.3,
) -> dict[str, Any]:
    """Compare E1 (temporal-only) vs E2 (temporal + YOLO rerank) on identical queries."""
    e1_by_qid = {row["query_id"]: row for row in e1_eval.get("per_query", [])}
    e2_by_qid = {row["query_id"]: row for row in e2_eval.get("per_query", [])}

    paired_rows: list[dict[str, Any]] = []
    e1_tax_counter: Counter[str] = Counter()
    e2_tax_counter: Counter[str] = Counter()

    improved_tiou = 0
    regressed_tiou = 0
    unchanged_tiou = 0
    improved_hit = 0
    regressed_hit = 0

    for q in queries:
        qid = q["query_id"]
        if qid not in e1_by_qid or qid not in e2_by_qid:
            raise ValueError(f"query_id {qid!r} missing from E1 or E2 evaluation rows")

        covered = (
            bool(index_coverage_by_qid.get(qid, True))
            if index_coverage_by_qid is not None
            else True
        )
        e1_row = e1_by_qid[qid]
        e2_row = e2_by_qid[qid]

        prov_list = (
            e2_provenance_by_qid.get(qid, ())
            if e2_provenance_by_qid is not None
            else ()
        )
        top1_prov = prov_list[0] if prov_list else None
        det_objs = (
            e2_detected_by_qid.get(qid)
            if e2_detected_by_qid is not None
            else None
        )

        c1 = classify_temporal_query_error(
            q,
            e1_row,
            covered_in_index=covered,
            tiou_threshold=tiou_threshold,
        )
        c2 = classify_temporal_query_error(
            q,
            e2_row,
            top1_provenance=top1_prov,
            detected_classes=det_objs,
            covered_in_index=covered,
            tiou_threshold=tiou_threshold,
        )

        e1_tax_counter[c1["failure_category"]] += 1
        e2_tax_counter[c2["failure_category"]] += 1

        delta = c2["best_tiou"] - c1["best_tiou"]
        if delta > 1e-6:
            improved_tiou += 1
            outcome = "improved"
        elif delta < -1e-6:
            regressed_tiou += 1
            outcome = "regressed"
        else:
            unchanged_tiou += 1
            outcome = "unchanged"

        hit1 = c1["best_tiou"] >= tiou_threshold - 1e-12
        hit2 = c2["best_tiou"] >= tiou_threshold - 1e-12
        if not hit1 and hit2:
            improved_hit += 1
        elif hit1 and not hit2:
            regressed_hit += 1

        paired_rows.append(
            {
                "query_id": qid,
                "video_id": q["video_id"],
                "split": q.get("split", "dev"),
                "query_vi": c2["query_vi"],
                "gt_windows": c2["gt_windows"],
                "e1_predicted_interval": c1["predicted_interval"],
                "e2_predicted_interval": c2["predicted_interval"],
                "e1_tiou": c1["best_tiou"],
                "e2_tiou": c2["best_tiou"],
                "delta_tiou": round(delta, 6),
                "detected_objects": c2["detected_objects"],
                "target_objects": c2["target_objects"],
                "unsupported_cues": c2["unsupported_cues"],
                "e1_failure_category": c1["failure_category"],
                "e2_failure_category": c2["failure_category"],
                "outcome": outcome,
            }
        )

    e1_miou = float(e1_eval["summary"]["all"]["miou"])
    e2_miou = float(e2_eval["summary"]["all"]["miou"])
    miou_delta = e2_miou - e1_miou

    if miou_delta > 1e-6:
        conclusion = "object_rerank_improved"
    elif miou_delta < -1e-6:
        conclusion = "object_rerank_degraded"
    else:
        conclusion = "object_rerank_neutral"

    return {
        "tiou_threshold": tiou_threshold,
        "n_queries": len(paired_rows),
        "summary": {
            "e1_miou": e1_miou,
            "e2_miou": e2_miou,
            "miou_delta": round(miou_delta, 6),
            "e1_r@1_tiou_0.3": float(e1_eval["summary"]["all"]["r@1_tiou_0.3"]),
            "e2_r@1_tiou_0.3": float(e2_eval["summary"]["all"]["r@1_tiou_0.3"]),
            "e1_r@1_tiou_0.5": float(e1_eval["summary"]["all"]["r@1_tiou_0.5"]),
            "e2_r@1_tiou_0.5": float(e2_eval["summary"]["all"]["r@1_tiou_0.5"]),
            "improved_tiou_count": improved_tiou,
            "regressed_tiou_count": regressed_tiou,
            "unchanged_tiou_count": unchanged_tiou,
            "improved_hit_count": improved_hit,
            "regressed_hit_count": regressed_hit,
            "e1_taxonomy_counts": dict(e1_tax_counter),
            "e2_taxonomy_counts": dict(e2_tax_counter),
            "empirical_conclusion": conclusion,
        },
        "per_query": paired_rows,
    }
