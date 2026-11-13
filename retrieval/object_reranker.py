"""P05 object-aware reranking over P03 temporal moments and P04 detections.

Design principles (per `REWIND_LITE_IMPROVEMENT_PLAN.md` §1.1, §3 P05):
1. Ablation-friendly: explicit switch ``enabled`` (default ``False``). When
   disabled, or when a Vietnamese query maps to no supported COCO classes, or
   when candidate moments carry no detections, the reranker falls back cleanly
   to the P03 temporal-only ordering without fabricating boxes or scores.
2. Academic honesty: object presence, same-frame subject/object co-occurrence,
   and coarse spatial proximity are treated strictly as **supporting object
   evidence**, NEVER as proof of action completion (`action_verified=False`).
3. Multi-person ambiguity: when multiple ``person`` boxes appear in the same
   frame, co-occurrence and spatial cues are explicitly discounted via
   ``multi_person_penalty`` rather than assuming which person performed the
   queried action.
4. Full score provenance: ``RerankProvenance`` records raw/normalized semantic
   score, object score components, combined score, matched/missing classes, and
   any unsupported query cues (actions/colors).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
import unicodedata
from typing import Any, Mapping, Sequence

from retrieval.temporal import TemporalMoment


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


# Explicit, inspectable Vietnamese phrase -> canonical COCO class dictionary.
# Longer phrases are matched before shorter substrings on word boundaries.
VIETNAMESE_COCO_LEXICON: tuple[tuple[str, str], ...] = (
    # Person / human subjects
    ("người đàn ông", "person"),
    ("người phụ nữ", "person"),
    ("chàng trai", "person"),
    ("cô gái", "person"),
    ("đầu bếp", "person"),
    ("em bé", "person"),
    ("đàn ông", "person"),
    ("phụ nữ", "person"),
    ("người", "person"),
    ("học sinh", "person"),
    ("sinh viên", "person"),
    ("cặp đôi", "person"),
    ("đôi nam nữ", "person"),
    ("đám đông", "person"),
    ("nhóm bạn", "person"),
    # Kitchen / food / containers
    ("chai nước", "bottle"),
    ("cái chai", "bottle"),
    ("chai", "bottle"),
    ("cái ly", "cup"),
    ("cái cốc", "cup"),
    ("ly nước", "cup"),
    ("cốc nước", "cup"),
    ("cốc", "cup"),
    ("ly", "cup"),
    ("cái bát", "bowl"),
    ("cái tô", "bowl"),
    ("cái chén", "bowl"),
    ("bát", "bowl"),
    ("tô", "bowl"),
    ("chén", "bowl"),
    ("con dao", "knife"),
    ("dao", "knife"),
    ("cái muỗng", "spoon"),
    ("cái thìa", "spoon"),
    ("muỗng", "spoon"),
    ("thìa", "spoon"),
    ("cái nĩa", "fork"),
    ("nĩa", "fork"),
    ("cái kéo", "scissors"),
    ("lò vi sóng", "microwave"),
    ("lò nướng", "oven"),
    ("tủ lạnh", "refrigerator"),
    ("bồn rửa", "sink"),
    ("quả chuối", "banana"),
    ("quả táo", "apple"),
    ("quả cam", "orange"),
    ("bánh ngọt", "cake"),
    ("bánh pizza", "pizza"),
    # Furniture / indoor objects
    ("bàn ăn", "dining table"),
    ("cái bàn", "dining table"),
    ("bàn", "dining table"),
    ("cái ghế", "chair"),
    ("ghế sofa", "couch"),
    ("sofa", "couch"),
    ("ghế", "chair"),
    ("cái giường", "bed"),
    ("giường", "bed"),
    ("chậu cây", "potted plant"),
    ("bình hoa", "vase"),
    ("đồng hồ", "clock"),
    ("quyển sách", "book"),
    ("cuốn sách", "book"),
    ("sách", "book"),
    # Electronics
    ("điện thoại", "cell phone"),
    ("máy tính xách tay", "laptop"),
    ("laptop", "laptop"),
    ("máy tính", "laptop"),
    ("bàn phím", "keyboard"),
    ("chuột máy tính", "mouse"),
    ("tivi", "tv"),
    ("màn hình", "tv"),
    ("điều khiển", "remote"),
    # Accessories / bags
    ("ba lô", "backpack"),
    ("balo", "backpack"),
    ("túi xách", "handbag"),
    ("cái túi", "handbag"),
    ("vali", "suitcase"),
    ("cái ô", "umbrella"),
    ("cái dù", "umbrella"),
    ("cà vạt", "tie"),
    # Vehicles / street
    ("xe máy", "motorcycle"),
    ("mô tô", "motorcycle"),
    ("xe đạp", "bicycle"),
    ("ô tô", "car"),
    ("xe hơi", "car"),
    ("xe buýt", "bus"),
    ("xe tải", "truck"),
    ("đèn giao thông", "traffic light"),
    # Animals
    ("con chó", "dog"),
    ("chó", "dog"),
    ("con mèo", "cat"),
    ("mèo", "cat"),
    ("con chim", "bird"),
)

# Cues that class-only COCO detection cannot verify by itself
_UNSUPPORTED_ACTION_TERMS: tuple[str, ...] = (
    "lấy",
    "cầm",
    "nắm",
    "đặt",
    "để",
    "mở",
    "đóng",
    "uống",
    "ăn",
    "cắt",
    "thái",
    "nấu",
    "xào",
    "rửa",
    "lau",
    "đi bộ",
    "chạy",
    "ngồi",
    "đứng",
    "nhìn",
    "đọc",
    "viết",
    "gõ",
    "nói chuyện",
)

_UNSUPPORTED_ATTRIBUTE_TERMS: tuple[str, ...] = (
    "áo xanh",
    "áo đỏ",
    "áo trắng",
    "áo đen",
    "áo vàng",
    "quần đen",
    "quần trắng",
    "màu xanh",
    "màu đỏ",
    "màu trắng",
    "màu đen",
    "màu vàng",
    "bên trái",
    "bên phải",
)


def _normalize_vi_text(text: str) -> str:
    norm = unicodedata.normalize("NFC", text).lower()
    norm = re.sub(r"[^\w\s]", " ", norm, flags=re.UNICODE)
    norm = re.sub(r"\s+", " ", norm).strip()
    return f" {norm} "


def _contains_phrase(padded_text: str, phrase: str) -> bool:
    norm_phrase = _normalize_vi_text(phrase).strip()
    if not norm_phrase:
        return False
    return f" {norm_phrase} " in padded_text


_VI_NUMBER_WORDS: dict[str, int] = {
    "một": 1,
    "hai": 2,
    "ba": 3,
    "bốn": 4,
    "tư": 4,
    "năm": 5,
    "sáu": 6,
    "bảy": 7,
    "tám": 8,
    "chín": 9,
    "mười": 10,
}

_VI_PERSON_NOUN_PATTERN = (
    r"(?:người đàn ông|người phụ nữ|chàng trai|cô gái|học sinh|sinh viên|"
    r"em bé|đầu bếp|đàn ông|phụ nữ|người)"
)

_VI_MULTI_PERSON_PHRASES: tuple[str, ...] = (
    "cặp đôi",
    "đôi nam nữ",
    "nhóm người",
    "nhóm học sinh",
    "nhóm bạn",
    "đám đông",
    "đông người",
    "nhiều người",
)


def _extract_person_count(padded_text: str) -> int | None:
    """Extract explicit or collective person count from normalized Vietnamese query text."""
    counts: list[int] = []
    num_words = "|".join(re.escape(w) for w in _VI_NUMBER_WORDS)
    pattern = rf"(?<!\w)(\d+|{num_words})\s+{_VI_PERSON_NOUN_PATTERN}(?!\w)"
    for match in re.finditer(pattern, padded_text):
        token = match.group(1)
        if token.isdigit():
            val = int(token)
        else:
            val = _VI_NUMBER_WORDS.get(token, 0)
        if val >= 1:
            counts.append(val)

    for phrase in _VI_MULTI_PERSON_PHRASES:
        if _contains_phrase(padded_text, phrase):
            counts.append(2)

    return max(counts) if counts else None


@dataclass(frozen=True)
class QueryObjectSpec:
    """Transparent mapping of a Vietnamese query to COCO classes and limitations."""

    query_vi: str
    target_classes: tuple[str, ...]
    subject_class: str | None
    object_classes: tuple[str, ...]
    matched_phrases: tuple[str, ...]
    unsupported_cues: tuple[str, ...]
    mapping_supported: bool
    person_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_vietnamese_query_objects(
    query_vi: str,
    *,
    custom_lexicon: Sequence[tuple[str, str]] | None = None,
) -> QueryObjectSpec:
    """Extract canonical COCO classes from a Vietnamese query using explicit rules.

    Never invents unsupported classes. Explicitly flags action and attribute
    cues (e.g. ``action:lấy``, ``attribute:áo xanh``) that a class-only YOLO
    detector cannot verify.
    """
    if not isinstance(query_vi, str) or not query_vi.strip():
        raise ValueError("query_vi must be a nonempty string")

    lexicon = tuple(custom_lexicon) if custom_lexicon is not None else VIETNAMESE_COCO_LEXICON
    sorted_lexicon = sorted(lexicon, key=lambda item: len(item[0]), reverse=True)

    padded = _normalize_vi_text(query_vi)
    working = padded
    matched_classes: list[str] = []
    matched_phrases: list[str] = []

    for phrase, coco_class in sorted_lexicon:
        norm_phrase = _normalize_vi_text(phrase).strip()
        if not norm_phrase:
            continue
        token = f" {norm_phrase} "
        if token in working:
            matched_phrases.append(norm_phrase)
            if coco_class not in matched_classes:
                matched_classes.append(coco_class)
            # Mask matched span so shorter sub-phrases do not double-match
            working = working.replace(token, " ")

    unsupported: list[str] = []
    for act in _UNSUPPORTED_ACTION_TERMS:
        if _contains_phrase(padded, act):
            unsupported.append(f"action:{act}")
    for attr in _UNSUPPORTED_ATTRIBUTE_TERMS:
        if _contains_phrase(padded, attr):
            unsupported.append(f"attribute:{attr}")

    target_tuple = tuple(matched_classes)
    if "person" in target_tuple and len(target_tuple) >= 2:
        subject_cls: str | None = "person"
        object_cls = tuple(c for c in target_tuple if c != "person")
    else:
        subject_cls = None
        object_cls = target_tuple

    person_count = _extract_person_count(padded)

    return QueryObjectSpec(
        query_vi=query_vi.strip(),
        target_classes=target_tuple,
        subject_class=subject_cls,
        object_classes=object_cls,
        matched_phrases=tuple(matched_phrases),
        unsupported_cues=tuple(unsupported),
        mapping_supported=bool(target_tuple),
        person_count=person_count,
    )


@dataclass(frozen=True)
class ObjectRerankConfig:
    """Explicit, configurable weights for P05 object-aware reranking.

    Note: ``semantic_weight`` and ``object_weight`` are retained for backward
    contract compatibility; the active R05 combination formula uses additive
    object evidence ``combined = sem_norm + max_object_boost * obj_score`` so
    missing detector detections (``obj_score = 0``) never penalize high-semantic
    moments.
    """

    enabled: bool = False
    max_object_boost: float = 0.15
    semantic_weight: float = 0.75
    object_weight: float = 0.25
    presence_weight: float = 0.60
    cooccurrence_weight: float = 0.25
    spatial_proximity_weight: float = 0.15
    multi_person_penalty: float = 0.50
    generic_person_object_scale: float = 0.50
    min_detection_confidence: float = 0.25
    duplicate_box_iou_threshold: float = 0.85

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")
        if not _is_finite_number(self.max_object_boost) or not (
            0.0 <= float(self.max_object_boost) <= 0.5
        ):
            raise ValueError("max_object_boost must be in [0.0, 0.5]")
        for name in (
            "semantic_weight",
            "object_weight",
            "presence_weight",
            "cooccurrence_weight",
            "spatial_proximity_weight",
        ):
            val = getattr(self, name)
            if not _is_finite_number(val) or float(val) < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        if float(self.semantic_weight) + float(self.object_weight) <= 0.0:
            raise ValueError("semantic_weight + object_weight must be > 0")
        if float(self.presence_weight) + float(self.cooccurrence_weight) + float(
            self.spatial_proximity_weight
        ) <= 0.0:
            raise ValueError(
                "presence_weight + cooccurrence_weight + spatial_proximity_weight must be > 0"
            )
        if not _is_finite_number(self.multi_person_penalty) or not (
            0.0 <= float(self.multi_person_penalty) <= 1.0
        ):
            raise ValueError("multi_person_penalty must be in [0.0, 1.0]")
        if not _is_finite_number(self.generic_person_object_scale) or not (
            0.0 <= float(self.generic_person_object_scale) <= 1.0
        ):
            raise ValueError("generic_person_object_scale must be in [0.0, 1.0]")
        if not _is_finite_number(self.min_detection_confidence) or not (
            0.0 <= float(self.min_detection_confidence) <= 1.0
        ):
            raise ValueError("min_detection_confidence must be in [0.0, 1.0]")
        if not _is_finite_number(self.duplicate_box_iou_threshold) or not (
            0.0 < float(self.duplicate_box_iou_threshold) <= 1.0
        ):
            raise ValueError("duplicate_box_iou_threshold must be in (0.0, 1.0]")


@dataclass(frozen=True)
class RerankProvenance:
    """Detailed score breakdown for one temporal candidate moment."""

    video_id: str
    t_start: float
    t_end: float
    original_rank: int
    final_rank: int
    semantic_score_raw: float
    semantic_score_norm: float
    object_score: float
    presence_score: float
    cooccurrence_score: float
    spatial_score: float
    combined_score: float
    matched_classes: tuple[str, ...]
    missing_classes: tuple[str, ...]
    cooccurring_frame_ids: tuple[str, ...]
    multi_person_ambiguous: bool
    unsupported_query_cues: tuple[str, ...]
    object_boost: float = 0.0
    fallback_reason: str | None = None
    action_verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bbox_iou_xyxy(
    box_a: Sequence[float], box_b: Sequence[float]
) -> float:
    """2D IoU of two ``(x1, y1, x2, y2)`` bounding boxes."""
    if len(box_a) != 4 or len(box_b) != 4:
        raise ValueError("bounding boxes must have 4 coordinates (x1, y1, x2, y2)")
    ax1, ay1, ax2, ay2 = (float(v) for v in box_a)
    bx1, by1, bx2, by2 = (float(v) for v in box_b)
    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        return 0.0
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def _extract_norm_box(det: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    norm_box = det.get("bbox_norm_xyxy")
    if isinstance(norm_box, (list, tuple)) and len(norm_box) == 4:
        vals = [float(x) for x in norm_box]
        if all(math.isfinite(x) for x in vals) and 0.0 <= vals[0] < vals[2] <= 1.0 and 0.0 <= vals[1] < vals[3] <= 1.0:
            return (vals[0], vals[1], vals[2], vals[3])
    raw_box = det.get("bbox_xyxy") or det.get("bbox")
    img_size = det.get("image_size")
    if (
        isinstance(raw_box, (list, tuple))
        and len(raw_box) == 4
        and isinstance(img_size, (list, tuple))
        and len(img_size) == 2
    ):
        w, h = float(img_size[0]), float(img_size[1])
        if w > 0 and h > 0:
            x1, y1, x2, y2 = (float(v) for v in raw_box)
            nx1, ny1, nx2, ny2 = x1 / w, y1 / h, x2 / w, y2 / h
            if 0.0 <= nx1 < nx2 <= 1.0 + 1e-6 and 0.0 <= ny1 < ny2 <= 1.0 + 1e-6:
                return (
                    max(0.0, min(1.0, nx1)),
                    max(0.0, min(1.0, ny1)),
                    max(0.0, min(1.0, nx2)),
                    max(0.0, min(1.0, ny2)),
                )
    return None


def _deduplicate_frame_detections(
    detections: Sequence[Mapping[str, Any]],
    *,
    min_confidence: float,
    iou_threshold: float,
) -> list[dict[str, Any]]:
    """Filter low-confidence boxes and deduplicate near-identical same-class boxes per frame."""
    valid: list[dict[str, Any]] = []
    for det in detections:
        if not isinstance(det, Mapping):
            continue
        cls_name = det.get("class_name")
        conf = det.get("confidence")
        frame_id = det.get("frame_id")
        if (
            not isinstance(cls_name, str)
            or not cls_name.strip()
            or not _is_finite_number(conf)
            or float(conf) < min_confidence
            or float(conf) > 1.0
            or not isinstance(frame_id, str)
            or not frame_id.strip()
        ):
            continue
        norm_box = _extract_norm_box(det)
        if norm_box is None:
            continue
        entry = dict(det)
        entry["class_name"] = cls_name.strip().lower()
        entry["confidence"] = float(conf)
        entry["frame_id"] = frame_id.strip()
        entry["bbox_norm_xyxy"] = norm_box
        valid.append(entry)

    # Sort descending by confidence for deterministic per-frame class NMS
    valid.sort(
        key=lambda d: (
            d["frame_id"],
            d["class_name"],
            -d["confidence"],
            d["bbox_norm_xyxy"],
        )
    )
    deduped: list[dict[str, Any]] = []
    for cand in valid:
        if any(
            kept["frame_id"] == cand["frame_id"]
            and kept["class_name"] == cand["class_name"]
            and bbox_iou_xyxy(kept["bbox_norm_xyxy"], cand["bbox_norm_xyxy"]) >= iou_threshold
            for kept in deduped
        ):
            continue
        deduped.append(cand)
    return deduped


def _spatial_proximity(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Coarse spatial relation cue in [0, 1] between two normalized boxes.

    Measures normalized center proximity + box overlap; NEVER interpreted as
    action verification.
    """
    acx = 0.5 * (box_a[0] + box_a[2])
    acy = 0.5 * (box_a[1] + box_a[3])
    bcx = 0.5 * (box_b[0] + box_b[2])
    bcy = 0.5 * (box_b[1] + box_b[3])
    dist = math.hypot(acx - bcx, acy - bcy) / math.sqrt(2.0)
    center_prox = max(0.0, 1.0 - dist)
    iou = bbox_iou_xyxy(box_a, box_b)
    return float(min(1.0, 0.75 * center_prox + 0.25 * iou))


def _score_moment_objects(
    moment: TemporalMoment,
    query_spec: QueryObjectSpec,
    config: ObjectRerankConfig,
) -> tuple[float, float, float, float, tuple[str, ...], tuple[str, ...], tuple[str, ...], bool]:
    """Compute object score components in [0, 1] for one candidate moment."""
    target_classes = query_spec.target_classes
    if not target_classes or not moment.detections:
        return 0.0, 0.0, 0.0, 0.0, (), target_classes, (), False

    evidence_set = set(moment.evidence_frame_ids)
    raw_dets = [
        d
        for d in moment.detections
        if isinstance(d, Mapping) and d.get("frame_id") in evidence_set
    ]
    deduped = _deduplicate_frame_detections(
        raw_dets,
        min_confidence=float(config.min_detection_confidence),
        iou_threshold=float(config.duplicate_box_iou_threshold),
    )
    if not deduped:
        return 0.0, 0.0, 0.0, 0.0, (), target_classes, (), False

    # Group by frame_id
    by_frame: dict[str, list[dict[str, Any]]] = {}
    for det in deduped:
        by_frame.setdefault(det["frame_id"], []).append(det)

    # 1. Presence score across target classes
    best_conf_by_class: dict[str, float] = {}
    for cls in target_classes:
        confs = [d["confidence"] for d in deduped if d["class_name"] == cls]
        if confs:
            best_conf_by_class[cls] = float(max(confs))

    matched = tuple(cls for cls in target_classes if cls in best_conf_by_class)
    missing = tuple(cls for cls in target_classes if cls not in best_conf_by_class)
    coverage = len(matched) / len(target_classes)
    mean_conf = (
        sum(best_conf_by_class.values()) / len(target_classes)
        if best_conf_by_class
        else 0.0
    )
    presence_score = 0.5 * coverage + 0.5 * mean_conf

    # Check multi-person ambiguity across frames (unless query explicitly asks for >= 2 persons)
    req_person_count = query_spec.person_count
    expects_multi_person = req_person_count is not None and req_person_count >= 2

    has_multiple_persons = any(
        sum(1 for d in frame_dets if d["class_name"] == "person") >= 2
        for frame_dets in by_frame.values()
    )
    multi_person = False if expects_multi_person else has_multiple_persons
    ambiguity_factor = (
        1.0
        if expects_multi_person
        else (1.0 - 0.5 * float(config.multi_person_penalty) if multi_person else 1.0)
    )

    # 2. Co-occurrence and 3. Spatial proximity
    if len(target_classes) == 1:
        if expects_multi_person and target_classes[0] == "person":
            cooccurring_frames = tuple(
                fid
                for fid in moment.evidence_frame_ids
                if sum(1 for d in by_frame.get(fid, []) if d["class_name"] == "person")
                >= int(req_person_count)
            )
            if cooccurring_frames:
                cooccurrence_score = presence_score
                spatial_score = presence_score
            else:
                max_persons = max(
                    (
                        sum(1 for d in fd if d["class_name"] == "person")
                        for fd in by_frame.values()
                    ),
                    default=0,
                )
                count_ratio = min(1.0, max_persons / float(req_person_count))
                presence_score = presence_score * count_ratio
                cooccurrence_score = 0.0
                spatial_score = 0.0
        else:
            cooccurring_frames = tuple(
                fid
                for fid in moment.evidence_frame_ids
                if any(d["class_name"] == target_classes[0] for d in by_frame.get(fid, []))
            )
            cooccurrence_score = presence_score * ambiguity_factor
            spatial_score = presence_score * ambiguity_factor
    else:
        coocc_list: list[str] = []
        frame_coocc_scores: list[float] = []
        frame_spatial_scores: list[float] = []

        for fid in moment.evidence_frame_ids:
            f_dets = by_frame.get(fid, [])
            if not f_dets:
                continue
            present_in_frame = {d["class_name"] for d in f_dets}
            n_persons = sum(1 for d in f_dets if d["class_name"] == "person")
            person_count_ok = (
                not (expects_multi_person and "person" in target_classes)
                or n_persons >= int(req_person_count)
            )
            if all(cls in present_in_frame for cls in target_classes) and person_count_ok:
                coocc_list.append(fid)
                cls_max = [
                    max(d["confidence"] for d in f_dets if d["class_name"] == cls)
                    for cls in target_classes
                ]
                frame_discount = (
                    1.0
                    if expects_multi_person
                    else (
                        1.0 - float(config.multi_person_penalty)
                        if n_persons >= 2
                        else 1.0
                    )
                )
                frame_coocc_scores.append(
                    float(sum(cls_max) / len(cls_max)) * frame_discount
                )

                # Spatial proximity between subject (or first target class) and other target classes
                anchor_cls = query_spec.subject_class or target_classes[0]
                other_classes = [c for c in target_classes if c != anchor_cls]
                anchor_boxes = [
                    d["bbox_norm_xyxy"] for d in f_dets if d["class_name"] == anchor_cls
                ]
                prox_vals: list[float] = []
                for o_cls in other_classes:
                    o_boxes = [
                        d["bbox_norm_xyxy"] for d in f_dets if d["class_name"] == o_cls
                    ]
                    if anchor_boxes and o_boxes:
                        best_pair = max(
                            _spatial_proximity(ab, ob)
                            for ab in anchor_boxes
                            for ob in o_boxes
                        )
                        prox_vals.append(best_pair)
                if prox_vals:
                    frame_spatial_scores.append(
                        (sum(prox_vals) / len(prox_vals)) * frame_discount
                    )

        cooccurring_frames = tuple(coocc_list)
        cooccurrence_score = max(frame_coocc_scores, default=0.0)
        spatial_score = max(frame_spatial_scores, default=0.0)

    w_pres = float(config.presence_weight)
    w_cooc = float(config.cooccurrence_weight)
    w_spat = float(config.spatial_proximity_weight)
    w_total = w_pres + w_cooc + w_spat
    raw_object_score = (
        w_pres * presence_score + w_cooc * cooccurrence_score + w_spat * spatial_score
    ) / w_total

    # When query only matches generic 'person' (without specific objects like bottle, car, dog),
    # scale down object score so generic human presence does not override scene semantics.
    if target_classes == ("person",):
        object_score = raw_object_score * float(config.generic_person_object_scale)
    else:
        object_score = raw_object_score

    return (
        float(object_score),
        float(presence_score),
        float(cooccurrence_score),
        float(spatial_score),
        matched,
        missing,
        cooccurring_frames,
        multi_person,
    )


def _normalize_semantic_scores(moments: Sequence[TemporalMoment]) -> list[float]:
    """Normalize semantic scores continuously to [0, 1].

    When all candidate scores are equal (within 1e-12), returns 1.0 for every
    candidate. When ``hi > 0``, computes ``clip(s / hi, 0.0, 1.0)`` continuously
    without any threshold jump. When ``hi <= 0`` (all scores non-positive and
    not all equal), applies min-max scaling into ``[0.0, 1.0]``.
    """
    if not moments:
        return []
    raw = [float(m.score) for m in moments]
    lo = min(raw)
    hi = max(raw)
    if hi - lo <= 1e-12:
        return [1.0 for _ in raw]
    if hi > 0.0:
        return [float(min(1.0, max(0.0, s / hi))) for s in raw]
    return [float(min(1.0, max(0.0, (s - lo) / (hi - lo)))) for s in raw]


def rerank_temporal_moments(
    moments: Sequence[TemporalMoment],
    query_vi: str,
    *,
    config: ObjectRerankConfig = ObjectRerankConfig(),
    custom_lexicon: Sequence[tuple[str, str]] | None = None,
) -> tuple[list[TemporalMoment], list[RerankProvenance], QueryObjectSpec]:
    """Rerank P03 candidate moments using P04 object detections (ablation-safe).

    Returns ``(reranked_moments, provenances, query_object_spec)``.
     never fabricates bounding boxes or modifies interval boundaries.
    """
    if not isinstance(config, ObjectRerankConfig):
        raise ValueError("config must be an ObjectRerankConfig instance")
    query_spec = parse_vietnamese_query_objects(
        query_vi, custom_lexicon=custom_lexicon
    )
    if not moments:
        return [], [], query_spec

    for m in moments:
        if not isinstance(m, TemporalMoment):
            raise ValueError(f"expected TemporalMoment, got {m!r}")

    sem_norms = _normalize_semantic_scores(moments)

    # Check safe fallback conditions
    fallback_reason: str | None = None
    if not config.enabled:
        fallback_reason = "reranker_disabled"
    elif not query_spec.mapping_supported:
        fallback_reason = "no_mapped_classes"
    elif all(m.detections is None for m in moments):
        fallback_reason = "no_detections_available"

    if fallback_reason is not None:
        kept_moments: list[TemporalMoment] = []
        provenances: list[RerankProvenance] = []
        for idx, (m, sem_norm) in enumerate(zip(moments, sem_norms), 1):
            kept_moments.append(
                TemporalMoment(
                    video_id=m.video_id,
                    t_start=m.t_start,
                    t_end=m.t_end,
                    rank=idx,
                    score=m.score,
                    evidence_frame_ids=m.evidence_frame_ids,
                    score_type=m.score_type,
                    detections=m.detections,
                    peak_frame_id=getattr(m, "peak_frame_id", None),
                    peak_timestamp=getattr(m, "peak_timestamp", None),
                )
            )
            provenances.append(
                RerankProvenance(
                    video_id=m.video_id,
                    t_start=m.t_start,
                    t_end=m.t_end,
                    original_rank=m.rank,
                    final_rank=idx,
                    semantic_score_raw=m.score,
                    semantic_score_norm=sem_norm,
                    object_score=0.0,
                    presence_score=0.0,
                    cooccurrence_score=0.0,
                    spatial_score=0.0,
                    combined_score=m.score,
                    matched_classes=(),
                    missing_classes=query_spec.target_classes,
                    cooccurring_frame_ids=(),
                    multi_person_ambiguous=False,
                    unsupported_query_cues=query_spec.unsupported_cues,
                    fallback_reason=fallback_reason,
                    action_verified=False,
                )
            )
        return kept_moments, provenances, query_spec

    max_boost = float(config.max_object_boost)

    scored_entries: list[tuple[float, int, TemporalMoment, dict[str, Any]]] = []
    for m, sem_norm in zip(moments, sem_norms):
        (
            obj_score,
            pres_score,
            cooc_score,
            spat_score,
            matched,
            missing,
            coocc_fids,
            multi_person,
        ) = _score_moment_objects(m, query_spec, config)

        obj_boost = max_boost * obj_score
        combined = sem_norm + obj_boost
        scored_entries.append(
            (
                combined,
                m.rank,
                m,
                {
                    "sem_norm": sem_norm,
                    "obj_score": obj_score,
                    "obj_boost": obj_boost,
                    "pres_score": pres_score,
                    "cooc_score": cooc_score,
                    "spat_score": spat_score,
                    "matched": matched,
                    "missing": missing,
                    "coocc_fids": coocc_fids,
                    "multi_person": multi_person,
                },
            )
        )

    # Sort deterministically by descending combined score, then original rank & interval
    scored_entries.sort(
        key=lambda item: (
            -round(item[0], 9),
            item[1],
            item[2].t_start,
            item[2].t_end,
        )
    )

    reranked_moments: list[TemporalMoment] = []
    provenances = []
    for new_rank, (combined, orig_rank, m, info) in enumerate(scored_entries, 1):
        reranked_moments.append(
            TemporalMoment(
                video_id=m.video_id,
                t_start=m.t_start,
                t_end=m.t_end,
                rank=new_rank,
                score=float(combined),
                evidence_frame_ids=m.evidence_frame_ids,
                score_type="temporal_object_reranked",
                detections=m.detections,
                peak_frame_id=getattr(m, "peak_frame_id", None),
                peak_timestamp=getattr(m, "peak_timestamp", None),
            )
        )
        provenances.append(
            RerankProvenance(
                video_id=m.video_id,
                t_start=m.t_start,
                t_end=m.t_end,
                original_rank=orig_rank,
                final_rank=new_rank,
                semantic_score_raw=m.score,
                semantic_score_norm=info["sem_norm"],
                object_score=info["obj_score"],
                presence_score=info["pres_score"],
                cooccurrence_score=info["cooc_score"],
                spatial_score=info["spat_score"],
                combined_score=float(combined),
                matched_classes=info["matched"],
                missing_classes=info["missing"],
                cooccurring_frame_ids=info["coocc_fids"],
                multi_person_ambiguous=info["multi_person"],
                unsupported_query_cues=query_spec.unsupported_cues,
                object_boost=float(info["obj_boost"]),
                fallback_reason=None,
                action_verified=False,
            )
        )

    return reranked_moments, provenances, query_spec
