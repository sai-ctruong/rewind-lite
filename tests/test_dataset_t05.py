"""T05 real-source manifest, annotation, split and sampling validation."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import pytest

cv2 = pytest.importorskip("cv2")

from evaluation import dataset


def _video(tmp_path, name="v", n_frames=10):
    path = tmp_path / f"{name}.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             10, (32, 32))
    assert writer.isOpened()
    import numpy as np
    for _ in range(n_frames):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()
    return path


def _source(path, *, split="dev", group="scene-a", rights="confirmed"):
    return {
        "video_id": path.stem, "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "duration_s": 1.0, "fps": 10.0, "frame_count": 10,
        "source": "user-supplied local video",
        "rights": {"status": rights, "basis": "user confirmed ownership"},
        "split": split, "group_id": group,
    }


def _label(video_id="v", query="người áo đỏ", *, group="q1", split="dev"):
    return {
        "query_id": f"{video_id}-001", "video_id": video_id,
        "query": query, "time_windows": [[0.0, 0.5]],
        "category": "color", "split": split,
        "language": "vi",
        "paraphrase_group": group,
        "annotator": "A", "review_status": "verified",
        "reviewer": "B", "reviewed_at": "2026-11-06",
    }


def test_valid_source_label_and_sampling_coverage(tmp_path):
    video = _video(tmp_path)
    manifest = {"schema_version": 1, "videos": [_source(video)]}
    report = dataset.validate_dataset(manifest, [_label()], root=tmp_path,
                              sample_every_s=0.25)
    assert report.issues == []
    assert report.n_videos == 1 and report.n_labels == 1
    assert report.n_verified == 1
    assert report.n_nominal_covered == 1
    assert report.category_counts == {"color": 1}
    assert report.ready_for_accuracy is False  # 50 reviewed rows remain the gate


def test_manifest_checks_video_file_hash_duration_and_rights(tmp_path):
    video = _video(tmp_path)
    source = _source(video, rights="pending")
    source["sha256"] = "0" * 64
    source["duration_s"] = 2.0
    report = dataset.validate_dataset({"schema_version": 1, "videos": [source]},
                              [_label()], root=tmp_path)
    assert {issue.code for issue in report.issues} >= {
        "checksum_mismatch", "duration_mismatch", "rights_unconfirmed"}
    assert report.n_verified == 0
    source["path"] = "missing.mp4"
    report = dataset.validate_dataset({"schema_version": 1, "videos": [source]},
                              [_label()], root=tmp_path)
    assert "missing_video" in {issue.code for issue in report.issues}


def test_labels_require_review_fields_and_windows_within_duration(tmp_path):
    video = _video(tmp_path)
    label = _label()
    label["time_windows"] = [[0.9, 1.1]]
    label["reviewer"] = ""
    report = dataset.validate_dataset({"schema_version": 1, "videos": [_source(video)]},
                              [label], root=tmp_path)
    assert {issue.code for issue in report.issues} >= {
        "window_out_of_bounds", "missing_reviewer"}
    assert report.n_verified == 0


def test_split_rejects_shared_video_group_and_paraphrase_across_dev_test(tmp_path):
    a, b = _video(tmp_path, "a"), _video(tmp_path, "b")
    manifest = {"schema_version": 1,
                "videos": [_source(a, split="dev", group="same"),
                           _source(b, split="test", group="same")]}
    labels = [_label("a", "người áo đỏ", group="same-query", split="dev"),
              _label("b", "người áo đỏ đang đi", group="same-query", split="test")]
    report = dataset.validate_dataset(manifest, labels, root=tmp_path)
    assert {issue.code for issue in report.issues} >= {
        "split_group_leakage", "paraphrase_leakage"}


def test_empty_dataset_reports_zero_verified_without_fake_pass(tmp_path):
    report = dataset.validate_dataset({"schema_version": 1, "videos": []}, [],
                              root=tmp_path)
    assert report.n_verified == 0
    assert report.n_nominal_covered == 0
    assert report.ready_for_accuracy is False


def test_nominal_sampling_coverage_reports_uncovered_narrow_window(tmp_path):
    video = _video(tmp_path)
    label = _label()
    label["time_windows"] = [[0.3, 0.6]]
    manifest = {"schema_version": 1, "videos": [_source(video)]}
    slow = dataset.validate_dataset(manifest, [label], root=tmp_path,
                                    sample_every_s=1.0)
    fast = dataset.validate_dataset(manifest, [label], root=tmp_path,
                                    sample_every_s=0.25)
    assert slow.n_nominal_covered == 0
    assert fast.n_nominal_covered == 1


def test_same_query_across_splits_leaks_even_with_different_group_ids(tmp_path):
    a, b = _video(tmp_path, "a"), _video(tmp_path, "b")
    manifest = {"schema_version": 1,
                "videos": [_source(a, split="dev", group="a"),
                           _source(b, split="test", group="b")]}
    labels = [_label("a", "người áo đỏ", group="g-a", split="dev"),
              _label("b", "Người áo đỏ!", group="g-b", split="test")]
    report = dataset.validate_dataset(manifest, labels, root=tmp_path)
    assert "paraphrase_leakage" in {issue.code for issue in report.issues}


def test_dataset_cli_reports_blocked_count_without_failing_structure(tmp_path):
    video = _video(tmp_path)
    manifest = tmp_path / "manifest.json"
    labels = tmp_path / "labels.json"
    manifest.write_text(json.dumps({"schema_version": 1,
                                    "videos": [_source(video)]}), encoding="utf-8")
    labels.write_text("[]", encoding="utf-8")
    cmd = [sys.executable, "-m", "evaluation.dataset", "--manifest", str(manifest),
           "--labels", str(labels), "--root", str(tmp_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    report = json.loads(result.stdout)
    assert report["n_verified"] == 0 and report["ready_for_accuracy"] is False
    assert subprocess.run([*cmd, "--require-ready"], capture_output=True).returncode == 3


def test_contact_sheet_samples_video_and_marks_timestamps(tmp_path):
    from PIL import Image
    from scripts.contact_sheet import build_contact_sheet

    video = _video(tmp_path)
    output = tmp_path / "sheet.jpg"
    report = build_contact_sheet(video, output, every_s=0.25, max_frames=4,
                                 columns=2)
    assert output.is_file()
    assert report["timestamps"] == [0.0, 0.25, 0.5, 0.75]
    assert report["n_samples"] == 4
    with Image.open(output) as image:
        assert image.width > image.height


def test_contact_sheet_rejects_broken_video(tmp_path):
    from scripts.contact_sheet import build_contact_sheet

    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="video"):
        build_contact_sheet(broken, tmp_path / "sheet.jpg")


def test_contact_sheet_frame_cap_covers_end_of_longer_video(tmp_path):
    from scripts.contact_sheet import build_contact_sheet

    video = _video(tmp_path, n_frames=100)
    report = build_contact_sheet(video, tmp_path / "sheet.jpg", every_s=1,
                                 max_frames=4)
    assert report["n_samples"] == 4
    assert report["timestamps"][0] == 0.0
    assert report["timestamps"][-1] == 9.0


def test_validator_reports_malformed_json_fields_instead_of_crashing(tmp_path):
    video = _video(tmp_path)
    source = _source(video)
    source["split"] = ["dev"]
    label = _label()
    label["category"] = ["color"]
    label["review_status"] = ["verified"]
    report = dataset.validate_dataset({"schema_version": 1, "videos": [source]},
                                      [label], root=tmp_path)
    assert {issue.code for issue in report.issues} >= {
        "invalid_split", "invalid_category", "invalid_review_status"}


def test_verified_count_requires_vietnamese_language_declaration(tmp_path):
    video = _video(tmp_path)
    label = _label()
    label["language"] = "en"
    report = dataset.validate_dataset({"schema_version": 1, "videos": [_source(video)]},
                                      [label], root=tmp_path)
    assert "invalid_language" in {issue.code for issue in report.issues}
    assert report.n_verified == 0
