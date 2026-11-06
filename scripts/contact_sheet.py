"""Make timestamped contact sheets for human review of local source videos."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


def build_contact_sheet(
    video: str | Path, output: str | Path, *, every_s: float = 1.0,
    max_frames: int = 64, columns: int = 4,
) -> dict:
    """Sample a video's uniform time grid and label each thumbnail.

    The sheet is for visual review; its thumbnails do not certify annotations.
    """
    if not math.isfinite(every_s) or every_s <= 0:
        raise ValueError("every_s must be positive and finite")
    if max_frames < 1 or columns < 1:
        raise ValueError("max_frames and columns must be positive")
    video, output = Path(video), Path(output)
    capture = cv2.VideoCapture(str(video))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or fps <= 0 or frame_count <= 0:
            raise ValueError(f"unreadable video: {video}")
        duration = frame_count / fps
        grid_size = math.ceil(duration / every_s)
        sample_count = min(max_frames, grid_size)
        indices = ([0] if sample_count == 1 else
                   [round(i * (grid_size - 1) / (sample_count - 1))
                    for i in range(sample_count)])
        timestamps = [round(i * every_s, 6) for i in indices]
        tile_width, tile_height, caption_height = 320, 180, 26
        rows = math.ceil(len(timestamps) / columns)
        sheet = Image.new("RGB", (columns * tile_width,
                                  rows * (tile_height + caption_height)), "#171717")
        draw = ImageDraw.Draw(sheet)
        for position, timestamp in enumerate(timestamps):
            frame_index = min(frame_count - 1, round(timestamp * fps))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                raise ValueError(f"could not decode video frame at {timestamp:.3f}s: {video}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            thumb = Image.fromarray(rgb)
            thumb.thumbnail((tile_width, tile_height))
            x = (position % columns) * tile_width
            y = (position // columns) * (tile_height + caption_height)
            sheet.paste(thumb, (x + (tile_width - thumb.width) // 2,
                                y + (tile_height - thumb.height) // 2))
            draw.text((x + 8, y + tile_height + 5), f"{timestamp:.2f}s", fill="white")
        output.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output, quality=88)
        return {"video": str(video), "output": str(output),
                "n_samples": len(timestamps), "timestamps": timestamps,
                "duration_s": round(duration, 6)}
    finally:
        capture.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a timestamped video contact sheet")
    parser.add_argument("video")
    parser.add_argument("output")
    parser.add_argument("--every", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        report = build_contact_sheet(args.video, args.output, every_s=args.every,
                                     max_frames=args.max_frames, columns=args.columns)
    except ValueError as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
