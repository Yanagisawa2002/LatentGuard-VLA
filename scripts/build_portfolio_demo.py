"""Build the deterministic LatentGuard-VLA portfolio demo.

The renderer uses only committed, audited result values. It does not invoke a
simulator, load a checkpoint, or imply that the motion graphics are robot
footage. The MP4 is a release asset and remains outside the Git tree.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

WIDTH = 1280
HEIGHT = 720
FPS = 15
DURATION_SECONDS = 78

BACKGROUND = (7, 13, 29)
PANEL = (18, 29, 52)
PANEL_ALT = (24, 38, 65)
WHITE = (239, 246, 255)
MUTED = (148, 163, 184)
BLUE = (96, 165, 250)
CYAN = (45, 212, 191)
GREEN = (74, 222, 128)
AMBER = (251, 191, 36)
RED = (248, 113, 113)


@dataclass(frozen=True)
class Scene:
    """One timed section in the release demo."""

    start: float
    end: float
    eyebrow: str
    title: str
    subtitle: str
    kind: str


SCENES = (
    Scene(
        0,
        9,
        "THE QUESTION",
        "Can a verifier prevent a plausible robot action from failing?",
        "LatentGuard-VLA  |  ManiSkill PickCube  |  audited portfolio release",
        "question",
    ),
    Scene(
        9,
        21,
        "01  EXACT REPLAY",
        "Compare candidates from the same physical state",
        "Content-bound state > independent baseline/candidate replay > strong evidence",
        "replay",
    ),
    Scene(
        21,
        33,
        "02  VERIFIER",
        "Failure prediction worked - selective reliability did not",
        "Validation-only selection; untouched-test reporting",
        "verifier",
    ),
    Scene(
        33,
        45,
        "03  ONE-SHOT SELECTION",
        "Blind action choice improved",
        "All selectors saw the same frozen candidate pools",
        "oneshot",
    ),
    Scene(
        45,
        58,
        "04  CLOSED-LOOP STRESS TEST",
        "The one-shot win did not survive repeated intervention",
        "Negative result preserved - not hidden behind the earlier success",
        "closed_loop",
    ),
    Scene(
        58,
        72,
        "05  CONSERVATIVE REDESIGN",
        "Accept nominal by default; intervene only when the full gate fires",
        "Clean behavior recovered, but fault interception remained limited",
        "redesign",
    ),
    Scene(
        72,
        78,
        "TAKEAWAY",
        "Evidence first. Stress test the deployment claim.",
        "Keep the positive, negative, and partial result in one reproducible release.",
        "takeaway",
    ),
)


def _fonts() -> tuple[Any, Any, Any, Any, Any]:
    from PIL import ImageFont

    return (
        ImageFont.load_default(size=18),
        ImageFont.load_default(size=29),
        ImageFont.load_default(size=44),
        ImageFont.load_default(size=58),
        ImageFont.load_default(size=76),
    )


def _ease(value: float) -> float:
    clamped = max(0.0, min(1.0, value))
    return clamped * clamped * (3.0 - 2.0 * clamped)


def _mix(
    first: tuple[int, int, int], second: tuple[int, int, int], amount: float
) -> tuple[int, int, int]:
    return tuple(
        round(left + (right - left) * amount)
        for left, right in zip(first, second, strict=True)
    )


def _fit(draw: Any, text: str, font: Any, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if not current or draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _text_block(
    draw: Any,
    position: tuple[int, int],
    text: str,
    font: Any,
    fill: tuple[int, int, int],
    max_width: int,
    spacing: int = 10,
) -> int:
    x, y = position
    lines = _fit(draw, text, font, max_width)
    line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1]
    for index, line in enumerate(lines):
        draw.text((x, y + index * (line_height + spacing)), line, font=font, fill=fill)
    return y + len(lines) * (line_height + spacing)


def _panel(
    draw: Any, box: tuple[int, int, int, int], accent: tuple[int, int, int]
) -> None:
    draw.rounded_rectangle(box, radius=24, fill=PANEL, outline=PANEL_ALT, width=2)
    draw.rounded_rectangle(
        (box[0], box[1], box[0] + 8, box[3]),
        radius=4,
        fill=accent,
    )


def _metric(
    draw: Any,
    box: tuple[int, int, int, int],
    label: str,
    value: str,
    note: str,
    accent: tuple[int, int, int],
    reveal: float,
    fonts: tuple[Any, Any, Any, Any, Any],
) -> None:
    small, body, _heading, number, _hero = fonts
    _panel(draw, box, accent)
    draw.text((box[0] + 30, box[1] + 25), label, font=small, fill=MUTED)
    value_color = _mix(PANEL, accent, reveal)
    draw.text((box[0] + 30, box[1] + 64), value, font=number, fill=value_color)
    draw.text((box[0] + 30, box[3] - 49), note, font=body, fill=WHITE)


def _progress(draw: Any, second: float, fonts: tuple[Any, Any, Any, Any, Any]) -> None:
    small = fonts[0]
    labels = ("Replay", "Verify", "One-shot", "Closed loop", "Redesign")
    active = max(0, min(4, int((second - 9) // 12)))
    x0 = 78
    y = 665
    width = 1124
    draw.line((x0, y, x0 + width, y), fill=(51, 65, 85), width=3)
    for index, label in enumerate(labels):
        x = x0 + round(index * width / 4)
        color = CYAN if index <= active else (71, 85, 105)
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color)
        anchor = "la" if index == 0 else "ma" if index < 4 else "ra"
        draw.text((x, y + 17), label, font=small, fill=color, anchor=anchor)


def _header(
    draw: Any, scene: Scene, reveal: float, fonts: tuple[Any, Any, Any, Any, Any]
) -> None:
    small, body, heading, _number, _hero = fonts
    accent = _mix(BACKGROUND, CYAN, reveal)
    draw.text((78, 62), scene.eyebrow, font=small, fill=accent)
    y = _text_block(draw, (78, 96), scene.title, heading, WHITE, 1110, 7)
    _text_block(draw, (78, y + 6), scene.subtitle, body, MUTED, 1110, 5)


def _question(draw: Any, local: float, fonts: tuple[Any, Any, Any, Any, Any]) -> None:
    body, heading, hero = fonts[1], fonts[2], fonts[4]
    pulse = (math.sin(local * math.pi * 0.7) + 1.0) / 2.0
    x = 640
    draw.ellipse((x - 122, 286, x + 122, 530), outline=_mix(BLUE, CYAN, pulse), width=5)
    draw.text((x, 330), "?", font=hero, fill=WHITE, anchor="ma")
    draw.text(
        (x, 438),
        "action-conditioned verification",
        font=heading,
        fill=CYAN,
        anchor="ma",
    )
    draw.text((x, 496), "No general-safety claim", font=body, fill=MUTED, anchor="ma")


def _replay(draw: Any, local: float, fonts: tuple[Any, Any, Any, Any, Any]) -> None:
    body = fonts[1]
    labels = (
        ("Exact state", "content-bound", BLUE),
        ("Baseline", "independent session", CYAN),
        ("Candidate", "independent session", AMBER),
        ("Evidence", "terminal outcome", GREEN),
    )
    for index, (title, note, color) in enumerate(labels):
        x = 75 + index * 300
        reveal = _ease(local * 1.4 - index * 0.45)
        draw.rounded_rectangle(
            (x, 310, x + 230, 440), radius=20, fill=PANEL, outline=color, width=3
        )
        draw.text(
            (x + 115, 341),
            title,
            font=body,
            fill=_mix(PANEL, color, reveal),
            anchor="ma",
        )
        draw.text((x + 115, 391), note, font=fonts[0], fill=MUTED, anchor="ma")
        if index < 3:
            arrow = _mix(BACKGROUND, WHITE, _ease(local * 1.4 - index * 0.45 - 0.25))
            draw.line((x + 240, 375, x + 282, 375), fill=arrow, width=4)
            draw.polygon(((x + 282, 375), (x + 268, 366), (x + 268, 384)), fill=arrow)
    draw.text(
        (640, 500),
        "60 trajectories  |  360 anchors  |  2,880 strong outcomes",
        font=fonts[2],
        fill=WHITE,
        anchor="ma",
    )
    draw.text(
        (640, 557),
        "exact archive integrity + adapter-bound complete-state verification",
        font=body,
        fill=CYAN,
        anchor="ma",
    )


def _verifier(draw: Any, local: float, fonts: tuple[Any, Any, Any, Any, Any]) -> None:
    _metric(
        draw,
        (78, 300, 432, 550),
        "UNTOUCHED TEST",
        "0.8986",
        "failure AUPRC",
        GREEN,
        _ease(local),
        fonts,
    )
    _metric(
        draw,
        (463, 300, 817, 550),
        "RISK @ 80% COVERAGE",
        "3.38%",
        "selective risk",
        AMBER,
        _ease(local - 0.45),
        fonts,
    )
    _metric(
        draw,
        (848, 300, 1202, 550),
        "RISK @ 100% COVERAGE",
        "17.9%",
        "full coverage",
        RED,
        _ease(local - 0.9),
        fonts,
    )


def _bars(
    draw: Any,
    bars: tuple[tuple[str, float, tuple[int, int, int]], ...],
    local: float,
    fonts: tuple[Any, Any, Any, Any, Any],
    maximum: float = 100.0,
) -> None:
    body, heading = fonts[1], fonts[2]
    x0 = 350
    y0 = 300
    for index, (label, value, color) in enumerate(bars):
        y = y0 + index * 100
        reveal = _ease(local * 1.25 - index * 0.35)
        draw.text((x0 - 24, y + 20), label, font=body, fill=WHITE, anchor="ra")
        draw.rounded_rectangle((x0, y, 1020, y + 57), radius=14, fill=PANEL_ALT)
        endpoint = x0 + round(670 * value / maximum * reveal)
        if endpoint > x0:
            draw.rounded_rectangle((x0, y, endpoint, y + 57), radius=14, fill=color)
        draw.text(
            (1140, y + 28), f"{value:.2f}%", font=heading, fill=color, anchor="mm"
        )


def _oneshot(draw: Any, local: float, fonts: tuple[Any, Any, Any, Any, Any]) -> None:
    _bars(
        draw,
        (
            ("Random", 88.89, MUTED),
            ("Temporal verifier", 98.33, CYAN),
            ("Visual verifier", 98.89, GREEN),
        ),
        local,
        fonts,
    )
    draw.text(
        (640, 589),
        "one-shot success  |  temporal-minus-random 95% CI: [6.39, 12.50] pp",
        font=fonts[1],
        fill=WHITE,
        anchor="ma",
    )


def _closed_loop(
    draw: Any, local: float, fonts: tuple[Any, Any, Any, Any, Any]
) -> None:
    _bars(
        draw,
        (
            ("Fixed primary", 100.0, GREEN),
            ("Distilled visual", 73.33, RED),
            ("Intervention rate", 93.48, AMBER),
        ),
        local,
        fonts,
    )
    draw.text(
        (640, 589),
        "Repeated rescoring changed the state distribution and compounded errors",
        font=fonts[1],
        fill=RED,
        anchor="ma",
    )


def _redesign(draw: Any, local: float, fonts: tuple[Any, Any, Any, Any, Any]) -> None:
    _metric(
        draw,
        (78, 300, 342, 545),
        "CLEAN SUCCESS",
        "100%",
        "0% intervention",
        GREEN,
        _ease(local),
        fonts,
    )
    _metric(
        draw,
        (365, 300, 629, 545),
        "FAULT SUCCESS",
        "80%",
        "vs 73.33% nominal",
        CYAN,
        _ease(local - 0.35),
        fonts,
    )
    _metric(
        draw,
        (652, 300, 916, 545),
        "INTERVENTION",
        "1.17%",
        "fault schedule",
        AMBER,
        _ease(local - 0.7),
        fonts,
    )
    _metric(
        draw,
        (939, 300, 1203, 545),
        "OVERRIDE RECALL",
        "4.03%",
        "still unresolved",
        RED,
        _ease(local - 1.05),
        fonts,
    )
    draw.text(
        (640, 589),
        "Partial engineering success - not a solved safety claim",
        font=fonts[1],
        fill=WHITE,
        anchor="ma",
    )


def _takeaway(draw: Any, local: float, fonts: tuple[Any, Any, Any, Any, Any]) -> None:
    heading = fonts[2]
    items = (
        ("POSITIVE", "Exact evidence + blind one-shot selection", GREEN),
        ("NEGATIVE", "Repeated intervention regressed", RED),
        ("PARTIAL", "Conservative gate restored nominal behavior", AMBER),
    )
    for index, (label, text, color) in enumerate(items):
        y = 300 + index * 94
        reveal = _ease(local * 1.4 - index * 0.45)
        draw.rounded_rectangle((140, y, 1140, y + 68), radius=18, fill=PANEL)
        draw.text(
            (175, y + 34),
            label,
            font=fonts[0],
            fill=_mix(PANEL, color, reveal),
            anchor="lm",
        )
        draw.text((390, y + 34), text, font=heading, fill=WHITE, anchor="lm")


def render_frame(second: float) -> Any:
    """Render a single RGB frame at ``second``."""
    from PIL import Image, ImageDraw

    fonts = _fonts()
    scene = next(item for item in SCENES if second < item.end)
    local = second - scene.start
    reveal = _ease(local / 0.8)
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    for index in range(10):
        x = round((second * (8 + index) + index * 131) % (WIDTH + 220)) - 110
        y = 90 + index * 57
        draw.ellipse((x, y, x + 3, y + 3), fill=(30, 50, 78))

    _header(draw, scene, reveal, fonts)
    if scene.kind == "question":
        _question(draw, local, fonts)
    elif scene.kind == "replay":
        _replay(draw, local, fonts)
    elif scene.kind == "verifier":
        _verifier(draw, local, fonts)
    elif scene.kind == "oneshot":
        _oneshot(draw, local, fonts)
    elif scene.kind == "closed_loop":
        _closed_loop(draw, local, fonts)
    elif scene.kind == "redesign":
        _redesign(draw, local, fonts)
    else:
        _takeaway(draw, local, fonts)

    if scene.kind not in {"question", "takeaway"}:
        _progress(draw, second, fonts)
    draw.text(
        (1202, 63),
        f"{round(second):02d} / {DURATION_SECONDS}s",
        font=fonts[0],
        fill=MUTED,
        anchor="ra",
    )
    return image


def build_video(output: Path) -> None:
    """Encode the full release video to ``output``."""
    import imageio_ffmpeg

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(output),
        (WIDTH, HEIGHT),
        fps=FPS,
        codec="libx264",
        quality=7,
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        for frame_index in range(FPS * DURATION_SECONDS):
            frame = render_frame(frame_index / FPS)
            writer.send(frame.tobytes())
    finally:
        writer.close()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/latentguard-vla-78s-demo.mp4"),
    )
    parser.add_argument(
        "--poster-output",
        type=Path,
        default=Path("artifacts/latentguard-vla-demo-poster.png"),
    )
    parser.add_argument(
        "--poster-only",
        action="store_true",
        help="Render only the PNG poster and skip MP4 encoding.",
    )
    return parser.parse_args()


def main() -> int:
    """Render the poster and, unless requested otherwise, the release MP4."""
    args = parse_args()
    args.poster_output.parent.mkdir(parents=True, exist_ok=True)
    render_frame(61.5).save(args.poster_output)
    if not args.poster_only:
        build_video(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
