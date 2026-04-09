#!/usr/bin/env python3
"""Frame raw Streamlit screenshots for use in the project README.

Usage:
  python scripts/frame_readme_screenshots.py \
    --chatbot-input raw/chatbot.png \
    --explorer-input raw/explorer.png

Outputs:
  assets/readme/ui-chatbot-preview.png
  assets/readme/ui-explorer-preview.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFilter, ImageFont


CANVAS_BG = (246, 248, 251, 255)
FRAME_BG = (255, 255, 255, 255)
ACCENT = (99, 102, 241, 255)
TEXT = (17, 24, 39, 255)
MUTED = (107, 114, 128, 255)
TOPBAR = (248, 250, 252, 255)
BORDER = (226, 232, 240, 255)
SHADOW = (15, 23, 42, 70)


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        ])
    else:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ])
    for path in candidates:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


TITLE_FONT = _load_font(28, bold=True)
SUB_FONT = _load_font(18, bold=False)
LABEL_FONT = _load_font(16, bold=True)
SMALL_FONT = _load_font(14, bold=False)


def rounded_rectangle_mask(size: tuple[int, int], radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, size[0], size[1]), radius=radius, fill=255)
    return mask


class ScreenshotFramer:
    def __init__(self, target_width: int = 1600) -> None:
        self.target_width = target_width

    def frame(self, input_path: Path, output_path: Path, title: str, subtitle: str) -> None:
        image = Image.open(input_path).convert("RGBA")
        scale = self.target_width / image.width if image.width > self.target_width else 1.0
        if scale != 1.0:
            image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)

        margin = 56
        header_h = 88
        topbar_h = 44
        inner_pad = 18
        radius = 26

        frame_w = image.width + inner_pad * 2
        frame_h = image.height + inner_pad * 2 + topbar_h
        canvas_w = frame_w + margin * 2
        canvas_h = frame_h + margin * 2 + header_h

        canvas = Image.new("RGBA", (canvas_w, canvas_h), CANVAS_BG)
        draw = ImageDraw.Draw(canvas)

        # Header text
        draw.text((margin, 18), title, fill=TEXT, font=TITLE_FONT)
        draw.text((margin, 54), subtitle, fill=MUTED, font=SUB_FONT)

        frame_x = margin
        frame_y = header_h + margin // 2

        shadow = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        shadow_draw.rounded_rectangle((0, 0, frame_w, frame_h), radius=radius, fill=SHADOW)
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=18))
        canvas.alpha_composite(shadow, (frame_x + 10, frame_y + 12))

        frame = Image.new("RGBA", (frame_w, frame_h), FRAME_BG)
        frame_mask = rounded_rectangle_mask((frame_w, frame_h), radius)
        frame_draw = ImageDraw.Draw(frame)
        frame_draw.rounded_rectangle((0, 0, frame_w - 1, frame_h - 1), radius=radius, fill=FRAME_BG, outline=BORDER, width=1)
        frame_draw.rounded_rectangle((0, 0, frame_w - 1, topbar_h), radius=radius, fill=TOPBAR)
        frame_draw.rectangle((0, topbar_h // 2, frame_w, topbar_h), fill=TOPBAR)

        # Window dots
        dot_y = topbar_h // 2
        dot_x = 24
        dot_colors = [(239, 68, 68, 255), (245, 158, 11, 255), (34, 197, 94, 255)]
        for color in dot_colors:
            frame_draw.ellipse((dot_x - 6, dot_y - 6, dot_x + 6, dot_y + 6), fill=color)
            dot_x += 22

        # Title pill
        pill_x1 = frame_w - 280
        pill_y1 = 10
        pill_x2 = frame_w - 18
        pill_y2 = 34
        frame_draw.rounded_rectangle((pill_x1, pill_y1, pill_x2, pill_y2), radius=12, fill=(238, 242, 255, 255), outline=(199, 210, 254, 255))
        frame_draw.text((pill_x1 + 12, pill_y1 + 4), "README screenshot asset", fill=ACCENT, font=SMALL_FONT)

        frame.alpha_composite(image, (inner_pad, inner_pad + topbar_h))
        canvas.paste(frame, (frame_x, frame_y), frame_mask)

        # Footer label
        footer_text = output_path.name
        tw = draw.textlength(footer_text, font=LABEL_FONT)
        badge_w = int(tw + 28)
        badge_h = 34
        bx = canvas_w - badge_w - margin
        by = canvas_h - badge_h - 18
        draw.rounded_rectangle((bx, by, bx + badge_w, by + badge_h), radius=16, fill=(255, 255, 255, 220), outline=(214, 219, 229, 255))
        draw.text((bx + 14, by + 8), footer_text, fill=MUTED, font=SMALL_FONT)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frame raw Streamlit screenshots for README usage.")
    parser.add_argument("--chatbot-input", type=Path, required=True, help="Raw screenshot for chatbot mode.")
    parser.add_argument("--explorer-input", type=Path, required=True, help="Raw screenshot for explorer mode.")
    parser.add_argument("--assets-dir", type=Path, default=Path("assets/readme"), help="Target assets directory.")
    parser.add_argument("--width", type=int, default=1600, help="Max width of the framed screenshot content.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    framer = ScreenshotFramer(target_width=args.width)
    framer.frame(
        input_path=args.chatbot_input,
        output_path=args.assets_dir / "ui-chatbot-preview.png",
        title="Document Chatbot",
        subtitle="Grounded multi-turn Q&A over a PDF corpus, with visible sources and conversation continuity.",
    )
    framer.frame(
        input_path=args.explorer_input,
        output_path=args.assets_dir / "ui-explorer-preview.png",
        title="Pipeline Explorer",
        subtitle="Side-by-side comparison of Manual, LangGraph, and LlamaIndex pipelines on the same query.",
    )
    print(f"Saved framed assets to: {args.assets_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
