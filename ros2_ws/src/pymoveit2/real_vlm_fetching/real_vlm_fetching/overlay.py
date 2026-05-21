import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _clock_to_image_dir(clock: int):
    """Return (dx, dy) in image pixel coords (y-down). 12=up, 3=right, 6=down, 9=left."""
    angle_rad = math.radians((clock % 12) * 30)
    dx = math.sin(angle_rad)
    dy = -math.cos(angle_rad)
    # Snap near-zero to exactly 0.0
    dx = 0.0 if abs(dx) < 1e-9 else dx
    dy = 0.0 if abs(dy) < 1e-9 else dy
    return dx, dy


def _draw_arrow(draw, start, end, color, width=2, arrow_size=10):
    draw.line([start, end], fill=color, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)
    p1 = (
        end[0] - arrow_size * math.cos(angle - math.pi / 6),
        end[1] - arrow_size * math.sin(angle - math.pi / 6),
    )
    p2 = (
        end[0] - arrow_size * math.cos(angle + math.pi / 6),
        end[1] - arrow_size * math.sin(angle + math.pi / 6),
    )
    draw.polygon([end, p1, p2], fill=color)


def _load_font(size=14):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


def draw_clock_overlay(
    image_path,
    target_u: int,
    target_v: int,
    target_name: str = None,
    output_path=None,
    radius_px: int = 80,
) -> str:
    """
    Draw a 12-clock-direction arrow overlay on an image.

    Convention: 12=image up, 3=image right, 6=image down, 9=image left.
    Returns the path where the overlay image was saved.
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    cx, cy = float(target_u), float(target_v)

    # Target marker: circle + crosshair
    mr = 8
    draw.ellipse(
        [cx - mr, cy - mr, cx + mr, cy + mr],
        outline=(255, 40, 40), width=2,
    )
    draw.line([(cx - mr * 2, cy), (cx + mr * 2, cy)], fill=(255, 40, 40), width=2)
    draw.line([(cx, cy - mr * 2), (cx, cy + mr * 2)], fill=(255, 40, 40), width=2)

    if target_name:
        font_label = _load_font(13)
        draw.text((cx + mr + 4, cy - mr), target_name, fill=(255, 40, 40), font=font_label)

    # Clock arrows
    inner_r = max(mr * 2 + 4, int(radius_px * 0.30))
    font_clock = _load_font(14)
    arrow_color = (0, 210, 255)
    label_color = (255, 230, 0)

    for clock in range(1, 13):
        dx, dy = _clock_to_image_dir(clock)
        sx, sy = cx + inner_r * dx, cy + inner_r * dy
        ex, ey = cx + radius_px * dx, cy + radius_px * dy
        _draw_arrow(draw, (sx, sy), (ex, ey), color=arrow_color, width=2, arrow_size=9)

        lx = cx + (radius_px + 14) * dx
        ly = cy + (radius_px + 14) * dy
        text = str(clock)
        try:
            bbox = font_clock.getbbox(text)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = 8, 12
        draw.text((lx - tw / 2, ly - th / 2), text, fill=label_color, font=font_clock)

    if output_path is None:
        p = Path(image_path)
        output_path = str(p.parent / (p.stem + "_clock_overlay" + p.suffix))

    img.save(output_path)
    return str(output_path)
