from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .report_schema import ensure_dir


DEFAULT_DPI = 600


def figure_from_pixels(width_px: int, height_px: int, dpi: int = DEFAULT_DPI):
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, constrained_layout=True)
    return fig


def apply_axis_style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=20, pad=14)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, alpha=0.25, linewidth=0.8)


def save_figure_triplet(fig, base_path: Path, dpi: int = DEFAULT_DPI) -> dict[str, str]:
    ensure_dir(base_path.parent)
    png_path = base_path.with_suffix(".png")
    svg_path = base_path.with_suffix(".svg")
    pdf_path = base_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, facecolor="white")
    fig.savefig(svg_path, dpi=dpi, facecolor="white")
    fig.savefig(pdf_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return {
        "png": str(png_path),
        "svg": str(svg_path),
        "pdf": str(pdf_path),
    }


def write_plot_csv(csv_path: Path, rows: list[dict], fieldnames: list[str], origin_dir: Path) -> tuple[Path, Path]:
    import csv

    ensure_dir(csv_path.parent)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    ensure_dir(origin_dir)
    origin_csv = origin_dir / csv_path.name
    with origin_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path, origin_csv


def write_plot_metadata(metadata_path: Path, payload: dict) -> None:
    ensure_dir(metadata_path.parent)
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def note_figure(base_path: Path, title: str, lines: list[str], width_px: int, height_px: int, dpi: int = 300) -> dict[str, str]:
    fig = figure_from_pixels(width_px, height_px, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.03, 0.92, title, fontsize=26, fontweight="bold", va="top")
    for index, line in enumerate(lines):
        ax.text(0.03, 0.82 - index * 0.08, line, fontsize=18, va="top")
    return save_figure_triplet(fig, base_path, dpi=dpi)


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_contact_sheet(image_paths: list[Path], output_path: Path, columns: int, width_px: int, dpi: int = 300, bg_color=(255, 255, 255)) -> None:
    ensure_dir(output_path.parent)
    images = [load_image(path) for path in image_paths if path.exists()]
    if not images:
        blank = Image.new("RGB", (width_px, int(width_px * 0.6)), bg_color)
        blank.save(output_path, dpi=(dpi, dpi))
        return
    thumb_w = width_px // columns
    thumb_h = int(thumb_w * 9 / 16)
    rows = int(np.ceil(len(images) / columns))
    sheet = Image.new("RGB", (width_px, thumb_h * rows), bg_color)
    for index, image in enumerate(images):
        resized = image.resize((thumb_w, thumb_h))
        x = (index % columns) * thumb_w
        y = (index // columns) * thumb_h
        sheet.paste(resized, (x, y))
    sheet.save(output_path, dpi=(dpi, dpi))
