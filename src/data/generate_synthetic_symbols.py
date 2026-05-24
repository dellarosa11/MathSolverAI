from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data.class_config import (
    DIGIT_CLASSES,
    FOLDER_TO_LABEL,
    SYMBOL_FOLDER_TO_LABEL,
    get_default_classes,
    get_folder_name_for_label,
)


COMMON_FONT_DIRS = [
    Path("C:/Windows/Fonts"),
    Path("/Library/Fonts"),
    Path("/System/Library/Fonts"),
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
]

PREFERRED_FONT_NAMES = [
    "arial.ttf",
    "arialbd.ttf",
    "ariali.ttf",
    "calibri.ttf",
    "cambria.ttf",
    "consola.ttf",
    "consolab.ttf",
    "times.ttf",
    "timesbd.ttf",
    "verdana.ttf",
    "tahoma.ttf",
    "seguiemj.ttf",
]

LABEL_RENDER_VARIANTS = {
    "*": ["*", "\u00d7"],
    "/": ["/", "\u00f7"],
    "-": ["-", "\u2212"],
    "x": ["x", "X"],
}


def expand_requested_labels(requested_tokens: Sequence[str] | None) -> list[str]:
    if not requested_tokens:
        return get_default_classes()

    expanded: list[str] = []
    for token in requested_tokens:
        normalized = token.strip()
        if not normalized:
            continue

        lowered = normalized.lower()
        if lowered == "all":
            expanded.extend(get_default_classes())
            continue
        if lowered == "digits":
            expanded.extend(DIGIT_CLASSES)
            continue
        if lowered in {"operators", "symbols"}:
            expanded.extend(SYMBOL_FOLDER_TO_LABEL.values())
            continue
        if normalized in get_default_classes():
            expanded.append(normalized)
            continue
        if lowered in FOLDER_TO_LABEL:
            expanded.append(FOLDER_TO_LABEL[lowered])
            continue

        raise ValueError(f"Rotulo ou pasta nao suportado para geracao sintetica: {token}")

    unique_labels: list[str] = []
    seen: set[str] = set()
    for label in expanded:
        if label in seen:
            continue
        seen.add(label)
        unique_labels.append(label)
    return unique_labels


def discover_font_paths(font_dirs: Sequence[str | Path] | None = None, limit: int = 12) -> list[Path]:
    candidate_dirs = [Path(path) for path in font_dirs] if font_dirs else []
    if not candidate_dirs:
        candidate_dirs = list(COMMON_FONT_DIRS)

    discovered: list[Path] = []
    for directory in candidate_dirs:
        if not directory.exists():
            continue
        discovered.extend(path for path in directory.rglob("*") if path.suffix.lower() in {".ttf", ".otf"})

    if not discovered:
        return []

    preferred = {name.lower(): [] for name in PREFERRED_FONT_NAMES}
    fallback: list[Path] = []
    for path in discovered:
        name = path.name.lower()
        if name in preferred:
            preferred[name].append(path)
        else:
            fallback.append(path)

    ordered: list[Path] = []
    for preferred_name in PREFERRED_FONT_NAMES:
        ordered.extend(sorted(preferred.get(preferred_name, [])))

    remaining = sorted(path for path in fallback if path not in ordered)
    ordered.extend(remaining)
    return ordered[: max(1, limit)]


def _choose_render_text(label: str, rng: random.Random) -> str:
    variants = LABEL_RENDER_VARIANTS.get(label, [label])
    return rng.choice(variants)


def _load_font(
    font_paths: Sequence[Path],
    rng: random.Random,
    font_size: int,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if not font_paths:
        return ImageFont.load_default()

    chosen_path = rng.choice(list(font_paths))
    try:
        return ImageFont.truetype(str(chosen_path), size=font_size)
    except OSError:
        return ImageFont.load_default()


def _apply_random_effects(image: Image.Image, rng: random.Random) -> Image.Image:
    result = image

    if rng.random() < 0.35:
        result = result.filter(ImageFilter.MaxFilter(size=3))
    elif rng.random() < 0.15:
        result = result.filter(ImageFilter.MinFilter(size=3))

    if rng.random() < 0.2:
        result = result.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.1, 0.8)))

    array = np.array(result, dtype=np.uint8)
    if rng.random() < 0.4:
        noise_strength = rng.randint(2, 10)
        noise = np.random.default_rng(rng.randint(0, 1_000_000)).integers(
            low=0,
            high=noise_strength,
            size=array.shape,
            dtype=np.uint8,
        )
        array = np.clip(array + noise, 0, 255)

    return Image.fromarray(array.astype(np.uint8), mode="L")


def _render_manual_operator_image(
    label: str,
    rng: random.Random,
    canvas_size: int,
) -> Image.Image | None:
    if label not in {"+", "*", "/", "=", "x"}:
        return None

    image = Image.new("L", (canvas_size, canvas_size), color=0)
    draw = ImageDraw.Draw(image)
    center_x = canvas_size / 2 + rng.uniform(-canvas_size * 0.06, canvas_size * 0.06)
    center_y = canvas_size / 2 + rng.uniform(-canvas_size * 0.06, canvas_size * 0.06)
    half_span = rng.uniform(canvas_size * 0.16, canvas_size * 0.27)
    stroke_width = rng.randint(3, 7)

    if label == "+":
        draw.line(
            [(center_x - half_span, center_y), (center_x + half_span, center_y)],
            fill=255,
            width=stroke_width,
        )
        draw.line(
            [(center_x, center_y - half_span), (center_x, center_y + half_span)],
            fill=255,
            width=stroke_width,
        )
    elif label == "=":
        gap = rng.uniform(canvas_size * 0.06, canvas_size * 0.1)
        draw.line(
            [(center_x - half_span, center_y - gap), (center_x + half_span, center_y - gap)],
            fill=255,
            width=stroke_width,
        )
        draw.line(
            [(center_x - half_span, center_y + gap), (center_x + half_span, center_y + gap)],
            fill=255,
            width=stroke_width,
        )
    elif label == "*":
        if rng.random() < 0.5:
            draw.line(
                [(center_x - half_span, center_y - half_span), (center_x + half_span, center_y + half_span)],
                fill=255,
                width=stroke_width,
            )
            draw.line(
                [(center_x - half_span, center_y + half_span), (center_x + half_span, center_y - half_span)],
                fill=255,
                width=stroke_width,
            )
        else:
            draw.line(
                [(center_x - half_span, center_y), (center_x + half_span, center_y)],
                fill=255,
                width=stroke_width,
            )
            draw.line(
                [(center_x, center_y - half_span), (center_x, center_y + half_span)],
                fill=255,
                width=stroke_width,
            )
            draw.line(
                [(center_x - half_span * 0.8, center_y - half_span * 0.8), (center_x + half_span * 0.8, center_y + half_span * 0.8)],
                fill=255,
                width=max(2, stroke_width - 1),
            )
            draw.line(
                [(center_x - half_span * 0.8, center_y + half_span * 0.8), (center_x + half_span * 0.8, center_y - half_span * 0.8)],
                fill=255,
                width=max(2, stroke_width - 1),
            )
    elif label == "/":
        if rng.random() < 0.5:
            draw.line(
                [(center_x - half_span, center_y + half_span), (center_x + half_span, center_y - half_span)],
                fill=255,
                width=stroke_width,
            )
        else:
            dot_radius = max(3, stroke_width - 1)
            bar_half_span = half_span * 0.95
            draw.line(
                [(center_x - bar_half_span, center_y), (center_x + bar_half_span, center_y)],
                fill=255,
                width=stroke_width,
            )
            upper_center_y = center_y - half_span * 0.95
            lower_center_y = center_y + half_span * 0.95
            draw.ellipse(
                [
                    center_x - dot_radius,
                    upper_center_y - dot_radius,
                    center_x + dot_radius,
                    upper_center_y + dot_radius,
                ],
                fill=255,
            )
            draw.ellipse(
                [
                    center_x - dot_radius,
                    lower_center_y - dot_radius,
                    center_x + dot_radius,
                    lower_center_y + dot_radius,
                ],
                fill=255,
            )
    elif label == "x":
        x_span = half_span * rng.uniform(0.85, 1.1)
        y_span = half_span * rng.uniform(0.85, 1.15)
        draw.line(
            [(center_x - x_span, center_y - y_span), (center_x + x_span, center_y + y_span)],
            fill=255,
            width=stroke_width,
        )
        draw.line(
            [(center_x - x_span, center_y + y_span), (center_x + x_span, center_y - y_span)],
            fill=255,
            width=max(2, stroke_width - 1),
        )
        if rng.random() < 0.35:
            offset = rng.uniform(-canvas_size * 0.04, canvas_size * 0.04)
            draw.line(
                [(center_x - x_span * 0.55, center_y - y_span * 0.55 + offset), (center_x + x_span * 0.55, center_y + y_span * 0.55 + offset)],
                fill=255,
                width=max(1, stroke_width - 2),
            )

    rotation = rng.uniform(-20.0, 20.0)
    return image.rotate(
        rotation,
        resample=Image.Resampling.BILINEAR,
        fillcolor=0,
    )


def render_symbol_image(
    label: str,
    font_paths: Sequence[Path],
    rng: random.Random,
    canvas_size: int = 96,
) -> Image.Image:
    if label in {"+", "*", "/", "=", "x"} and rng.random() < 0.55:
        manual = _render_manual_operator_image(label, rng=rng, canvas_size=canvas_size)
        if manual is not None:
            return _apply_random_effects(manual, rng)

    background = Image.new("L", (canvas_size, canvas_size), color=0)
    draw = ImageDraw.Draw(background)

    render_text = _choose_render_text(label, rng)
    font_size = rng.randint(max(18, canvas_size // 2), max(24, int(canvas_size * 0.8)))
    font = _load_font(font_paths, rng, font_size)
    stroke_width = rng.randint(0, 2)

    bbox = draw.textbbox((0, 0), render_text, font=font, stroke_width=stroke_width)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    jitter_x = rng.randint(-canvas_size // 10, canvas_size // 10)
    jitter_y = rng.randint(-canvas_size // 10, canvas_size // 10)
    position = (
        (canvas_size - text_width) // 2 - bbox[0] + jitter_x,
        (canvas_size - text_height) // 2 - bbox[1] + jitter_y,
    )

    draw.text(
        position,
        render_text,
        fill=255,
        font=font,
        stroke_width=stroke_width,
        stroke_fill=255,
    )

    rotation = rng.uniform(-18.0, 18.0)
    rotated = background.rotate(
        rotation,
        resample=Image.Resampling.BILINEAR,
        fillcolor=0,
    )

    return _apply_random_effects(rotated, rng)


def ensure_output_dirs(output_dir: Path, labels: Iterable[str]) -> None:
    for split in ("train", "val"):
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for label in labels:
            (split_dir / get_folder_name_for_label(label)).mkdir(parents=True, exist_ok=True)


def clear_selected_dirs(output_dir: Path, labels: Iterable[str]) -> None:
    for split in ("train", "val"):
        for label in labels:
            label_dir = output_dir / split / get_folder_name_for_label(label)
            if label_dir.exists():
                shutil.rmtree(label_dir)


def generate_split(
    labels: Sequence[str],
    split: str,
    count_per_label: int,
    output_dir: Path,
    font_paths: Sequence[Path],
    rng: random.Random,
    canvas_size: int,
    prefix: str,
) -> dict[str, int]:
    counts: dict[str, int] = {label: 0 for label in labels}

    for label in labels:
        label_dir = output_dir / split / get_folder_name_for_label(label)
        label_dir.mkdir(parents=True, exist_ok=True)
        safe_label = get_folder_name_for_label(label)

        for index in range(count_per_label):
            image = render_symbol_image(label, font_paths=font_paths, rng=rng, canvas_size=canvas_size)
            filename = f"{prefix}_{safe_label}_{index:04d}.png"
            image.save(label_dir / filename)
            counts[label] += 1

    return counts


def write_manifest(
    manifest_path: Path,
    *,
    labels: Sequence[str],
    train_counts: dict[str, int],
    val_counts: dict[str, int],
    font_paths: Sequence[Path],
    seed: int,
    canvas_size: int,
) -> None:
    manifest = {
        "labels": list(labels),
        "seed": seed,
        "canvas_size": canvas_size,
        "fonts": [str(path) for path in font_paths],
        "train_counts": train_counts,
        "val_counts": val_counts,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gera simbolos sinteticos para reforcar o dataset do MathSolverAI.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "data" / "symbols",
        help="Diretorio base de saida contendo train/ e val/.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=["all"],
        help="Lista de rotulos ou nomes de pasta. Aceita tambem: all, digits, operators.",
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=200,
        help="Quantidade de imagens sinteticas por classe no split de treino.",
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=40,
        help="Quantidade de imagens sinteticas por classe no split de validacao.",
    )
    parser.add_argument(
        "--canvas-size",
        type=int,
        default=96,
        help="Tamanho do canvas sintetico antes do preprocessamento do dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para reproducibilidade.",
    )
    parser.add_argument(
        "--font-dir",
        action="append",
        default=[],
        help="Diretorio adicional de fontes. Pode ser informado multiplas vezes.",
    )
    parser.add_argument(
        "--font-limit",
        type=int,
        default=12,
        help="Quantidade maxima de fontes carregadas para sorteio.",
    )
    parser.add_argument(
        "--prefix",
        default="synthetic",
        help="Prefixo dos arquivos gerados.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove as pastas das classes selecionadas antes de gerar novos exemplos.",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    labels = expand_requested_labels(args.labels)
    rng = random.Random(args.seed)
    font_paths = discover_font_paths(font_dirs=args.font_dir, limit=args.font_limit)

    if args.clean_output:
        clear_selected_dirs(args.output_dir, labels)

    ensure_output_dirs(args.output_dir, labels)
    train_counts = generate_split(
        labels=labels,
        split="train",
        count_per_label=args.train_count,
        output_dir=args.output_dir,
        font_paths=font_paths,
        rng=rng,
        canvas_size=args.canvas_size,
        prefix=args.prefix,
    )
    val_counts = generate_split(
        labels=labels,
        split="val",
        count_per_label=args.val_count,
        output_dir=args.output_dir,
        font_paths=font_paths,
        rng=rng,
        canvas_size=args.canvas_size,
        prefix=args.prefix,
    )

    manifest_path = args.output_dir / "synthetic_manifest.json"
    write_manifest(
        manifest_path,
        labels=labels,
        train_counts=train_counts,
        val_counts=val_counts,
        font_paths=font_paths,
        seed=args.seed,
        canvas_size=args.canvas_size,
    )

    print(f"[INFO] Labels gerados: {', '.join(labels)}")
    print(f"[INFO] Fontes utilizadas: {len(font_paths)}")
    print(f"[INFO] Manifesto salvo em: {manifest_path}")
    print("[SUCESSO] Geracao sintetica concluida.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
