"""Monta um conjunto de calibração (h5 no formato lido por `_quantize_qnn.py`).

Formato de saída:
  data/0/sample_0, sample_1, ...   float32 [1, 3, 640, 640] em [0, 1]

O pré-processamento escolhido em `--mode` precisa ser **o mesmo que o app usa em
inferência**: o asset A16W8 atual foi calibrado com imagens 640x640 sem padding
(stretch); inferir com letterbox sobre uma calibração stretch coloca o modelo
fora de distribuição (medido em `_benchmark_preprocess.py`).

Uso:
  python _build_calibration.py --mode letterbox --out /tmp/calib_letterbox.h5
  python _build_calibration.py --mode stretch --out calib.h5 --frames pasta/
  python _build_calibration.py --mode letterbox --out calib.h5 --frames pasta/ --no-augment
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGES = [
    ROOT / "android" / "app" / "src" / "main" / "assets" / "selftest_scissors.jpg",
    ROOT / "_test_mayo_surgical_scissors.jpg",
    ROOT / "_test_surgical_scissors_07.jpg",
    ROOT / "_test_instruments_theatre.jpg",
    ROOT / "_test_mayo_scissors.jpg",
    ROOT / "_test_bus.jpg",
    ROOT / "_test_zidane.jpg",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SIZE = 640
PAD = 114


def preprocess(img: Image.Image, mode: str) -> np.ndarray:
    img = img.convert("RGB")
    if mode == "stretch":
        prepared = img.resize((SIZE, SIZE), Image.BILINEAR)
    else:
        scale = SIZE / max(img.width, img.height)
        width = max(1, round(img.width * scale))
        height = max(1, round(img.height * scale))
        prepared = Image.new("RGB", (SIZE, SIZE), (PAD, PAD, PAD))
        prepared.paste(img.resize((width, height), Image.BILINEAR), ((SIZE - width) // 2, (SIZE - height) // 2))
    array = np.asarray(prepared, dtype=np.float32) / 255.0
    return array.transpose(2, 0, 1)[None]  # [1, 3, 640, 640]


def augmentations(img: Image.Image, enabled: bool) -> list[Image.Image]:
    if not enabled:
        return [img]
    variants = []
    for angle in (0, 90, 180, 270):
        rotated = img.rotate(angle, expand=True) if angle else img
        variants.append(rotated)
        variants.append(rotated.transpose(Image.FLIP_LEFT_RIGHT))
    return variants


def crop_aspect(img: Image.Image, spec: str) -> Image.Image:
    """Recorta o centro da imagem para o aspecto `W:H` (ex.: 3:4)."""
    target_w, target_h = (int(value) for value in spec.split(":"))
    target = target_w / target_h
    current = img.width / img.height
    if current > target:
        width = round(img.height * target)
        left = (img.width - width) // 2
        return img.crop((left, 0, left + width, img.height))
    height = round(img.width / target)
    top = (img.height - height) // 2
    return img.crop((0, top, img.width, top + height))


def collect_images(frames_dir: Path | None, extra: list[Path], frames_stride: int = 1) -> tuple[list[Path], set[Path]]:
    paths = [p for p in DEFAULT_IMAGES if p.is_file()]
    paths.extend(p for p in extra if p.is_file())
    frame_paths: set[Path] = set()
    if frames_dir is not None:
        found = sorted(p for p in frames_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        frame_paths = set(found[:: max(1, frames_stride)])
        paths.extend(sorted(frame_paths))
    return paths, frame_paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["stretch", "letterbox"], default="letterbox")
    parser.add_argument("--out", type=Path, required=True, help="caminho do .h5 de saída")
    parser.add_argument("--frames", type=Path, default=None, help="pasta com frames reais capturados")
    parser.add_argument("--frames-stride", type=int, default=1, help="usa 1 a cada N frames (evita amostras quase idênticas)")
    parser.add_argument(
        "--frames-aspect",
        default=None,
        help="recorta o centro dos frames para este aspecto (ex.: 3:4) antes do pré-processamento",
    )
    parser.add_argument("--extra", type=Path, nargs="*", default=[], help="imagens adicionais")
    parser.add_argument("--no-augment", action="store_true", help="não augmentar (rotações/flip)")
    args = parser.parse_args()

    images, frame_paths = collect_images(args.frames, args.extra, args.frames_stride)
    if not images:
        raise SystemExit("Nenhuma imagem de entrada encontrada.")

    # Escrita incremental: um conjunto real passa de 1 GB e não cabe confortavelmente na RAM.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with h5py.File(args.out, "w") as handle:
        group = handle.create_group("data").create_group("0")
        for path in images:
            with Image.open(path) as img:
                img = img.convert("RGB")
                if args.frames_aspect and path in frame_paths:
                    img = crop_aspect(img, args.frames_aspect)
                for variant in augmentations(img, enabled=not args.no_augment):
                    group.create_dataset(f"sample_{count}", data=preprocess(variant, args.mode))
                    count += 1

    print(f"calibração {args.mode}: {count} amostras de {len(images)} imagens -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
