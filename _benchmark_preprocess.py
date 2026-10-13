"""Benchmark offline do pré-processamento: stretch (atual) x letterbox.

Motivação: o app hoje redimensiona o frame direto para 640x640 (stretch), o que
distorce o aspecto em até 1,78x no frame retrato 720x1280 da câmera. O treino
padrão do YOLO usa letterbox (aspecto preservado, padding cinza). Este script
mede o efeito nos dois modelos disponíveis:

  - float de referência: models/joonhaim_yolo/best_fixed.onnx
  - asset quantizado A16W8: android/app/src/main/assets/models/detector_int8.onnx

Métricas por imagem: classe/score vencedores, nº de âncoras acima do limiar do
app (0,25) e a caixa vencedora (aspecto w/h). Roda em duas modalidades:

  - "natural": a imagem de teste no aspecto original
  - "camera 720x1280": cover-crop retrato (o que a câmera do POCO F7 entrega)

Uso:
  python _benchmark_preprocess.py
  python _benchmark_preprocess.py --camera      # só a simulação retrato
  python _benchmark_preprocess.py --model int8  # só o asset quantizado
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent
SELFTEST = ROOT / "android" / "app" / "src" / "main" / "assets" / "selftest_scissors.jpg"
EXTRA_IMAGES = [
    ROOT / "_test_mayo_surgical_scissors.jpg",
    ROOT / "_test_surgical_scissors_07.jpg",
    ROOT / "_test_instruments_theatre.jpg",
    ROOT / "_test_mayo_scissors.jpg",
    # imagens negativas (sem instrumentos) podem ser adicionadas aqui,
    # por exemplo um quadro da propria bancada com o campo vazio
]
MODELS = {
    "float": ROOT / "models" / "joonhaim_yolo" / "best_fixed.onnx",
    "int8": ROOT / "android" / "app" / "src" / "main" / "assets" / "models" / "detector_int8.onnx",
}
INSTRUMENT_IMAGES = {
    SELFTEST.name,
    "_test_mayo_surgical_scissors.jpg",
    "_test_surgical_scissors_07.jpg",
    "_test_instruments_theatre.jpg",
    "_test_mayo_scissors.jpg",
}

SIZE = 640
PAD = 114  # padding cinza padrão do letterbox de treino do YOLO
CONFIDENCE_THRESHOLD = 0.25


def portrait_frame(img: Image.Image) -> Image.Image:
    """Simula o frame retrato da câmera girando a foto (sem corte nem perda)."""
    return img.rotate(90, expand=True) if img.width >= img.height else img


def _to_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def preprocess_stretch(img: Image.Image) -> np.ndarray:
    return _to_array(img.resize((SIZE, SIZE), Image.BILINEAR))


def preprocess_letterbox(img: Image.Image) -> np.ndarray:
    scale = SIZE / max(img.width, img.height)
    width = max(1, round(img.width * scale))
    height = max(1, round(img.height * scale))
    canvas = Image.new("RGB", (SIZE, SIZE), (PAD, PAD, PAD))
    canvas.paste(img.resize((width, height), Image.BILINEAR), ((SIZE - width) // 2, (SIZE - height) // 2))
    return _to_array(canvas)


VARIANTS = {"stretch": preprocess_stretch, "letterbox": preprocess_letterbox}


class Detector:
    def __init__(self, path: Path) -> None:
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.names = [output.name for output in self.session.get_outputs()]

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        inputs = image.transpose(2, 0, 1)[None]
        outputs = self.session.run(None, {"images": inputs})
        if set(self.names) == {"boxes", "scores"}:
            boxes = outputs[self.names.index("boxes")]
            scores = outputs[self.names.index("scores")]
        else:
            boxes, scores = outputs[0][:, :4, :], outputs[0][:, 4:, :]
        return boxes[0], scores[0]  # (4, 8400), (5, 8400)


def describe(boxes: np.ndarray, scores: np.ndarray) -> dict:
    per_class = scores.max(axis=1)
    top = int(np.argmax(per_class))
    anchor = int(np.argmax(scores[top]))
    box = boxes[:, anchor]
    return {
        "top": top,
        "score": float(per_class[top]),
        "anchors": int((scores.max(axis=0) > CONFIDENCE_THRESHOLD).sum()),
        "box": [float(v) for v in box],
        "ratio": float(box[2] / max(box[3], 1e-6)),
    }


def run_mode(model_name: str, detector: Detector, mode: str, camera: bool) -> None:
    print(f"\n== {model_name} | {mode} ==")
    print(f"{'imagem':38s} {'variante':10s} {'classe':>6s} {'score':>7s} {'>0,25':>6s}  caixa (cx,cy,w,h)          w/h")
    instrument_scores: dict[str, list[float]] = {name: [] for name in VARIANTS}
    negative_scores: dict[str, list[float]] = {name: [] for name in VARIANTS}

    for path in [SELFTEST, *EXTRA_IMAGES]:
        if not path.exists():
            continue
        image = Image.open(path).convert("RGB")
        if camera:
            image = portrait_frame(image)
        for variant_name, transform in VARIANTS.items():
            boxes, scores = detector(transform(image))
            info = describe(boxes, scores)
            cx, cy, w, h = info["box"]
            print(
                f"{path.name:38s} {variant_name:10s} {info['top']:6d} {info['score']:7.3f} {info['anchors']:6d}"
                f"  {cx:6.0f},{cy:6.0f},{w:6.0f},{h:6.0f}  {info['ratio']:5.2f}"
            )
            target = instrument_scores if path.name in INSTRUMENT_IMAGES else negative_scores
            target[variant_name].append(info["score"])

    print("-" * 100)
    for variant_name in VARIANTS:
        mean_instrument = np.mean(instrument_scores[variant_name]) if instrument_scores[variant_name] else 0.0
        mean_negative = np.mean(negative_scores[variant_name]) if negative_scores[variant_name] else 0.0
        print(f"  {variant_name:10s}: score medio instrumentos={mean_instrument:.3f} | negativas={mean_negative:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["all", *MODELS], default="all")
    parser.add_argument("--camera", action="store_true", help="só a simulação retrato 720x1280")
    args = parser.parse_args()

    model_names = MODELS if args.model == "all" else {args.model: MODELS[args.model]}
    for model_name, path in model_names.items():
        if not path.is_file():
            print(f"[{model_name}] ausente: {path}")
            continue
        detector = Detector(path)
        if args.camera:
            run_mode(model_name, detector, "camera 720x1280", camera=True)
        else:
            run_mode(model_name, detector, "natural", camera=False)
            run_mode(model_name, detector, "camera 720x1280", camera=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
