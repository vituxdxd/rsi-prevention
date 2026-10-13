"""Separa a saída do detector INT8 em `boxes` e `scores` com escalas QDQ próprias.

Problema (ver DIAGNOSTICO_CONFIANCA_ZERO.md):
o grafo termina em Concat(caixas 0..640 px, scores 0..1) -> QuantizeLinear
(escala ~2,5, zp 0) -> DequantizeLinear `output0`. Como a escala é única e
dominada pelas caixas, todo score < ~1,25 é arredondado para 0 no uint8 e a
NPU devolve confiança 0,000 — o self-test reprova o backend e o app não inicia.

Correção:
remover o Concat/QuantizeLinear/DequantizeLinear final e expor as duas partes
ANTES dele, cada uma com a sua escala própria (o DQ que já existia em cada
ramo):

  boxes  [1, 4, 8400] float32 — escala ~2,5  (pixels)
  scores [1, 5, 8400] float32 — escala 1/256 (probabilidades)

Base escolhida: job de quantização `j56873jng` (calibrado com imagens reais de
tesoura), que se mostrou melhor que o `j5wl7dkmp` (ruído) em todas as imagens de
teste medidas — ver relatório impresso por este script.

Uso:
  python _fix_output_split.py                 # corrige e instala nos assets
  python _fix_output_split.py --check         # valida o asset atual (sem editar)
  python _fix_output_split.py --source caminho/model.onnx   # outra base
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "android" / "app" / "src" / "main" / "assets" / "models"
ASSET_ONNX = ASSETS / "detector_int8.onnx"
ASSET_DATA = ASSETS / "model.data"
ASSET_DATA_STALE = ASSETS / "detector_int8.data"
PREVIEW = ROOT / "models" / "_output_split_preview"

DEFAULT_SOURCE = (
    ROOT
    / "models"
    / "qdq_v73"
    / "j56873jng"
    / "extracted"
    / "job_j56873jng_qdq_onnx"
    / "model.onnx"
)

SELFTEST = ROOT / "android" / "app" / "src" / "main" / "assets" / "selftest_scissors.jpg"
EXTRA_IMAGES = [
    ROOT / "_test_mayo_surgical_scissors.jpg",
    ROOT / "_test_surgical_scissors_07.jpg",
    ROOT / "_test_instruments_theatre.jpg",
    # imagens negativas (sem instrumentos) podem ser adicionadas aqui,
    # por exemplo um quadro da propria bancada com o campo vazio
]

NUM_BOXES = 4
NUM_CLASSES = 5
NUM_PREDICTIONS = 8400
SHEAR_CLASS = 3
LETTERBOX_PAD = 114


def producer(model: onnx.ModelProto, tensor_name: str):
    for node in model.graph.node:
        if tensor_name in node.output:
            return node
    return None


def scale_of(initializers: dict, node) -> float:
    scale = initializers.get(node.input[1])
    return float(numpy_helper.to_array(scale).reshape(-1)[0]) if scale is not None else float("nan")


def find_final_chain(model: onnx.ModelProto):
    """Devolve (concat, quantize, dequantize, dq_boxes, dq_scores)."""
    outputs = [o.name for o in model.graph.output]
    if len(outputs) != 1:
        raise SystemExit(f"Esperava 1 saída no grafo de origem, achei {outputs}")
    dq = producer(model, outputs[0])
    if dq is None or dq.op_type != "DequantizeLinear":
        raise SystemExit(f"Saída '{outputs[0]}' não é produzida por DequantizeLinear (achei {dq and dq.op_type})")
    q = producer(model, dq.input[0])
    if q is None or q.op_type != "QuantizeLinear":
        raise SystemExit(f"Entrada do DQ final não é QuantizeLinear (achei {q and q.op_type})")
    concat = producer(model, q.input[0])
    if concat is None or concat.op_type != "Concat" or len(concat.input) != 2:
        raise SystemExit(f"A entrada do Q final não é um Concat com 2 ramos (achei {concat and concat.op_type})")
    inits = {i.name: i for i in model.graph.initializer}
    branches = []
    for name in concat.input:
        dqb = producer(model, name)
        if dqb is None or dqb.op_type != "DequantizeLinear":
            raise SystemExit(f"Ramo '{name}' do Concat final não vem de um DequantizeLinear")
        qn = producer(model, dqb.input[0])
        upstream = producer(model, qn.input[0]) if qn is not None and qn.input else None
        is_scores = upstream is not None and upstream.op_type == "Sigmoid"
        branches.append((dqb, scale_of(inits, dqb), is_scores))
    scores_branch = [b for b in branches if b[2]]
    boxes_branch = [b for b in branches if not b[2]]
    if len(scores_branch) != 1 or len(boxes_branch) != 1:
        raise SystemExit(
            "Não consegui classificar os ramos do Concat final: "
            f"{[(b[0].output[0], b[1], b[2]) for b in branches]}"
        )
    (dq_scores, s_scores, _) = scores_branch[0]
    (dq_boxes, s_boxes, _) = boxes_branch[0]
    if not (s_scores < s_boxes):
        raise SystemExit(f"Escalas inesperadas nos ramos do Concat: scores={s_scores} boxes={s_boxes}")
    return concat, q, dq, dq_boxes, dq_scores, s_boxes, s_scores


def split_output(model: onnx.ModelProto) -> onnx.ModelProto:
    concat, q, dq, dq_boxes, dq_scores, s_boxes, s_scores = find_final_chain(model)
    removed = {id(concat), id(q), id(dq)}

    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name == dq_boxes.output[0]:
                node.input[i] = "boxes"
            elif name == dq_scores.output[0]:
                node.input[i] = "scores"
    dq_boxes.output[0] = "boxes"
    dq_scores.output[0] = "scores"

    kept_nodes = [n for n in model.graph.node if id(n) not in removed]
    model.graph.ClearField("node")
    model.graph.node.extend(kept_nodes)

    used = {name for n in model.graph.node for name in n.input}
    dropped_inits = [i.name for i in model.graph.initializer if i.name not in used]
    kept_inits = [i for i in model.graph.initializer if i.name in used]
    model.graph.ClearField("initializer")
    model.graph.initializer.extend(kept_inits)

    kept_vi = [v for v in model.graph.value_info if v.name not in {dq.output[0], q.output[0], concat.output[0], "boxes", "scores"}]
    model.graph.ClearField("value_info")
    model.graph.value_info.extend(kept_vi)

    model.graph.ClearField("output")
    model.graph.output.extend(
        [
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, NUM_BOXES, NUM_PREDICTIONS]),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, NUM_CLASSES, NUM_PREDICTIONS]),
        ]
    )

    print(f"  removidos: Concat '{concat.output[0]}' + Q '{q.output[0]}' + DQ '{dq.output[0]}'")
    print(f"  initializers removidos ({len(dropped_inits)}): {dropped_inits[:6]}{' ...' if len(dropped_inits) > 6 else ''}")
    print(f"  boxes  escala {s_boxes:.6f} | scores escala {s_scores:.6f}")
    return model


def run_cpu(model_path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def preprocess(image_path: Path, mode: str = "stretch") -> np.ndarray:
    from PIL import Image

    image = Image.open(str(image_path)).convert("RGB")
    if mode == "stretch":
        image = image.resize((640, 640), Image.BILINEAR)
        return (np.asarray(image, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]

    # letterbox: mesma geometria do app (aspecto preservado, padding cinza).
    scale = 640 / max(image.width, image.height)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    canvas = Image.new("RGB", (640, 640), (LETTERBOX_PAD, LETTERBOX_PAD, LETTERBOX_PAD))
    canvas.paste(image.resize((width, height), Image.BILINEAR), ((640 - width) // 2, (640 - height) // 2))
    return (np.asarray(canvas, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]


def validate(model_path: Path, mode: str = "stretch") -> bool:
    print(f"\n== Validação CPU: {model_path} (preprocess={mode}) ==")
    sess = run_cpu(model_path)
    names = [o.name for o in sess.get_outputs()]
    shapes = {o.name: o.shape for o in sess.get_outputs()}
    print(f"  saídas: {names} {shapes}")
    if set(names) != {"boxes", "scores"}:
        print("  FALHOU: esperava exatamente as saídas boxes/scores")
        return False
    if tuple(shapes["boxes"]) != (1, NUM_BOXES, NUM_PREDICTIONS) or tuple(shapes["scores"]) != (1, NUM_CLASSES, NUM_PREDICTIONS):
        print("  FALHOU: shapes inesperados")
        return False

    ok = True
    for image in [SELFTEST, *EXTRA_IMAGES]:
        if not image.exists():
            continue
        boxes, scores = sess.run(None, {"images": preprocess(image, mode)})
        per_class = [float(scores[0, c].max()) for c in range(NUM_CLASSES)]
        top = int(np.argmax(per_class))
        n_above = int((scores > 0.25).sum())
        box = boxes[0, :, int(np.argmax(scores[0, top]))]
        print(
            f"  {image.name:38s} top classe {top} score {per_class[top]:.4f} | "
            + " ".join(f"{v:.4f}" for v in per_class)
            + f" | >0.25 {n_above:4d} | caixa x={box[0]:.0f} y={box[1]:.0f} w={box[2]:.0f} h={box[3]:.0f}"
        )
        if image == SELFTEST:
            if top != SHEAR_CLASS or per_class[top] < 0.3:
                print("  FALHOU: self-test exige classe 3 (shear) com score >= 0.3")
                ok = False
            if boxes.min() < 0.0 or boxes.max() > 700.0:
                print(f"  FALHOU: caixas fora da faixa esperada [{boxes.min():.1f}, {boxes.max():.1f}]")
                ok = False
    return ok


def install_assets(preview_onnx: Path, preview_data: Path) -> None:
    print("\n== Instalando nos assets ==")
    shutil.copy2(preview_onnx, ASSET_ONNX)
    shutil.copy2(preview_data, ASSET_DATA)
    print(f"  {ASSET_ONNX} ({ASSET_ONNX.stat().st_size:,} bytes)")
    print(f"  {ASSET_DATA} ({ASSET_DATA.stat().st_size:,} bytes)")
    if ASSET_DATA_STALE.exists():
        ASSET_DATA_STALE.unlink()
        print(f"  removido duplicado obsoleto: {ASSET_DATA_STALE.name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--check", action="store_true", help="apenas valida o asset atual")
    parser.add_argument("--no-install", action="store_true", help="gera só o preview, sem copiar para os assets")
    parser.add_argument(
        "--preprocess",
        choices=["stretch", "letterbox"],
        default="stretch",
        help="geometria de entrada usada para validar (o app precisa usar a mesma)",
    )
    args = parser.parse_args()

    if args.check:
        return 0 if validate(ASSET_ONNX, args.preprocess) else 1

    if not args.source.is_file():
        raise SystemExit(f"Modelo de origem ausente: {args.source}")

    print(f"== Carregando {args.source} ==")
    model = onnx.load(str(args.source))
    print(f"  nós {len(model.graph.node)}, saída {[o.name for o in model.graph.output]}")

    print("\n== Separando a saída final ==")
    model = split_output(model)
    onnx.checker.check_model(model)
    print("  onnx.checker: OK")

    if PREVIEW.exists():
        shutil.rmtree(PREVIEW)
    PREVIEW.mkdir(parents=True)
    onnx.save_model(
        model,
        str(PREVIEW / "detector_int8.onnx"),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.data",
        size_threshold=1024,
    )
    print(f"  preview salvo em {PREVIEW}")

    if not validate(PREVIEW / "detector_int8.onnx", args.preprocess):
        return 1

    if args.no_install:
        print(f"\n--no-install: preview em {PREVIEW} (assets intactos).")
        return 0

    install_assets(PREVIEW / "detector_int8.onnx", PREVIEW / "model.data")
    print("\nPronto.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
