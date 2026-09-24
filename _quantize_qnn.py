"""Quantiza o detector com a config recomendada para QNN (`get_qnn_qdq_config`).

Motivação: a quantização 8-bit das ativações não preserva os logits positivos da
cabeça de classes (bias ~-6 e soma de 128 termos; ver
DIAGNOSTICO_CONFIANCA_ZERO.md). Ativações 16-bit (`QUInt16`, A16W8) reduzem o
erro de ativação o suficiente para o head.

Uso:
  python _quantize_qnn.py --activations int8
  python _quantize_qnn.py --activations int16
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
import onnx
from onnx import numpy_helper
from onnx.external_data_helper import convert_model_from_external_data
from onnxruntime.quantization import CalibrationMethod, QuantType, quantize
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config

import _fix_output_split as split

ROOT = Path(__file__).resolve().parent
FLOAT_ONNX = ROOT / "models" / "joonhaim_yolo" / "best_fixed.onnx"
CALIB_H5 = ROOT / "models" / "_calib_download" / "dataset-d7x843459.h5"
OUT_ROOT = ROOT / "models" / "_qnn_quant"


class CalibReader(CalibrationDataReader):
    def __init__(self, h5_path: Path):
        self.samples = []
        with h5py.File(h5_path, "r") as f:
            names = sorted(f["data/0"].keys(), key=lambda n: int(n.split("_")[1]))
            for name in names:
                self.samples.append(f[f"data/0/{name}"][()])
        self.index = 0

    def get_next(self):
        if self.index >= len(self.samples):
            return None
        sample = {"images": self.samples[self.index].astype(np.float32)}
        self.index += 1
        return sample

    def rewind(self):
        self.index = 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", choices=["int8", "int16"], default="int16")
    args = parser.parse_args()

    activation_type = QuantType.QUInt8 if args.activations == "int8" else QuantType.QUInt16
    out_dir = OUT_ROOT / args.activations
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    qdq = out_dir / "best_qnn_qdq.onnx"
    print(f"== get_qnn_qdq_config ({args.activations}) ==")
    config = get_qnn_qdq_config(
        model_input=str(FLOAT_ONNX),
        calibration_data_reader=CalibReader(CALIB_H5),
        calibrate_method=CalibrationMethod.MinMax,
        activation_type=activation_type,
        weight_type=QuantType.QInt8,
        per_channel=True,
        activation_symmetric=False,
    )
    quantize(
        model_input=str(FLOAT_ONNX),
        model_output=str(qdq),
        quant_config=config,
    )
    print(f"  {qdq} ({qdq.stat().st_size:,} bytes)")
    for extra in out_dir.iterdir():
        if extra.name != qdq.name:
            print(f"  {extra.name} ({extra.stat().st_size:,} bytes)")

    m = onnx.load(str(qdq))
    domains = sorted({n.domain for n in m.graph.node if n.domain})
    print(f"  domínios: {domains}")
    inits = {i.name: i for i in m.graph.initializer}
    for init in m.graph.initializer:
        if not init.name.endswith("_scale"):
            continue
        base = init.name[: -len("_scale")]
        if not any(k in base for k in ("Concat_1", "cv3.2.2", "Sigmoid_output")):
            continue
        arr = numpy_helper.to_array(init)
        if arr.size > 1:
            print(f"  encoding {base:50s} scale(per-channel) {arr.reshape(-1)[:4]}")
            continue
        zp_name = base + "_zero_point"
        z = int(numpy_helper.to_array(inits[zp_name]).reshape(-1)[0]) if zp_name in inits else 0
        s = float(arr.reshape(-1)[0])
        bits = 65535 if args.activations == "int16" else 255
        print(f"  encoding {base:50s} scale={s:.6f} zp={z} range=[{-z*s:.3f}, {(bits-z)*s:.3f}]")

    print("\n== Aplicando split boxes/scores ==")
    model = onnx.load(str(qdq))
    model = split.split_output(model)
    onnx.checker.check_model(model)
    onnx.save_model(
        model,
        str(out_dir / "best_qnn_split.onnx"),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.data",
        size_threshold=1024,
    )

    print("\n== Validação CPU ==")
    if not split.validate(out_dir / "best_qnn_split.onnx"):
        return 1

    embedded = out_dir / "best_qnn_split_embedded.onnx"
    mm = onnx.load(str(out_dir / "best_qnn_split.onnx"))
    convert_model_from_external_data(mm)
    onnx.save_model(mm, str(embedded))
    print(f"\n  embedded: {embedded} ({embedded.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
