"""Quantiza o best_fixed.onnx localmente (QDQ INT8, MinMax) e valida no CPU.

Motivação: o job de quantização do QAI Hub escolheu para os logits da cabeça de
classes a faixa [-36,2, 0,0] (corta todo logit positivo), enquanto a faixa real
medida nas 20 imagens de calibração é [-13,8, +1,7]. Este script refaz a
quantização com `quantize_static` (MinMax, sem percentil) para preservar os
positivos, e aplica o split boxes/scores (`_fix_output_split.py`).

Uso:
  python _quantize_local.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np
import onnx
import onnxruntime as ort
from onnx.external_data_helper import convert_model_from_external_data
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.calibrate import CalibrationDataReader

import _fix_output_split as split

ROOT = Path(__file__).resolve().parent
FLOAT_ONNX = ROOT / "models" / "joonhaim_yolo" / "best_fixed.onnx"
CALIB_H5 = ROOT / "models" / "_calib_download" / "dataset-d7x843459.h5"
OUT_DIR = ROOT / "models" / "_local_quant"


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
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    qdq = OUT_DIR / "best_local_qdq.onnx"
    print(f"== quantize_static MinMax: {FLOAT_ONNX} ==")
    quantize_static(
        model_input=str(FLOAT_ONNX),
        model_output=str(qdq),
        calibration_data_reader=CalibReader(CALIB_H5),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    print(f"  {qdq} ({qdq.stat().st_size:,} bytes)")
    for f in OUT_DIR.iterdir():
        if f.suffix == ".data":
            print(f"  {f.name} ({f.stat().st_size:,} bytes)")

    m = onnx.load(str(qdq))
    inits = {i.name: i for i in m.graph.initializer}
    from onnx import numpy_helper

    for init in m.graph.initializer:
        if not init.name.endswith("_scale"):
            continue
        base = init.name[: -len("_scale")]
        if not any(k in base for k in ("Concat_1", "Reshape_3", "Reshape_4", "Reshape_5", "cv3.2.2", "Sigmoid")):
            continue
        zp_name = base + "_zero_point"
        if zp_name in inits:
            s = float(numpy_helper.to_array(init).reshape(-1)[0])
            z = int(numpy_helper.to_array(inits[zp_name]).reshape(-1)[0])
            print(f"  encoding {base:55s} scale={s:.6f} zp={z} range=[{-z*s:.3f}, {(255-z)*s:.3f}]")

    print("\n== Aplicando split boxes/scores ==")
    model = onnx.load(str(qdq))
    model = split.split_output(model)
    onnx.checker.check_model(model)
    onnx.save_model(
        model,
        str(OUT_DIR / "best_local_split.onnx"),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.data",
        size_threshold=1024,
    )

    print("\n== Validação CPU ==")
    if not split.validate(OUT_DIR / "best_local_split.onnx"):
        return 1

    embedded = OUT_DIR / "best_local_split_embedded.onnx"
    mm = onnx.load(str(OUT_DIR / "best_local_split.onnx"))
    convert_model_from_external_data(mm)
    onnx.save_model(mm, str(embedded))
    print(f"\n  embedded: {embedded} ({embedded.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
