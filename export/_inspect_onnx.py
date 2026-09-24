"""Diagnostico temporario: inspeciona um .onnx exportado (I/O, QDQ, quantizacao)."""

import sys

import onnx
from onnx import TensorProto

DTYPE_NAMES = {
    TensorProto.FLOAT: "float32",
    TensorProto.UINT8: "uint8",
    TensorProto.INT8: "int8",
    TensorProto.UINT16: "uint16",
    TensorProto.INT16: "int16",
    TensorProto.INT32: "int32",
    TensorProto.INT64: "int64",
    TensorProto.FLOAT16: "float16",
    TensorProto.BOOL: "bool",
}


def tensor_type(value_info):
    tt = value_info.type.tensor_type
    dims = []
    for d in tt.shape.dim:
        dims.append(d.dim_value if d.HasField("dim_value") else (d.dim_param or "?"))
    return DTYPE_NAMES.get(tt.elem_type, str(tt.elem_type)), dims


def find_qdq_params(graph, tensor_name, initializers):
    """Procura QuantizeLinear/DequantizeLinear que toca `tensor_name` e extrai scale/zp."""
    results = []
    for node in graph.node:
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear"):
            continue
        if tensor_name not in list(node.input):
            continue
        scale_name = node.input[1] if len(node.input) > 1 else None
        zp_name = node.input[2] if len(node.input) > 2 else None
        scale = initializers.get(scale_name)
        zp = initializers.get(zp_name)
        axis = None
        for attr in node.attribute:
            if attr.name == "axis":
                axis = attr.i
        results.append(
            (
                node.op_type,
                "output=" + node.output[0],
                f"scale={scale}",
                f"zero_point={zp}",
                f"axis={axis}",
            )
        )
    return results


def main(path):
    model = onnx.load(path, load_external_data=True)
    print(f"arquivo: {path}")
    print(f"ir_version: {model.ir_version}")
    print(f"opset: {[(o.domain or 'ai.onnx', o.version) for o in model.opset_import]}")
    print(f"stage: {model.graph.name}")
    print()

    initializers = {}
    for init in model.graph.initializer:
        import numpy as np

        try:
            initializers[init.name] = np.frombuffer(init.raw_data, dtype=np.float32).tolist()[:4] if init.raw_data else onnx.numpy_helper.to_array(init).flatten().tolist()[:4]
        except Exception:  # noqa: BLE001
            initializers[init.name] = "<erro>"

    print("== INPUTS ==")
    for i in model.graph.input:
        dtype, dims = tensor_type(i)
        print(f"  {i.name}: {dtype} {dims}")
        for row in find_qdq_params(model.graph, i.name, initializers):
            print(f"      QDQ: {row}")
    print()
    print("== OUTPUTS ==")
    for o in model.graph.output:
        dtype, dims = tensor_type(o)
        print(f"  {o.name}: {dtype} {dims}")
        for row in find_qdq_params(model.graph, o.name, initializers):
            print(f"      QDQ: {row}")
    print()

    op_counts = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    print("== OPS ==")
    for op, count in sorted(op_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {op}: {count}")
    print()
    print(f"initializers: {len(model.graph.initializer)}")
    total_bytes = sum(len(init.raw_data) for init in model.graph.initializer)
    print(f"tamanho total dos initializers (bytes): {total_bytes}")


if __name__ == "__main__":
    main(sys.argv[1])
