"""Diagnostico temporario: inspeciona um modelo EPContext gerado pelo QNN EP do ORT."""

import sys

import onnx

DT = {
    onnx.TensorProto.FLOAT: "float32", onnx.TensorProto.UINT8: "uint8",
    onnx.TensorProto.INT8: "int8", onnx.TensorProto.INT32: "int32",
    onnx.TensorProto.INT64: "int64", onnx.TensorProto.STRING: "string",
}


def main(path):
    model = onnx.load(path, load_external_data=False)
    g = model.graph
    print(f"arquivo: {path}")
    print(f"ir_version={model.ir_version}  opset={[(o.domain or 'ai.onnx', o.version) for o in model.opset_import]}")
    print(f"producer={model.producer_name!r} {model.producer_version!r}")
    print()

    print("== nodes ==")
    for n in g.node:
        print(f"  op_type={n.op_type!r} domain={n.domain!r} name={n.name!r} inputs={len(n.input)} outputs={len(n.output)}")
        for a in n.attribute:
            kind = a.type
            size = ""
            if a.HasField("s"):
                size = f" s_len={len(a.s)}"
            elif a.HasField("i"):
                size = f" i={a.i}"
            elif a.HasField("t"):
                size = f" t_dims={list(a.t.dims)} t_type={DT.get(a.t.data_type, a.t.data_type)} t_raw_len={len(a.t.raw_data)}"
            print(f"      attr {a.name!r} (type={kind}){size}")
    print()

    print("== graph inputs ==")
    for i in g.input:
        tt = i.type.tensor_type
        print(f"  {i.name!r} dtype={DT.get(tt.elem_type, tt.elem_type)} dims={[d.dim_value or d.dim_param for d in tt.shape.dim]}")
    print("== graph outputs ==")
    for o in g.output:
        tt = o.type.tensor_type
        print(f"  {o.name!r} dtype={DT.get(tt.elem_type, tt.elem_type)} dims={[d.dim_value or d.dim_param for d in tt.shape.dim]}")
    print()
    print(f"initializers={len(g.initializer)}  value_info={len(g.value_info)}")
    total_ext = sum(1 for init in g.initializer if init.data_location == onnx.TensorProto.EXTERNAL)
    print(f"initializers com external data: {total_ext}")
    for init in g.initializer[:5]:
        if init.data_location == onnx.TensorProto.EXTERNAL:
            print(f"  {init.name!r} externo: {[(e.key, e.value) for e in init.external_data]}")
    md = {p.key: p.value for p in model.metadata_props}
    if md:
        print("== metadata_props ==")
        for k, v in md.items():
            print(f"  {k} = {str(v)[:200]}")


if __name__ == "__main__":
    main(sys.argv[1])
