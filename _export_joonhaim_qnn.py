"""
Exporta o YOLO26s (best.onnx sem NMS) para INT8 QDQ via Qualcomm AI Hub.
O modelo resultante roda na NPU Hexagon V73 do POCO F7 (SM8735).
"""
import qai_hub as hub

def main():
    onnx_path = "models/joonhaim_yolo/best.onnx"
    
    print("1. Enviando modelo ONNX para o QAI Hub...")
    source_model = hub.upload_model(onnx_path)
    
    # Snapdragon 8 Gen 3 = Hexagon V73 (mesmo que o SM8735 do POCO F7)
    target_device = hub.Device("Samsung Galaxy S24")
    
    compile_options = (
        "--target_runtime precompiled_qnn_onnx "
        "--quantize_full_type int8 "
        "--quantize_io "
    )
    
    print("2. Submetendo compile job (quantização INT8)...")
    compile_job = hub.submit_compile_job(
        model=source_model,
        device=target_device,
        input_specs=dict(images=(1, 3, 640, 640)),
        options=compile_options
    )
    
    print(f"Job: {compile_job.url}")
    print("3. Aguardando compilação...")
    
    compiled_model = compile_job.get_target_model()
    
    print("4. Baixando modelo INT8...")
    download_path = compile_job.download_target_model("models/compiled")
    print(f"Modelo INT8 baixado para: {download_path}")

if __name__ == "__main__":
    main()
