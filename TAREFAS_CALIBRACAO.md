# Tarefas de calibração e próximos passos

Roadmap do trabalho a partir da build atual (**letterbox 4:3 + A16W8 recalibrado**,
assets validados). Números de referência medidos no POCO F7 com `DetectStats`:

| métrica | baseline | Fase 1 (perf) | Fase 2 (letterbox 4:3) |
|---|---|---|---|
| fps de detecção | 7–11 | 16,6–19,8 | **25,1** |
| decode (câmera → bitmap em pé) | — | 15–16 ms | **8,5 ms** |
| pré-processamento | ~94 ms | 6,0 ms | 6,2 ms |
| inferência (NPU) | 12 ms | 11,5 ms | 12,1 ms |
| total medido | ~120 ms | ~33,5 ms | **28,2 ms** |
| confiança máxima | ~0,85 (saturada) | 0,85 | **0,86–0,93** |
| candidatos fantasma descartados | 20 | 92 | **0** (sessão medida) |

Referência de qualidade do asset atual em CPU (`_fix_output_split.py --check
--preprocess letterbox`): tesouras 0,88–0,95; teatro 0,9578; negativas 0,03–0,24.

## Bloco A — Calibração na bancada realista (prioridade)

- [ ] Montar a bancada definitiva (mesa, instrumentos, iluminação) e capturar ~70
      frames com **só a transmissão ligada** (monitoramento desligado, frames sem
      caixas): `adb forward tcp:18080 tcp:8080` + loop de
      `curl -s -o frames/frame_$i.jpg http://127.0.0.1:18080/frame.jpg`
- [ ] Montar a calibração letterbox na geometria 4:3 (recorte central dos frames):
      `python _build_calibration.py --mode letterbox --frames frames/
      --frames-stride 3 --frames-aspect 3:4 --out models/_calib_download/calib_letterbox_bancada.h5`
- [ ] Quantizar A16W8 e validar em CPU:
      `python _quantize_qnn.py --activations int16 --calib <h5>
      --out models/_qnn_quant/letterbox_bancada --preprocess letterbox`
- [ ] Critérios de aceite (CPU): selftest tesoura (classe 3) ≥ 0,5; negativas < 0,3;
      caixa da tesoura com w/h ≈ 3,5–4,5 (aspecto real do instrumento)
- [ ] Instalar assets (`detector_int8.onnx` + `model.data`), rebuild e
      `adb install -r`; confirmar sanidade ≥ 0,9 no aparelho
- [ ] Medição de 90 s no aparelho (transmissão + monitoramento): fps ≥ 25,
      pré ≤ 7 ms, confiança máxima ≥ 0,9 nos instrumentos, 0 fantasmas
- [ ] Validação funcional: instrumentos confirmam (8/8, certeza ~3 s),
      retirada/devolução registradas, checagem final correta

## Bloco B — Retune de limiares com dados reais

- [ ] Coletar a distribuição de scores por track durante a medição (reais vs blips)
- [ ] Revisar `CONFIDENCE_THRESHOLD` (0,25), `MIN_NEW_DETECTION_CONFIDENCE` (0,35),
      `MIN_PEAK_CONFIDENCE` (0,55), `MIN_MEDIAN_CONFIDENCE` (0,40) e
      `HIGH_CONFIDENCE_THRESHOLD` (0,65) com essa distribuição
- [ ] Decidir se mantém o requisito de ~3 s de certeza sustentada para confirmar

## Bloco C — Performance (Fase 3, só se ainda precisar)

- [ ] Medir o que não está contabilizado (~7 ms/frame): cópia do streamer,
      dispatch para a UI e overhead do CameraX
- [ ] Avaliar `OUTPUT_IMAGE_FORMAT_RGBA_8888` + bitmap reutilizado para eliminar
      `toBitmap()` + rotação por frame (decode de 8,5 ms)
- [ ] Considerar `getBufferRef()` (zero-copy) no Postprocessor apenas se o
      pós-processamento voltar a pesar (hoje 0,5–0,8 ms)

## Bloco D — Validações pendentes / decisões

- [ ] A/B do metering AE/AF (mudança revertida no commit `77f2140`): duas builds,
      comparar confiança máxima e fps
- [ ] Decidir sobre center-crop (revertido): perde ~25% do FOV; só reconsiderar
      com recalibração e se a bancada permitir
- [ ] Confirmar o FOV real da bancada (cobre a mesa inteira? sobra margem?)

## Observações

- A geometria de calibração **precisa** ser igual à de inferência (letterbox 4:3);
  misturar degrada as confianças — medido em `_benchmark_preprocess.py`.
- Fluxo completo de captura/calibração/validação: seção "Calibração e validação
  do modelo" do `README.md`.
