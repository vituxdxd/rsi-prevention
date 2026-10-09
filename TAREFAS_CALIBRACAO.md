# Tarefas de calibração e próximos passos

Última atualização: 09/10/2026.

Estado: **letterbox 4:3 + A16W8 recalibrado na bancada real** (assets commitados
em `8512eb6`; branch `feature/calibracao-bancada`). A limpeza do repositório
(venv quebrado, h5 intermediários, build do Gradle, docs antigos e experimento
de baixo contraste) foi feita em `bf7741b`.

## Resultados medidos no POCO F7

Sessão da bancada (28/09: 65 janelas de 5 s ≈ 5,4 min, transmissão + monitoramento):

| métrica | média | mín–máx |
|---|---|---|
| fps de detecção | **27,0** | 20,3–30,1 |
| decode (câmera → bitmap em pé) | 11,6 ms | 8,3–17,7 |
| pré-processamento | 7,2 ms | 5,5–10,7 |
| inferência (NPU) | 14,4 ms | 10,3–16,1 |
| pós-processamento | 1,2 ms | 0,9–1,8 |
| tracker | 0,34 ms | 0,1–1,4 |
| total | 34,7 ms | 29,6–46,6 |
| detecções/frame | 4,3 | 0,3–6,2 |
| confiança máxima | 0,84 | 0,13–0,95 |
| transmissão | 26,3 fps · encode 11,9 ms · qualidade 70 estável | — |

Comparação com o baseline (mesma câmera, antes das fases): **7–11 → 27 fps**;
pré-processamento **94 → 7 ms**; confiança saturada em 0,85 → 0,84 de média com
máximos de 0,95.

Validação CPU do asset instalado (`_fix_output_split.py --check --preprocess
letterbox`): tesoura 0,9518 · theatre 0,9578 · negativas 0,24 e 0,03.
No aparelho: sanidade **0,943**; contexto QNN compilado em 9,2 s.

## Bloco A — Calibração na bancada realista ✅ (28/09)

- [x] Capturar ~70 frames com só a transmissão ligada (guardados em
      `models/_calib_download/frames_bancada/`)
- [x] Montar a calibração letterbox 4:3 (`_build_calibration.py`, 248 amostras)
- [x] Quantizar A16W8 e validar em CPU (`_quantize_qnn.py`)
- [x] Critérios de aceite (CPU): tesoura 0,95 ✓ · negativas < 0,3 ✓ · w/h 3,95 ✓
- [x] Instalar assets, rebuild, `adb install -r`; sanidade 0,943 no aparelho ✓
- [x] Medição na bancada (5,4 min): fps 27 de média, pré 7,2 ms, confiança máx. 0,95 ✓
- [ ] Validação funcional com protocolo controlado (parado → retira → devolve):
      a medição livre mostrou 21 confirmações para ~7 instrumentos no campo —
      possível duplicação de tracks (ver Bloco E)

## Bloco B — Retune de limiares com dados reais

- [x] Coletar a distribuição de scores na medição da bancada (DetectStats + tracker)
- [ ] Revisar `CONFIDENCE_THRESHOLD` (0,25), `MIN_NEW_DETECTION_CONFIDENCE` (0,35),
      `MIN_PEAK_CONFIDENCE` (0,55), `MIN_MEDIAN_CONFIDENCE` (0,40) e
      `HIGH_CONFIDENCE_THRESHOLD` (0,65)
- [ ] Decidir se mantém o requisito de ~3 s de certeza sustentada para confirmar

## Bloco C — Performance (Fase 3, só se ainda precisar)

- [ ] Medir o que não está contabilizado (~7 ms/frame): cópia do streamer,
      dispatch para a UI e overhead do CameraX
- [ ] Avaliar `OUTPUT_IMAGE_FORMAT_RGBA_8888` + bitmap reutilizado (decode 11,6 ms)
- [ ] Considerar `getBufferRef()` (zero-copy) no Postprocessor apenas se o
      pós-processamento voltar a pesar (hoje ~1,2 ms)

## Bloco D — Validações pendentes / decisões

- [ ] A/B do metering AE/AF (revertido em `77f2140`): duas builds, comparar
      confiança máxima e fps
- [ ] Decidir sobre center-crop (revertido): perde ~25% do FOV; só reconsiderar
      com recalibração e se a bancada permitir
- [ ] Confirmar o FOV real da bancada (cobre a mesa inteira? sobra margem?)

## Bloco E — Tracking: duplicação em instrumentos fora das 5 classes

Achado da medição (28/09): afastadores, gancho e espelho — que **não** são
classes do treino (clamp, needle_holder, scalpel, shear, tweezer) — geram
**caixas parciais**. Duas caixas do mesmo objeto com IoU baixo sobrevivem ao NMS
e podem virar dois tracks confirmados: 21 confirmações para ~7 instrumentos e 6
tracks confirmados que sumiram em menos de 10 s.

- [ ] Confirmar com protocolo controlado (sem mexer na bancada) se os eventos
      são falsos
- [ ] Se confirmado: deduplicar no nível do track (caixas contidas/muito
      sobrepostas por vários segundos viram um único track) ou NMS mais
      agressivo para caixas parciais

## Bloco F — Conjunto de imagens de validação/calibração

- [ ] Substituir as negativas genéricas (`_test_bus.jpg`, `_test_zidane.jpg`)
      por negativas cirúrgicas (campo vazio, tecido sem instrumento, mão
      enluvada) e **re-quantizar** o asset para manter calibração ≡ inferência.
      As 4 imagens cirúrgicas (`_test_*`) ficam: são o painel de regressão e
      parte do conjunto de calibração.

## Observações

- A geometria de calibração **precisa** ser igual à de inferência (letterbox 4:3);
  misturar degrada as confianças — medido em `_benchmark_preprocess.py`.
- Fluxo completo de captura/calibração/validação: seção "Calibração e validação
  do modelo" do `README.md`.
- Materiais do paper (`referencias/`, `imagens/`, edital, template, `.docx`) são
  locais e ficam fora do versionamento (`.gitignore`).
