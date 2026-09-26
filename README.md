# RSI Prevention

App Android (Kotlin) para **prevenção de retenção inadvertida de instrumentos cirúrgicos** — o objetivo é monitorar a mesa cirúrgica, registrar retirada/devolução de instrumentos e fazer a conferência final, reduzindo o risco de corpo estranho retido.

A inferência de visão computacional roda **estritamente na NPU Qualcomm Hexagon** (QNN HTP), sem fallback para CPU: se a NPU não aceitar o modelo, o app falha explicitamente. Toda a UI é em **Português do Brasil**.

## Alvo

| Item | Valor |
|---|---|
| Aparelho | POCO F7 (Xiaomi `onyx`, modelo `25053PC47G`) |
| SoC | Snapdragon 8s Gen 4 (SM8735) |
| NPU | Qualcomm Hexagon HTP **V73** |
| Android | 16 (HyperOS 2), `minSdk 31`, `targetSdk 34` |
| Runtime | ONNX Runtime Android + QNN EP (`com.microsoft.onnxruntime:onnxruntime-android-qnn:1.29.0`) |

## Pipeline

```
CameraX (ImageAnalysis 960×720, KEEP_ONLY_LATEST)
      │  frame → Bitmap "em pé" → letterbox 640×640 (aspecto preservado,
      │  padding cinza 114; RGB, [0,1], NCHW)
      ▼
Detector QNN/NPU  detector_int8.onnx  (A16W8)
      │  saídas: boxes [1,4,8400] (cx,cy,w,h)  +  scores [1,5,8400]
      ▼
Postprocessor (CPU/Kotlin)
      │  máx. score entre as 5 classes brutas → classe única
      │  filtro de confiança + NMS por classe + NMS entre classes
      ▼
InstrumentTracker (CPU/Kotlin)
│  associação global + modelo de movimento + template visual
│  confirmação K-de-N + certeza sustentada + votação de classe + ciclo de vida REMOVED/RETURNED
      ▼
OverlayView (caixas) + lista de instrumentos + checagem final
```

### Modelo

- Base: [joonhaim/surgical-tool-recognition-yolo26s](https://huggingface.co/joonhaim/surgical-tool-recognition-yolo26s) (`best.pt`), exportada sem NMS para `best_fixed.onnx`.
- Quantização **A16W8** (ativações 16-bit, pesos 8-bit) via `get_qnn_qdq_config` + `quantize_static` (`_quantize_qnn.py`), calibração MinMax sobre um conjunto montado por `_build_calibration.py` (frames reais da bancada + imagens de teste, com augmentações).
- **Geometria de calibração = geometria de inferência**: o asset precisa ser calibrado com o mesmo pré-processamento do app. O asset anterior (stretch) perdia confiança ao rodar com letterbox (imagem de teatro: 0,85 → 0,54 em CPU) — medido por `_benchmark_preprocess.py`.
- Entrada **letterbox 4:3** (`Letterbox.kt`): o frame retrato (análise 960×720 → 720×960) vira conteúdo 480×640 centralizado, sem distorcer o aspecto. O *stretch* anterior distorcia até 1,78× e deixava a caixa com aspecto errado (tesoura Mayo: w/h 2,7 contra ~4 reais); com letterbox a caixa fica 3,9–4,0.
- Grafo **QDQ portátil** (`com.microsoft` QuantizeLinear/DequantizeLinear) compilado pelo QNN EP **no próprio aparelho** na primeira execução; o contexto fica em cache (`files/models/detector_int8_ctx.onnx`).
- Saída dividida em `boxes` e `scores` (`_fix_output_split.py`): num tensor único `[1,9,8400]` caixas e scores eram requantizados com uma escala só (dominada pelas caixas, ~640 px), zerando as confianças.
- Por que A16W8 e não INT8 puro: as ativações 8-bit destruíam os logits da cabeça de classes (bias ≈ -6 e somas de 128 termos) e o runtime QNN divergia numericamente.

### Classe única

O modelo bruto distingue 5 instrumentos (`clamp`, `needle_holder`, `scalpel`, `shear`, `tweezer`), mas para retenção de corpos estranhos a subclasse não importa — e a classificação fina errava muito (ex.: tesoura × pinça). O `Postprocessor` colapsa as 5 pontuações no **maior score** e emite tudo como **`instrumento_cirurgico` → "Instrumento Cirúrgico"**.

O NMS, porém, roda **antes** do colapso, com a **classe bruta** vencedora de cada âncora: assim dois instrumentos diferentes que se sobrepõem (IoU entre 0,45 e 0,55) não são fundidos numa caixa só — só duplicatas do mesmo instrumento bruto (IoU > 0,45) e caixas quase idênticas entre classes (IoU > 0,55) são suprimidas.

## Comportamento (limiares)

| Etapa | Constante | Valor |
|---|---|---|
| Detecção por frame | `Postprocessor.CONFIDENCE_THRESHOLD` | 0,25 |
| NMS | `NMS_IOU_THRESHOLD` / `CROSS_CLASS_NMS_IOU_THRESHOLD` | 0,45 / 0,55 |
| Novo candidato | `MIN_NEW_DETECTION_CONFIDENCE` | 0,35 |
| Confirmação | `CONFIRM_HITS` / `CONFIRM_WINDOW` | 4 / 8 frames |
| Falha máxima na janela | `MAX_MISS_RUN_IN_WINDOW` | 1 frame |
| Confiança pico | `MIN_PEAK_CONFIDENCE` | 0,55 |
| Confiança mediana | `MIN_MEDIAN_CONFIDENCE` | 0,40 |
| Certeza sustentada | `HIGH_CONFIDENCE_THRESHOLD` / `MIN_HIGH_CONFIDENCE_DURATION_MS` | 0,65 / 3 s |
| Retirada | `REMOVE_AFTER_MISSED_MS` + `MIN_ON_TABLE_MS_FOR_REMOVAL` | 1,2 s + 1,2 s |
| Sanidade da NPU (startup) | `ModelSelfTest.MIN_CONFIDENCE` | 0,04 |

A confirmação não é só K-de-N: o candidato acumula tempo apenas enquanto a confiança fica ≥ 0,65 (frames de confiança baixa ou ausência descontam o mesmo tempo). Assim um pico isolado de confiança — mesmo forte — não vira "Instrumento Cirúrgico"; são necessários ~3 s de certeza alta e persistente.

Track confirmado que some antes de 1,2 s (ou com pico baixo) é **descartado em silêncio** — não vira "retirado" nem entra na lista. Objetos reais geram eventos de retirada/devolução.

Na tela, a lista de instrumentos mostra a **data/hora do último evento** de cada track, e a barra superior tem um painel discreto (oculto por padrão, toque para expandir) com o **log da ativação da NPU Hexagon** (ADSP_LIBRARY_PATH, escolha do backend, compilação/carregamento do contexto e verificação de sanidade).

## Instrumentação (logcat)

| Tag | Conteúdo | Cadência |
|---|---|---|
| `DetectStats` | fps de detecção e ms por etapa (decode, pré-processamento com sub-etapas escala/getPixels/montagem NCHW, inferência, pós-processamento, tracker), detecções/frame e confiança máxima | 5 s (monitoramento ligado) |
| `StreamStats` | fps de captura/publicados, descartados, ms de encode e qualidade JPEG | 5 s (transmissão ligada) |
| `Tracker` / `NpuLog` | confirmação/retirada/devolução de instrumentos e ativação da NPU | por evento |

O template visual do tracker ([Appearance]) é extraído dos **pixels 640×640 do modelo** (o mesmo espaço das caixas) — antes era amostrado no bitmap "em pé" da câmera, o que apontava para a região errada sempre que o frame não era quadrado.

## Transmissão Wi-Fi (navegador)

Sem nenhum app no outro aparelho: o celular roda um **servidor HTTP embutido** e qualquer Android/iPhone na mesma rede abre a página no navegador.

1. Ative o **hotspot** no POCO F7 (Configurações do Android).
2. Ligue o switch **"Transmitir (Wi-Fi)"** na tela principal.
3. No outro aparelho, conecte-se ao hotspot e abra a URL exibida (ex.: `http://192.168.43.1:8080`) — ou toque no status para **mostrar o QR code** e aponte a câmera.

| Aspecto | Implementação |
|---|---|
| Rotas | `/` (página), `/stream` (MJPEG), `/frame.jpg` (último frame) |
| Android/Chrome | MJPEG nativo (`multipart/x-mixed-replace`) — push contínuo |
| iPhone/Safari | **Não suporta MJPEG**; a página detecta iOS e cai automaticamente para polling em `/frame.jpg` |
| Alvo | 960×720 (4:3, aspecto do sensor) @ 30 fps, JPEG qualidade 70 (adaptativa 70↔50 sob carga) |
| Performance | Encode em thread própria (`rsi-stream`, prioridade FOREGROUND); frames velhos descartados ([LatestFrameSlot]); `TCP_NODELAY`; sem alocação por frame no hot path |
| Diagnóstico | Logcat `StreamStats` a cada 5 s: fps de captura, fps publicados, descartados, ms de encode e qualidade |
| Limites | Até 4 espectadores MJPEG; o app precisa ficar em primeiro plano (a câmera para se a tela apagar — `KEEP_SCREEN_ON` é ligado durante a transmissão); sem autenticação (rede local) |

Ajuste fino: `CameraController.ANALYSIS_TARGET_SIZE` (960×720; o fallback da câmera escolhe o mais próximo) e `FrameStreamer.DEFAULT_TARGET_FPS` / `START_QUALITY`.

## Estrutura

```
android/app/src/main/java/com/rsiprevention/app/
├── MainActivity.kt                 # tela única: preview + overlay + lista + checagem final
├── camera/                         # CameraX + conversão de frame (FrameConverter, bitmap reutilizado)
├── detection/
│   ├── Preprocessor.kt             # Bitmap → tensor float32 NCHW [0,1] + pixels do modelo (buffers reutilizados)
│   ├── YoloWorldDetector.kt        # sessão ONNX → boxes/scores + tempo por etapa
│   ├── Postprocessor.kt            # filtro, classe única, NMS (uma passada)
│   ├── ModelSelfTest.kt            # sanidade da NPU (imagem de referência)
│   ├── Box.kt / BoxMath.kt / Detection.kt
│   ├── Letterbox.kt                # transformação frame ↔ modelo (aspecto preservado)
│   ├── DetectionFrame.kt / DetectionResult.kt / ModelFrame.kt
├── ort/                            # ORT + QNN (sessões, cache de contexto, backend estrito, log da NPU)
├── tracking/
│   ├── InstrumentTracker.kt        # associação, confirmação, ciclo de vida
│   ├── TrackedInstrument.kt        # estado do track (movimento, votos, template)
│   ├── Appearance.kt               # dHash + histograma RGB do crop (espaço do modelo)
│   └── InstrumentStatus.kt
├── perf/
│   └── PipelineStats.kt            # tempos por etapa do pipeline (log DetectStats)
├── stream/
│   ├── StreamServer.kt             # HTTP embutido: página, MJPEG e /frame.jpg (fallback iOS)
│   ├── FrameStreamer.kt            # encode JPEG (30 fps) com caixas, thread própria
│   ├── LatestFrameSlot.kt          # descarta frames velhos (latência não acumula)
│   ├── StreamQualityController.kt  # degradação/recuperação adaptativa da qualidade JPEG
│   ├── StreamOverlay.kt            # snapshot imutável das caixas (sem race com o tracker)
│   └── HotspotAddress.kt           # IP local do hotspot para montar a URL
├── ui/                             # OverlayView.kt + QrCodeDialog.kt (QR da URL)
└── vocabulary/InstrumentVocabulary.kt   # rótulos pt-BR (classe única)

android/app/src/test/               # testes JVM (tracker, pós-processador, letterbox, transmissão)
_build_calibration.py               # monta o conjunto de calibração (.h5) na geometria do app
_quantize_qnn.py                    # quantização A16W8 (get_qnn_qdq_config)
_benchmark_preprocess.py            # compara stretch × letterbox (float e asset quantizado)
_fix_output_split.py                # separa boxes/scores; valida o asset (--check, --preprocess)
_quantize_local.py                  # experimento INT8 MinMax (documentação)
_export_joonhaim_qnn.py             # tentativa QAI Hub (histórico, não usado)
_test_*.jpg                         # imagens para validação do detector
```

## Build

```powershell
$env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
cd android
.\gradlew testDebugUnitTest     # testes JVM (tracker, pós-processador, letterbox, transmissão)
.\gradlew assembleDebug         # APK em app/build/outputs/apk/debug/app-debug.apk
```

## Deploy

Com **"Instalar via USB"** habilitado nas opções de desenvolvedor, `adb install -r` funciona no POCO F7 (HyperOS 2) e é o caminho mais rápido. Sem ele, o HyperOS bloqueia (`INSTALL_FAILED_USER_RESTRICTED`) e a instalação é manual:

```powershell
adb -s <serial> install -r android\app\build\outputs\apk\debug\app-debug.apk
# alternativa manual:
adb -s <serial> push android\app\build\outputs\apk\debug\app-debug.apk /sdcard/Download/rsi-app.apk
```

O cache de contexto QNN (`detector_int8_ctx.onnx`) é recompilado automaticamente na primeira abertura após trocar o modelo (a assinatura `nome:tamanho` dos assets muda).

## Calibração e validação do modelo

```bash
# 1) frames reais da bancada (transmissão ligada, monitoramento desligado):
#    amostrar /frame.jpg via USB, ex.:  adb forward tcp:18080 tcp:8080
for i in $(seq 1 70); do curl -s -o frames/frame_$i.jpg http://127.0.0.1:18080/frame.jpg; sleep 0.3; done

# 2) conjunto de calibração na geometria do app (letterbox, recorte 3:4 dos frames):
python _build_calibration.py --mode letterbox --frames frames/ --frames-stride 3 \
    --frames-aspect 3:4 --out calib.h5

# 3) quantizar (A16W8) e validar em CPU:
python _quantize_qnn.py --activations int16 --calib calib.h5 \
    --out models/_qnn_quant/letterbox_final --preprocess letterbox

# 4) instalar os assets e revalidar os bytes que vão no APK:
cp models/_qnn_quant/letterbox_final/best_qnn_split.onnx android/app/src/main/assets/models/detector_int8.onnx
cp models/_qnn_quant/letterbox_final/model.data android/app/src/main/assets/models/model.data
python _fix_output_split.py --check --preprocess letterbox
```

## Testes

- `InstrumentTrackerTest` — 13 testes JVM: falso positivo curto/persistente, certeza sustentada (pico curto e alternância de confiança não confirmam; queda breve não descarta o acumulado), confirmação e identidade após oclusão, troca de classe sem duplicar, descarte silencioso de fantasma, evento de retirada, dois objetos próximos, checagem final, caixas degeneradas e registro de data/hora dos eventos.
- `PostprocessorTest` — 7 testes JVM: filtro de confiança, colapso em classe única, NMS por classe bruta (instrumentos diferentes sobrepostos não são fundidos), clamp das caixas ao espaço 640×640 e confiança máxima do frame na mesma passada.
- `AppearanceTest` — 5 testes JVM da extração no espaço do modelo: mesma região ⇒ mesmo template, regiões visualmente diferentes ⇒ similaridade menor, independência da posição no frame, caixa parcialmente fora não quebra.
- `LetterboxTest` — 5 testes JVM da geometria: escala uniforme em retrato/paisagem, padding centralizado, conversão frame ↔ modelo ida e volta.
- `PipelineStatsTest` — 5 testes JVM: média por etapa, fps do intervalo, reinício da janela após report e após pausa longa do monitoramento.
- `LatestFrameSlotTest`, `StreamQualityControllerTest`, `HttpAndMjpegTest`, `HotspotAddressTest` — 17 testes JVM da transmissão: descarte do frame antigo, adaptação de qualidade, parser HTTP, formato MJPEG e escolha do IP do hotspot.

## Limitações conhecidas

- **Sem fallback**: o app exige a NPU Hexagon; em outro aparelho/SoC o modelo pode não compilar (o binário de contexto QNN é específico do SoC, e a quantização foi validada no V73).
- A calibração atual do A16W8 usa frames reais de uma bancada improvisada + imagens de teste; a recalibração com a bancada definitiva é o próximo passo (mesmo fluxo da seção "Calibração e validação do modelo").
- O app não distingue o tipo de instrumento (decisão de projeto: classe única).
