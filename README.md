# Monitoramento de instrumentos cirúrgicos por visão computacional embarcada

Aplicativo Android (Kotlin) para prevenir a retenção inadvertida de instrumentos cirúrgicos (RSI, do inglês *Retained Surgical Items*). Roda visão computacional inteiramente no aparelho: detecta e rastreia instrumentos sobre a mesa operatória, registra retiradas e devoluções e faz a checagem final dos que não voltaram.

A inferência usa a NPU Qualcomm Hexagon (QNN HTP), sem fallback para CPU: se a NPU não aceitar o modelo, o aplicativo falha de forma explícita, em vez de rodar na CPU sem avisar. Nenhuma imagem é enviada para servidores externos: o processamento roda no aparelho e a transmissão Wi-Fi, quando ligada, fica restrita à rede local.

Números medidos em bancada com um POCO F7, em 5,4 minutos de operação contínua: 27,0 quadros por segundo ponta a ponta e 34,7 ms de latência média por quadro (14,4 ms na NPU).

## Como funciona

```
CameraX (ImageAnalysis 960×720, KEEP_ONLY_LATEST)
      │  quadro → bitmap "em pé" → letterbox 640×640 (aspecto preservado, padding cinza)
      ▼
Detector na NPU (detector_int8.onnx, A16W8)
      │  saídas: boxes [1,4,8400] + scores [1,5,8400]
      ▼
Pós-processamento (Kotlin)
      │  maior score entre as 5 classes → classe única, filtro de confiança e NMS
      ▼
Rastreador (Kotlin)
      │  associação global, modelo de movimento e template visual
      │  confirmação temporal e ciclo de vida (retirado / devolvido)
      ▼
Overlay (caixas), lista de instrumentos e checagem final
```

O rastreamento não depende de um único quadro. Um candidato só vira instrumento confirmado depois de aparecer em 4 dos últimos 8 quadros, com pico de confiança ≥ 0,55, mediana ≥ 0,40 e 3 segundos acumulados de confiança ≥ 0,65. Quadros de confiança baixa ou ausentes descontam esse tempo acumulado, então um pico isolado não confirma nada. Detecções curtas são descartadas em silêncio, sem entrar na lista nem gerar evento.

### Classe única

O modelo bruto distingue cinco instrumentos (`clamp`, `needle_holder`, `scalpel`, `shear`, `tweezer`), mas para a prevenção de retenção a subclasse não importa, e a classificação fina errava com frequência (tesoura e pinça, por exemplo). O pós-processamento colapsa as cinco pontuações na maior delas e emite tudo como "Instrumento Cirúrgico". A classe bruta vencedora é preservada internamente e usada na associação entre quadros. O NMS roda antes do colapso, com a classe bruta de cada âncora, para não fundir dois instrumentos diferentes que se sobrepõem.

### Limiares

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
| Retirada | `REMOVE_AFTER_MISSED_MS` / `MIN_ON_TABLE_MS_FOR_REMOVAL` | 1,2 s / 1,2 s |
| Sanidade da NPU (startup) | `ModelSelfTest.MIN_CONFIDENCE` | 0,04 |

### Instrumentação

O aplicativo mede o próprio desempenho e escreve no logcat:

| Tag | Conteúdo | Cadência |
|---|---|---|
| `DetectStats` | fps e tempo por etapa (decodificação, pré-processamento, inferência, pós-processamento, rastreamento), detecções por quadro e confiança máxima | 5 s |
| `StreamStats` | fps de captura e de publicação, frames descartados, tempo de encode e qualidade JPEG | 5 s |
| `Tracker` / `NpuLog` | confirmação, retirada e devolução de instrumentos, além da ativação da NPU | por evento |

A tela principal tem um painel oculto por padrão com o log da ativação da NPU (caminho das bibliotecas, escolha do backend, compilação ou carregamento do contexto e verificação de sanidade).

## Alvo

| Item | Valor |
|---|---|
| Aparelho | POCO F7 (Xiaomi `onyx`, modelo `25053PC47G`) |
| SoC | Snapdragon 8s Gen 4 (SM8735) |
| NPU | Qualcomm Hexagon HTP V73 |
| Android | 16 (HyperOS 3), `minSdk 31`, `targetSdk 34` |
| Runtime | ONNX Runtime Android + QNN EP (`com.microsoft.onnxruntime:onnxruntime-android-qnn:1.29.0`) |

## Modelo e quantização

- Base: [joonhaim/surgical-tool-recognition-yolo26s](https://huggingface.co/joonhaim/surgical-tool-recognition-yolo26s) (YOLO26s), exportado sem NMS para `best_fixed.onnx`.
- Quantização A16W8 (ativações de 16 bits, pesos de 8 bits), com grafo QDQ e calibração MinMax sobre quadros reais de bancada e imagens de referência. O conjunto de calibração é montado por `_build_calibration.py`.
- A calibração usa a mesma geometria de entrada da inferência: um asset calibrado com stretch perdia confiança ao rodar com letterbox (0,85 para 0,54 em uma imagem de teste). A comparação está em `_benchmark_preprocess.py`.
- Entrada em letterbox 4:3: o quadro retrato (análise 960×720, em pé 720×960) vira conteúdo de 480×640 centralizado, sem distorcer o aspecto. O stretch anterior distorcia até 1,78× e deixava as caixas com aspecto errado.
- O grafo QDQ é portátil e compilado pela QNN no próprio aparelho na primeira execução. O contexto fica em cache em `files/models/detector_int8_ctx.onnx`.
- As saídas são separadas em `boxes` e `scores` porque, em um tensor único `[1,9,8400]`, caixas e pontuações eram requantizadas com uma escala só (dominada pelas caixas, na casa de 640 px) e as confianças zeravam.
- Quantização A16W8 em vez de INT8: com ativações de 8 bits os logits da cabeça de classes se degradavam (bias próximo de -6, somas de 128 termos) e o runtime da QNN divergia numericamente.

## Transmissão Wi-Fi

Sem instalar nada no outro aparelho: o celular sobe um servidor HTTP embutido e qualquer Android ou iPhone na mesma rede abre a página no navegador.

1. Ative o hotspot no POCO F7.
2. Ligue "Transmitir (Wi-Fi)" na tela principal.
3. No outro aparelho, conecte-se ao hotspot e abra a URL exibida (por exemplo, `http://192.168.43.1:8080`). Toque no status para mostrar o QR code.

| Aspecto | Implementação |
|---|---|
| Rotas | `/` (página), `/stream` (MJPEG), `/frame.jpg` (último frame) |
| Android/Chrome | MJPEG nativo (`multipart/x-mixed-replace`) |
| iPhone/Safari | Não suporta MJPEG, então a página detecta iOS e passa a buscar `/frame.jpg` periodicamente |
| Vídeo | 960×720 (4:3), 30 fps alvo, JPEG qualidade 70 (cai para 50 sob carga) |
| Limites | Até 4 espectadores MJPEG. O app precisa ficar em primeiro plano e não há autenticação (rede local) |

O encode roda em thread própria (`rsi-stream`, prioridade FOREGROUND), com descarte de quadros antigos para a latência não acumular e `TCP_NODELAY` nos sockets.

## Estrutura do repositório

```
android/app/src/main/java/com/rsiprevention/app/
├── MainActivity.kt                 # tela única: preview, overlay, lista e checagem final
├── camera/                         # CameraX e conversão de frame (bitmap reutilizado)
├── detection/
│   ├── Preprocessor.kt             # bitmap → tensor float32 NCHW + pixels do modelo
│   ├── InstrumentDetector.kt       # sessão ONNX, tempo por etapa
│   ├── Postprocessor.kt            # filtro, classe única e NMS
│   ├── ModelSelfTest.kt            # sanidade da NPU na inicialização
│   ├── Letterbox.kt                # transformação frame ↔ modelo
│   └── Box.kt / BoxMath.kt / Detection.kt / DetectionFrame.kt / DetectionResult.kt / ModelFrame.kt
├── ort/                            # ORT e QNN: sessões, cache de contexto, backend estrito
├── tracking/                       # InstrumentTracker, TrackedInstrument, Appearance, InstrumentStatus
├── perf/PipelineStats.kt           # tempos por etapa (log DetectStats)
├── stream/                         # servidor HTTP, encode JPEG, slot de último frame, QR
├── ui/                             # OverlayView e diálogo do QR code
└── vocabulary/InstrumentVocabulary.kt

android/app/src/test/               # 54 testes JVM
_build_calibration.py               # monta o conjunto de calibração na geometria do app
_quantize_qnn.py                    # quantização A16W8
_benchmark_preprocess.py            # compara stretch e letterbox
_fix_output_split.py                # separa boxes/scores e valida o asset
_test_*.jpg                         # imagens para validação do detector
LICENSE                             # MIT
```

## Build e testes

Requisitos: JDK 17 ou superior e Android SDK (platform 34).

```bash
cd android
./gradlew testDebugUnitTest     # 54 testes JVM
./gradlew assembleDebug         # gera app/build/outputs/apk/debug/app-debug.apk
```

No Windows (PowerShell):

```powershell
$env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
cd android
.\gradlew testDebugUnitTest
.\gradlew assembleDebug
```

## Deploy no aparelho

Com "Instalar via USB" habilitado nas opções de desenvolvedor, `adb install -r` funciona no POCO F7 (HyperOS 3). Sem esse ajuste o HyperOS bloqueia a instalação com `INSTALL_FAILED_USER_RESTRICTED`, e o caminho é enviar o APK e instalar manualmente:

```bash
adb install -r android/app/build/outputs/apk/debug/app-debug.apk

# alternativa manual
adb push android/app/build/outputs/apk/debug/app-debug.apk /sdcard/Download/rsi-app.apk
```

O contexto da QNN é recompilado automaticamente na primeira abertura depois de trocar o modelo, porque a assinatura dos assets muda.

## Calibração e validação do modelo

```bash
# 1) capturar frames reais da bancada (transmitir ligado, monitoramento desligado)
adb forward tcp:18080 tcp:8080
for i in $(seq 1 70)
do
  curl -s -o frames/frame_$i.jpg http://127.0.0.1:18080/frame.jpg
  sleep 0.3
done

# 2) montar o conjunto de calibração na geometria do app
python _build_calibration.py --mode letterbox --frames frames/ --frames-stride 3 \
    --frames-aspect 3:4 --out calib.h5

# 3) quantizar e validar em CPU
python _quantize_qnn.py --activations int16 --calib calib.h5 \
    --out models/_qnn_quant/letterbox_final --preprocess letterbox

# 4) instalar os assets e revalidar os bytes que vão no APK
cp models/_qnn_quant/letterbox_final/best_qnn_split.onnx android/app/src/main/assets/models/detector_int8.onnx
cp models/_qnn_quant/letterbox_final/model.data android/app/src/main/assets/models/model.data
python _fix_output_split.py --check --preprocess letterbox
```

## Testes automatizados

54 testes JVM, sem aparelho:

| Suíte | Testes | Cobre |
|---|---|---|
| `InstrumentTrackerTest` | 13 | falso positivo curto e persistente, certeza sustentada, identidade após oclusão, troca de classe sem duplicar, descarte silencioso, retirada, checagem final, caixas degeneradas, data e hora dos eventos |
| `PostprocessorTest` | 9 | filtro de confiança, classe única, NMS por classe bruta, limites de 640×640, confiança máxima na mesma passada, `rawClassId` |
| `AppearanceTest` | 5 | extração no espaço do modelo, similaridade entre regiões, caixa parcialmente fora do quadro |
| `LetterboxTest` | 5 | escala em retrato e paisagem, padding centralizado, conversão frame ↔ modelo |
| `PipelineStatsTest` | 5 | média por etapa, fps do intervalo, reinício da janela |
| `LatestFrameSlotTest`, `StreamQualityControllerTest`, `HttpAndMjpegTest`, `HotspotAddressTest` | 17 | descarte de frame antigo, qualidade JPEG adaptativa, parser HTTP, formato MJPEG, IP do hotspot |

## Limitações

- O aplicativo exige a NPU Hexagon. Em outro aparelho ou SoC o modelo pode não compilar, já que o contexto da QNN é específico do SoC, e a quantização foi validada no V73.
- A validação foi feita em bancada, sem cenário cirúrgico real e sem anotação de referência nos quadros, então não há precisão, revocação ou mAP. O desempenho é descrito por métricas de confiança e pelo comportamento do rastreamento.
- Oclusões e perdas momentâneas de detecção podem interromper o rastreamento de um instrumento por alguns segundos.
- Instrumentos fora das cinco classes de treino, como afastadores, podem gerar caixas parciais com risco de registro duplicado do mesmo objeto.
- Não houve medição de consumo de energia nem de temperatura. A vantagem energética da NPU em relação a GPUs de servidor é argumento da literatura e da arquitetura local, não resultado medido aqui.
- O app não distingue o tipo de instrumento, por decisão de projeto.

## Reprodutibilidade

- Modelo embarcado (conferência por SHA-256):
  - `detector_int8.onnx`: `eb023290ff7848a3f63e9b7ac2c4fa4ed97d01a0b63e1ccc02667d0e23f8a855`
  - `model.data`: `b31541b23071f95291993d255bdc7778a329925f78d74d2f4bb791b076d00ae4`
- Validação rápida do asset em CPU: `python _fix_output_split.py --check --preprocess letterbox`.
- Os quadros de captura e os registros brutos de desempenho ficam fora do repositório. O modelo quantizado e as imagens de referência estão versionados.

## Licença

MIT. Livre para usar, modificar e redistribuir, mantendo os créditos. Ver [`LICENSE`](LICENSE).

## Aviso

Protótipo de pesquisa. Não é um dispositivo médico regulado e não substitui a contagem manual de instrumentos nem os protocolos de segurança do paciente. Uso clínico depende de validação prospectiva com aprovação de Comitê de Ética.
