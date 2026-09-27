package com.rsiprevention.app.camera

import android.content.Context
import android.os.Process
import android.os.SystemClock
import android.util.Log
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.rsiprevention.app.detection.InstrumentDetector
import com.rsiprevention.app.perf.PipelineStats
import com.rsiprevention.app.stream.FrameStreamer
import com.rsiprevention.app.stream.StreamOverlay
import com.rsiprevention.app.tracking.InstrumentStatus
import com.rsiprevention.app.tracking.InstrumentTracker
import com.rsiprevention.app.tracking.TrackedInstrument
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * Liga o `ImageAnalysis` do CameraX (`STRATEGY_KEEP_ONLY_LATEST`, doc de
 * deploy seção 6) a um executor dedicado de inferência (nunca
 * `Dispatchers.Main`/main thread — seção 9.4) que roda o pipeline completo:
 * frame -> [InstrumentDetector] -> [InstrumentTracker].
 *
 * A **transmissão** ([FrameStreamer]/`StreamServer`) é independente do
 * monitoramento: [streamingEnabled] liga o envio de frames ao servidor HTTP,
 * enquanto [monitoringEnabled] liga a detecção/tracking na NPU. Com os dois
 * ligados, o mesmo bitmap "em pé" alimenta a NPU e o stream (uma conversão por
 * frame); sem monitoramento, nenhuma inferência roda e só o vídeo é enviado.
 *
 * A resolução da análise é **4:3 (960×720)** — o aspecto nativo do sensor. Em
 * retrato (720×960) o letterbox do `Preprocessor`/`Letterbox` aproveita melhor o
 * quadro (escala uniforme de 0,667; conteúdo 480×640) do que um quadro 16:9, que
 * resultaria em escala 0,5 (conteúdo 360×640), e ainda mostra mais campo da
 * mesa. A NPU continua recebendo 640×640.
 */
class CameraController(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: PreviewView,
    private val detector: InstrumentDetector,
    private val tracker: InstrumentTracker,
    private val onFrameProcessed: (instruments: List<TrackedInstrument>, frameWidth: Int, frameHeight: Int) -> Unit
) {
    @Volatile
    var monitoringEnabled: Boolean = false

    @Volatile
    var streamingEnabled: Boolean = false

    /** Streamer ativo (definido pela Activity quando a transmissão é ligada). */
    @Volatile
    var streamer: FrameStreamer? = null

    private val inferenceExecutor: ExecutorService = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "rsi-inference").apply {
            // Inferência é o caminho crítico da captura; o encode JPEG do stream
            // roda em thread própria (rsi-stream) para não competir por este loop.
            Process.setThreadPriority(Process.THREAD_PRIORITY_MORE_FAVORABLE)
        }
    }
    private val mainExecutor = ContextCompat.getMainExecutor(context)
    private var cameraProvider: ProcessCameraProvider? = null

    /** Conversão de frame com bitmap interno reutilizado (não reciclar o retorno). */
    private val frameConverter = FrameConverter()

    /** Estatísticas por etapa do pipeline (log `DetectStats`). */
    private val stats = PipelineStats()

    fun start() {
        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener(
            {
                val provider = providerFuture.get()
                cameraProvider = provider
                bindUseCases(provider)
            },
            mainExecutor
        )
    }

    private fun bindUseCases(provider: ProcessCameraProvider) {
        val preview = Preview.Builder().build().apply {
            setSurfaceProvider(previewView.surfaceProvider)
        }

        val analysis = ImageAnalysis.Builder()
            .setResolutionSelector(
                ResolutionSelector.Builder()
                    .setResolutionStrategy(
                        ResolutionStrategy(
                            ANALYSIS_TARGET_SIZE,
                            ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                        )
                    )
                    .build()
            )
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
        analysis.setAnalyzer(inferenceExecutor) { imageProxy -> processFrame(imageProxy) }

        provider.unbindAll()
        provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
    }

    private fun processFrame(imageProxy: ImageProxy) {
        try {
            val streaming = streamingEnabled
            if (monitoringEnabled || streaming) {
                val decodeStartNs = System.nanoTime()
                val bitmap = frameConverter.toUpright(imageProxy)
                val decodeNs = System.nanoTime() - decodeStartNs

                val instruments: List<TrackedInstrument> = if (monitoringEnabled) {
                    val detection = detector.detectFrame(bitmap)

                    val trackerStartNs = System.nanoTime()
                    tracker.update(detection.detections, detection.model)
                    val trackerNs = System.nanoTime() - trackerStartNs

                    stats.record(
                        decodeNs = decodeNs,
                        preprocessNs = detection.preprocessNs,
                        inferenceNs = detection.inferenceNs,
                        postprocessNs = detection.postprocessNs,
                        trackerNs = trackerNs,
                        detections = detection.detections.size,
                        maxConfidence = detection.maxConfidence,
                        nowMs = SystemClock.elapsedRealtime(),
                        preprocessScaleNs = detection.preprocessScaleNs,
                        preprocessPixelsNs = detection.preprocessPixelsNs,
                        preprocessFillNs = detection.preprocessFillNs
                    )
                    logStatsIfDue()

                    tracker.activeInstruments
                } else {
                    // Sem monitoramento não há detecção fresca: o stream sai
                    // sem caixas (evita mostrar posição antiga de instrumento).
                    emptyList()
                }

                if (streaming) {
                    streamer?.offer(bitmap, snapshotOverlays(instruments))
                }

                if (monitoringEnabled) {
                    val width = bitmap.width
                    val height = bitmap.height
                    mainExecutor.execute { onFrameProcessed(instruments, width, height) }
                }
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Erro ao processar frame de câmera", t)
        } finally {
            imageProxy.close()
        }
    }

    /**
     * Loga a média do intervalo (`DetectStats`) a cada
     * [PipelineStats.DEFAULT_INTERVAL_MS]: fps de detecção e ms por etapa. É a
     * base para decidir onde otimizar (pré-processamento, NPU, tracker).
     */
    private fun logStatsIfDue() {
        val snapshot = stats.reportIfDue(SystemClock.elapsedRealtime()) ?: return
        Log.i(
            TAG_STATS,
            "fps=${"%.1f".format(snapshot.fps)} " +
                "decode=${"%.1f".format(snapshot.decodeMs)}ms " +
                "pre=${"%.1f".format(snapshot.preprocessMs)}ms" +
                "(scale=${"%.1f".format(snapshot.preprocessScaleMs)} " +
                "px=${"%.1f".format(snapshot.preprocessPixelsMs)} " +
                "fill=${"%.1f".format(snapshot.preprocessFillMs)}) " +
                "infer=${"%.1f".format(snapshot.inferenceMs)}ms " +
                "post=${"%.1f".format(snapshot.postprocessMs)}ms " +
                "track=${"%.1f".format(snapshot.trackerMs)}ms " +
                "total=${"%.1f".format(snapshot.totalMs)}ms " +
                "dets=${"%.1f".format(snapshot.detectionsPerFrame)} " +
                "maxConf=${"%.2f".format(snapshot.maxConfidence)}"
        )
    }

    /**
     * Snapshot imutável das caixas visíveis para o streamer (montado aqui, na
     * thread de inferência, logo após o `tracker.update` — o streamer nunca lê
     * o estado mutável do tracker).
     */
    private fun snapshotOverlays(instruments: List<TrackedInstrument>): List<StreamOverlay> =
        instruments.mapNotNull { instrument ->
            when (instrument.status) {
                InstrumentStatus.ON_TABLE, InstrumentStatus.RETURNED ->
                    StreamOverlay(
                        box = instrument.box,
                        label = "${instrument.displayLabel} • ${(instrument.confidence * 100).toInt()}%"
                    )

                else -> null
            }
        }

    /**
     * Chamar em `onDestroy`/fim de sessão: solta a câmera e encerra o executor
     * de inferência, **esperando o frame em andamento terminar**.
     *
     * `shutdown()` sozinho não interrompe a tarefa já em execução: se a Activity
     * fechasse as sessões ONNX em seguida, um `OrtSession.run` ainda rodando na
     * NPU seria interrompido no meio — crash nativo (SIGSEGV em
     * `libonnxruntime.so`, observado na thread `rsi-inference`). O
     * [awaitTermination] garante que, ao retornar, nenhum frame está mais usando
     * a sessão; só então `modelSessions.close()` pode ser chamado.
     */
    fun shutdown() {
        cameraProvider?.unbindAll()
        cameraProvider = null
        inferenceExecutor.shutdown()
        runCatching { inferenceExecutor.awaitTermination(SHUTDOWN_TIMEOUT_SECONDS, TimeUnit.SECONDS) }
    }

    companion object {
        private const val TAG = "CameraController"
        private const val TAG_STATS = "DetectStats"

        /** Tempo máximo de espera do frame em andamento no shutdown (frames levam dezenas de ms). */
        private const val SHUTDOWN_TIMEOUT_SECONDS = 2L

        /** Resolução alvo da análise (4:3 nativo); a NPU sempre recebe 640×640. */
        private val ANALYSIS_TARGET_SIZE = Size(960, 720)
    }
}
