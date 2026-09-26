package com.rsiprevention.app.camera

import android.content.Context
import android.os.Process
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
import com.rsiprevention.app.detection.YoloWorldDetector
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
 * frame -> [YoloWorldDetector] -> [InstrumentTracker].
 *
 * A **transmissão** ([FrameStreamer]/`StreamServer`) é independente do
 * monitoramento: [streamingEnabled] liga o envio de frames ao servidor HTTP,
 * enquanto [monitoringEnabled] liga a detecção/tracking na NPU. Com os dois
 * ligados, o mesmo bitmap "em pé" alimenta a NPU e o stream (uma conversão por
 * frame); sem monitoramento, nenhuma inferência roda e só o vídeo é enviado.
 *
 * A resolução da análise é fixada perto de 720p para dar nitidez ao stream.
 * A NPU continua recebendo 640×640 (o `Preprocessor` escala). Se a medição no
 * aparelho mostrar queda de fps, o degrau seguinte é reduzir esta resolução
 * para 960×540 (`ANALYSIS_TARGET_SIZE`, ver README).
 */
class CameraController(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val previewView: PreviewView,
    private val detector: YoloWorldDetector,
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
                val bitmap = ImageUtils.imageProxyToUprightBitmap(imageProxy)
                try {
                    val instruments: List<TrackedInstrument> = if (monitoringEnabled) {
                        val detections = detector.detect(bitmap)
                        tracker.update(detections, bitmap)
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
                } finally {
                    // O bitmap já foi consumido (detector copia os pixels para o
                    // tensor; tracker extrai os templates de aparência; o streamer
                    // faz a própria cópia) — recicla para não acumular memória
                    // nativa frame a frame.
                    bitmap.recycle()
                }
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Erro ao processar frame de câmera", t)
        } finally {
            imageProxy.close()
        }
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

        /** Tempo máximo de espera do frame em andamento no shutdown (frames levam dezenas de ms). */
        private const val SHUTDOWN_TIMEOUT_SECONDS = 2L

        /** Resolução alvo da análise (stream 720p); a NPU sempre recebe 640×640. */
        private val ANALYSIS_TARGET_SIZE = Size(1280, 720)
    }
}
