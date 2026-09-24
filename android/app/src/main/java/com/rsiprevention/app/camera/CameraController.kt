package com.rsiprevention.app.camera

import android.content.Context
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.rsiprevention.app.detection.YoloWorldDetector
import com.rsiprevention.app.tracking.InstrumentTracker
import com.rsiprevention.app.tracking.TrackedInstrument
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Liga o `ImageAnalysis` do CameraX (`STRATEGY_KEEP_ONLY_LATEST`, doc de
 * deploy seção 6) a um executor dedicado de inferência (nunca
 * `Dispatchers.Main`/main thread — seção 9.4) que roda o pipeline completo:
 * frame -> [YoloWorldDetector] -> [InstrumentTracker].
 *
 * O preview da câmera roda continuamente assim que [start] é chamado (após a
 * permissão de câmera concedida); a detecção/tracking só processa frames
 * quando [monitoringEnabled] está ativo (botão "Iniciar monitoramento").
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

    private val inferenceExecutor: ExecutorService = Executors.newSingleThreadExecutor()
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
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
        analysis.setAnalyzer(inferenceExecutor) { imageProxy -> processFrame(imageProxy) }

        provider.unbindAll()
        provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
    }

    private fun processFrame(imageProxy: ImageProxy) {
        try {
            if (monitoringEnabled) {
                val bitmap = ImageUtils.imageProxyToUprightBitmap(imageProxy)
                val detections = detector.detect(bitmap)
                tracker.update(detections, bitmap)

                val instruments = tracker.activeInstruments
                val width = bitmap.width
                val height = bitmap.height
                mainExecutor.execute { onFrameProcessed(instruments, width, height) }
            }
        } catch (t: Throwable) {
            Log.e(TAG, "Erro ao processar frame de câmera", t)
        } finally {
            imageProxy.close()
        }
    }

    /** Chamar em `onDestroy`/fim de sessão: solta a câmera e encerra o executor de inferência. */
    fun shutdown() {
        cameraProvider?.unbindAll()
        inferenceExecutor.shutdown()
    }

    companion object {
        private const val TAG = "CameraController"
    }
}
