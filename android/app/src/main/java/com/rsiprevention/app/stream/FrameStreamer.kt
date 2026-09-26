package com.rsiprevention.app.stream

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import android.os.Process
import android.os.SystemClock
import android.util.Log
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong

/**
 * Codifica os frames da câmera em JPEG (com as caixas de detecção desenhadas)
 * e publica para o [StreamServer], visando [targetFps] no Snapdragon 8s Gen 4.
 *
 * Desenho de performance:
 *  - **Thread própria** (`rsi-stream`, prioridade FOREGROUND) — o encode nunca
 *    bloqueia a thread de inferência da NPU; o frame é oferecido por cópia.
 *  - **Sempre o frame mais recente** ([LatestFrameSlot]): sob carga, frames
 *    velhos são descartados em vez de enfileirados (latência não acumula).
 *  - **Sem alocação no hot path**: `Canvas`/`Paint`/`ByteArrayOutputStream`
 *    reutilizados; o array JPEG é compartilhado entre os clientes do servidor.
 *  - **Throttle** por orçamento de frame (`1000/targetFps` ms) e **qualidade
 *    adaptativa** ([StreamQualityController]) para segurar o fps.
 *  - Contadores (`StreamStats`, logcat) a cada 5 s: fps de captura, fps
 *    publicados, descartados e tempo de encode — base para calibrar o degrau
 *    (720p30 → 960×540 → detectar a cada N frames).
 *
 * As caixas chegam como [StreamOverlay] imutável, montadas na thread de
 * inferência — o streamer nunca lê o tracker.
 */
class FrameStreamer(
    private val boxColor: Int,
    private val onJpeg: (ByteArray) -> Unit,
    private val targetFps: Int = DEFAULT_TARGET_FPS
) {
    companion object {
        private const val TAG = "StreamStats"
        const val DEFAULT_TARGET_FPS = 30
        private const val START_QUALITY = 70
        private const val STATS_INTERVAL_MS = 5000L
        private const val MODEL_INPUT_SIZE = 640f
        private const val STROKE_FRACTION = 0.006f
        private const val TEXT_FRACTION = 0.035f
        private const val LABEL_PAD_FRACTION = 0.008f
    }

    private data class Frame(val bitmap: Bitmap, val overlays: List<StreamOverlay>)

    private val slot = LatestFrameSlot<Frame> { it.bitmap.recycle() }
    private val executor = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "rsi-stream").apply {
            Process.setThreadPriority(Process.THREAD_PRIORITY_FOREGROUND)
        }
    }

    private val quality = StreamQualityController(startQuality = START_QUALITY)
    private val jpegBuffer = ByteArrayOutputStream(256 * 1024)
    private val canvas = Canvas()
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        color = boxColor
    }
    private val labelBackgroundPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = boxColor
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        typeface = Typeface.DEFAULT_BOLD
    }

    private val frameIntervalMs = 1000L / targetFps.coerceAtLeast(1)
    private val offeredCount = AtomicLong()
    private val publishedCount = AtomicLong()

    @Volatile
    private var running = false

    private var lastPublishMs = 0L
    private var encodeMsEwma = 0f
    private var lastStatsMs = SystemClock.elapsedRealtime()
    private var lastOffered = 0L
    private var lastPublished = 0L

    /** Inicia a thread de encode. Idempotente. */
    fun start() {
        if (running) return
        running = true
        executor.execute { encodeLoop() }
    }

    /** Encerra a thread e descarta o frame pendente. Idempotente. */
    fun stop() {
        if (!running) return
        running = false
        slot.close()
        executor.shutdown()
    }

    /**
     * Oferece um frame recém-capturado (thread de inferência). Faz uma cópia
     * leve — o bitmap original é reciclado pelo chamador logo em seguida.
     * Não bloqueia: se a thread de encode estiver ocupada, o frame anterior
     * pendente é descartado.
     */
    fun offer(bitmap: Bitmap, overlays: List<StreamOverlay>) {
        if (!running) return
        // isMutable = true: o Canvas desenha as caixas direto na cópia — uma
        // cópia imutável não aceita desenho (origem do bug "sem caixas no stream").
        val copy = runCatching { bitmap.copy(Bitmap.Config.ARGB_8888, true) }.getOrNull() ?: return
        offeredCount.incrementAndGet()
        slot.offer(Frame(copy, overlays))
    }

    private fun encodeLoop() {
        while (running) {
            val frame = slot.take() ?: break
            try {
                publish(frame)
            } catch (t: Throwable) {
                Log.w(TAG, "Falha ao codificar/publicar frame: ${t.message}")
            } finally {
                frame.bitmap.recycle()
            }
            logStatsIfDue()
        }
    }

    private fun publish(frame: Frame) {
        val startedNs = System.nanoTime()
        // Uma falha no desenho não pode derrubar o frame: o vídeo sai sem as
        // caixas (e o erro fica no log), em vez de o stream congelar.
        runCatching { drawOverlays(frame.bitmap, frame.overlays) }
            .onFailure { Log.w(TAG, "Falha ao desenhar caixas no stream", it) }

        jpegBuffer.reset()
        frame.bitmap.compress(Bitmap.CompressFormat.JPEG, quality.quality, jpegBuffer)
        val encodeMs = ((System.nanoTime() - startedNs) / 1_000_000).toFloat()

        encodeMsEwma = if (encodeMsEwma == 0f) encodeMs else 0.8f * encodeMsEwma + 0.2f * encodeMs
        quality.onEncode(
            encodeMs = encodeMsEwma.toLong(),
            budgetMs = frameIntervalMs,
            nowMs = SystemClock.elapsedRealtime()
        )

        onJpeg(jpegBuffer.toByteArray())
        publishedCount.incrementAndGet()
        throttle()
    }

    /**
     * Mantém o ritmo do alvo de fps; o slot descarta sobras do lado do produtor.
     * Usa `elapsedRealtime` (monotônico) — um ajuste do relógio de parede não
     * pode "congelar" o stream com um sleep gigante.
     */
    private fun throttle() {
        val now = SystemClock.elapsedRealtime()
        val nextAllowed = lastPublishMs + frameIntervalMs
        lastPublishMs = now
        val waitMs = nextAllowed - now
        if (waitMs > 0) SystemClock.sleep(waitMs)
    }

    private fun drawOverlays(bitmap: Bitmap, overlays: List<StreamOverlay>) {
        if (overlays.isEmpty()) return

        val scaleX = bitmap.width / MODEL_INPUT_SIZE
        val scaleY = bitmap.height / MODEL_INPUT_SIZE
        val stroke = bitmap.width * STROKE_FRACTION
        val textSize = bitmap.width * TEXT_FRACTION
        val padding = bitmap.width * LABEL_PAD_FRACTION

        boxPaint.strokeWidth = stroke
        textPaint.textSize = textSize

        canvas.setBitmap(bitmap)
        try {
            for (overlay in overlays) {
                val box = overlay.box
                val left = box.left * scaleX
                val top = box.top * scaleY
                val right = box.right * scaleX
                val bottom = box.bottom * scaleY

                canvas.drawRect(left, top, right, bottom, boxPaint)
                drawLabel(overlay.label, left, top, bottom, padding)
            }
        } finally {
            canvas.setBitmap(null)
        }
    }

    private fun drawLabel(label: String, boxLeft: Float, boxTop: Float, boxBottom: Float, padding: Float) {
        val textWidth = textPaint.measureText(label)
        val textHeight = textPaint.textSize
        // Acima da caixa; se não couber (perto do topo), logo abaixo.
        val labelBottom =
            if (boxTop - textHeight - padding * 2 >= 0) boxTop else boxBottom + textHeight + padding * 2
        val labelTop = labelBottom - textHeight - padding * 2

        canvas.drawRect(boxLeft, labelTop, boxLeft + textWidth + padding * 2, labelBottom, labelBackgroundPaint)
        canvas.drawText(
            label,
            boxLeft + padding,
            labelBottom - padding - textPaint.descent(),
            textPaint
        )
    }

    private fun logStatsIfDue() {
        val now = SystemClock.elapsedRealtime()
        val elapsed = now - lastStatsMs
        if (elapsed < STATS_INTERVAL_MS) return

        val offered = offeredCount.get()
        val published = publishedCount.get()
        val windowOffered = offered - lastOffered
        val windowPublished = published - lastPublished
        lastOffered = offered
        lastPublished = published
        lastStatsMs = now

        Log.i(
            TAG,
            "captura=${windowOffered * 1000 / elapsed}fps " +
                "publicados=${windowPublished * 1000 / elapsed}fps " +
                "descartados=${windowOffered - windowPublished} " +
                "encode=${"%.1f".format(encodeMsEwma)}ms " +
                "qualidade=${quality.quality}"
        )
    }
}
