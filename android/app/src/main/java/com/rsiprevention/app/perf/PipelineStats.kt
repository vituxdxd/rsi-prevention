package com.rsiprevention.app.perf

/**
 * Acumula o tempo de cada etapa do pipeline de detecção e devolve a **média do
 * intervalo** a cada [intervalMs] — insumo do log `DetectStats` (logcat) que
 * orienta o ajuste de performance deste projeto.
 *
 * Etapas medidas por frame (ver `CameraController`):
 *  1. decode — `ImageProxy` → bitmap "em pé" (`FrameConverter`)
 *  2. pré-processamento — bitmap → tensor NCHW + pixels do modelo (`Preprocessor`)
 *  3. inferência — `session.run` na NPU Hexagon
 *  4. pós-processamento — decodificação das saídas + NMS (`Postprocessor`)
 *  5. tracker — associação/confirmação (`InstrumentTracker`)
 *
 * Puro (sem Android, recebe `nowMs`) — testável em JVM. Thread-safe por
 * `@Synchronized` (hoje só a thread de inferência chama, mas o custo é nulo).
 */
class PipelineStats(private val intervalMs: Long = DEFAULT_INTERVAL_MS) {

    /** Média do intervalo encerrado. Tempos em ms; [fps] de frames processados. */
    data class Snapshot(
        val frames: Long,
        val fps: Float,
        val decodeMs: Float,
        val preprocessMs: Float,
        /** Sub-etapas do pré-processamento (escala, getPixels, montagem NCHW). */
        val preprocessScaleMs: Float,
        val preprocessPixelsMs: Float,
        val preprocessFillMs: Float,
        val inferenceMs: Float,
        val postprocessMs: Float,
        val trackerMs: Float,
        val totalMs: Float,
        val detectionsPerFrame: Float,
        val maxConfidence: Float
    )

    private var windowStartMs = 0L
    private var lastRecordMs = 0L
    private var initialized = false

    private var frames = 0L
    private var decodeSumNs = 0L
    private var preprocessSumNs = 0L
    private var preprocessScaleSumNs = 0L
    private var preprocessPixelsSumNs = 0L
    private var preprocessFillSumNs = 0L
    private var inferenceSumNs = 0L
    private var postprocessSumNs = 0L
    private var trackerSumNs = 0L
    private var detectionCount = 0L
    private var maxConfidenceSum = 0f

    @Synchronized
    fun record(
        decodeNs: Long,
        preprocessNs: Long,
        inferenceNs: Long,
        postprocessNs: Long,
        trackerNs: Long,
        detections: Int,
        maxConfidence: Float,
        nowMs: Long,
        preprocessScaleNs: Long = 0L,
        preprocessPixelsNs: Long = 0L,
        preprocessFillNs: Long = 0L
    ) {
        // Pausa longa no monitoramento (ex.: desligou/religou): a média não pode
        // ser diluída no intervalo parado — recomeça a janela.
        if (initialized && nowMs - lastRecordMs > intervalMs) resetWindow(nowMs)
        if (!initialized) {
            resetWindow(nowMs)
            initialized = true
        }

        frames++
        decodeSumNs += decodeNs
        preprocessSumNs += preprocessNs
        preprocessScaleSumNs += preprocessScaleNs
        preprocessPixelsSumNs += preprocessPixelsNs
        preprocessFillSumNs += preprocessFillNs
        inferenceSumNs += inferenceNs
        postprocessSumNs += postprocessNs
        trackerSumNs += trackerNs
        detectionCount += detections
        maxConfidenceSum += maxConfidence
        lastRecordMs = nowMs
    }

    /**
     * @return a média do intervalo se [intervalMs] venceu desde o primeiro
     *   frame registrado, reiniciando a janela; `null` caso contrário.
     */
    @Synchronized
    fun reportIfDue(nowMs: Long): Snapshot? {
        if (!initialized || frames == 0L) return null
        val elapsedMs = nowMs - windowStartMs
        if (elapsedMs < intervalMs) return null

        val frameCount = frames
        val snapshot = Snapshot(
            frames = frameCount,
            fps = frameCount * 1000f / elapsedMs,
            decodeMs = averageMs(decodeSumNs, frameCount),
            preprocessMs = averageMs(preprocessSumNs, frameCount),
            preprocessScaleMs = averageMs(preprocessScaleSumNs, frameCount),
            preprocessPixelsMs = averageMs(preprocessPixelsSumNs, frameCount),
            preprocessFillMs = averageMs(preprocessFillSumNs, frameCount),
            inferenceMs = averageMs(inferenceSumNs, frameCount),
            postprocessMs = averageMs(postprocessSumNs, frameCount),
            trackerMs = averageMs(trackerSumNs, frameCount),
            totalMs = averageMs(
                decodeSumNs + preprocessSumNs + inferenceSumNs + postprocessSumNs + trackerSumNs,
                frameCount
            ),
            detectionsPerFrame = detectionCount / frameCount.toFloat(),
            maxConfidence = maxConfidenceSum / frameCount
        )
        resetWindow(nowMs)
        return snapshot
    }

    @Synchronized
    fun reset() {
        resetWindow(0L)
        initialized = false
    }

    private fun resetWindow(startMs: Long) {
        windowStartMs = startMs
        lastRecordMs = startMs
        frames = 0L
        decodeSumNs = 0L
        preprocessSumNs = 0L
        preprocessScaleSumNs = 0L
        preprocessPixelsSumNs = 0L
        preprocessFillSumNs = 0L
        inferenceSumNs = 0L
        postprocessSumNs = 0L
        trackerSumNs = 0L
        detectionCount = 0L
        maxConfidenceSum = 0f
    }

    private fun averageMs(sumNs: Long, frameCount: Long): Float =
        if (frameCount == 0L) 0f else sumNs / frameCount / 1_000_000f

    companion object {
        const val DEFAULT_INTERVAL_MS = 5000L
    }
}
