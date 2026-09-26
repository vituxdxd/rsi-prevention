package com.rsiprevention.app.perf

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * Testes do [PipelineStats]: média por etapa, fps do intervalo, reinício da
 * janela após report e após pausa longa do monitoramento.
 */
class PipelineStatsTest {

    private fun PipelineStats.recordFrame(
        decodeMs: Long = 1,
        preprocessMs: Long = 2,
        scaleMs: Long = 1,
        pixelsMs: Long = 0,
        fillMs: Long = 1,
        inferenceMs: Long = 3,
        postprocessMs: Long = 4,
        trackerMs: Long = 5,
        detections: Int = 1,
        maxConfidence: Float = 0.5f,
        nowMs: Long
    ) = record(
        decodeNs = decodeMs * 1_000_000,
        preprocessNs = preprocessMs * 1_000_000,
        inferenceNs = inferenceMs * 1_000_000,
        postprocessNs = postprocessMs * 1_000_000,
        trackerNs = trackerMs * 1_000_000,
        detections = detections,
        maxConfidence = maxConfidence,
        nowMs = nowMs,
        preprocessScaleNs = scaleMs * 1_000_000,
        preprocessPixelsNs = pixelsMs * 1_000_000,
        preprocessFillNs = fillMs * 1_000_000
    )

    @Test
    fun `nada a reportar antes do intervalo`() {
        val stats = PipelineStats(intervalMs = 1000)
        stats.recordFrame(nowMs = 0)

        assertNull(stats.reportIfDue(999))
    }

    @Test
    fun `media por etapa e fps no fim do intervalo`() {
        val stats = PipelineStats(intervalMs = 1000)
        repeat(10) { index ->
            stats.recordFrame(detections = 2, maxConfidence = 0.6f, nowMs = index * 100L)
        }

        val snapshot = stats.reportIfDue(1000)!!

        assertEquals(10L, snapshot.frames)
        assertEquals(10f, snapshot.fps, 1e-3f)
        assertEquals(1f, snapshot.decodeMs, 1e-3f)
        assertEquals(2f, snapshot.preprocessMs, 1e-3f)
        assertEquals(1f, snapshot.preprocessScaleMs, 1e-3f)
        assertEquals(0f, snapshot.preprocessPixelsMs, 1e-3f)
        assertEquals(1f, snapshot.preprocessFillMs, 1e-3f)
        assertEquals(3f, snapshot.inferenceMs, 1e-3f)
        assertEquals(4f, snapshot.postprocessMs, 1e-3f)
        assertEquals(5f, snapshot.trackerMs, 1e-3f)
        assertEquals(15f, snapshot.totalMs, 1e-3f)
        assertEquals(2f, snapshot.detectionsPerFrame, 1e-3f)
        assertEquals(0.6f, snapshot.maxConfidence, 1e-3f)
    }

    @Test
    fun `depois de reportar a janela reinicia`() {
        val stats = PipelineStats(intervalMs = 1000)
        stats.recordFrame(nowMs = 0)
        stats.recordFrame(nowMs = 999)
        assertEquals(2L, stats.reportIfDue(1000)!!.frames)

        assertNull(stats.reportIfDue(1500))

        stats.recordFrame(nowMs = 1600)
        assertEquals(1L, stats.reportIfDue(2000)!!.frames)
    }

    @Test
    fun `pausa longa reinicia a janela em vez de diluir a media`() {
        val stats = PipelineStats(intervalMs = 1000)
        stats.recordFrame(decodeMs = 100, nowMs = 0)
        // Pausa de 60 s (monitoramento desligado): o próximo frame começa uma janela nova.
        stats.recordFrame(decodeMs = 1, nowMs = 60_000)

        val snapshot = stats.reportIfDue(61_000)!!

        assertEquals(1L, snapshot.frames)
        assertEquals(1f, snapshot.decodeMs, 1e-3f)
    }

    @Test
    fun `reset limpa a janela`() {
        val stats = PipelineStats(intervalMs = 1000)
        stats.recordFrame(nowMs = 0)

        stats.reset()

        assertNull(stats.reportIfDue(5000))
    }
}
