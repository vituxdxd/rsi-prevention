package com.rsiprevention.app.tracking

import com.rsiprevention.app.detection.Box
import com.rsiprevention.app.detection.Detection
import com.rsiprevention.app.vocabulary.InstrumentVocabulary
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Testes de comportamento do [InstrumentTracker] com detecções sintéticas
 * (sem bitmap, portanto sem template visual). Cobrem os casos que geravam
 * fantasmas na lista: falso positivo curto, troca de classe, oclusão e
 * descarte silencioso.
 */
class InstrumentTrackerTest {

    private var nowMs = 0L
    private val frameMs = 66L

    private fun tick() {
        nowMs += frameMs
    }

    private fun det(classId: Int, confidence: Float, left: Float, top: Float, right: Float, bottom: Float) =
        Detection(
            classId = classId,
            className = InstrumentVocabulary.MODEL_RAW_CLASSES[classId],
            confidence = confidence,
            box = Box(left, top, right, bottom)
        )

    private fun shear(confidence: Float = 0.8f, left: Float = 100f) =
        det(3, confidence, left, 100f, left + 120f, 180f)

    private fun clamp(left: Float = 300f) =
        det(0, 0.8f, left, 100f, left + 120f, 180f)

    @Test
    fun `falso positivo curto nunca confirma`() {
        val tracker = InstrumentTracker()
        repeat(3) {
            tick()
            tracker.update(listOf(shear(confidence = 0.30f)), null, nowMs)
        }
        assertTrue(tracker.activeInstruments.isEmpty())
    }

    @Test
    fun `falso positivo persistente mas de baixa confianca nao confirma`() {
        val tracker = InstrumentTracker()
        repeat(12) {
            tick()
            tracker.update(listOf(shear(confidence = 0.45f)), null, nowMs)
        }
        assertTrue(tracker.activeInstruments.isEmpty())
    }

    @Test
    fun `objeto real confirma e mantem identidade apos oclusao`() {
        val tracker = InstrumentTracker()
        repeat(6) {
            tick()
            tracker.update(listOf(shear(left = 100f + it * 4f)), null, nowMs)
        }

        val confirmed = tracker.activeInstruments
        assertEquals(1, confirmed.size)
        assertEquals(InstrumentStatus.ON_TABLE, confirmed[0].status)
        val id = confirmed[0].id

        repeat(2) {
            tick()
            tracker.update(emptyList(), null, nowMs)
        }
        repeat(4) {
            tick()
            tracker.update(listOf(shear(left = 130f)), null, nowMs)
        }

        val after = tracker.activeInstruments
        assertEquals(1, after.size)
        assertEquals(id, after[0].id)
    }

    @Test
    fun `troca de classe nao duplica o track e estabiliza o rotulo`() {
        val tracker = InstrumentTracker()
        repeat(6) {
            tick()
            tracker.update(listOf(shear()), null, nowMs)
        }
        assertEquals(1, tracker.activeInstruments.size)
        assertEquals(3, tracker.activeInstruments[0].classId)

        repeat(8) {
            tick()
            tracker.update(listOf(det(4, 0.8f, 100f, 100f, 220f, 180f)), null, nowMs)
        }

        val after = tracker.activeInstruments
        assertEquals(1, after.size)
        assertEquals(4, after[0].classId)
    }

    @Test
    fun `fantasma confirmado de vida curta e descartado em silencio`() {
        var removalEvents = 0
        val tracker = InstrumentTracker(onInstrumentRemoved = { removalEvents++ })

        repeat(6) {
            tick()
            tracker.update(listOf(shear(confidence = 0.7f)), null, nowMs)
        }
        assertEquals(1, tracker.activeInstruments.size)

        repeat(25) {
            tick()
            tracker.update(emptyList(), null, nowMs)
        }

        assertTrue(tracker.activeInstruments.isEmpty())
        assertEquals(0, removalEvents)
    }

    @Test
    fun `objeto real de vida longa gera evento de retirada`() {
        var removalEvents = 0
        val tracker = InstrumentTracker(onInstrumentRemoved = { removalEvents++ })

        repeat(40) {
            tick()
            tracker.update(listOf(shear()), null, nowMs)
        }
        repeat(25) {
            tick()
            tracker.update(emptyList(), null, nowMs)
        }

        assertEquals(1, removalEvents)
        val removed = tracker.activeInstruments
        assertEquals(1, removed.size)
        assertEquals(InstrumentStatus.REMOVED, removed[0].status)
    }

    @Test
    fun `dois objetos proximos mantem identidades separadas`() {
        val tracker = InstrumentTracker()
        repeat(6) {
            tick()
            tracker.update(listOf(shear(left = 100f), clamp(left = 300f)), null, nowMs)
        }

        val before = tracker.activeInstruments.associate { it.classId to it.id }
        assertEquals(2, before.size)

        repeat(6) {
            tick()
            tracker.update(listOf(shear(left = 112f), clamp(left = 288f)), null, nowMs)
        }

        val after = tracker.activeInstruments.associate { it.classId to it.id }
        assertEquals(before[3], after[3])
        assertEquals(before[0], after[0])
    }

    @Test
    fun `checagem final marca removidos que nao voltaram`() {
        val tracker = InstrumentTracker()
        repeat(40) {
            tick()
            tracker.update(listOf(shear()), null, nowMs)
        }
        repeat(25) {
            tick()
            tracker.update(emptyList(), null, nowMs)
        }

        val notReturned = tracker.performFinalCheck()
        assertEquals(1, notReturned.size)
        assertEquals(InstrumentStatus.NOT_RETURNED, notReturned[0].status)
    }

    @Test
    fun `caixas degeneradas sao ignoradas`() {
        val tracker = InstrumentTracker()
        repeat(10) {
            tick()
            tracker.update(listOf(det(3, 0.9f, 100f, 100f, 101f, 101f)), null, nowMs)
        }
        assertTrue(tracker.activeInstruments.isEmpty())
    }
}
