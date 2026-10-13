package com.rsiprevention.app.stream

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Semântica do slot de "último valor": o consumidor sempre pega o mais
 * recente e os intermediários são descartados (com callback de liberação de
 * recursos, ex.: `Bitmap.recycle()`).
 */
class LatestFrameSlotTest {

    @Test
    fun `take devolve o item mais recente e descarta o anterior`() {
        val dropped = mutableListOf<String>()
        val slot = LatestFrameSlot<String>(dropped::add)

        slot.offer("a")
        slot.offer("b")

        assertEquals("b", slot.take())
        assertEquals(listOf("a"), dropped)
    }

    @Test
    fun `offer apos fechar descarta o item e take devolve null`() {
        val dropped = mutableListOf<String>()
        val slot = LatestFrameSlot<String>(dropped::add)

        slot.close()
        slot.offer("x")

        assertNull(slot.take())
        assertTrue(dropped.contains("x"))
    }

    @Test
    fun `close descarta o pendente e libera o take`() {
        val dropped = mutableListOf<String>()
        val slot = LatestFrameSlot<String>(dropped::add)

        slot.offer("a")
        slot.close()

        assertNull(slot.take())
        assertEquals(listOf("a"), dropped)
    }
}
