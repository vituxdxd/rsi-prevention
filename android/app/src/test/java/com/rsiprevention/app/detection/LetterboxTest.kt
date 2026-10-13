package com.rsiprevention.app.detection

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Testes da geometria letterbox: escala uniforme, padding centralizado e
 * conversão frame ↔ modelo ida e volta.
 */
class LetterboxTest {

    @Test
    fun `frame retrato 720x1280 usa escala uniforme e padding lateral`() {
        val transform = Letterbox.forFrame(720, 1280)

        assertEquals(0.5f, transform.scale, 1e-6f)
        assertEquals(140f, transform.padX, 1e-6f)
        assertEquals(0f, transform.padY, 1e-6f)
        // Conteúdo 360x640 dentro de 640x640.
        assertEquals(140f, transform.toModelX(0f), 1e-6f)
        assertEquals(500f, transform.toModelX(720f), 1e-6f)
        assertEquals(0f, transform.toModelY(0f), 1e-6f)
        assertEquals(640f, transform.toModelY(1280f), 1e-6f)
    }

    @Test
    fun `frame paisagem 1280x720 usa padding vertical`() {
        val transform = Letterbox.forFrame(1280, 720)

        assertEquals(0.5f, transform.scale, 1e-6f)
        assertEquals(0f, transform.padX, 1e-6f)
        assertEquals(140f, transform.padY, 1e-6f)
        assertEquals(640f, transform.toModelX(1280f), 1e-6f)
        assertEquals(500f, transform.toModelY(720f), 1e-6f)
    }

    @Test
    fun `frame quadrado nao tem padding nem escala`() {
        val transform = Letterbox.forFrame(640, 640)

        assertEquals(1f, transform.scale, 1e-6f)
        assertEquals(0f, transform.padX, 1e-6f)
        assertEquals(0f, transform.padY, 1e-6f)
    }

    @Test
    fun `conversao frame modelo ida e volta e estavel`() {
        val transform = Letterbox.forFrame(720, 1280)

        for (x in listOf(0f, 123.5f, 360f, 719.9f)) {
            assertEquals(x, transform.toFrameX(transform.toModelX(x)), 1e-3f)
        }
        for (y in listOf(0f, 250f, 900.25f, 1279.9f)) {
            assertEquals(y, transform.toFrameY(transform.toModelY(y)), 1e-3f)
        }
    }

    @Test
    fun `conteudo fica centralizado com padding igual nos dois lados`() {
        val transform = Letterbox.forFrame(720, 1280)

        val leftEdge = transform.toModelX(0f)
        val rightPadding = Letterbox.MODEL_SIZE - transform.toModelX(720f)
        assertEquals(leftEdge, rightPadding, 1e-6f)
        assertTrue(leftEdge > 0f)
    }
}
