package com.rsiprevention.app.stream

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Degradação/recuperação adaptativa da qualidade JPEG: se o encode passa de
 * 70% do orçamento do frame, cai um degrau (no máximo 1×/s); se fica abaixo de
 * 40%, sobe — sempre dentro de [50, 70].
 */
class StreamQualityControllerTest {

    private val budgetMs = 33L

    @Test
    fun `qualidade cai quando o encode estoura o orcamento`() {
        val controller = StreamQualityController(startQuality = 70)

        assertEquals(60, controller.onEncode(encodeMs = 30, budgetMs = budgetMs, nowMs = 0))
        // Dentro do cooldown de 1 s: não muda de novo.
        assertEquals(60, controller.onEncode(encodeMs = 30, budgetMs = budgetMs, nowMs = 500))
        assertEquals(50, controller.onEncode(encodeMs = 30, budgetMs = budgetMs, nowMs = 1500))
        // Já no mínimo: permanece.
        assertEquals(50, controller.onEncode(encodeMs = 30, budgetMs = budgetMs, nowMs = 3000))
    }

    @Test
    fun `qualidade sobe quando o encode folga`() {
        val controller = StreamQualityController(startQuality = 50)

        assertEquals(60, controller.onEncode(encodeMs = 5, budgetMs = budgetMs, nowMs = 0))
        assertEquals(70, controller.onEncode(encodeMs = 5, budgetMs = budgetMs, nowMs = 1000))
        // Já no máximo: permanece.
        assertEquals(70, controller.onEncode(encodeMs = 5, budgetMs = budgetMs, nowMs = 2000))
    }

    @Test
    fun `encode dentro da faixa nao muda a qualidade`() {
        val controller = StreamQualityController(startQuality = 60)

        assertEquals(60, controller.onEncode(encodeMs = 20, budgetMs = budgetMs, nowMs = 0))
    }

    @Test
    fun `qualidade inicial e limitada a faixa`() {
        val controller = StreamQualityController(startQuality = 95)
        assertEquals(70, controller.quality)
    }
}
