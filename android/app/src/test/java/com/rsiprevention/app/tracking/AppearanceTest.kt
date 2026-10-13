package com.rsiprevention.app.tracking

import com.rsiprevention.app.detection.Box
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Testes do [Appearance] sobre pixels sintéticos (o objeto agora é puro: recebe
 * o buffer ARGB 640×640 do modelo, o mesmo espaço das caixas do detector).
 *
 * A regressão que motivou esta API: as coordenadas do modelo eram usadas como
 * pixels do bitmap "em pé" da câmera, amostrando a região errada sempre que o
 * frame não era 640×640.
 */
class AppearanceTest {

    private val size = 640

    /** Metade esquerda de [leftColor], metade direita de [rightColor]. */
    private fun splitPixels(leftColor: Int, rightColor: Int): IntArray =
        IntArray(size * size) { index -> if (index % size < size / 2) leftColor else rightColor }

    private fun solidPixels(color: Int): IntArray = IntArray(size * size) { color }

    private fun box(left: Float, top: Float, right: Float, bottom: Float) = Box(left, top, right, bottom)

    @Test
    fun `a mesma regiao produz o mesmo template`() {
        val pixels = splitPixels(0xFF000000.toInt(), 0xFFFFFFFF.toInt())
        val region = box(100f, 100f, 300f, 300f)

        val a = Appearance.extract(pixels, size, region)
        val b = Appearance.extract(pixels, size, region)

        assertEquals(1f, Appearance.similarity(a, b), 1e-6f)
        assertEquals(a, b)
    }

    @Test
    fun `regioes visualmente diferentes tem similaridade menor`() {
        val pixels = splitPixels(0xFF101010.toInt(), 0xFFF0F0F0.toInt())
        val leftRegion = box(0f, 200f, 200f, 400f)
        val rightRegion = box(440f, 200f, 640f, 400f)

        val left = Appearance.extract(pixels, size, leftRegion)
        val right = Appearance.extract(pixels, size, rightRegion)

        assertTrue(Appearance.similarity(left, left) > Appearance.similarity(left, right))
        assertTrue(Appearance.similarity(left, right) < 0.9f)
    }

    @Test
    fun `regioes iguais em pontos diferentes do frame sao parecidas`() {
        // Faixas verticais uniformes: dois recortes na mesma faixa têm o mesmo
        // conteúdo, independente da posição — a extração não pode depender de a
        // caixa estar na origem do frame.
        val pixels = IntArray(size * size) { index ->
            if (index % size < size / 2) -0x1000000 else -0x1
        }
        val topLeft = Appearance.extract(pixels, size, box(10f, 10f, 210f, 120f))
        val bottomLeft = Appearance.extract(pixels, size, box(10f, 520f, 210f, 630f))

        assertEquals(1f, Appearance.similarity(topLeft, bottomLeft), 1e-6f)
    }

    @Test
    fun `caixa parcialmente fora do frame nao quebra e usa a parte valida`() {
        val pixels = solidPixels(0xFF336699.toInt())

        val template = Appearance.extract(pixels, size, box(-50f, -50f, 50f, 50f))

        assertEquals(48, template.histogram.size)
        assertTrue(Appearance.similarity(template, template) == 1f)
    }

    @Test
    fun `blend com template identico mantem a assinatura`() {
        val pixels = solidPixels(0xFF204060.toInt())
        val template = Appearance.extract(pixels, size, box(200f, 200f, 400f, 400f))

        val blended = Appearance.blend(template, template)

        assertEquals(template.hash, blended.hash)
        assertTrue(Appearance.similarity(template, blended) > 0.99f)
    }
}
