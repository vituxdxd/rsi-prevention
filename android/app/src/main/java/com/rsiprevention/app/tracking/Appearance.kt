package com.rsiprevention.app.tracking

import com.rsiprevention.app.detection.Box

/**
 * Assinatura visual barata de um crop: dHash de 64 bits (estrutura/textura,
 * robusta a mudanças de iluminação e escala) + histograma RGB normalizado
 * (16 bins por canal) sobre a região central da caixa (menos fundo).
 *
 * Usada pelo [InstrumentTracker] para complementar a geometria na associação
 * entre frames e para rejeitar associações em que o conteúdo visual muda.
 *
 * Os pixels vêm do frame **no espaço do modelo** (640×640, ver
 * `com.rsiprevention.app.detection.ModelFrame`) — o mesmo espaço das caixas do
 * detector. Antes a extração recebia o bitmap "em pé" da câmera e usava as
 * coordenadas do modelo direto como pixels dele, amostrando a região errada
 * sempre que o frame não era 640×640.
 *
 * Puro (sem `android.graphics`) — testável em JVM.
 */
data class AppearanceTemplate(
    val hash: Long,
    val histogram: FloatArray
) {
    override fun equals(other: Any?): Boolean =
        other is AppearanceTemplate && other.hash == hash && other.histogram.contentEquals(histogram)

    override fun hashCode(): Int = 31 * hash.hashCode() + histogram.contentHashCode()
}

object Appearance {

    private const val HIST_BINS = 16
    private const val HIST_SIZE = HIST_BINS * 3

    private const val HASH_COLS = 9
    private const val HASH_ROWS = 8

    /** Fração retirada de cada lado para focar a amostragem no instrumento. */
    private const val CENTER_SHRINK = 0.15f

    /** Passo de amostragem do histograma (px) — trade-off precisão/velocidade. */
    private const val HIST_STEP = 2

    /**
     * @param pixels ARGB linha a linha de [size]×[size]
     * @param box caixa no mesmo espaço dos pixels (0..[size])
     */
    fun extract(pixels: IntArray, size: Int, box: Box): AppearanceTemplate {
        val region = centerRegion(box)
        return AppearanceTemplate(
            hash = dHash(pixels, size, region),
            histogram = colorHistogram(pixels, size, region)
        )
    }

    /**
     * Similaridade em [0, 1]: metade estrutura (Hamming do dHash) e metade cor
     * (interseção de histogramas).
     */
    fun similarity(a: AppearanceTemplate, b: AppearanceTemplate): Float =
        0.5f * hashSimilarity(a.hash, b.hash) + 0.5f * histogramIntersection(a.histogram, b.histogram)

    /** Atualiza o template devagar, aceitando a nova assinatura só se ainda parecida. */
    fun blend(old: AppearanceTemplate, new: AppearanceTemplate): AppearanceTemplate {
        val histogram = FloatArray(HIST_SIZE) {
            0.7f * old.histogram[it] + 0.3f * new.histogram[it]
        }
        val hash = if (hashSimilarity(old.hash, new.hash) >= 0.55f) new.hash else old.hash
        return AppearanceTemplate(hash, histogram)
    }

    private fun hashSimilarity(a: Long, b: Long): Float =
        1f - java.lang.Long.bitCount(a xor b) / 64f

    private fun histogramIntersection(h1: FloatArray, h2: FloatArray): Float {
        var intersection = 0f
        for (i in h1.indices) intersection += minOf(h1[i], h2[i])
        return intersection.coerceIn(0f, 1f)
    }

    private fun centerRegion(box: Box): Box {
        val dx = box.width * CENTER_SHRINK
        val dy = box.height * CENTER_SHRINK
        return Box(box.left + dx, box.top + dy, box.right - dx, box.bottom - dy)
    }

    /** dHash: 8 comparações horizontais por linha em uma grade 9x8 de luma. */
    private fun dHash(pixels: IntArray, size: Int, region: Box): Long {
        val left = region.left.toInt().coerceIn(0, size - 1)
        val top = region.top.toInt().coerceIn(0, size - 1)
        val right = region.right.toInt().coerceIn(left + 1, size)
        val bottom = region.bottom.toInt().coerceIn(top + 1, size)
        val spanX = (right - left).coerceAtLeast(1)
        val spanY = (bottom - top).coerceAtLeast(1)

        val luma = IntArray(HASH_COLS * HASH_ROWS)
        for (row in 0 until HASH_ROWS) {
            val y = (top + spanY * (row + 0.5f) / HASH_ROWS).toInt().coerceIn(top, bottom - 1)
            for (col in 0 until HASH_COLS) {
                val x = (left + spanX * (col + 0.5f) / HASH_COLS).toInt().coerceIn(left, right - 1)
                luma[row * HASH_COLS + col] = lumaOf(pixels[y * size + x])
            }
        }

        var hash = 0L
        for (row in 0 until HASH_ROWS) {
            for (col in 0 until HASH_COLS - 1) {
                if (luma[row * HASH_COLS + col] > luma[row * HASH_COLS + col + 1]) {
                    hash = hash or (1L shl (row * (HASH_COLS - 1) + col))
                }
            }
        }
        return hash
    }

    private fun colorHistogram(pixels: IntArray, size: Int, region: Box): FloatArray {
        val left = region.left.toInt().coerceIn(0, size - 1)
        val top = region.top.toInt().coerceIn(0, size - 1)
        val right = region.right.toInt().coerceIn(left + 1, size)
        val bottom = region.bottom.toInt().coerceIn(top + 1, size)

        val histogram = FloatArray(HIST_SIZE)
        var samples = 0
        var y = top
        while (y < bottom) {
            val rowOffset = y * size
            var x = left
            while (x < right) {
                val pixel = pixels[rowOffset + x]
                val r = (pixel shr 16) and 0xFF
                val g = (pixel shr 8) and 0xFF
                val b = pixel and 0xFF
                histogram[r * HIST_BINS / 256] += 1f
                histogram[HIST_BINS + g * HIST_BINS / 256] += 1f
                histogram[2 * HIST_BINS + b * HIST_BINS / 256] += 1f
                samples++
                x += HIST_STEP
            }
            y += HIST_STEP
        }

        if (samples > 0) {
            val total = samples.toFloat()
            for (i in histogram.indices) histogram[i] /= total
        }
        return histogram
    }

    private fun lumaOf(pixel: Int): Int {
        val r = (pixel shr 16) and 0xFF
        val g = (pixel shr 8) and 0xFF
        val b = pixel and 0xFF
        return (r * 299 + g * 587 + b * 114) / 1000
    }
}
