package com.rsiprevention.app.detection

import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Geometria de caixas — usada no NMS do [Postprocessor] e na associação entre
 * frames do [com.rsiprevention.app.tracking.InstrumentTracker].
 */
object BoxMath {
    fun iou(a: Box, b: Box): Float {
        val intersectionLeft = max(a.left, b.left)
        val intersectionTop = max(a.top, b.top)
        val intersectionRight = min(a.right, b.right)
        val intersectionBottom = min(a.bottom, b.bottom)

        val intersectionWidth = max(0f, intersectionRight - intersectionLeft)
        val intersectionHeight = max(0f, intersectionBottom - intersectionTop)
        val intersectionArea = intersectionWidth * intersectionHeight

        val unionArea = a.area + b.area - intersectionArea
        return if (unionArea <= 0f) 0f else intersectionArea / unionArea
    }

    /** Distância entre centros (px). */
    fun centerDistance(a: Box, b: Box): Float {
        val dx = a.centerX - b.centerX
        val dy = a.centerY - b.centerY
        return sqrt(dx * dx + dy * dy)
    }

    /**
     * Similaridade de tamanho em [0, 1]: 1 = mesma largura e altura;
     * cai conforme a razão entre as dimensões se afasta de 1.
     */
    fun sizeSimilarity(a: Box, b: Box): Float {
        val wRatio = min(a.width, b.width) / max(a.width, b.width).coerceAtLeast(1f)
        val hRatio = min(a.height, b.height) / max(a.height, b.height).coerceAtLeast(1f)
        return (wRatio * hRatio).coerceIn(0f, 1f)
    }

    /** Verdadeiro se a caixa tem área e proporção plausíveis para um instrumento. */
    fun isPlausibleInstrument(box: Box, minArea: Float, maxArea: Float, maxAspectRatio: Float): Boolean {
        if (box.area < minArea || box.area > maxArea) return false
        val longSide = max(box.width, box.height)
        val shortSide = min(box.width, box.height)
        if (shortSide < 1f) return false
        return longSide / shortSide <= maxAspectRatio
    }
}
