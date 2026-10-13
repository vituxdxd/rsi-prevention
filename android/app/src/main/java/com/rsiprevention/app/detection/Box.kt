package com.rsiprevention.app.detection

/**
 * Caixa retangular em pixels no espaço de entrada do modelo (640x640),
 * independente de `android.graphics.RectF` para que a lógica de tracking possa
 * ser testada em JVM puro (ver `InstrumentTrackerTest`).
 */
data class Box(
    val left: Float,
    val top: Float,
    val right: Float,
    val bottom: Float
) {
    val width: Float get() = (right - left).coerceAtLeast(0f)
    val height: Float get() = (bottom - top).coerceAtLeast(0f)
    val area: Float get() = width * height
    val centerX: Float get() = (left + right) / 2f
    val centerY: Float get() = (top + bottom) / 2f

    fun translated(dx: Float, dy: Float): Box =
        Box(left + dx, top + dy, right + dx, bottom + dy)

    /** Escala em torno do centro (largura/altura multiplicadas por [sx]/[sy]). */
    fun scaledAroundCenter(sx: Float, sy: Float): Box {
        val halfW = width * sx / 2f
        val halfH = height * sy / 2f
        return Box(centerX - halfW, centerY - halfH, centerX + halfW, centerY + halfH)
    }

    /** Interpolação linear com outra caixa ([t] = 0 mantém esta; [t] = 1 usa [other]). */
    fun lerp(other: Box, t: Float): Box = Box(
        left + (other.left - left) * t,
        top + (other.top - top) * t,
        right + (other.right - right) * t,
        bottom + (other.bottom - bottom) * t
    )
}
