package com.rsiprevention.app.detection

/**
 * Resultado de uma detecção pontual do detector em um único frame, antes do
 * tracking entre frames (ver [com.rsiprevention.app.tracking.InstrumentTracker]).
 *
 * @param box coordenadas em pixels no espaço da imagem de entrada do modelo (640x640).
 */
data class Detection(
    val classId: Int,
    val className: String,
    val confidence: Float,
    val box: Box
)
