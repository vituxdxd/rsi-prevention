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
    val box: Box,
    /**
     * Classe bruta vencedora do modelo (0..4: clamp, needle_holder, scalpel,
     * shear, tweezer). Usada pelo [com.rsiprevention.app.tracking.InstrumentTracker]
     * para votação de classe e para discriminar instrumentos sobrepostos na
     * associação entre frames. O [classId] colapsa para a classe única exibida na
     * UI; este campo preserva a informação fina do modelo sem impactar a exibição.
     */
    val rawClassId: Int = classId
)
