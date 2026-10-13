package com.rsiprevention.app.detection

/**
 * Saída do [Postprocessor] para um frame: as detecções finais (classe única,
 * após confiança + NMS) e a **maior confiança crua** do frame (sem NMS), usada
 * pela verificação de sanidade do backend (ver `ModelSelfTest`).
 *
 * As duas informações saem da mesma passada sobre as 5×8400 âncoras — antes
 * eram duas varreduras completas do tensor.
 */
data class DetectionResult(
    val detections: List<Detection>,
    val maxConfidence: Float
)
