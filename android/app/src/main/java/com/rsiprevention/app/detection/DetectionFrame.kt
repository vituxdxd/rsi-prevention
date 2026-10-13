package com.rsiprevention.app.detection

/**
 * Um ciclo completo de detecção em um frame, com o **tempo de cada etapa**
 * separado — insumo da instrumentação do pipeline (ver
 * `com.rsiprevention.app.perf.PipelineStats`).
 *
 * Contrato de posse: [model] referencia os buffers reutilizados do
 * [Preprocessor] e só é válido até a próxima detecção (mesma thread).
 */
class DetectionFrame(
    val detections: List<Detection>,
    /** Maior confiança crua do frame, sem NMS (ver [Postprocessor]). */
    val maxConfidence: Float,
    /** Pixels do frame no espaço do modelo (640×640). */
    val model: ModelFrame,
    /** Pré-processamento + criação do tensor de entrada (~ms em ns). */
    val preprocessNs: Long,
    /** Sub-etapas do pré-processamento: escala, `getPixels` e montagem NCHW. */
    val preprocessScaleNs: Long,
    val preprocessPixelsNs: Long,
    val preprocessFillNs: Long,
    /** Execução da sessão ONNX na NPU (`session.run`). */
    val inferenceNs: Long,
    /** Decodificação das saídas + NMS. */
    val postprocessNs: Long
)
