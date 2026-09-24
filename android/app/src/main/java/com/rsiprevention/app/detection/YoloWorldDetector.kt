package com.rsiprevention.app.detection

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap

/**
 * Orquestra um ciclo completo de detecção: pré-processamento -> `detector.onnx`
 * (QNN/NPU) -> pós-processamento (confiança + NMS em CPU). Não é thread-safe
 * por si só: assume que é chamado sempre a partir do mesmo executor dedicado de
 * inferência (nunca `Dispatchers.Main`, ver doc de deploy seção 9.4) e nunca
 * concorrentemente — o mesmo padrão do `ImageAnalysis` do CameraX com
 * `STRATEGY_KEEP_ONLY_LATEST` (no máximo um frame em voo por vez).
 */
class YoloWorldDetector(
    private val env: OrtEnvironment,
    private val detectorSession: OrtSession
) {
    /**
     * Bloqueia até a NPU retornar — nunca chamar na thread principal.
     *
     * Devolve também a **maior confiança crua** do frame, usada pela verificação
     * de sanidade do backend (ver `ModelSelfTest`).
     */
    fun detectWithScore(uprightBitmap: Bitmap): Pair<List<Detection>, Float> {
        val imageBytes = Preprocessor.preprocess(uprightBitmap)

        OnnxTensor.createTensor(env, imageBytes, IMAGE_SHAPE, OnnxJavaType.FLOAT).use { imageTensor ->
            val inputs = mapOf("images" to imageTensor)
            detectorSession.run(inputs).use { result ->
                val boxesTensor = result.tensor(BOXES_OUTPUT)
                val scoresTensor = result.tensor(SCORES_OUTPUT)

                val maxConfidence = Postprocessor.maxConfidence(scoresTensor)
                val detections = Postprocessor.process(boxesTensor, scoresTensor)
                return detections to maxConfidence
            }
        }
    }

    private fun OrtSession.Result.tensor(name: String): OnnxTensor =
        get(name).orElse(null) as? OnnxTensor
            ?: throw IllegalStateException(
                "Saída '$name' ausente no modelo. Saídas disponíveis: ${map { it.key }}"
            )

    /** Bloqueia até a NPU retornar — nunca chamar na thread principal. */
    fun detect(uprightBitmap: Bitmap): List<Detection> = detectWithScore(uprightBitmap).first

    companion object {
        const val BOXES_OUTPUT = "boxes"
        const val SCORES_OUTPUT = "scores"
        private val IMAGE_SHAPE = longArrayOf(
            1, 3, Preprocessor.MODEL_INPUT_SIZE.toLong(), Preprocessor.MODEL_INPUT_SIZE.toLong()
        )
    }
}
