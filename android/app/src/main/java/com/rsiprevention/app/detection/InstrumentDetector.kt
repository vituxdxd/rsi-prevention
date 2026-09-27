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
 *
 * **Nota de nome**: na fase inicial o projeto usava YOLO-World (vocabulário
 * aberto, prompts de texto + encoder CLIP) e esta classe se chamava
 * `YoloWorldDetector`. Depois da migração para o YOLO26s de classe única
 * (`detector_int8.onnx`, ver README) o caminho de texto foi removido e o nome
 * ficou obsoleto — a classe é só o orquestrador do detector.
 */
class InstrumentDetector(
    private val env: OrtEnvironment,
    private val detectorSession: OrtSession
) {
    /**
     * Tensor de entrada reutilizado entre frames: envolve o mesmo [ByteBuffer]
     * direto que o [Preprocessor] preenche (sem cópia). Eliminamos a criação e
     * destruição de um `OrtValue` nativo a cada frame via JNI (~30 µs + GC).
     *
     * O contrato é seguro porque:
     * - O pipeline roda em single-thread (executor `rsi-inference`).
     * - O [Preprocessor] preenche o buffer e retorna; o tensor é consumido
     *   pela NPU antes do próximo frame ser preprocessado.
     */
    private val reusableTensor: OnnxTensor = OnnxTensor.createTensor(
        env, Preprocessor.inputBuffer, IMAGE_SHAPE, OnnxJavaType.FLOAT
    )

    /** Mapa de entrada reutilizado (evita alocar HashMap + Entry por frame). */
    private val inputMap: Map<String, OnnxTensor> = mapOf("images" to reusableTensor)

    /**
     * Bloqueia até a NPU retornar — nunca chamar na thread principal.
     *
     * Devolve as detecções, a maior confiança crua do frame (verificação de
     * sanidade do backend em `ModelSelfTest`), o frame no espaço do modelo
     * (usado pelo tracker para o template de aparência) e o tempo de cada etapa
     * (instrumentação).
     */
    fun detectFrame(uprightBitmap: Bitmap): DetectionFrame {
        val preprocessStartNs = System.nanoTime()
        val preprocessed = Preprocessor.preprocess(uprightBitmap)
        val preprocessNs = System.nanoTime() - preprocessStartNs

        // O Preprocessor já preencheu o mesmo ByteBuffer que o reusableTensor
        // referencia — basta rodar a sessão sem criar um tensor novo.
        val inferenceStartNs = System.nanoTime()
        detectorSession.run(inputMap).use { result ->
            val inferenceNs = System.nanoTime() - inferenceStartNs
            val boxesTensor = result.tensor(BOXES_OUTPUT)
            val scoresTensor = result.tensor(SCORES_OUTPUT)

            val postprocessStartNs = System.nanoTime()
            val processed = Postprocessor.process(boxesTensor, scoresTensor)
            val postprocessNs = System.nanoTime() - postprocessStartNs

            return DetectionFrame(
                detections = processed.detections,
                maxConfidence = processed.maxConfidence,
                model = preprocessed.model,
                preprocessNs = preprocessNs,
                preprocessScaleNs = preprocessed.scaleNs,
                preprocessPixelsNs = preprocessed.pixelsNs,
                preprocessFillNs = preprocessed.fillNs,
                inferenceNs = inferenceNs,
                postprocessNs = postprocessNs
            )
        }
    }

    /** Libera o tensor reutilizado (chamar no shutdown, após `awaitTermination`). */
    fun close() {
        reusableTensor.close()
    }

    private fun OrtSession.Result.tensor(name: String): OnnxTensor =
        get(name).orElse(null) as? OnnxTensor
            ?: throw IllegalStateException(
                "Saída '$name' ausente no modelo. Saídas disponíveis: ${map { it.key }}"
            )

    companion object {
        const val BOXES_OUTPUT = "boxes"
        const val SCORES_OUTPUT = "scores"
        private val IMAGE_SHAPE = longArrayOf(
            1, 3, Preprocessor.MODEL_INPUT_SIZE.toLong(), Preprocessor.MODEL_INPUT_SIZE.toLong()
        )
    }
}
