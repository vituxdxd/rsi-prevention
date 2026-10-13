package com.rsiprevention.app.detection

import com.rsiprevention.app.vocabulary.InstrumentVocabulary
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.FloatBuffer

/**
 * Testes do [Postprocessor] sobre buffers sintéticos — sem ORT nem NPU.
 *
 * Exercitam o **mesmo caminho usado em produção** ([Postprocessor.processBuffers],
 * que recebe os `FloatBuffer` das saídas do ONNX), com as dimensões fixas do
 * `detector_int8.onnx`: 5 classes × 8400 âncoras.
 */
class PostprocessorTest {

    /** Layout `[coord][prediction]`: 4 × 8400. */
    private fun emptyBoxes(): FloatBuffer = FloatBuffer.allocate(BOX_COORDS * NUM_PREDICTIONS)

    /** Layout `[class][prediction]`: 5 × 8400. */
    private fun emptyScores(): FloatBuffer = FloatBuffer.allocate(RAW_CLASSES * NUM_PREDICTIONS)

    private fun setAnchor(
        boxes: FloatBuffer,
        scores: FloatBuffer,
        index: Int,
        cx: Float,
        cy: Float,
        w: Float,
        h: Float,
        classScores: FloatArray
    ) {
        require(classScores.size == RAW_CLASSES) { "classScores deve ter $RAW_CLASSES entradas" }
        boxes.put(index, cx)
        boxes.put(NUM_PREDICTIONS + index, cy)
        boxes.put(2 * NUM_PREDICTIONS + index, w)
        boxes.put(3 * NUM_PREDICTIONS + index, h)
        for (c in classScores.indices) scores.put(c * NUM_PREDICTIONS + index, classScores[c])
    }

    @Test
    fun `ancora abaixo do limiar de confianca e descartada`() {
        val boxes = emptyBoxes()
        val scores = emptyScores()
        setAnchor(boxes, scores, 0, 100f, 100f, 50f, 50f, floatArrayOf(0.10f, 0.24f, 0.05f, 0.20f, 0.15f))

        assertTrue(Postprocessor.processBuffers(boxes, scores).detections.isEmpty())
    }

    @Test
    fun `colapsa as 5 classes brutas no maior score e emite classe unica`() {
        val boxes = emptyBoxes()
        val scores = emptyScores()
        setAnchor(boxes, scores, 0, 100f, 100f, 50f, 50f, floatArrayOf(0.10f, 0.60f, 0.20f, 0.30f, 0.15f))

        val detections = Postprocessor.processBuffers(boxes, scores).detections

        assertEquals(1, detections.size)
        val detection = detections.single()
        assertEquals(0.60f, detection.confidence, 1e-6f)
        assertEquals(InstrumentVocabulary.SINGLE_CLASS_ID, detection.classId)
        assertEquals(InstrumentVocabulary.SINGLE_CLASS_NAME, detection.className)
        assertEquals(1, detection.rawClassId)
    }

    @Test
    fun `nms suprime caixa sobreposta de menor confianca`() {
        val boxes = emptyBoxes()
        val scores = emptyScores()
        // IoU ≈ 0,68 entre as duas caixas (acima do limiar de NMS 0,45).
        setAnchor(boxes, scores, 0, 100f, 100f, 50f, 50f, floatArrayOf(0.8f, 0f, 0f, 0f, 0f))
        setAnchor(boxes, scores, 1, 105f, 105f, 50f, 50f, floatArrayOf(0.6f, 0f, 0f, 0f, 0f))

        val detections = Postprocessor.processBuffers(boxes, scores).detections

        assertEquals(1, detections.size)
        assertEquals(0.8f, detections.single().confidence, 1e-6f)
    }

    @Test
    fun `nms suprime duplicata da mesma classe com iou moderado`() {
        val boxes = emptyBoxes()
        val scores = emptyScores()
        // Mesma classe, IoU ≈ 0,50: entre o limiar por classe (0,45) e o entre
        // classes (0,55). Duplicatas do MESMO instrumento não podem sobreviver —
        // o tracker as transformaria em tracks fantasma (um instrumento contado
        // duas vezes). Regressão contra Soft-NMS.
        setAnchor(boxes, scores, 0, 150f, 150f, 100f, 100f, floatArrayOf(0.8f, 0f, 0f, 0f, 0f))
        setAnchor(boxes, scores, 1, 183f, 150f, 100f, 100f, floatArrayOf(0.7f, 0f, 0f, 0f, 0f))

        val detections = Postprocessor.processBuffers(boxes, scores).detections

        assertEquals(1, detections.size)
        assertEquals(0.8f, detections.single().confidence, 1e-6f)
    }

    @Test
    fun `instrumentos diferentes sobrepostos nao sao fundidos pelo nms`() {
        val boxes = emptyBoxes()
        val scores = emptyScores()
        // IoU ≈ 0,50: entre o limiar de mesma classe (0,45) e o entre classes (0,55).
        // Duas tesouras/pinça próximas não podem virar uma caixa só por causa do
        // colapso em classe única — o NMS acontece antes, com a classe bruta.
        setAnchor(boxes, scores, 0, 150f, 150f, 100f, 100f, floatArrayOf(0f, 0f, 0f, 0.8f, 0f))
        setAnchor(boxes, scores, 1, 183f, 150f, 100f, 100f, floatArrayOf(0f, 0f, 0f, 0f, 0.7f))

        val detections = Postprocessor.processBuffers(boxes, scores).detections

        assertEquals(2, detections.size)
    }

    @Test
    fun `caixas distantes sao mantidas`() {
        val boxes = emptyBoxes()
        val scores = emptyScores()
        setAnchor(boxes, scores, 0, 100f, 100f, 50f, 50f, floatArrayOf(0.8f, 0f, 0f, 0f, 0f))
        setAnchor(boxes, scores, 1, 400f, 400f, 50f, 50f, floatArrayOf(0.6f, 0f, 0f, 0f, 0f))

        assertEquals(2, Postprocessor.processBuffers(boxes, scores).detections.size)
    }

    @Test
    fun `coordenadas sao limitadas ao espaco do modelo 640x640`() {
        val boxes = emptyBoxes()
        val scores = emptyScores()
        // left/top negativos e right/bottom além de 640.
        setAnchor(boxes, scores, 0, 10f, 10f, 50f, 50f, floatArrayOf(0.9f, 0f, 0f, 0f, 0f))
        setAnchor(boxes, scores, 1, 630f, 630f, 50f, 50f, floatArrayOf(0.7f, 0f, 0f, 0f, 0f))

        val detections = Postprocessor.processBuffers(boxes, scores).detections
            .sortedByDescending { it.confidence }

        assertEquals(2, detections.size)
        val first = detections[0].box
        assertEquals(0f, first.left, 1e-6f)
        assertEquals(0f, first.top, 1e-6f)
        assertEquals(35f, first.right, 1e-6f)
        assertEquals(35f, first.bottom, 1e-6f)
        val second = detections[1].box
        assertEquals(640f, second.right, 1e-6f)
        assertEquals(640f, second.bottom, 1e-6f)
    }

    @Test
    fun `maior confianca do frame inclui ancoras abaixo do limiar de deteccao`() {
        val boxes = emptyBoxes()
        val scores = emptyScores()
        // Nenhuma âncora acima de 0,25 (lista de detecções vazia), mas o pico do
        // frame (0,24) precisa sair da mesma varredura — é ele que a verificação
        // de sanidade do backend usa para reprovar modelos degenerados.
        setAnchor(boxes, scores, 0, 100f, 100f, 50f, 50f, floatArrayOf(0.10f, 0.24f, 0.05f, 0.20f, 0.15f))
        scores.put(2 * NUM_PREDICTIONS + 123, 0.22f)
        scores.put(4 * NUM_PREDICTIONS + 8000, 0.23f)

        val result = Postprocessor.processBuffers(boxes, scores)

        assertTrue(result.detections.isEmpty())
        assertEquals(0.24f, result.maxConfidence, 1e-6f)
    }

    @Test
    fun `rawClassId preserva a classe bruta do modelo`() {
        val boxes = emptyBoxes()
        val scores = emptyScores()
        // A classe bruta vencedora é 3 (shear, com score 0.80).
        setAnchor(boxes, scores, 0, 100f, 100f, 50f, 50f, floatArrayOf(0.10f, 0.20f, 0.05f, 0.80f, 0.15f))

        val detection = Postprocessor.processBuffers(boxes, scores).detections.single()

        // classId colapsa para classe única (0), mas rawClassId preserva o 3.
        assertEquals(InstrumentVocabulary.SINGLE_CLASS_ID, detection.classId)
        assertEquals(3, detection.rawClassId)
    }

    companion object {
        // Espelham as constantes privadas de Postprocessor (saídas fixas do modelo).
        private const val RAW_CLASSES = 5
        private const val BOX_COORDS = 4
        private const val NUM_PREDICTIONS = 8400
    }
}
