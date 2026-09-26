package com.rsiprevention.app.detection

import com.rsiprevention.app.vocabulary.InstrumentVocabulary
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Testes da lógica pura do [Postprocessor] (filtro de confiança, colapso em
 * classe única, NMS) sobre tensores sintéticos — sem ORT nem NPU.
 *
 * As dimensões 5 classes × 8400 âncoras são as saídas fixas do
 * `detector_int8.onnx` (ver KDoc de [Postprocessor]).
 */
class PostprocessorTest {

    private fun emptyBoxes(): Array<FloatArray> = Array(BOX_COORDS) { FloatArray(NUM_PREDICTIONS) }
    private fun emptyScores(): Array<FloatArray> = Array(RAW_CLASSES) { FloatArray(NUM_PREDICTIONS) }

    private fun setAnchor(
        boxes: Array<FloatArray>,
        scores: Array<FloatArray>,
        index: Int,
        cx: Float,
        cy: Float,
        w: Float,
        h: Float,
        classScores: FloatArray
    ) {
        require(classScores.size == RAW_CLASSES) { "classScores deve ter $RAW_CLASSES entradas" }
        boxes[0][index] = cx
        boxes[1][index] = cy
        boxes[2][index] = w
        boxes[3][index] = h
        for (c in classScores.indices) scores[c][index] = classScores[c]
    }

    @Test
    fun `ancora abaixo do limiar de confianca e descartada`() {
        val boxes = emptyBoxes()
        val scores = emptyScores()
        setAnchor(boxes, scores, 0, 100f, 100f, 50f, 50f, floatArrayOf(0.10f, 0.24f, 0.05f, 0.20f, 0.15f))

        assertTrue(Postprocessor.process(boxes, scores).isEmpty())
    }

    @Test
    fun `colapsa as 5 classes brutas no maior score e emite classe unica`() {
        val boxes = emptyBoxes()
        val scores = emptyScores()
        setAnchor(boxes, scores, 0, 100f, 100f, 50f, 50f, floatArrayOf(0.10f, 0.60f, 0.20f, 0.30f, 0.15f))

        val detections = Postprocessor.process(boxes, scores)

        assertEquals(1, detections.size)
        val detection = detections.single()
        assertEquals(0.60f, detection.confidence, 1e-6f)
        assertEquals(InstrumentVocabulary.SINGLE_CLASS_ID, detection.classId)
        assertEquals(InstrumentVocabulary.SINGLE_CLASS_NAME, detection.className)
    }

    @Test
    fun `nms suprime caixa sobreposta de menor confianca`() {
        val boxes = emptyBoxes()
        val scores = emptyScores()
        // IoU ≈ 0,68 entre as duas caixas (acima do limiar de NMS 0,45).
        setAnchor(boxes, scores, 0, 100f, 100f, 50f, 50f, floatArrayOf(0.8f, 0f, 0f, 0f, 0f))
        setAnchor(boxes, scores, 1, 105f, 105f, 50f, 50f, floatArrayOf(0.6f, 0f, 0f, 0f, 0f))

        val detections = Postprocessor.process(boxes, scores)

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

        val detections = Postprocessor.process(boxes, scores)

        assertEquals(2, detections.size)
    }

    @Test
    fun `caixas distantes sao mantidas`() {
        val boxes = emptyBoxes()
        val scores = emptyScores()
        setAnchor(boxes, scores, 0, 100f, 100f, 50f, 50f, floatArrayOf(0.8f, 0f, 0f, 0f, 0f))
        setAnchor(boxes, scores, 1, 400f, 400f, 50f, 50f, floatArrayOf(0.6f, 0f, 0f, 0f, 0f))

        assertEquals(2, Postprocessor.process(boxes, scores).size)
    }

    @Test
    fun `coordenadas sao limitadas ao espaco do modelo 640x640`() {
        val boxes = emptyBoxes()
        val scores = emptyScores()
        // left/top negativos e right/bottom além de 640.
        setAnchor(boxes, scores, 0, 10f, 10f, 50f, 50f, floatArrayOf(0.9f, 0f, 0f, 0f, 0f))
        setAnchor(boxes, scores, 1, 630f, 630f, 50f, 50f, floatArrayOf(0.7f, 0f, 0f, 0f, 0f))

        val detections = Postprocessor.process(boxes, scores).sortedByDescending { it.confidence }

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
    fun `maxConfidence retorna o maior score bruto do frame`() {
        val scores = emptyScores()
        scores[2][123] = 0.42f
        scores[4][8000] = 0.87f
        scores[0][5] = 0.10f

        assertEquals(0.87f, Postprocessor.maxConfidence(scores), 1e-6f)
    }

    companion object {
        // Espelham as constantes privadas de Postprocessor (saídas fixas do modelo).
        private const val RAW_CLASSES = 5
        private const val BOX_COORDS = 4
        private const val NUM_PREDICTIONS = 8400
    }
}
