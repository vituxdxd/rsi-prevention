package com.rsiprevention.app.detection

import ai.onnxruntime.OnnxTensor
import com.rsiprevention.app.vocabulary.InstrumentVocabulary

/**
 * Decodifica as duas saídas do YOLO26s (`detector_int8.onnx`) em uma lista de
 * [Detection]. O grafo é exportado **sem NMS end-to-end** e com as saídas
 * separadas:
 *
 *   - `boxes`  `[1, 4, 8400]` float32: cx, cy, w, h em pixels (0..640)
 *   - `scores` `[1, 5, 8400]` float32: probabilidade de cada classe bruta por âncora
 *
 * A separação existe porque caixas e scores num único tensor `[1, 9, 8400]`
 * seriam requantizados com uma escala só (dominada pelas caixas, ~640), o que
 * arredonda todo score para zero no INT8 — ver DIAGNOSTICO_CONFIANCA_ZERO.md.
 *
 * **Classe única com NMS por classe bruta**: para prevenção de retenção de
 * corpos estranhos a subclasse do instrumento não importa (e o modelo confunde
 * muito entre classes parecidas), então a [Detection] emitida é sempre a classe
 * única [InstrumentVocabulary.SINGLE_CLASS_NAME], com o **maior score** entre
 * as 5 pontuações brutas.
 *
 * O NMS, porém, usa a **classe bruta vencedora de cada âncora** (antes do
 * colapso): a primeira passada remove duplicatas do mesmo instrumento bruto
 * (IoU > [NMS_IOU_THRESHOLD]) e a segunda remove caixas quase idênticas entre
 * classes diferentes (IoU > [CROSS_CLASS_NMS_IOU_THRESHOLD] — o modelo às vezes
 * dispara `shear` e `tweezer` na mesma região). Fazer o NMS *depois* do colapso
 * suprimiria também **instrumentos diferentes que se sobrepõem** com IoU > 0,45
 * — dois instrumentos próximos/cruzados sumiriam da tela como um só.
 *
 * Filtro de confiança + NMS rodam aqui, em CPU/Kotlin.
 */
object Postprocessor {
    private const val CONFIDENCE_THRESHOLD = 0.25f
    private const val NMS_IOU_THRESHOLD = 0.45f

    /**
     * IoU para supressão **entre classes diferentes** (segunda passada do NMS).
     * O modelo às vezes dispara `shear` e `tweezer` na mesma região; sem isso os
     * dois viram tracks e um deles aparece como "fantasma" com a classe errada.
     */
    private const val CROSS_CLASS_NMS_IOU_THRESHOLD = 0.55f

    /** Classes brutas do modelo: clamp, needle_holder, scalpel, shear, tweezer. */
    private const val RAW_CLASSES = 5
    private const val NUM_PREDICTIONS = 8400

    /**
     * Maior confiança crua do frame (sem NMS). Usada pelo selftest para
     * verificar se o backend está funcionando.
     */
    fun maxConfidence(scoresTensor: OnnxTensor): Float {
        // scores shape: [1, 5, 8400]; o ORT dequantiza INT8→float automaticamente
        // para grafos QDQ
        return maxConfidence((scoresTensor.value as Array<Array<FloatArray>>)[0])
    }

    /** Versão pura de [maxConfidence] (sem ORT) — testável em JVM. */
    fun maxConfidence(scores: Array<FloatArray>): Float {
        var maxConf = 0f

        for (c in 0 until RAW_CLASSES) {
            for (j in 0 until NUM_PREDICTIONS) {
                val score = scores[c][j]
                if (score > maxConf) maxConf = score
            }
        }
        return maxConf
    }

    /**
     * Decodifica `boxes` [1, 4, 8400] + `scores` [1, 5, 8400], filtra por
     * confiança e aplica NMS.
     */
    fun process(boxesTensor: OnnxTensor, scoresTensor: OnnxTensor): List<Detection> {
        // boxes[coord][prediction] e scores[class][prediction] — já dequantizados pelo ORT
        return process(
            (boxesTensor.value as Array<Array<FloatArray>>)[0],
            (scoresTensor.value as Array<Array<FloatArray>>)[0]
        )
    }

    /**
     * Versão pura de [process] (sem ORT) — testável em JVM.
     *
     * @param boxes `[4][8400]` cx, cy, w, h em pixels (0..640)
     * @param scores `[5][8400]` probabilidade de cada classe bruta por âncora
     */
    fun process(boxes: Array<FloatArray>, scores: Array<FloatArray>): List<Detection> {
        val candidates = ArrayList<RawCandidate>()
        val imgSize = Preprocessor.MODEL_INPUT_SIZE.toFloat()

        for (j in 0 until NUM_PREDICTIONS) {
            // Maior score entre as 5 classes brutas — e a classe vencedora, que o
            // NMS usa para não fundir instrumentos diferentes que se sobrepõem.
            var bestScore = scores[0][j]
            var bestClass = 0
            for (c in 1 until RAW_CLASSES) {
                val score = scores[c][j]
                if (score > bestScore) {
                    bestScore = score
                    bestClass = c
                }
            }

            if (bestScore < CONFIDENCE_THRESHOLD) continue

            // cx, cy, w, h → left, top, right, bottom
            val cx = boxes[0][j]
            val cy = boxes[1][j]
            val w  = boxes[2][j]
            val h  = boxes[3][j]

            val box = Box(
                (cx - w / 2f).coerceIn(0f, imgSize),
                (cy - h / 2f).coerceIn(0f, imgSize),
                (cx + w / 2f).coerceIn(0f, imgSize),
                (cy + h / 2f).coerceIn(0f, imgSize)
            )

            candidates.add(RawCandidate(box = box, confidence = bestScore, rawClass = bestClass))
        }

        // NMS com as classes brutas; só depois as detecções viram a classe única.
        return nmsCrossClass(nmsPerClass(candidates)).map { candidate ->
            Detection(
                classId = InstrumentVocabulary.SINGLE_CLASS_ID,
                className = InstrumentVocabulary.SINGLE_CLASS_NAME,
                confidence = candidate.confidence,
                box = candidate.box
            )
        }
    }

    /** Caixa decodificada com a classe bruta vencedora, antes do colapso em classe única. */
    private data class RawCandidate(val box: Box, val confidence: Float, val rawClass: Int)

    /** NMS clássico (maior confiança primeiro, suprime sobreposições) por classe bruta. */
    private fun nmsPerClass(candidates: List<RawCandidate>): List<RawCandidate> {
        val kept = ArrayList<RawCandidate>()
        for ((_, group) in candidates.groupBy { it.rawClass }) {
            val remaining = group.sortedByDescending { it.confidence }.toMutableList()
            while (remaining.isNotEmpty()) {
                val best = remaining.removeAt(0)
                kept.add(best)
                remaining.removeAll { BoxMath.iou(best.box, it.box) > NMS_IOU_THRESHOLD }
            }
        }
        return kept
    }

    /**
     * Segunda passada: suprime caixas de **outra classe bruta** muito
     * sobrepostas, mantendo a de maior confiança (a vencedora da região).
     */
    private fun nmsCrossClass(candidates: List<RawCandidate>): List<RawCandidate> {
        val kept = ArrayList<RawCandidate>()
        for (candidate in candidates.sortedByDescending { it.confidence }) {
            if (kept.none { BoxMath.iou(it.box, candidate.box) > CROSS_CLASS_NMS_IOU_THRESHOLD }) {
                kept.add(candidate)
            }
        }
        return kept
    }
}
