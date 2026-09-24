package com.rsiprevention.app.detection

import android.content.Context
import android.graphics.BitmapFactory
import android.util.Log

/**
 * Verificação de sanidade do detector: roda o modelo numa imagem de referência
 * (uma tesoura Mayo sobre fundo claro, embarcada em `assets/selftest_scissors.jpg`)
 * e exige uma confiança mínima conhecida.
 *
 * Por que isso existe (medido neste projeto):
 *  - o export `w8a8` do AI Hub devolveu **0.0** em todos os candidatos: as
 *    caixas e os scores eram requantizados juntos, com a escala das caixas
 *    (~640 px), arredondando todo score para 0 (ver
 *    DIAGNOSTICO_CONFIANCA_ZERO.md e a separação `boxes`/`scores` em
 *    [Postprocessor]);
 *  - um execution provider pode **não reivindicar nó nenhum** e o ONNX Runtime
 *    executar tudo no CPU em silêncio (ver [com.rsiprevention.app.ort.QnnSessionFactory]).
 *
 * Ou seja: criar a sessão sem erro não prova que o backend funciona. Esta
 * verificação é usada para escolher o backend (a sessão só é aceita se detectar
 * a referência) e serve de teste ponta a ponta do conjunto
 * `detector + pré-processamento + pós-processamento`.
 */
object ModelSelfTest {

    private const val TAG = "NPU_CHECK"
    private const val ASSET_NAME = "selftest_scissors.jpg"

    /**
     * Confiança mínima na imagem de referência para aprovar um backend.
     *
     * A imagem de selftest (selftest_scissors.jpg) é uma tesoura cirúrgica
     * Mayo; no grafo A16W8 atual a classe `shear` (3) dá ~0.85 de confiança
     * (praticamente igual ao float de referência, ~0.87).
     *
     * O limiar 0.04 aceita o HTP (0.85 > 0.04) e rejeita modelos degenerados
     * (0.0 < 0.04). O selftest existe para rejeitar modelos completamente
     * quebrados, não para validar qualidade de detecção.
     */
    private const val MIN_CONFIDENCE = 0.04f

    /** @return `true` se o detector aprovou a imagem de referência. */
    fun passes(context: Context, detector: YoloWorldDetector): Boolean {
        val bitmap = runCatching {
            context.assets.open(ASSET_NAME).use { BitmapFactory.decodeStream(it) }
        }.getOrNull()

        if (bitmap == null) {
            Log.w(TAG, "sem assets/$ASSET_NAME — verificação de sanidade ignorada")
            return true
        }

        val (detections, maxConfidence) = runCatching { detector.detectWithScore(bitmap) }
            .getOrElse { throwable ->
                Log.w(TAG, "verificação de sanidade falhou na execução: ${throwable.message}")
                return false
            }

        val ok = maxConfidence >= MIN_CONFIDENCE
        Log.i(
            TAG,
            "verificação de sanidade: confiança máxima %.3f (mínimo %.2f), %d detecção(ões) -> %s".format(
                maxConfidence, MIN_CONFIDENCE, detections.size, if (ok) "aprovado" else "reprovado"
            )
        )
        return ok
    }
}