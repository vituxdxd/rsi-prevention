package com.rsiprevention.app.stream

/**
 * Ajuste adaptativo da qualidade JPEG do stream conforme o tempo de encode
 * medido (média móvel), para segurar o alvo de fps mesmo com a CPU ocupada
 * (ex.: picos da NPU/tracker no Snapdragon 8s Gen 4).
 *
 * Regras: se o encode passa de 70% do orçamento do frame, cai um degrau; se
 * fica abaixo de 40%, sobe um degrau. No máximo um ajuste por
 * [adjustIntervalMs] (evita oscilação). Qualidade limitada a
 * [minQuality]..[maxQuality].
 *
 * Pura (recebe `nowMs`) — testável em JVM.
 */
class StreamQualityController(
    private val minQuality: Int = 50,
    private val maxQuality: Int = 70,
    startQuality: Int = 70,
    private val step: Int = 10,
    private val adjustIntervalMs: Long = 1000L
) {
    var quality: Int = startQuality.coerceIn(minQuality, maxQuality)
        private set

    /** `null` até o primeiro ajuste (evita overflow ao comparar com o "sem ajuste ainda"). */
    private var lastAdjustMs: Long? = null

    /**
     * @param encodeMs tempo (média móvel) do último encode
     * @param budgetMs orçamento por frame (ex.: 33 ms para 30 fps)
     * @return a qualidade a usar no próximo encode
     */
    fun onEncode(encodeMs: Long, budgetMs: Long, nowMs: Long): Int {
        val lastAdjust = lastAdjustMs
        if (lastAdjust != null && nowMs - lastAdjust < adjustIntervalMs) return quality

        val target = when {
            encodeMs * 10 > budgetMs * 7 -> quality - step
            encodeMs * 10 < budgetMs * 4 -> quality + step
            else -> quality
        }.coerceIn(minQuality, maxQuality)

        if (target != quality) {
            quality = target
            lastAdjustMs = nowMs
        }
        return quality
    }
}
