package com.rsiprevention.app.tracking

import com.rsiprevention.app.detection.Box
import com.rsiprevention.app.vocabulary.InstrumentVocabulary

/**
 * Um objeto rastreado ao longo de múltiplos frames.
 *
 * Ciclo de vida:
 *  1. **CANDIDATE** — detectado mas ainda não confirmado (precisa de K de N
 *     frames com presença quase contínua e confiança mediana suficiente)
 *  2. **ON_TABLE** — confirmado, visível na mesa
 *  3. **REMOVED** — ausente por tempo suficiente **e** com histórico confiável
 *     (objetos curtos/pouco confiáveis são descartados em silêncio)
 *  4. **RETURNED** — reapareceu após remoção
 *  5. **NOT_RETURNED** — checagem final: foi removido e nunca voltou
 *
 * Além do estado, o track mantém um modelo simples de movimento (velocidade em
 * px/ms), um template visual ([AppearanceTemplate]) e uma votação de classe,
 * usados na associação entre frames (ver [InstrumentTracker]).
 */
data class TrackedInstrument(
    val id: Int,
    var classId: Int,
    var className: String,
    var box: Box,
    var confidence: Float,
    var status: InstrumentStatus,
    val firstSeenMs: Long,
    var lastSeenMs: Long,
    var onTableSinceMs: Long = 0L,
    var framesSinceSeen: Int = 0,
    var wasEverRemoved: Boolean = false,

    // --- Movimento (px/ms) ---
    var vx: Float = 0f,
    var vy: Float = 0f,
    var vw: Float = 0f,
    var vh: Float = 0f,

    // --- Confirmação temporal (janela deslizante de [InstrumentTracker.CONFIRM_WINDOW] frames) ---
    /** Bit i = o track casou com detecção há i frames (bit 0 = frame atual). */
    var hitMask: Int = 0,
    /** Frames processados desde a criação do track. */
    var ageFrames: Int = 1,
    var peakConfidence: Float = confidence,
    /** Últimas confianças casadas (anel) para a mediana. */
    val confidenceSamples: FloatArray = FloatArray(InstrumentTracker.CONFIRM_WINDOW),
    var confidenceSampleCount: Int = 0,

    // --- Classe ---
    /** Votos decaídos por classe bruta do modelo, ponderados pela confiança das detecções. */
    val classVotes: FloatArray = FloatArray(InstrumentVocabulary.MODEL_RAW_CLASSES.size),
    var classFlipStreak: Int = 0,

    // --- Template visual ---
    var appearance: AppearanceTemplate? = null,

    /** Marcado para descarte silencioso (falso positivo que não qualifica como retirada real). */
    var discarded: Boolean = false
) {
    /** Rótulo pt-BR pronto para exibição (nunca mostrar [className] bruto na UI). */
    val displayLabel: String
        get() = InstrumentVocabulary.displayLabel(className)

    /** Se o objeto já foi confirmado (apareceu em frames suficientes). */
    val isConfirmed: Boolean
        get() = status != InstrumentStatus.CANDIDATE

    /** Quantos frames da janela atual tiveram match. */
    val hitsInWindow: Int
        get() = hitMask.countOneBits()

    /** Mediana das últimas confianças casadas (0 se ainda não houve match). */
    val medianConfidence: Float
        get() {
            val n = minOf(confidenceSampleCount, confidenceSamples.size)
            if (n == 0) return 0f
            val sorted = confidenceSamples.copyOf(n)
            sorted.sort()
            return if (n % 2 == 1) sorted[n / 2] else (sorted[n / 2 - 1] + sorted[n / 2]) / 2f
        }

    /** Visível na UI mesmo sem match neste frame (evita piscar em perda momentânea). */
    fun isCurrentlyVisible(nowMs: Long, graceMs: Long): Boolean =
        framesSinceSeen == 0 || nowMs - lastSeenMs <= graceMs
}
