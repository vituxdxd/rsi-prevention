package com.rsiprevention.app.tracking

import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.rsiprevention.app.detection.Box
import com.rsiprevention.app.detection.BoxMath
import com.rsiprevention.app.detection.Detection
import com.rsiprevention.app.vocabulary.InstrumentVocabulary
import kotlin.math.sqrt

/**
 * Tracker multi-objeto para detecções ruidosas na NPU Hexagon.
 *
 * Melhorias sobre a versão anterior (que deixava passar fantasmas e trocas de
 * classe):
 *
 *  1. **Associação global**: todos os pares (track × detecção) são pontuados e
 *     atribuídos do melhor para o pior, garantindo que cada detecção vá para o
 *     track mais provável (sem “pegar o primeiro que servir”).
 *
 *  2. **Modelo de movimento**: cada track guarda velocidade (px/ms) e a caixa é
 *     comparada na posição **predita** — essencial para objetos que se movem
 *     rápido entre frames e para não trocar dois instrumentos próximos.
 *
 *  3. **Template visual**: dHash de 64 bits + histograma RGB central
 *     ([Appearance]) com limiar mínimo de similaridade. Associação com conteúdo
 *     visual muito diferente é rejeitada mesmo com IoU alto.
 *
 *  4. **Votação de classe**: o rótulo exibido é o da classe com mais votos
 *     ponderados por confiança, com histerese — quando o modelo alterna entre
 *     classes parecidas (ex.: tesoura × pinça), o track **não** duplica e o
 *     rótulo só muda após [CLASS_FLIP_FRAMES] frames consistentes.
 *
 *  5. **Confirmação K-de-N + certeza sustentada**: candidato só vira ON_TABLE
 *     com presença em [CONFIRM_HITS] de [CONFIRM_WINDOW] frames, falha máxima
 *     de [MAX_MISS_RUN_IN_WINDOW] frame, confiança **pico** ≥
 *     [MIN_PEAK_CONFIDENCE], **mediana** ≥ [MIN_MEDIAN_CONFIDENCE] **e** pelo
 *     menos [MIN_HIGH_CONFIDENCE_DURATION_MS] ms acumulados de confiança ≥
 *     [HIGH_CONFIDENCE_THRESHOLD] — frames de confiança baixa ou ausência
 *     descontam o acumulado. Um pico isolado (mesmo alto) não confirma; certeza
 *     alta e persistente por segundos sim. Candidatos que não se sustentam são
 *     purgados (ver [PURGE_UNSEEN_CANDIDATE_AFTER_MS] e [CANDIDATE_MAX_AGE_MS]).
 *
 *  6. **Descarte silencioso**: track confirmado que some antes de
 *     [MIN_ON_TABLE_MS_FOR_REMOVAL] ms ou com pico baixo é removido sem gerar
 *     evento de “retirada” (não polui a lista nem os avisos). Objetos reais
 *     que ficam na mesa geram REMOVED/RETURNED normalmente.
 */
class InstrumentTracker(
    private val onInstrumentRemoved: ((TrackedInstrument) -> Unit)? = null,
    private val onInstrumentReturned: ((TrackedInstrument) -> Unit)? = null
) {
    private val tracked = mutableListOf<TrackedInstrument>()
    private var nextId = 0
    private var lastUpdateMs = 0L

    companion object {
        private const val TAG = "Tracker"

        // --- Confirmação ---
        const val CONFIRM_WINDOW = 8
        const val CONFIRM_HITS = 4
        const val MAX_MISS_RUN_IN_WINDOW = 1
        const val MIN_AGE_FRAMES = 4
        const val MIN_NEW_DETECTION_CONFIDENCE = 0.35f
        const val MIN_PEAK_CONFIDENCE = 0.55f
        const val MIN_MEDIAN_CONFIDENCE = 0.40f

        /**
         * Certeza sustentada: o K-de-N acima só cobre a presença recente
         * (~[CONFIRM_WINDOW] frames). Para virar instrumento confirmado, o
         * candidato também precisa acumular [MIN_HIGH_CONFIDENCE_DURATION_MS] ms
         * de presença com confiança ≥ [HIGH_CONFIDENCE_THRESHOLD]. O integrador
         * desconta o mesmo tempo em frames com confiança baixa ou sem detecção —
         * um pico isolado (mesmo alto) não soma; certeza persistente sim.
         */
        const val HIGH_CONFIDENCE_THRESHOLD = 0.65f
        const val MIN_HIGH_CONFIDENCE_DURATION_MS = 3000f
        /** Passo máximo de integração/desconto por frame (protege contra pausas longas do monitoramento). */
        const val MAX_CONFIDENCE_STEP_MS = 250f

        // --- Associação ---
        const val MIN_MATCH_SCORE = 0.30f
        const val MIN_APPEARANCE_SIMILARITY = 0.35f
        const val GATE_IOU = 0.05f
        const val GATE_CENTER_DISTANCE = 0.60f
        const val BOX_SMOOTHING = 0.65f
        const val VELOCITY_SMOOTHING = 0.30f
        const val MAX_VELOCITY = 2.0f
        const val MAX_PREDICTION_MS = 1000L

        // --- Ciclo de vida (tempo) ---
        const val VISIBLE_GRACE_MS = 450L
        const val REMOVE_AFTER_MISSED_MS = 1200L
        /** Candidato sem nenhuma detecção por este tempo é descartado (fantasma). */
        const val PURGE_UNSEEN_CANDIDATE_AFTER_MS = 900L
        /** Candidato que nunca atinge a certeza sustentada é descartado (libera slots). */
        const val CANDIDATE_MAX_AGE_MS = 10_000L
        const val MIN_ON_TABLE_MS_FOR_REMOVAL = 1200L

        // --- Plausibilidade geométrica ---
        const val MIN_BOX_AREA = 300f
        const val MAX_BOX_AREA = 0.75f * 640f * 640f
        const val MAX_ASPECT_RATIO = 14f
        const val MAX_TRACKS = 12

        // --- Votação de classe ---
        const val CLASS_VOTE_DECAY = 0.85f
        const val CLASS_FLIP_MARGIN = 0.20f
        const val CLASS_FLIP_FRAMES = 3

        private const val WINDOW_MASK = (1 shl CONFIRM_WINDOW) - 1
    }

    /**
     * Instrumentos para exibição: confirmados e visíveis agora (ou dentro da
     * carência de [VISIBLE_GRACE_MS]) mais o histórico de removidos/não
     * devolvidos (usado pela checagem final).
     */
    val activeInstruments: List<TrackedInstrument>
        get() = tracked.filter {
            it.isConfirmed && (
                it.isCurrentlyVisible(lastUpdateMs, VISIBLE_GRACE_MS) ||
                    it.status == InstrumentStatus.REMOVED ||
                    it.status == InstrumentStatus.NOT_RETURNED
                )
        }

    @Synchronized
    fun update(
        detections: List<Detection>,
        frameBitmap: Bitmap? = null,
        nowMs: Long = SystemClock.elapsedRealtime(),
        wallNowMs: Long = System.currentTimeMillis()
    ) {
        // Incremento de tempo desde o frame anterior processado, limitado para
        // que uma pausa longa do monitoramento não infle o integrador de certeza.
        val frameDtMs = (nowMs - lastUpdateMs).coerceIn(0L, MAX_CONFIDENCE_STEP_MS.toLong()).toFloat()
        lastUpdateMs = nowMs

        val plausible = detections.filter {
            BoxMath.isPlausibleInstrument(it.box, MIN_BOX_AREA, MAX_BOX_AREA, MAX_ASPECT_RATIO)
        }

        // Assinatura visual por detecção (uma extração por detecção, não por par).
        val detectionAppearance: Map<Int, AppearanceTemplate> = if (frameBitmap != null) {
            plausible.mapIndexedNotNull { index, detection ->
                runCatching { Appearance.extract(frameBitmap, detection.box) }
                    .getOrNull()
                    ?.let { index to it }
            }.toMap()
        } else {
            emptyMap()
        }

        val predicted = tracked.map { predictedBox(it, nowMs) }

        // 1. Pontuação de todos os pares e atribuição global (melhor primeiro).
        data class Pairing(val trackIndex: Int, val detectionIndex: Int, val score: Float)

        val pairings = ArrayList<Pairing>()
        for (trackIndex in tracked.indices) {
            for (detectionIndex in plausible.indices) {
                val score = matchScore(
                    tracked[trackIndex],
                    plausible[detectionIndex],
                    detectionAppearance[detectionIndex],
                    predicted[trackIndex]
                )
                if (score >= MIN_MATCH_SCORE) {
                    pairings.add(Pairing(trackIndex, detectionIndex, score))
                }
            }
        }
        pairings.sortByDescending { it.score }

        val matchedTracks = BooleanArray(tracked.size)
        val matchedDetections = BooleanArray(plausible.size)
        for (pairing in pairings) {
            if (matchedTracks[pairing.trackIndex] || matchedDetections[pairing.detectionIndex]) continue
            matchedTracks[pairing.trackIndex] = true
            matchedDetections[pairing.detectionIndex] = true
            applyMatch(
                tracked[pairing.trackIndex],
                plausible[pairing.detectionIndex],
                detectionAppearance[pairing.detectionIndex],
                nowMs,
                wallNowMs
            )
        }

        // 2. Tracks sem match: ausência.
        for (trackIndex in tracked.indices) {
            if (!matchedTracks[trackIndex]) registerMiss(tracked[trackIndex], nowMs, frameDtMs, wallNowMs)
        }

        // 3. Detecções sem match: novo candidato (se confiável e couber).
        for (detectionIndex in plausible.indices) {
            if (matchedDetections[detectionIndex]) continue
            val detection = plausible[detectionIndex]
            if (detection.confidence < MIN_NEW_DETECTION_CONFIDENCE) continue
            val activeTracks = tracked.count {
                it.status != InstrumentStatus.REMOVED && it.status != InstrumentStatus.NOT_RETURNED
            }
            if (activeTracks >= MAX_TRACKS) break
            tracked.add(newCandidate(detection, detectionAppearance[detectionIndex], nowMs))
        }

        // 4. Purga: falsos positivos descartados e candidatos que não se sustentam.
        val iterator = tracked.iterator()
        while (iterator.hasNext()) {
            val track = iterator.next()
            val purge = track.discarded ||
                (track.status == InstrumentStatus.CANDIDATE &&
                    (nowMs - track.lastSeenMs >= PURGE_UNSEEN_CANDIDATE_AFTER_MS ||
                        nowMs - track.firstSeenMs >= CANDIDATE_MAX_AGE_MS))
            if (!purge) continue
            if (track.status == InstrumentStatus.CANDIDATE && !track.discarded) {
                // Diagnóstico: por que um objeto real pode não ter virado instrumento
                // (confiança que não se manteve ≥ limiar pelo tempo mínimo).
                Log.i(
                    TAG,
                    "Candidato descartado sem confirmar: ${track.className} id=${track.id} " +
                        "(certeza=${"%.1f".format(track.highConfidenceMs / 1000f)}s de " +
                        "${"%.1f".format(MIN_HIGH_CONFIDENCE_DURATION_MS / 1000f)}s, " +
                        "pico=${"%.1f".format(track.peakConfidence * 100)}%)"
                )
            }
            iterator.remove()
        }
    }

    private fun applyMatch(
        track: TrackedInstrument,
        detection: Detection,
        appearance: AppearanceTemplate?,
        nowMs: Long,
        wallNowMs: Long
    ) {
        val dtMs = (nowMs - track.lastSeenMs).coerceAtLeast(1L).toFloat()

        // Velocidade suavizada (px/ms), limitada para não explodir.
        track.vx = smoothVelocity(track.vx, (detection.box.centerX - track.box.centerX) / dtMs)
        track.vy = smoothVelocity(track.vy, (detection.box.centerY - track.box.centerY) / dtMs)
        track.vw = smoothVelocity(track.vw, (detection.box.width - track.box.width) / dtMs)
        track.vh = smoothVelocity(track.vh, (detection.box.height - track.box.height) / dtMs)

        // Caixa e confiança suavizadas (display estável).
        track.box = track.box.lerp(detection.box, BOX_SMOOTHING)
        track.confidence = if (track.confidence <= 0f) {
            detection.confidence
        } else {
            0.5f * track.confidence + 0.5f * detection.confidence
        }
        if (detection.confidence > track.peakConfidence) {
            track.peakConfidence = detection.confidence
        }

        track.confidenceSamples[track.confidenceSampleCount % CONFIRM_WINDOW] = detection.confidence
        track.confidenceSampleCount++

        track.hitMask = ((track.hitMask shl 1) or 1) and WINDOW_MASK
        track.ageFrames++
        track.framesSinceSeen = 0
        track.lastSeenMs = nowMs

        updateClassVotes(track, detection)

        // Integrador de certeza sustentada: alta confiança soma o tempo do frame,
        // confiança baixa desconta o mesmo tempo (nunca abaixo de zero).
        val stepMs = dtMs.coerceIn(1f, MAX_CONFIDENCE_STEP_MS)
        track.highConfidenceMs = if (detection.confidence >= HIGH_CONFIDENCE_THRESHOLD) {
            track.highConfidenceMs + stepMs
        } else {
            (track.highConfidenceMs - stepMs).coerceAtLeast(0f)
        }

        if (appearance != null) {
            track.appearance = track.appearance?.let { Appearance.blend(it, appearance) } ?: appearance
        }

        when (track.status) {
            InstrumentStatus.CANDIDATE -> tryConfirm(track, nowMs, wallNowMs)
            InstrumentStatus.REMOVED, InstrumentStatus.NOT_RETURNED -> {
                track.status = InstrumentStatus.RETURNED
                track.onTableSinceMs = nowMs
                track.returnedAtWallMs = wallNowMs
                Log.i(TAG, "Devolvido: ${track.className} id=${track.id}")
                onInstrumentReturned?.invoke(track)
            }
            else -> Unit
        }
    }

    private fun tryConfirm(track: TrackedInstrument, nowMs: Long, wallNowMs: Long) {
        val confirmed = track.hitsInWindow >= CONFIRM_HITS &&
            longestMissRun(track.hitMask, CONFIRM_WINDOW, track.ageFrames) <= MAX_MISS_RUN_IN_WINDOW &&
            track.ageFrames >= MIN_AGE_FRAMES &&
            track.peakConfidence >= MIN_PEAK_CONFIDENCE &&
            track.medianConfidence >= MIN_MEDIAN_CONFIDENCE &&
            track.highConfidenceMs >= MIN_HIGH_CONFIDENCE_DURATION_MS
        if (!confirmed) return

        track.status = InstrumentStatus.ON_TABLE
        track.onTableSinceMs = nowMs
        track.confirmedAtWallMs = wallNowMs
        Log.i(
            TAG,
            "Confirmado: ${track.className} id=${track.id} " +
                "(${track.hitsInWindow}/${CONFIRM_WINDOW} frames, " +
                "pico=${"%.1f".format(track.peakConfidence * 100)}%, " +
                "mediana=${"%.1f".format(track.medianConfidence * 100)}%, " +
                "certeza=${"%.1f".format(track.highConfidenceMs / 1000f)}s)"
        )
    }

    private fun registerMiss(track: TrackedInstrument, nowMs: Long, frameDtMs: Float, wallNowMs: Long) {
        track.framesSinceSeen++
        track.hitMask = (track.hitMask shl 1) and WINDOW_MASK
        track.highConfidenceMs = (track.highConfidenceMs - frameDtMs).coerceAtLeast(0f)

        if (track.status == InstrumentStatus.CANDIDATE) return

        val missedForMs = nowMs - track.lastSeenMs
        if (missedForMs < REMOVE_AFTER_MISSED_MS) return
        if (track.status == InstrumentStatus.REMOVED || track.status == InstrumentStatus.NOT_RETURNED) return

        val onTableMs = track.lastSeenMs - track.onTableSinceMs
        val qualifiesAsRemoval = track.peakConfidence >= MIN_PEAK_CONFIDENCE &&
            onTableMs >= MIN_ON_TABLE_MS_FOR_REMOVAL
        if (qualifiesAsRemoval) {
            track.status = InstrumentStatus.REMOVED
            track.wasEverRemoved = true
            track.removedAtWallMs = wallNowMs
            Log.i(TAG, "Retirado: ${track.className} id=${track.id}")
            onInstrumentRemoved?.invoke(track)
        } else {
            track.discarded = true
            Log.i(TAG, "Descartado (curto/baixa confiança): ${track.className} id=${track.id}")
        }
    }

    private fun updateClassVotes(track: TrackedInstrument, detection: Detection) {
        for (i in track.classVotes.indices) track.classVotes[i] *= CLASS_VOTE_DECAY
        if (detection.classId in track.classVotes.indices) {
            track.classVotes[detection.classId] += detection.confidence
        }

        val bestId = track.classVotes.indices.maxByOrNull { track.classVotes[it] } ?: return
        if (bestId == track.classId) {
            track.classFlipStreak = 0
            return
        }

        val bestVotes = track.classVotes[bestId]
        val currentVotes = track.classVotes.getOrElse(track.classId) { 0f }
        if (bestVotes - currentVotes >= CLASS_FLIP_MARGIN) {
            track.classFlipStreak++
            if (track.classFlipStreak >= CLASS_FLIP_FRAMES) {
                track.classId = bestId
                track.className = InstrumentVocabulary.MODEL_RAW_CLASSES[bestId]
                track.classFlipStreak = 0
                Log.i(TAG, "Classe estabilizada: ${track.className} id=${track.id}")
            }
        } else {
            track.classFlipStreak = 0
        }
    }

    private fun matchScore(
        track: TrackedInstrument,
        detection: Detection,
        appearance: AppearanceTemplate?,
        predictedBox: Box
    ): Float {
        val iou = BoxMath.iou(predictedBox, detection.box)
        val centerDistance = BoxMath.centerDistance(predictedBox, detection.box) /
            sqrt(predictedBox.area.coerceAtLeast(1f))
        if (iou < GATE_IOU && centerDistance > GATE_CENTER_DISTANCE) return 0f

        val centerScore = (1f - centerDistance).coerceIn(0f, 1f)
        val sizeScore = BoxMath.sizeSimilarity(predictedBox, detection.box)

        val appearanceScore = if (track.appearance != null && appearance != null) {
            Appearance.similarity(track.appearance!!, appearance).also {
                if (it < MIN_APPEARANCE_SIMILARITY) return 0f
            }
        } else {
            0.5f
        }

        val classScore = if (detection.classId == track.classId) 1f else 0.55f

        return 0.40f * iou + 0.15f * centerScore + 0.15f * sizeScore +
            0.20f * appearanceScore + 0.10f * classScore
    }

    private fun predictedBox(track: TrackedInstrument, nowMs: Long): Box {
        val dt = (nowMs - track.lastSeenMs).coerceIn(0L, MAX_PREDICTION_MS).toFloat()
        val width = track.box.width.coerceAtLeast(1f)
        val height = track.box.height.coerceAtLeast(1f)
        val scaleX = ((width + track.vw * dt) / width).coerceIn(0.5f, 2f)
        val scaleY = ((height + track.vh * dt) / height).coerceIn(0.5f, 2f)
        return track.box
            .scaledAroundCenter(scaleX, scaleY)
            .translated(track.vx * dt, track.vy * dt)
    }

    private fun smoothVelocity(current: Float, observed: Float): Float {
        val smoothed = (1f - VELOCITY_SMOOTHING) * current + VELOCITY_SMOOTHING * observed
        return smoothed.coerceIn(-MAX_VELOCITY, MAX_VELOCITY)
    }

    private fun longestMissRun(mask: Int, window: Int, validFrames: Int): Int {
        var best = 0
        var run = 0
        val considered = minOf(window, validFrames)
        for (i in 0 until considered) {
            if ((mask shr i) and 1 == 1) {
                run = 0
            } else {
                run++
                if (run > best) best = run
            }
        }
        return best
    }

    private fun newCandidate(
        detection: Detection,
        appearance: AppearanceTemplate?,
        nowMs: Long
    ) = TrackedInstrument(
        id = nextId++,
        classId = detection.classId,
        className = detection.className,
        box = detection.box,
        confidence = detection.confidence,
        status = InstrumentStatus.CANDIDATE,
        firstSeenMs = nowMs,
        lastSeenMs = nowMs,
        hitMask = 1,
        ageFrames = 1,
        peakConfidence = detection.confidence,
        classVotes = FloatArray(InstrumentVocabulary.MODEL_RAW_CLASSES.size).also {
            if (detection.classId in it.indices) it[detection.classId] = detection.confidence
        },
        appearance = appearance
    ).also {
        it.confidenceSamples[0] = detection.confidence
        it.confidenceSampleCount = 1
    }

    @Synchronized
    fun performFinalCheck(): List<TrackedInstrument> {
        val notReturned = tracked.filter { it.isConfirmed && it.status == InstrumentStatus.REMOVED }
        val wallNow = System.currentTimeMillis()
        notReturned.forEach {
            it.status = InstrumentStatus.NOT_RETURNED
            it.notReturnedAtWallMs = wallNow
        }
        return notReturned
    }

    @Synchronized
    fun reset() {
        tracked.clear()
        nextId = 0
    }
}
