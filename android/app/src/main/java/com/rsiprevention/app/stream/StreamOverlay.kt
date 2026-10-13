package com.rsiprevention.app.stream

import com.rsiprevention.app.detection.Box

/**
 * Snapshot imutável de uma caixa a desenhar no frame transmitido.
 *
 * É montado na **thread de inferência** (logo após o `tracker.update`), para
 * que a thread de encode/rede nunca leia o estado mutável do tracker — sem
 * corrida de dados e sem precisar sincronizar o tracker com o streaming.
 */
data class StreamOverlay(val box: Box, val label: String)
