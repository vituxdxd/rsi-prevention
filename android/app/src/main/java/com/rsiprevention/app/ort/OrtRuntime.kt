package com.rsiprevention.app.ort

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel

/**
 * Ponto único de acesso ao [OrtEnvironment] do processo.
 *
 * Regra do projeto (doc de deploy, seção 9.3): manter um único `OrtEnvironment`
 * para todo o app — nunca criar um novo por frame ou por tela.
 *
 * O nível de log do runtime fica em WARNING: o nível VERBOSE (usado durante o
 * diagnóstico do backend QNN) gera milhares de linhas de logcat por inferência
 * — medido no aparelho, degrada visivelmente o pipeline por frame.
 */
object OrtRuntime {
    /**
     * Regra do projeto: um único [OrtEnvironment] para todo o app.
     *
     * Nível WARNING: o nível VERBOSE (usado no diagnóstico do backend QNN com as
     * sondas `_probe_*.py` / `QnnOptionProbe`, já removida) gera ~90 mil linhas de
     * logcat por sessão criada e degrada visivelmente o pipeline por frame.
     */
    val environment: OrtEnvironment by lazy { OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING) }
}
