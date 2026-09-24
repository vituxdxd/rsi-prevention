package com.rsiprevention.app.ort

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.os.SystemClock
import android.util.Log
import java.io.File

/**
 * Cria as sessões de inferência e escolhe o **melhor backend que realmente
 * funciona** para o detector deste projeto.
 *
 * ## Cadeia de backends
 *
 * O caminho previsto pelo projeto (doc de deploy, seção 5) é a NPU Hexagon via
 * QNN HTP. Este aparelho (POCO F7, SM8735, HTP V73) **não suporta FP16** no
 * DSP, então o grafo precisa ser QDQ de inteiros. O `detector_int8.onnx` atual
 * usa **ativações 16-bit (A16W8)** — a quantização int8 pura degradava os
 * logits da cabeça de classes (ver `_quantize_qnn.py` e
 * DIAGNOSTICO_CONFIANCA_ZERO.md) — e é separado em `boxes`/`scores`
 * (ver `_fix_output_split.py`).
 *
 * [CANDIDATES] é percorrida em ordem de backend e a sessão só é aceita se
 * passar na [verify] (verificação de sanidade com imagem de referência,
 * `ModelSelfTest`). Hoje só existe o candidato HTP — sem fallback: se a NPU não
 * rodar, o app falha explicitamente.
 *
 * Dois modos de falha já observados motivam isso: um EP que não reivindica nó
 * nenhum (o ORT executa tudo no CPU em silêncio) e um grafo que roda mas devolve
 * saída degenerada. A verificação é feita sobre a **confiança máxima de detecção**
 * (e não só sobre "a sessão criou").
 *
 * ## Cache de contexto QNN
 *
 * O cache só é usado no caminho da NPU (é o único backend que compila o grafo
 * demoradamente no aparelho): **geração** na primeira execução
 * (`ep.context_enable=1` + `ep.context_file_path` + `ep.context_embed_mode=1`) e
 * **carregamento** nas seguintes (sessão aberta direto do `*_ctx.onnx`, sem
 * nenhuma opção `ep.context_*` — passar `ep.context_enable` de novo faria o ORT
 * falhar com "Failed to generate EP context model since the file '...' exists
 * already").
 *
 * O cache é descartado quando os modelos em `assets/models/` mudam de fato (novo
 * export) — e **mantido** quando o APK é reinstalado sem trocar os modelos, para
 * não recompilar por causa de uma mudança só de código (ver
 * [ModelSessions.installModelAssets]).
 */
object QnnSessionFactory {

    private const val TAG = "NPU_CHECK"

    /** Backend efetivamente usado pela sessão devolvida por [createCheckedSession]. */
    enum class Backend(val label: String) {
        /** Hexagon NPU (QNN HTP) — o backend previsto pelo projeto (doc seção 5). */
        HTP("NPU Hexagon (QNN HTP)"),

        /** Adreno GPU via QNN GPU — mesma limitação da NPU (exige grafo int8). */
        QNN_GPU("GPU Adreno (QNN GPU)"),

        /** Android NN API (pode acelerar em GPU/DSP/CPU do driver). */
        NNAPI("NNAPI (Android NN API)"),

        /** XNNPACK — kernels otimizados em CPU (fp32 nativo deste detector). */
        XNNPACK("XNNPACK (CPU otimizado)"),

        /** ONNX Runtime CPU EP puro — sempre disponível, o mais lento. */
        CPU("CPU (ONNX Runtime)")
    }

    data class Session(val session: OrtSession, val backend: Backend, val modelFile: File)

    /**
     * Um candidato da cadeia.
     *
     * @param qnnOptions opções do QNN EP, ou `null` quando o backend não é QNN.
     * @param strict exige que o EP reivindique **todos** os nós do grafo
     *   (`session.disable_cpu_ep_fallback=1`). Sem isso o ORT aceita em silêncio
     *   uma sessão em que nada foi reivindicado e roda tudo no CPU (mensagem do
     *   ORT: "This session contains graph nodes that are assigned to the default
     *   CPU EP, but fallback to CPU EP has been explicitly disabled by the
     *   user."). Não se aplica a NNAPI/XNNPACK/CPU, que aceleram parte do grafo
     *   por definição.
     * @param useContextCache liga o cache de contexto QNN (só a NPU compila o
     *   grafo demoradamente no aparelho).
     */
    private data class Candidate(
        val backend: Backend,
        val qnnOptions: Map<String, String>?,
        val strict: Boolean,
        val useContextCache: Boolean = false
    )

    private val CANDIDATES = listOf(
        Candidate(
            backend = Backend.HTP,
            qnnOptions = mapOf(
                "backend_path" to "libQnnHtp.so",
                "htp_performance_mode" to "burst",
                "htp_graph_finalization_optimization_mode" to "3"
                // SM8735 NÃO suporta FP16 no HTP ("The SocModel doesn't support FP16").
                // O modelo é um grafo QDQ INT8 — a precisão é definida pelos
                // pares QuantizeLinear/DequantizeLinear do próprio grafo.
            ),
            strict = true,
            useContextCache = true
        )
        // Sem fallback: se o HTP não funcionar, o app falha explicitamente.
        // Isso garante certeza absoluta de que a inferência roda na NPU Hexagon.
    )

    /**
     * Percorre [CANDIDATES] e devolve a primeira sessão que **passar na [verify]**.
     *
     * @param modelFile grafo `.onnx` do detector (`assets/models/detector_int8.onnx`),
     *   em armazenamento real (não asset): os pesos ficam em `model.data`, irmão
     *   referenciado por caminho relativo dentro do `.onnx`, e o ONNX Runtime só
     *   resolve esse arquivo a partir de um caminho real em disco (ver [ModelSessions]).
     * @param contextCacheFile cache do contexto QNN compilado (só caminho HTP; um
     *   por variante, ex. `detector_int8_ctx.onnx`).
     * @param verify verificação de sanidade da sessão (ver `ModelSelfTest`).
     *   Recebe a sessão recém-criada e diz se ela detecta corretamente. Devolve
     *   `null` quando nenhum backend passou na verificação.
     *
     *   Histórico da NPU neste aparelho:
     *   - Float é rejeitado: o validador do DSP responde "The SocModel doesn't
     *     support FP16" para nós em precisão float (verificado com
     *     `enable_htp_fp16_precision=1`, com `htp_bf16_enable=1` +
     *     `soc_model` ≥ 88 e com todas as `htp_arch` 68/69/73/75/81). O grafo
     *     utilizável na NPU é o QDQ de inteiros (`detector_int8.onnx`, A16W8).
     *   - `session.disable_cpu_ep_fallback=1` é setado nas opções QNN: sem ele o
     *     ORT roda tudo no CPU EP em silêncio — "sessão criada" não significa
     *     nada por si só.
     */
    fun createCheckedSession(
        env: OrtEnvironment,
        modelFile: File,
        contextCacheFile: File,
        verify: (OrtSession) -> Boolean
    ): Session? {
        for (candidate in CANDIDATES) {
            val started = SystemClock.elapsedRealtime()
            var session: OrtSession? = null
            try {
                session = createSession(env, modelFile, contextCacheFile, candidate)
                val created = SystemClock.elapsedRealtime() - started
                val verifyStarted = SystemClock.elapsedRealtime()
                val approved = verify(session)
                val verified = SystemClock.elapsedRealtime() - verifyStarted
                Log.i(
                    TAG,
                    "${candidate.backend.label}: sessão em $created ms, verificação em $verified ms -> " +
                        if (approved) "APROVADO" else "reprovado"
                )
                if (approved) {
                    return Session(session, candidate.backend, modelFile)
                }
            } catch (t: Throwable) {
                Log.w(TAG, "${candidate.backend.label} indisponível: ${t.firstLine()}")
            }
            session?.close()
        }
        Log.w(TAG, "Nenhum backend passou na verificação para ${modelFile.name}")
        return null
    }

    private fun Throwable.firstLine(): String = message?.lineSequence()?.firstOrNull() ?: toString()

    /**
     * Cria a sessão de um candidato, aplicando o EP correspondente.
     *
     * @param candidate candidato da cadeia (ver [CANDIDATES]).
     */
    private fun createSession(
        env: OrtEnvironment,
        modelFile: File,
        contextCacheFile: File,
        candidate: Candidate
    ): OrtSession {
        val useContextCache = candidate.useContextCache
        val loadingCachedContext = useContextCache && contextCacheFile.isFile
        val modelPath = (if (loadingCachedContext) contextCacheFile else modelFile).absolutePath

        return OrtSession.SessionOptions().use { sessionOptions ->
            when {
                // addQnn(...) é o método público real da API Java do ONNX Runtime
                // para o QNN Execution Provider (o doc usa "addExecutionProvider"
                // apenas como pseudocódigo ilustrativo; addQnn é a assinatura real).
                candidate.qnnOptions != null -> sessionOptions.addQnn(candidate.qnnOptions)
                candidate.backend == Backend.NNAPI -> sessionOptions.addNnapi()
                candidate.backend == Backend.XNNPACK -> sessionOptions.addXnnpack(emptyMap<String, String>())
                else -> Unit // CPU EP: nenhum EP adicional
            }

            // Ver KDoc de [Candidate.strict]: sem isso o ORT esconde um fallback
            // total para o CPU como se a sessão acelerada tivesse funcionado.
            if (candidate.strict) {
                sessionOptions.addConfigEntry("session.disable_cpu_ep_fallback", "1")
            }

            if (useContextCache && !loadingCachedContext) {
                // As opções `ep.context_*` servem **apenas para GERAR** o modelo de
                // contexto (doc do ORT, "EP Context Design": "ep.context_enable —
                // Used only for EP context model generation"). Se forem passadas
                // também ao carregar um `*_ctx.onnx` existente, o ORT tenta gerar
                // o modelo de novo e falha com: "Failed to generate EP context
                // model since the file '...' exists already".
                // Por isso o modo carregamento (acima) não recebe nenhuma delas —
                // é exatamente o fluxo canônico da doc: abrir a sessão a partir do
                // `*_ctx.onnx` apenas com o execution provider configurado.
                sessionOptions.addConfigEntry("ep.context_enable", "1")
                sessionOptions.addConfigEntry("ep.context_file_path", contextCacheFile.absolutePath)
                sessionOptions.addConfigEntry("ep.context_embed_mode", "1")
            }

            env.createSession(modelPath, sessionOptions).also {
                if (loadingCachedContext) {
                    Log.i(TAG, "Contexto QNN compilado carregado do cache (${contextCacheFile.name})")
                } else if (useContextCache) {
                    Log.i(
                        TAG,
                        "Grafo compilado na NPU agora; contexto salvo em ${contextCacheFile.name} " +
                            "para as próximas aberturas"
                    )
                }
            }
        }
    }
}
