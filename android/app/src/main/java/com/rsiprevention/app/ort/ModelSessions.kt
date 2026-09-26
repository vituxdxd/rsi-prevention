package com.rsiprevention.app.ort

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.io.File

/**
 * Carrega o grafo ONNX/QNN do detector (doc de deploy, seção 0/4) a partir dos
 * assets e mantém a sessão viva por toda a vida do app — nunca recriada por
 * frame (seção 9.3).
 *
 * O grafo é um ONNX QDQ "portátil" (sem binário de contexto QNN pré-compilado).
 * O QNN Execution Provider compila o grafo para o HTP no próprio aparelho, na
 * criação da sessão (ver [QnnSessionFactory]).
 *
 * O grafo é um par de arquivos: `detector_int8.onnx` (estrutura) + `model.data`
 * (pesos em external data, referenciado por caminho relativo dentro do próprio
 * `.onnx`). Assets do APK vivem dentro do `.apk` (zip) e não são um caminho de
 * arquivo real, então o ONNX Runtime não consegue resolver esse arquivo-irmão a
 * partir de um asset. Por isso todo o conteúdo de `assets/models/` é copiado
 * para armazenamento interno real (`context.filesDir`) antes de criar a sessão.
 *
 * A primeira execução após a troca do modelo compila o grafo para a HTP e grava
 * o contexto compilado em `detector_int8_ctx.onnx` (mesma pasta) — nas aberturas
 * seguintes a sessão carrega desse cache, sem recompilar (ver [QnnSessionFactory]).
 */
class ModelSessions private constructor(
    val detectorSession: OrtSession,
    /** Arquivo `.onnx` do detector efetivamente aprovado pela verificação. */
    val detectorModelFile: File,
    /** Backend em que o detector está rodando (NPU, ver [QnnSessionFactory]). */
    val backend: QnnSessionFactory.Backend
) {
    fun close() {
        runCatching { detectorSession.close() }
    }

    companion object {
        private const val TAG = "ModelSessions"
        private const val ASSETS_MODELS_DIR = "models"
        private const val DETECTOR_INT8_FILE = "detector_int8.onnx"
        // Cache do contexto QNN compilado no aparelho (ver [QnnSessionFactory]).
        private const val DETECTOR_INT8_CONTEXT_CACHE_FILE = "detector_int8_ctx.onnx"
        // Assinatura dos assets de modelo da última cópia (ver installModelAssets):
        // gerada/atualizada a cada execução; o cache de contexto QNN (`*_ctx.onnx`)
        // fica na mesma pasta e só é descartado quando a assinatura muda.
        private const val SIGNATURE_MARKER_FILE = ".models_signature"

        /**
         * Deve ser chamado uma única vez (ex.: ao entrar na tela de monitoramento)
         * fora da thread principal — copiar os assets e compilar o grafo na NPU
         * (1ª execução) são operações demoradas.
         *
         * @param verify verificação de sanidade usada para aprovar o backend
         *   (ver `ModelSelfTest`): recebe a sessão candidata do detector e diz
         *   se ela detecta corretamente.
         */
        fun create(context: Context, env: OrtEnvironment, verify: (OrtSession) -> Boolean): ModelSessions {
            val modelsDir = installModelAssets(context)

            NpuLog.i("Escolhendo backend para o detector (verificação de sanidade na imagem de referência) ...")
            val detector = loadDetectorSession(env, modelsDir, verify)
            NpuLog.i(
                "${detector.modelFile.name} pronto no backend ${detector.backend.label} — " +
                    "entradas: ${detector.session.inputNames}, saídas: ${detector.session.outputNames}"
            )

            return ModelSessions(detector.session, detector.modelFile, detector.backend)
        }

        /**
         * Cria a sessão do `detector_int8.onnx` (único que a NPU aceita),
         * percorrendo a cadeia de backends (ver
         * [QnnSessionFactory.createCheckedSession]).
         *
         * Regra de cache: se o cache de contexto existir mas a NPU não for
         * aprovada com ele (arquivo corrompido por um fechamento do app no meio
         * da gravação, por exemplo), o cache é descartado e o grafo é recompilado
         * — evitando que um cache inválido deixe o app sem acelerador até uma
         * limpeza manual de dados.
         */
        private fun loadDetectorSession(
            env: OrtEnvironment,
            modelsDir: File,
            verify: (OrtSession) -> Boolean
        ): QnnSessionFactory.Session {
            val modelFile = File(modelsDir, DETECTOR_INT8_FILE)
            val contextCacheFile = File(modelsDir, DETECTOR_INT8_CONTEXT_CACHE_FILE)

            require(modelFile.isFile) {
                "Detector ausente em assets/$ASSETS_MODELS_DIR — copie $DETECTOR_INT8_FILE antes de compilar o app."
            }

            var result = QnnSessionFactory.createCheckedSession(env, modelFile, contextCacheFile, verify)

            if (result != null && result.backend != QnnSessionFactory.Backend.HTP && contextCacheFile.exists()) {
                NpuLog.w("Cache de contexto ${contextCacheFile.name} reprovado; descartando e tentando recompilar...")
                contextCacheFile.delete()
                result.session.close()
                result = QnnSessionFactory.createCheckedSession(env, modelFile, contextCacheFile, verify)
            }

            return result ?: throw IllegalStateException("Nenhum backend (NPU/CPU) aprovou o detector.")
        }

        /**
         * Garante que `assets/models/` esteja copiado em `context.filesDir/models/`
         * (grafo `.onnx` + pesos em external data `.data` irmãos).
         *
         * Essa pasta também guarda o **cache de contexto QNN** compilado no aparelho
         * (`*_ctx.onnx`, ver [QnnSessionFactory]) — que pode levar vários minutos
         * para ser gerado. Por isso a decisão de recopiar não pode ser "mudou o
         * APK": é preciso distinguir
         *  - reinstalação com os MESMOS modelos (ex.: mudança só de código) =>
         *    mantém a pasta e o cache, sem recompilar o grafo na NPU;
         *  - novo export (modelos diferentes) => apaga tudo (incluindo o cache, que
         *    ficaria inválido) e recopia.
         *
         * Critério: um marcador (`.models_signature`) guarda a assinatura
         * `nome:tamanho|...` (ordenada) dos assets da última cópia. Se a assinatura
         * bater, a pasta é reutilizada. Se o marcador estiver ausente/antigo (ex.:
         * primeira execução após uma atualização do app) mas todos os arquivos já
         * estiverem na pasta com os tamanhos exatos dos assets, a pasta também é
         * reutilizada e o marcador é regravado. Caso contrário, recopia do zero.
         */
        private fun installModelAssets(context: Context): File {
            val destDir = File(context.filesDir, ASSETS_MODELS_DIR)
            val assetNames = context.assets.list(ASSETS_MODELS_DIR).orEmpty().toList()
            require(assetNames.isNotEmpty()) {
                "Nenhum arquivo em assets/$ASSETS_MODELS_DIR — exporte os grafos e copie-os antes de compilar o app."
            }

            val signature = modelAssetsSignature(context, assetNames)
            val marker = File(destDir, SIGNATURE_MARKER_FILE)

            if (destDir.isDirectory && destFilesMatchAssets(context, destDir, assetNames)) {
                val markerMatches = marker.let { it.exists() && it.readText() == signature }
                if (markerMatches) return destDir
                NpuLog.i("Modelos em disco idênticos aos assets — mantendo a cópia e o cache de contexto QNN e atualizando o marcador.")
                marker.writeText(signature)
                return destDir
            }

            NpuLog.i("Modelos ausentes/desatualizados — copiando assets/$ASSETS_MODELS_DIR do zero (cache de contexto QNN será descartado).")
            destDir.deleteRecursively()
            destDir.mkdirs()

            for (name in assetNames) {
                val assetPath = "$ASSETS_MODELS_DIR/$name"
                val destFile = File(destDir, name)
                Log.i(TAG, "Copiando $assetPath -> ${destFile.absolutePath}")
                context.assets.open(assetPath).use { input ->
                    destFile.outputStream().use { output -> input.copyTo(output) }
                }
            }

            marker.writeText(signature)
            return destDir
        }

        /**
         * Assinatura `nome:tamanho|...` (ordenada por nome) dos assets de modelo.
         * `openFd().length` lê o tamanho real do arquivo no APK **sem descompactar**
         * (todos os arquivos de `assets/models/` estão em `noCompress`); se algum
         * deixar de estar, o `runCatching` devolve `-1` e a pasta é recopiada
         * (comportamento seguro).
         */
        private fun modelAssetsSignature(context: Context, assetNames: List<String>): String =
            assetNames.sorted().joinToString("|") { name ->
                "$name:${modelAssetLength(context, name)}"
            }

        private fun modelAssetLength(context: Context, name: String): Long =
            runCatching { context.assets.openFd("$ASSETS_MODELS_DIR/$name").length }.getOrElse { -1L }

        /** Todos os assets existem em [destDir] com os mesmos nomes e tamanhos? */
        private fun destFilesMatchAssets(context: Context, destDir: File, assetNames: List<String>): Boolean =
            assetNames.all { name ->
                val destFile = File(destDir, name)
                destFile.isFile && destFile.length() == modelAssetLength(context, name)
            }
    }
}
