package com.rsiprevention.app

import android.app.Application
import android.system.Os
import android.util.Log

/**
 * Ponto de entrada do processo (referenciado no manifesto). A inicialização
 * pesada (sessões QNN, tokenizer CLIP, embeddings de texto) roda em
 * [MainActivity] via coroutine fora da thread principal (doc de deploy seção
 * 9.3/9.4) — é lá que existe a única tela (de monitoramento) deste app.
 */
class RsiApplication : Application() {

    override fun onCreate() {
        super.onCreate()
        configureAdspLibraryPath()
    }

    /**
     * O backend QNN HTP (libQnnHtpV73Stub.so) localiza libcdsprpc.so e o skel
     * real que roda no Hexagon via a variável de ambiente ADSP_LIBRARY_PATH.
     * Sem isso o carregamento remoto falha ("Unable to load Remote symbols
     * 4000"), o router HTP não consegue identificar o SoC e a sessão QNN
     * inteira cai com QNN_BACKEND_ERROR_CANNOT_INITIALIZE. Precisa ser
     * definida antes de qualquer criação de sessão ONNX Runtime com QNN —
     * por isso fica aqui, no ponto mais cedo possível do processo.
     */
    private fun configureAdspLibraryPath() {
        val nativeLibDir = applicationInfo.nativeLibraryDir
        val systemAdspDirs =
            "/odm/lib/rfsa/adsp;/vendor/lib/rfsa/adsp/;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"
        val path = "$nativeLibDir;$systemAdspDirs"
        Os.setenv("ADSP_LIBRARY_PATH", path, true)
        Log.i("NPU_CHECK", "ADSP_LIBRARY_PATH definido: $path")
    }
}
