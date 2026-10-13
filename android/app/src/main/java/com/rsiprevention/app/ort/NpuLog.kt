package com.rsiprevention.app.ort

import android.util.Log

/**
 * Log unificado da NPU: escreve no logcat (tag `NPU_CHECK`, como antes) e no
 * [NpuActivationLog], que alimenta o painel discreto da tela principal.
 */
object NpuLog {

    private const val TAG = "NPU_CHECK"

    fun i(message: String) {
        Log.i(TAG, message)
        NpuActivationLog.log(message)
    }

    fun w(message: String) {
        Log.w(TAG, message)
        NpuActivationLog.log(message)
    }
}
