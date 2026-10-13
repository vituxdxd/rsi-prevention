package com.rsiprevention.app.alert

import android.content.Context
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Handler
import android.os.Looper
import android.os.VibrationEffect
import android.os.VibratorManager

/**
 * Alerta sonoro + tátil de prevenção de RSI: disparado quando a checagem final
 * ([com.rsiprevention.app.tracking.InstrumentTracker.performFinalCheck]) indica
 * que ao menos um instrumento retirado não retornou à mesa. A parte visual
 * (diálogo com a lista de instrumentos) fica na Activity, que já possui o
 * contexto de UI (ver strings `check_dialog_*`).
 *
 * `minSdk` deste projeto é 31, então `VibratorManager` (API 31+) é usado sem
 * necessidade de fallback para a API antiga de `Vibrator`.
 */
class AlertManager(context: Context) {
    private val appContext = context.applicationContext

    fun alertNotReturned() {
        vibrate()
        playAlertTone()
    }

    private fun vibrate() {
        val manager = appContext.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
        val pattern = longArrayOf(0, 300, 150, 300, 150, 300)
        manager.defaultVibrator.vibrate(VibrationEffect.createWaveform(pattern, -1))
    }

    private fun playAlertTone() {
        val toneGenerator = ToneGenerator(AudioManager.STREAM_ALARM, ToneGenerator.MAX_VOLUME)
        toneGenerator.startTone(ToneGenerator.TONE_SUP_ERROR, ALERT_TONE_DURATION_MS)
        // ToneGenerator precisa ser liberado após o tom terminar de tocar;
        // release() imediato cortaria o som.
        Handler(Looper.getMainLooper()).postDelayed(
            { toneGenerator.release() },
            (ALERT_TONE_DURATION_MS + 100).toLong()
        )
    }

    companion object {
        private const val ALERT_TONE_DURATION_MS = 800
    }
}
