package com.rsiprevention.app

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.lifecycle.lifecycleScope
import com.rsiprevention.app.alert.AlertManager
import com.rsiprevention.app.camera.CameraController
import com.rsiprevention.app.databinding.ActivityMainBinding
import com.rsiprevention.app.detection.ModelSelfTest
import com.rsiprevention.app.detection.YoloWorldDetector
import com.rsiprevention.app.ort.ModelSessions
import com.rsiprevention.app.ort.NpuActivationLog
import com.rsiprevention.app.ort.NpuLog
import com.rsiprevention.app.ort.OrtRuntime
import com.rsiprevention.app.ort.QnnSessionFactory
import com.rsiprevention.app.stream.FrameStreamer
import com.rsiprevention.app.stream.HotspotAddress
import com.rsiprevention.app.stream.StreamServer
import com.rsiprevention.app.tracking.InstrumentStatus
import com.rsiprevention.app.tracking.InstrumentTracker
import com.rsiprevention.app.tracking.TrackedInstrument
import com.rsiprevention.app.ui.QrCodeDialog
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Única tela deste app (doc de deploy seção 0/6): preview de câmera + overlay
 * de detecção/tracking + controles de monitoramento e checagem final de RSI.
 *
 * Fluxo: permissão de câmera -> carregar a sessão QNN do detector (uma única
 * vez, fora da thread principal) -> ligar CameraX -> a cada frame processado
 * (enquanto o monitoramento está ativo), atualizar overlay + lista de
 * instrumentos.
 *
 * A lista de instrumentos mostra data/hora do último evento de cada track, e a
 * barra superior tem um painel discreto (expansível/ocultável) com o log da
 * ativação da NPU Hexagon ([NpuActivationLog]).
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    private var modelSessions: ModelSessions? = null
    private var cameraController: CameraController? = null
    private lateinit var tracker: InstrumentTracker
    private lateinit var alertManager: AlertManager

    /** Evita inicializações concorrentes do pipeline (ex.: toques repetidos no retry). */
    private var pipelineInitInFlight = false

    private var streamServer: StreamServer? = null
    private var streamer: FrameStreamer? = null

    /** URL exibida na tela enquanto a transmissão está ativa (para o QR). */
    private var streamUrl: String? = null

    /** Recebe as atualizações do [NpuActivationLog] (de qualquer thread) e redesenha na UI. */
    private val npuLogListener: (List<NpuActivationLog.Entry>) -> Unit = { entries ->
        runOnUiThread { renderNpuLog(entries) }
    }

    private val cameraPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) initializePipeline() else showPermissionDenied()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        alertManager = AlertManager(this)
        setupButtons()
        setupNpuLogPanel()
        enterFullscreen()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            initializePipeline()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    /**
     * Tela cheia: esconde a barra de status (hora, Wi-Fi, operadora). A barra
     * reaparece temporariamente por um gesto no topo e some sozinha de novo.
     * Reaplicado no [onResume] porque alguns sistemas (HyperOS) a mostram de
     * volta ao voltar do segundo plano.
     */
    private fun enterFullscreen() {
        WindowInsetsControllerCompat(window, window.decorView).apply {
            hide(WindowInsetsCompat.Type.statusBars())
            systemBarsBehavior = WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        }
    }

    override fun onResume() {
        super.onResume()
        enterFullscreen()
    }

    private fun setupButtons() {
        binding.startStopButton.setOnClickListener { toggleMonitoring() }
        binding.finalCheckButton.setOnClickListener { performFinalCheck() }
        // Sem botão dedicado de "tentar novamente" no layout: permite reabrir o
        // pedido de permissão tocando na própria mensagem de status.
        binding.monitoringStatusText.setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
        // Retry da inicialização do pipeline (NPU) tocando no status de erro.
        binding.npuStatusText.setOnClickListener {
            if (modelSessions == null &&
                ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
            ) {
                binding.npuStatusText.text = getString(R.string.npu_status_initializing)
                initializePipeline()
            }
        }
        // Transmissão Wi-Fi (independente do monitoramento).
        binding.streamSwitch.setOnCheckedChangeListener { _, checked -> setStreaming(checked) }
        binding.streamStatusText.setOnClickListener {
            streamUrl?.let { url -> QrCodeDialog.show(this, url) }
        }
    }

    /** Painel discreto do log da NPU: começa oculto; o toque alterna mostrar/ocultar. */
    private fun setupNpuLogPanel() {
        binding.npuLogToggle.setOnClickListener {
            val expand = binding.npuLogScroll.visibility != View.VISIBLE
            binding.npuLogScroll.visibility = if (expand) View.VISIBLE else View.GONE
            binding.npuLogToggle.text = getString(
                if (expand) R.string.npu_log_toggle_hide else R.string.npu_log_toggle_show
            )
            if (expand) scrollNpuLogToEnd()
        }
        NpuActivationLog.addListener(npuLogListener)
    }

    private fun renderNpuLog(entries: List<NpuActivationLog.Entry>) {
        binding.npuLogText.text = if (entries.isEmpty()) {
            getString(R.string.npu_log_empty)
        } else {
            entries.joinToString("\n") { "${NpuActivationLog.formatTime(it.wallMs)} ${it.message}" }
        }
        if (binding.npuLogScroll.visibility == View.VISIBLE) scrollNpuLogToEnd()
    }

    private fun scrollNpuLogToEnd() {
        binding.npuLogScroll.post { binding.npuLogScroll.fullScroll(View.FOCUS_DOWN) }
    }

    private fun showPermissionDenied() {
        binding.npuStatusText.text = getString(R.string.npu_status_error)
        binding.monitoringStatusText.text = getString(R.string.camera_permission_denied)
    }

    private fun showPipelineInitError() {
        binding.npuStatusText.text = getString(R.string.npu_status_error_retry)
    }

    /** Carrega a sessão QNN do detector fora da thread principal (doc seção 9.3/9.4). */
    private fun initializePipeline() {
        if (pipelineInitInFlight) return
        pipelineInitInFlight = true
        lifecycleScope.launch(Dispatchers.Default) {
            try {
                val sessions = ModelSessions.create(applicationContext, OrtRuntime.environment) { session ->
                    ModelSelfTest.passes(
                        applicationContext,
                        YoloWorldDetector(
                            OrtRuntime.environment,
                            session
                        )
                    )
                }
                val detector = YoloWorldDetector(OrtRuntime.environment, sessions.detectorSession)

                withContext(Dispatchers.Main) {
                    pipelineInitInFlight = false
                    if (isDestroyed || isFinishing) {
                        // A tela morreu durante a carga (que pode levar minutos na
                        // 1ª compilação NPU): fecha as sessões em vez de vazá-las
                        // ligando a câmera numa Activity destruída.
                        sessions.close()
                        return@withContext
                    }
                    modelSessions = sessions
                    onPipelineReady(detector, sessions.backend)
                }
            } catch (t: Throwable) {
                Log.e(TAG, "Falha ao inicializar pipeline de detecção", t)
                NpuLog.w("Falha ao inicializar o pipeline: ${t.message?.lineSequence()?.firstOrNull() ?: t}")
                withContext(Dispatchers.Main) {
                    pipelineInitInFlight = false
                    if (!isDestroyed && !isFinishing) showPipelineInitError()
                }
            }
        }
    }

    private fun onPipelineReady(detector: YoloWorldDetector, backend: QnnSessionFactory.Backend) {
        binding.npuStatusText.text = getString(
            when (backend) {
                QnnSessionFactory.Backend.HTP -> R.string.npu_status_active
                QnnSessionFactory.Backend.QNN_GPU -> R.string.npu_status_gpu
                QnnSessionFactory.Backend.NNAPI -> R.string.npu_status_nnapi
                QnnSessionFactory.Backend.XNNPACK -> R.string.npu_status_xnnpack
                QnnSessionFactory.Backend.CPU -> R.string.npu_status_cpu
            }
        )
        NpuLog.i(getString(R.string.backend_confirmation_format, backend.label))

        tracker = InstrumentTracker(
            onInstrumentRemoved = { instrument ->
                runOnUiThread {
                    Toast.makeText(this, "${getString(R.string.instrument_removed_log)}: ${instrument.displayLabel}", Toast.LENGTH_SHORT).show()
                }
            },
            onInstrumentReturned = { instrument ->
                runOnUiThread {
                    Toast.makeText(this, "${getString(R.string.instrument_returned_log)}: ${instrument.displayLabel}", Toast.LENGTH_SHORT).show()
                }
            }
        )

        cameraController = CameraController(
            context = applicationContext,
            lifecycleOwner = this,
            previewView = binding.previewView,
            detector = detector,
            tracker = tracker,
            onFrameProcessed = ::onFrameProcessed
        ).also { it.start() }

        // A câmera já está no ar: a transmissão pode ser ligada pelo usuário.
        binding.streamSwitch.isEnabled = true
    }

    private fun setStreaming(enabled: Boolean) {
        if (enabled == (streamServer != null)) return
        if (enabled) startStreaming() else stopStreaming()
    }

    private fun startStreaming() {
        val controller = cameraController
        if (controller == null) {
            binding.streamSwitch.isChecked = false
            Toast.makeText(this, getString(R.string.stream_unavailable), Toast.LENGTH_SHORT).show()
            return
        }

        val server = runCatching { StreamServer().also { it.start() } }.getOrElse { t ->
            Log.e(TAG, "Falha ao abrir o servidor de transmissão", t)
            binding.streamSwitch.isChecked = false
            Toast.makeText(this, getString(R.string.stream_error), Toast.LENGTH_SHORT).show()
            return
        }
        val frameStreamer = FrameStreamer(
            boxColor = ContextCompat.getColor(this, R.color.overlay_box_on_table),
            onJpeg = server::publishFrame
        ).also { it.start() }

        streamServer = server
        streamer = frameStreamer
        controller.streamer = frameStreamer
        controller.streamingEnabled = true

        // A câmera precisa do app em primeiro plano: mantém a tela acesa.
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        val ip = HotspotAddress.find() ?: HotspotAddress.FALLBACK_IP
        val url = HotspotAddress.url(ip, StreamServer.DEFAULT_PORT)
        streamUrl = url
        binding.streamStatusText.text = getString(R.string.stream_status_on, url)
        Log.i(TAG, "Transmissão iniciada: $url")
    }

    private fun stopStreaming() {
        cameraController?.streamingEnabled = false
        cameraController?.streamer = null
        streamer?.stop()
        streamer = null
        streamServer?.stop()
        streamServer = null
        streamUrl = null

        window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        binding.streamSwitch.isChecked = false
        binding.streamStatusText.text = getString(R.string.stream_status_off)
        Log.i(TAG, "Transmissão encerrada")
    }

    private fun onFrameProcessed(instruments: List<TrackedInstrument>, frameWidth: Int, frameHeight: Int) {
        binding.overlayView.update(instruments, frameWidth, frameHeight)
        updateInstrumentsListText(instruments)
    }

    /**
     * Assinatura barata do que a lista exibe (ids, status e instantes de evento).
     * A lista só é remontada quando algo muda de fato — antes, sort + join +
     * `getString` rodavam a cada frame processado na thread principal.
     */
    private var lastListSignature: Long = Long.MIN_VALUE

    private fun updateInstrumentsListText(instruments: List<TrackedInstrument>) {
        val signature = listSignature(instruments)
        if (signature == lastListSignature) return
        lastListSignature = signature

        binding.instrumentsListText.text = if (instruments.isEmpty()) {
            getString(R.string.instruments_list_empty)
        } else {
            instruments
                .sortedWith(compareBy({ it.status.ordinal }, { it.displayLabel }, { eventWallMs(it) }))
                .joinToString("\n") { instrument ->
                    val statusText = getString(instrument.status.displayNameResId)
                    val time = eventWallMs(instrument)
                        .takeIf { it > 0L }
                        ?.let(NpuActivationLog::formatTime)
                        ?: "--:--:--"
                    "$time — ${instrument.displayLabel} — $statusText"
                }
        }
    }

    /** Hash do conteúdo exibido; colisões são inofensivas (some só até o próximo evento). */
    private fun listSignature(instruments: List<TrackedInstrument>): Long {
        var hash = 1125899906842597L
        hash = 31 * hash + instruments.size
        for (instrument in instruments) {
            hash = 31 * hash + instrument.id
            hash = 31 * hash + instrument.status.ordinal
            hash = 31 * hash + eventWallMs(instrument)
        }
        return hash
    }

    /** Instante (relógio de parede) do último evento de status do instrumento. */
    private fun eventWallMs(instrument: TrackedInstrument): Long = when (instrument.status) {
        InstrumentStatus.ON_TABLE -> instrument.confirmedAtWallMs
        InstrumentStatus.REMOVED -> instrument.removedAtWallMs
        InstrumentStatus.RETURNED -> instrument.returnedAtWallMs
        InstrumentStatus.NOT_RETURNED -> instrument.notReturnedAtWallMs
        // Candidatos não entram na lista (ver InstrumentTracker.activeInstruments).
        InstrumentStatus.CANDIDATE -> instrument.confirmedAtWallMs
    }

    private fun toggleMonitoring() {
        val controller = cameraController ?: return
        val enabling = !controller.monitoringEnabled
        controller.monitoringEnabled = enabling

        binding.startStopButton.text = getString(
            if (enabling) R.string.button_stop_monitoring else R.string.button_start_monitoring
        )
        binding.monitoringStatusText.text = getString(
            if (enabling) R.string.monitoring_active else R.string.monitoring_idle
        )
        if (enabling) {
            binding.finalCheckButton.isEnabled = true
        } else {
            // Sem frames novos processados, as últimas caixas ficariam congeladas
            // sobre o preview ao vivo — limpa o overlay ao parar.
            binding.overlayView.update(emptyList(), 1, 1)
        }
    }

    private fun performFinalCheck() {
        val notReturned = tracker.performFinalCheck()
        updateInstrumentsListText(tracker.activeInstruments)

        val message = if (notReturned.isEmpty()) {
            getString(R.string.check_dialog_all_ok)
        } else {
            alertManager.alertNotReturned()
            buildString {
                append(getString(R.string.check_dialog_missing_header))
                notReturned.groupBy { it.displayLabel }.forEach { (label, group) ->
                    append("\n• $label")
                    if (group.size > 1) append(" (${group.size})")
                }
            }
        }

        AlertDialog.Builder(this)
            .setTitle(R.string.check_dialog_title)
            .setMessage(message)
            .setPositiveButton(R.string.button_ok, null)
            .setNeutralButton(R.string.button_reset_session) { _, _ -> resetSession() }
            .setCancelable(false)
            .show()
    }

    private fun resetSession() {
        tracker.reset()
        cameraController?.monitoringEnabled = false
        binding.startStopButton.text = getString(R.string.button_start_monitoring)
        binding.monitoringStatusText.text = getString(R.string.monitoring_idle)
        binding.finalCheckButton.isEnabled = false
        binding.instrumentsListText.text = getString(R.string.instruments_list_empty)
        binding.overlayView.update(emptyList(), 1, 1)
    }

    override fun onDestroy() {
        super.onDestroy()
        stopStreaming()
        NpuActivationLog.removeListener(npuLogListener)
        cameraController?.shutdown()
        modelSessions?.close()
    }

    companion object {
        private const val TAG = "MainActivity"
    }
}
