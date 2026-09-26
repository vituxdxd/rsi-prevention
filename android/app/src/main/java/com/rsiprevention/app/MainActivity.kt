package com.rsiprevention.app

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
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
import com.rsiprevention.app.tracking.InstrumentStatus
import com.rsiprevention.app.tracking.InstrumentTracker
import com.rsiprevention.app.tracking.TrackedInstrument
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

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            initializePipeline()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
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
    }

    private fun onFrameProcessed(instruments: List<TrackedInstrument>, frameWidth: Int, frameHeight: Int) {
        binding.overlayView.update(instruments, frameWidth, frameHeight)
        updateInstrumentsListText(instruments)
    }

    private fun updateInstrumentsListText(instruments: List<TrackedInstrument>) {
        val newText = if (instruments.isEmpty()) {
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
                    "${instrument.displayLabel} — $statusText — $time"
                }
        }
        // Chamado a cada frame processado: evita re-layout da lista quando nada mudou.
        if (binding.instrumentsListText.text.toString() != newText) {
            binding.instrumentsListText.text = newText
        }
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
        NpuActivationLog.removeListener(npuLogListener)
        cameraController?.shutdown()
        modelSessions?.close()
    }

    companion object {
        private const val TAG = "MainActivity"
    }
}
