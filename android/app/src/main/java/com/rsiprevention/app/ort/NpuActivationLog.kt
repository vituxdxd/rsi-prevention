package com.rsiprevention.app.ort

import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.concurrent.CopyOnWriteArrayList

/**
 * Log em memória da ativação da NPU Hexagon, exibido discretamente na tela
 * principal (seção expansível/ocultável, ver `MainActivity`).
 *
 * Alimentado por [NpuLog] nos pontos-chave da inicialização: configuração do
 * `ADSP_LIBRARY_PATH`, escolha do backend (QNN HTP), compilação/carregamento do
 * contexto em cache e verificação de sanidade. Mantém apenas as últimas
 * [MAX_ENTRIES] entradas (não cresce sem limite).
 */
object NpuActivationLog {

    data class Entry(val wallMs: Long, val message: String)

    private const val MAX_ENTRIES = 100
    private val timeFormat = DateTimeFormatter.ofPattern("dd/MM HH:mm:ss")

    private val entries = ArrayDeque<Entry>()
    private val listeners = CopyOnWriteArrayList<(List<Entry>) -> Unit>()

    /** Registra uma entrada e notifica os listeners com o snapshot atualizado. */
    fun log(message: String) {
        val snapshot: List<Entry>
        synchronized(entries) {
            entries.addLast(Entry(System.currentTimeMillis(), message))
            while (entries.size > MAX_ENTRIES) entries.removeFirst()
            snapshot = entries.toList()
        }
        for (listener in listeners) listener(snapshot)
    }

    fun snapshot(): List<Entry> = synchronized(entries) { entries.toList() }

    /** Formato único de data/hora do log (também usado pela lista de instrumentos). */
    fun formatTime(wallMs: Long): String =
        timeFormat.format(Instant.ofEpochMilli(wallMs).atZone(ZoneId.systemDefault()))

    /** Registra um listener e já o chama com o conteúdo atual. */
    fun addListener(listener: (List<Entry>) -> Unit) {
        listeners.add(listener)
        listener(snapshot())
    }

    fun removeListener(listener: (List<Entry>) -> Unit) {
        listeners.remove(listener)
    }
}
