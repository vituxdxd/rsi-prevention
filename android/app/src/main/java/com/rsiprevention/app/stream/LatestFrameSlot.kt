package com.rsiprevention.app.stream

/**
 * Slot de "último valor": o produtor publica e o consumidor pega sempre o mais
 * recente, descartando os intermediários. Usado para o frame de vídeo — o
 * encoder nunca fica atrás de uma fila; sob carga, ele simplesmente pula
 * frames velhos (o vídeo atrasa um frame, não acumula latência).
 *
 * Thread-safe. Itens substituídos/descartados são entregues a [onDropped]
 * (para devolver recursos, ex.: `Bitmap.recycle()`).
 */
class LatestFrameSlot<T : Any>(private val onDropped: (T) -> Unit = {}) {

    private val lock = Object()
    private var value: T? = null

    @Volatile
    private var open = true

    /** Publica [item]; o anterior (se ainda não consumido) é descartado. */
    fun offer(item: T) {
        synchronized(lock) {
            if (!open) {
                onDropped(item)
                return
            }
            value?.let(onDropped)
            value = item
            lock.notifyAll()
        }
    }

    /**
     * Bloqueia até haver um item ou o slot fechar.
     * @return o item mais recente, ou `null` se o slot foi fechado.
     */
    fun take(): T? {
        synchronized(lock) {
            while (open && value == null) lock.wait()
            val item = value
            value = null
            return item
        }
    }

    /** Fecha o slot: desperta consumidores e descarta o item pendente. */
    fun close() {
        synchronized(lock) {
            open = false
            value?.let(onDropped)
            value = null
            lock.notifyAll()
        }
    }
}
