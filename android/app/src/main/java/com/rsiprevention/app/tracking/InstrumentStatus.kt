package com.rsiprevention.app.tracking

/**
 * Estado de rastreamento de um instrumento cirúrgico entre frames.
 * Exibição exclusivamente em pt-BR (ver instrucao_ui_ptbr_yolo_world.md).
 */
enum class InstrumentStatus(val displayNameResId: Int) {
    /** Detectado mas aguardando confirmação temporal (precisa de K de N frames). */
    CANDIDATE(com.rsiprevention.app.R.string.status_on_table), // Não exibido na UI

    /** Confirmado e presente na área da mesa no frame atual. */
    ON_TABLE(com.rsiprevention.app.R.string.status_on_table),

    /** Estava na mesa e não é mais detectado na área da mesa (saiu de cena). */
    REMOVED(com.rsiprevention.app.R.string.status_removed),

    /** Havia sido removido e voltou a ser detectado na área da mesa. */
    RETURNED(com.rsiprevention.app.R.string.status_returned),

    /** Resultado da checagem final: foi removido e nunca retornou até o fechamento. */
    NOT_RETURNED(com.rsiprevention.app.R.string.status_not_returned)
}
