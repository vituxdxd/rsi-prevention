package com.rsiprevention.app.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.util.AttributeSet
import android.view.View
import androidx.annotation.MainThread
import androidx.core.content.ContextCompat
import com.rsiprevention.app.R
import com.rsiprevention.app.tracking.InstrumentStatus
import com.rsiprevention.app.tracking.TrackedInstrument
import kotlin.math.max

/**
 * View de overlay desenhada sobre o [androidx.camera.view.PreviewView]: caixas +
 * rótulos pt-BR dos instrumentos atualmente detectados (ver instrucao_ui_ptbr_yolo_world.md).
 *
 * Contrato de coordenadas (importante para quem chamar [update]):
 * - [TrackedInstrument.box] está no espaço de entrada do modelo, um quadrado
 *   640x640 obtido por *stretch-resize* (sem letterboxing) do frame de câmera
 *   já corrigido de rotação (ver Preprocessor). Esta view desfaz o stretch
 *   (proporção frame/640) e depois aplica um recorte central (center-crop),
 *   equivalente ao `scaleType="fillCenter"` do `PreviewView`, para que as
 *   caixas caiam visualmente sobre os instrumentos reais na pré-visualização.
 * - `frameWidth`/`frameHeight` em [update] devem ser as dimensões do frame já
 *   em orientação "em pé" (pós-rotação), as mesmas vistas pelo modelo e pelo
 *   `PreviewView`.
 *
 * Apenas instrumentos atualmente visíveis no frame ([InstrumentStatus.ON_TABLE]
 * e [InstrumentStatus.RETURNED]) recebem caixa — instrumentos removidos não
 * têm posição atual válida; seu histórico/status continua visível na lista de
 * texto da tela principal, não como caixa "fantasma" sobre a imagem ao vivo.
 */
class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var instruments: List<TrackedInstrument> = emptyList()
    private var frameWidth: Int = 0
    private var frameHeight: Int = 0

    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = STROKE_WIDTH_PX
    }
    private val labelBackgroundPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
    }
    private val labelTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = ContextCompat.getColor(context, R.color.overlay_text)
        textSize = TEXT_SIZE_PX
        typeface = Typeface.DEFAULT_BOLD
    }

    /** Cor única de todas as caixas (a classe exibida é sempre a mesma, "Instrumento Cirúrgico"). */
    private val boxColor = ContextCompat.getColor(context, R.color.overlay_box_on_table)

    /** Chamado pela Activity, na thread principal, a cada novo resultado de tracking. */
    @MainThread
    fun update(instruments: List<TrackedInstrument>, frameWidth: Int, frameHeight: Int) {
        this.instruments = instruments
        this.frameWidth = frameWidth
        this.frameHeight = frameHeight
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (frameWidth <= 0 || frameHeight <= 0 || width <= 0 || height <= 0) return

        val stretchX = frameWidth / MODEL_INPUT_SIZE
        val stretchY = frameHeight / MODEL_INPUT_SIZE
        val cropScale = max(width / frameWidth.toFloat(), height / frameHeight.toFloat())
        val cropOffsetX = (width - frameWidth * cropScale) / 2f
        val cropOffsetY = (height - frameHeight * cropScale) / 2f

        for (instrument in instruments) {
            // Cor padronizada: a subclasse/status não altera a cor da caixa (a
            // lista de texto é quem informa o estado de cada instrumento).
            val color = when (instrument.status) {
                InstrumentStatus.CANDIDATE -> continue // não confirmado ainda
                InstrumentStatus.ON_TABLE, InstrumentStatus.RETURNED -> boxColor
                InstrumentStatus.REMOVED, InstrumentStatus.NOT_RETURNED -> continue
            }

            val box = instrument.box
            val left = (box.left * stretchX) * cropScale + cropOffsetX
            val top = (box.top * stretchY) * cropScale + cropOffsetY
            val right = (box.right * stretchX) * cropScale + cropOffsetX
            val bottom = (box.bottom * stretchY) * cropScale + cropOffsetY
            val viewBox = RectF(left, top, right, bottom)

            boxPaint.color = color
            canvas.drawRect(viewBox, boxPaint)
            drawLabel(canvas, viewBox, color, labelFor(instrument))
        }
    }

    private fun labelFor(instrument: TrackedInstrument): String {
        val confidencePercent = (instrument.confidence * 100).toInt()
        return "${instrument.displayLabel} • $confidencePercent%"
    }

    private fun drawLabel(canvas: Canvas, box: RectF, color: Int, text: String) {
        val textWidth = labelTextPaint.measureText(text)
        val textHeight = labelTextPaint.textSize

        // Rótulo acima da caixa; se não couber (perto do topo da tela), desenha logo abaixo.
        val labelBottom = if (box.top - textHeight - LABEL_PADDING_PX * 2 >= 0) box.top else box.bottom + textHeight + LABEL_PADDING_PX * 2
        val labelTop = labelBottom - textHeight - LABEL_PADDING_PX * 2
        val labelRect = RectF(box.left, labelTop, box.left + textWidth + LABEL_PADDING_PX * 2, labelBottom)

        labelBackgroundPaint.color = color
        canvas.drawRect(labelRect, labelBackgroundPaint)
        canvas.drawText(text, labelRect.left + LABEL_PADDING_PX, labelRect.bottom - LABEL_PADDING_PX, labelTextPaint)
    }

    companion object {
        private const val MODEL_INPUT_SIZE = 640f
        private const val STROKE_WIDTH_PX = 6f
        private const val TEXT_SIZE_PX = 36f
        private const val LABEL_PADDING_PX = 8f
    }
}
