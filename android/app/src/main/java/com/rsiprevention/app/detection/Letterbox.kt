package com.rsiprevention.app.detection

import kotlin.math.max

/**
 * Geometria do pré-processamento **letterbox**: o frame é redimensionado
 * preservando o aspecto para caber em 640×640 e o restante é preenchido com
 * cinza 114 (padrão do letterbox de treino do YOLO). Diferente do *stretch*
 * (redimensionar direto para 640×640), não distorce o aspecto — no frame
 * retrato da câmera a distorção do stretch chegava a 1,78×.
 *
 * É a **fonte única da transformação** frame ↔ modelo, usada em três lugares:
 *  - [Preprocessor]: desenha o frame no bitmap 640×640;
 *  - `OverlayView`: leva as caixas do modelo de volta ao preview;
 *  - `FrameStreamer`: idem no frame transmitido.
 *
 * Puro (sem `android.graphics`) — testável em JVM. A conversão usa a escala
 * ideal (float); o desenho arredonda o retângulo de conteúdo para pixel
 * inteiro, uma diferença sub-pixel irrelevante para caixas de detecção.
 */
object Letterbox {
    const val MODEL_SIZE = 640

    /** Cinza 114/255 — valor padrão do padding de treino do Ultralytics. */
    const val PAD_COLOR = 0xFF727272.toInt()

    /** Transformação frame → modelo: `model = frame * scale + pad`. */
    data class Transform(val scale: Float, val padX: Float, val padY: Float) {
        fun toModelX(frameX: Float): Float = frameX * scale + padX
        fun toModelY(frameY: Float): Float = frameY * scale + padY
        fun toFrameX(modelX: Float): Float = (modelX - padX) / scale
        fun toFrameY(modelY: Float): Float = (modelY - padY) / scale
    }

    /** Transformação letterbox para um frame de [width]×[height] pixels. */
    fun forFrame(width: Int, height: Int): Transform {
        require(width > 0 && height > 0) { "frame inválido: ${width}x$height" }
        val scale = MODEL_SIZE.toFloat() / max(width, height)
        return Transform(
            scale = scale,
            padX = (MODEL_SIZE - width * scale) / 2f,
            padY = (MODEL_SIZE - height * scale) / 2f
        )
    }
}
