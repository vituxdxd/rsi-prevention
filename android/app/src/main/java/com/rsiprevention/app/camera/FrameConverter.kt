package com.rsiprevention.app.camera

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import androidx.camera.core.ImageProxy

/**
 * Converte frames `YUV_420_888` (entregues pelo `ImageAnalysis` do CameraX,
 * ver doc de deploy seção 6) em um [Bitmap] RGB "em pé" (já com a rotação do
 * sensor aplicada), pronto para o
 * [com.rsiprevention.app.detection.Preprocessor].
 *
 * Usa o caminho nativo [ImageProxy.toBitmap] do CameraX (YUV → RGB interno, sem
 * round-trip por JPEG — a conversão anterior via `YuvImage.compressToJpeg` +
 * `BitmapFactory.decodeByteArray` era o ponto mais caro do pipeline por frame).
 *
 * `bitmap.width`/`bitmap.height` do resultado são, por definição, as dimensões
 * "upright" que devem ser repassadas para `OverlayView.update(...)`.
 *
 * **O bitmap devolvido é interno e reutilizado** (o `createBitmap` da rotação
 * alocava ~3,7 MB por frame): nunca reciclar; ele é sobrescrito no próximo
 * frame. Consumidores que precisem manter o frame (o streamer) fazem a própria
 * cópia antes de retornar deste método.
 *
 * Não é thread-safe: usar sempre na mesma thread do pipeline (ver
 * [CameraController]).
 */
class FrameConverter {

    private var scratch: Bitmap? = null
    private val canvas = Canvas()
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    /** @return bitmap "em pé" **interno e reutilizado** — não reciclar. */
    fun toUpright(imageProxy: ImageProxy): Bitmap {
        val source = imageProxy.toBitmap()
        val rotationDegrees = imageProxy.imageInfo.rotationDegrees

        val swap = rotationDegrees == 90 || rotationDegrees == 270
        val outWidth = if (swap) source.height else source.width
        val outHeight = if (swap) source.width else source.height

        val target = ensureScratch(outWidth, outHeight)
        val matrix = rotationMatrix(rotationDegrees, source.width, source.height)

        canvas.setBitmap(target)
        try {
            canvas.drawBitmap(source, matrix, paint)
        } finally {
            canvas.setBitmap(null)
            source.recycle()
        }
        return target
    }

    private fun ensureScratch(width: Int, height: Int): Bitmap {
        val current = scratch
        if (current != null && !current.isRecycled && current.width == width && current.height == height) {
            return current
        }
        current?.recycle()
        return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also { scratch = it }
    }

    /**
     * Matriz equivalente ao `Bitmap.createBitmap(src, ..., matrix, true)`, mas
     * desenhando em um destino já dimensionado — a translação reposiciona a
     * imagem rotacionada (a rotação do Canvas gira em torno da origem).
     */
    private fun rotationMatrix(rotationDegrees: Int, sourceWidth: Int, sourceHeight: Int): Matrix {
        val matrix = Matrix()
        when (rotationDegrees % 360) {
            90 -> {
                matrix.postRotate(90f)
                matrix.postTranslate(sourceHeight.toFloat(), 0f)
            }
            180 -> {
                matrix.postRotate(180f)
                matrix.postTranslate(sourceWidth.toFloat(), sourceHeight.toFloat())
            }
            270 -> {
                matrix.postRotate(270f)
                matrix.postTranslate(0f, sourceWidth.toFloat())
            }
        }
        return matrix
    }
}
