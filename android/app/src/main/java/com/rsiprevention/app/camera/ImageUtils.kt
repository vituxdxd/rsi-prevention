package com.rsiprevention.app.camera

import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.ImageProxy

/**
 * Conversão de frames `YUV_420_888` (entregues pelo `ImageAnalysis` do CameraX,
 * ver doc de deploy seção 6) para um [Bitmap] RGB "em pé" (já com a rotação do
 * sensor aplicada), pronto para o [com.rsiprevention.app.detection.Preprocessor].
 *
 * Usa o caminho nativo [ImageProxy.toBitmap] do CameraX (YUV → RGB interno, sem
 * round-trip por JPEG — a conversão anterior via `YuvImage.compressToJpeg` +
 * `BitmapFactory.decodeByteArray` era o ponto mais caro do pipeline por frame).
 *
 * `bitmap.width`/`bitmap.height` do resultado são, por definição, as dimensões
 * "upright" que devem ser repassadas para `OverlayView.update(...)`.
 *
 * O bitmap devolvido é de propriedade do chamador: deve ser reciclado após o uso
 * (ver `CameraController`).
 */
object ImageUtils {

    fun imageProxyToUprightBitmap(imageProxy: ImageProxy): Bitmap {
        val bitmap = imageProxy.toBitmap()

        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        if (rotationDegrees == 0) return bitmap

        val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        if (rotated != bitmap) bitmap.recycle()
        return rotated
    }
}
