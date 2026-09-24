package com.rsiprevention.app.camera

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer

/**
 * Conversão de frames `YUV_420_888` (entregues pelo `ImageAnalysis` do CameraX,
 * ver doc de deploy seção 6) para um [Bitmap] RGB "em pé" (já com a rotação do
 * sensor aplicada), pronto para o [com.rsiprevention.app.detection.Preprocessor].
 *
 * `bitmap.width`/`bitmap.height` do resultado são, por definição, as dimensões
 * "upright" que devem ser repassadas para `OverlayView.update(...)`.
 */
object ImageUtils {

    private const val JPEG_QUALITY = 90

    fun imageProxyToUprightBitmap(imageProxy: ImageProxy): Bitmap {
        val nv21 = yuv420888ToNv21(imageProxy)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)

        val jpegBytes = ByteArrayOutputStream().use { out ->
            yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), JPEG_QUALITY, out)
            out.toByteArray()
        }
        val bitmap = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
            ?: error("Falha ao decodificar frame de câmera (YUV -> JPEG -> Bitmap)")

        val rotationDegrees = imageProxy.imageInfo.rotationDegrees
        if (rotationDegrees == 0) return bitmap

        val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        bitmap.recycle()
        return rotated
    }

    /**
     * Converte os 3 planos `YUV_420_888` (respeitando `rowStride`/`pixelStride`
     * individuais de cada plano, que variam por fabricante/dispositivo) para o
     * layout semiplanar NV21 (Y contíguo, seguido de V,U intercalados).
     */
    private fun yuv420888ToNv21(imageProxy: ImageProxy): ByteArray {
        val width = imageProxy.width
        val height = imageProxy.height
        val nv21 = ByteArray(width * height * 3 / 2)

        val yPlane = imageProxy.planes[0]
        val uPlane = imageProxy.planes[1]
        val vPlane = imageProxy.planes[2]

        copyYPlane(yPlane.buffer, yPlane.rowStride, yPlane.pixelStride, width, height, nv21)
        copyChromaPlanesInterleaved(
            vBuffer = vPlane.buffer,
            vRowStride = vPlane.rowStride,
            vPixelStride = vPlane.pixelStride,
            uBuffer = uPlane.buffer,
            uRowStride = uPlane.rowStride,
            uPixelStride = uPlane.pixelStride,
            width = width,
            height = height,
            nv21 = nv21,
            chromaStartOffset = width * height
        )
        return nv21
    }

    private fun copyYPlane(buffer: ByteBuffer, rowStride: Int, pixelStride: Int, width: Int, height: Int, nv21: ByteArray) {
        var outIndex = 0
        for (row in 0 until height) {
            val rowStart = row * rowStride
            if (pixelStride == 1) {
                for (col in 0 until width) {
                    nv21[outIndex++] = buffer.get(rowStart + col)
                }
            } else {
                for (col in 0 until width) {
                    nv21[outIndex++] = buffer.get(rowStart + col * pixelStride)
                }
            }
        }
    }

    /** NV21 intercala V,U por par de pixels 2x2 (sub-amostragem 4:2:0). */
    private fun copyChromaPlanesInterleaved(
        vBuffer: ByteBuffer,
        vRowStride: Int,
        vPixelStride: Int,
        uBuffer: ByteBuffer,
        uRowStride: Int,
        uPixelStride: Int,
        width: Int,
        height: Int,
        nv21: ByteArray,
        chromaStartOffset: Int
    ) {
        val chromaWidth = width / 2
        val chromaHeight = height / 2
        var outIndex = chromaStartOffset
        for (row in 0 until chromaHeight) {
            val vRowStart = row * vRowStride
            val uRowStart = row * uRowStride
            for (col in 0 until chromaWidth) {
                nv21[outIndex++] = vBuffer.get(vRowStart + col * vPixelStride)
                nv21[outIndex++] = uBuffer.get(uRowStart + col * uPixelStride)
            }
        }
    }
}
