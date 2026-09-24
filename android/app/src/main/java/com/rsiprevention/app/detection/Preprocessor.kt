package com.rsiprevention.app.detection

import android.graphics.Bitmap
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Converte um [Bitmap] RGB "em pé" na entrada `images` exigida pelo YOLO26s
 * quantizado QDQ (`detector_int8.onnx`, ativações 16-bit / pesos 8-bit):
 * tensor float32 NCHW `[1,3,640,640]` normalizado para [0, 1].
 *
 * O grafo QDQ tem pares QuantizeLinear/DequantizeLinear internos que convertem
 * os floats para inteiro dentro do grafo. A entrada e saída do ONNX são
 * float32. O QNN Execution Provider compila esse grafo QDQ para a NPU Hexagon.
 */
object Preprocessor {
    const val MODEL_INPUT_SIZE = 640

    /** @return ByteBuffer direto, float32, layout NCHW, normalizado [0,1]. */
    fun preprocess(bitmap: Bitmap): ByteBuffer {
        val needsResize = bitmap.width != MODEL_INPUT_SIZE || bitmap.height != MODEL_INPUT_SIZE
        val resized = if (needsResize) {
            Bitmap.createScaledBitmap(bitmap, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, true)
        } else {
            bitmap
        }

        val planeSize = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE
        val pixels = IntArray(planeSize)
        resized.getPixels(pixels, 0, MODEL_INPUT_SIZE, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)

        val buffer = ByteBuffer.allocateDirect(4 * 3 * planeSize).order(ByteOrder.nativeOrder())
        val floatBuffer = buffer.asFloatBuffer()
        for (pixel in pixels) floatBuffer.put(((pixel shr 16) and 0xFF) / 255f)
        for (pixel in pixels) floatBuffer.put(((pixel shr 8) and 0xFF) / 255f)
        for (pixel in pixels) floatBuffer.put((pixel and 0xFF) / 255f)
        buffer.rewind()

        if (needsResize) resized.recycle()
        return buffer
    }
}
