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
 *
 * **Buffers reutilizados entre frames**: o pipeline roda sempre no mesmo
 * executor single-thread de inferência (ver `CameraController`) e o buffer
 * devolvido é consumido sincronamente pelo `session.run` antes do próximo
 * frame começar — então um único par de buffers (~5 MB) pode ser reusado sem
 * sincronização extra, em vez de alocar ~5 MB por frame e pressionar o GC.
 */
object Preprocessor {
    const val MODEL_INPUT_SIZE = 640
    private const val PLANE_SIZE = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE

    private val pixels = IntArray(PLANE_SIZE)
    private val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(4 * 3 * PLANE_SIZE).order(ByteOrder.nativeOrder())

    /**
     * @return ByteBuffer direto, float32, layout NCHW, normalizado [0,1].
     *   **Reutilizado entre chamadas** — não guardar referência além da inferência
     *   do frame atual.
     */
    @Synchronized
    fun preprocess(bitmap: Bitmap): ByteBuffer {
        val needsResize = bitmap.width != MODEL_INPUT_SIZE || bitmap.height != MODEL_INPUT_SIZE
        val resized = if (needsResize) {
            Bitmap.createScaledBitmap(bitmap, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, true)
        } else {
            bitmap
        }

        resized.getPixels(pixels, 0, MODEL_INPUT_SIZE, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)

        inputBuffer.clear()
        val floatBuffer = inputBuffer.asFloatBuffer()
        for (pixel in pixels) floatBuffer.put(((pixel shr 16) and 0xFF) / 255f)
        for (pixel in pixels) floatBuffer.put(((pixel shr 8) and 0xFF) / 255f)
        for (pixel in pixels) floatBuffer.put((pixel and 0xFF) / 255f)
        inputBuffer.rewind()

        if (needsResize) resized.recycle()
        return inputBuffer
    }
}
