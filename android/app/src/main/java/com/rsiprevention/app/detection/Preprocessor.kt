package com.rsiprevention.app.detection

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.roundToInt

/**
 * Converte um [Bitmap] RGB "em pé" na entrada `images` exigida pelo YOLO26s
 * quantizado QDQ (`detector_int8.onnx`, ativações 16-bit / pesos 8-bit):
 * tensor float32 NCHW `[1,3,640,640]` normalizado para [0, 1].
 *
 * O grafo QDQ tem pares QuantizeLinear/DequantizeLinear internos que convertem
 * os floats para inteiro dentro do grafo. A entrada e saída do ONNX são
 * float32. O QNN Execution Provider compila esse grafo QDQ para a NPU Hexagon.
 *
 * **Geometria letterbox** ([Letterbox]): o frame é redimensionado preservando o
 * aspecto e centralizado com padding cinza 114 (padrão do treino YOLO) — não há
 * distorção de aspecto. O asset A16W8 é calibrado com essa mesma geometria (ver
 * `_build_calibration.py`/`_quantize_qnn.py`); misturar geometrias de calibração
 * e inferência degrada as confianças (medido em `_benchmark_preprocess.py`).
 *
 * **Buffers e bitmap reutilizados entre frames**: o pipeline roda sempre no
 * mesmo executor single-thread de inferência (ver `CameraController`) e o
 * resultado é consumido sincronamente antes do próximo frame começar — então um
 * único jogo de buffers (~5 MB + bitmap 640×640) é reusado sem sincronização
 * extra, em vez de alocar ~6,5 MB por frame e pressionar o GC.
 */
object Preprocessor {
    const val MODEL_INPUT_SIZE = 640
    private const val PLANE_SIZE = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE
    private const val INV_255 = 1f / 255f

    private val pixels = IntArray(PLANE_SIZE)

    /**
     * NCHW planar em heap: o preenchimento é feito com escritas de array (baratas)
     * e copiado para o buffer direto em **uma** chamada `put(float[])` (cópia em
     * bloco). O caminho anterior fazia 3 puts absolutos por pixel (~1,23 milhão
     * de chamadas por frame) e era o gargalo do pipeline — medido em 94 ms/frame
     * no POCO F7 (`DetectStats`).
     */
    private val floatPixels = FloatArray(3 * PLANE_SIZE)

    /**
     * Buffer direto compartilhado com o [YoloWorldDetector]: o tensor de entrada
     * ONNX é criado uma vez e referencia este buffer. O [preprocess] preenche-o
     * a cada frame; a NPU consome antes do próximo preenchimento.
     */
    internal val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(4 * 3 * PLANE_SIZE).order(ByteOrder.nativeOrder())
    private val floatBuffer: FloatBuffer = inputBuffer.asFloatBuffer()

    // Letterbox para 640×640 em bitmap/canvas reutilizados (o `createScaledBitmap`
    // alocava um bitmap novo a cada frame).
    private val scaleCanvas = Canvas()
    private val scalePaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val scaleDestination = Rect()
    private var scaleScratch: Bitmap? = null

    /**
     * @return tensor direto (float32, NCHW, [0,1]) + os pixels 640×640 do
     *   modelo + o tempo de cada sub-etapa (instrumentação). **Reutilizados
     *   entre chamadas** — não guardar referência além do processamento do
     *   frame atual (ver [PreprocessedFrame]).
     */
    @Synchronized
    fun preprocess(bitmap: Bitmap): PreprocessedFrame {
        val scaleStartNs = System.nanoTime()
        val modelBitmap = if (bitmap.width == MODEL_INPUT_SIZE && bitmap.height == MODEL_INPUT_SIZE) {
            bitmap
        } else {
            letterboxScale(bitmap)
        }
        val scaleNs = System.nanoTime() - scaleStartNs

        val pixelsStartNs = System.nanoTime()
        modelBitmap.getPixels(pixels, 0, MODEL_INPUT_SIZE, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
        val pixelsNs = System.nanoTime() - pixelsStartNs

        // NCHW planar: uma passada sobre os pixels preenchendo o array em heap e
        // uma cópia em bloco para o buffer direto.
        val fillStartNs = System.nanoTime()
        for (i in pixels.indices) {
            val pixel = pixels[i]
            floatPixels[i] = ((pixel shr 16) and 0xFF) * INV_255
            floatPixels[PLANE_SIZE + i] = ((pixel shr 8) and 0xFF) * INV_255
            floatPixels[2 * PLANE_SIZE + i] = (pixel and 0xFF) * INV_255
        }
        floatBuffer.rewind()
        floatBuffer.put(floatPixels)
        inputBuffer.position(0)
        val fillNs = System.nanoTime() - fillStartNs

        return PreprocessedFrame(inputBuffer, ModelFrame(pixels, MODEL_INPUT_SIZE), scaleNs, pixelsNs, fillNs)
    }

    /**
     * Letterbox: redimensiona preservando aspecto, centraliza com padding cinza 114.
     * Frame 720×960 → conteúdo 480×640 centralizado em 640×640 (25% padding).
     */
    private fun letterboxScale(bitmap: Bitmap): Bitmap {
        val transform = Letterbox.forFrame(bitmap.width, bitmap.height)
        val contentWidth = (bitmap.width * transform.scale).roundToInt().coerceIn(1, MODEL_INPUT_SIZE)
        val contentHeight = (bitmap.height * transform.scale).roundToInt().coerceIn(1, MODEL_INPUT_SIZE)
        val left = (MODEL_INPUT_SIZE - contentWidth) / 2
        val top = (MODEL_INPUT_SIZE - contentHeight) / 2
        scaleDestination.set(left, top, left + contentWidth, top + contentHeight)

        val scratch = ensureScratch()
        scaleCanvas.setBitmap(scratch)
        try {
            scaleCanvas.drawColor(Letterbox.PAD_COLOR)
            scaleCanvas.drawBitmap(bitmap, null, scaleDestination, scalePaint)
        } finally {
            scaleCanvas.setBitmap(null)
        }
        return scratch
    }

    private fun ensureScratch(): Bitmap {
        return scaleScratch ?: Bitmap
            .createBitmap(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, Bitmap.Config.ARGB_8888)
            .also { scaleScratch = it }
    }
}

/**
 * Resultado do [Preprocessor]: tensor de entrada do ONNX + referência espacial
 * do frame no espaço do modelo, com o tempo de cada sub-etapa (instrumentação).
 *
 * Contrato de posse (ver KDoc de [Preprocessor]): os buffers são reutilizados na
 * próxima chamada — consumir imediatamente, na mesma thread.
 */
class PreprocessedFrame internal constructor(
    /** Tensor direto, float32, layout NCHW, normalizado [0,1]. */
    val tensor: ByteBuffer,
    /** Pixels 640×640 do frame (mesmo espaço das caixas do detector). */
    val model: ModelFrame,
    /** Bitmap → 640×640 (escala). */
    val scaleNs: Long,
    /** `getPixels` do bitmap 640×640. */
    val pixelsNs: Long,
    /** Montagem NCHW + cópia para o buffer direto. */
    val fillNs: Long
)
