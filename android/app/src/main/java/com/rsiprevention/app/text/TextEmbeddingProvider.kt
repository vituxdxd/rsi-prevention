package com.rsiprevention.app.text

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.util.Log
import com.rsiprevention.app.vocabulary.InstrumentVocabulary
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.roundToInt

/**
 * Calcula (uma única vez, no início da sessão de monitoramento — doc de deploy
 * seção 10: "text_encoder não roda por frame") os embeddings de texto para
 * [InstrumentVocabulary.CANONICAL_PROMPTS] e monta o tensor `txt_feats` fixo
 * que o `detector` espera a cada frame.
 *
 * O grafo `detector.onnx` foi exportado com `num_classes=80` fixo (não é
 * configurável em runtime — ver `YoloWorldDetector.get_input_spec` no
 * `qai_hub_models`), mas este projeto só monitora
 * [InstrumentVocabulary.CANONICAL_PROMPTS] (9 prompts). Solução: "telhar"
 * (tile/ciclar) os 9 prompts para preencher os 80 slots exigidos — o slot `s`
 * corresponde sempre ao prompt canônico `s % 9`. [slotToPromptIndex] é a fonte
 * única de verdade dessa correspondência: o `Postprocessor` usa esse mesmo
 * mapeamento para traduzir o `class_idx` (0..79) devolvido pelo detector de
 * volta ao instrumento real.
 *
 * Requantização: `text_encoder.onnx` e `detector.onnx` foram compilados/
 * quantizados de forma independente. Ambos expõem `txt_feats` como uint8, mas
 * com parâmetros de quantização (escala/zero-point) ligeiramente diferentes
 * (confirmado nos `metadata.json` de cada export) — por isso o valor lido do
 * `text_encoder` é desquantizado com os parâmetros dele e requantizado com os
 * parâmetros do `detector` antes de ser usado como entrada.
 *
 * **Caminho usado em produção: [createFromQuantizedAsset].** O
 * `text_encoder.onnx` exportado com `--quantize w8a8` tem os embeddings
 * destruídos pela quantização (medido neste projeto: cosseno -0.21 contra os
 * embeddings fp32 do mesmo grafo, ou seja, praticamente ortogonais), e os
 * `scores` do detector caem para zero por causa disso. Como o vocabulário é
 * fixo, os embeddings são calculados uma única vez no PC, em fp32, e embarcados
 * em `assets/models/txt_feats.u8` (ver `_gen_txt_feats.py`). [create] fica
 * apenas para o dia em que o `text_encoder` for re-exportado em fp16 (a
 * recomendação da própria Qualcomm para esse grafo).
 */
class TextEmbeddingProvider private constructor(
    /** Tensor `txt_feats` [1,80,512] uint8, já nos parâmetros de quantização do `detector`, pronto para reuso a cada frame. */
    val quantizedTxtFeats: ByteBuffer,
    private val slotToPromptIndex: IntArray
) {
    /** Índice em [InstrumentVocabulary.CANONICAL_PROMPTS] correspondente a um `class_idx` (0..79) devolvido pelo detector. */
    fun promptIndexForSlot(slot: Int): Int = slotToPromptIndex[slot]

    companion object {
        private const val TAG = "TextEmbeddingProvider"
        const val NUM_SLOTS = 80
        private const val EMBEDDING_DIM = 512
        private const val CONTEXT_LENGTH = 77

        // Parâmetros de quantização do OUTPUT txt_feats do text_encoder.onnx
        // exportado (export/output/text_encoder_onnx/yolo_world-onnx-w8a8/metadata.json).
        private const val TEXT_ENCODER_OUTPUT_SCALE = 0.0034038955345749855f
        private const val TEXT_ENCODER_OUTPUT_ZERO_POINT = 76

        // Parâmetros de quantização do INPUT txt_feats do detector.onnx
        // exportado (export/output/detector_onnx/yolo_world-onnx-w8a8/metadata.json).
        private const val DETECTOR_TXT_FEATS_SCALE = 0.003409650409594178f
        private const val DETECTOR_TXT_FEATS_ZERO_POINT = 76

        /**
         * Carrega os embeddings pré-calculados de `assets/models/txt_feats.u8`
         * (`[80,512]` uint8, gerado por `_gen_txt_feats.py`), já nos parâmetros de
         * quantização do `detector` — exatamente o tensor que [create] montaria em
         * runtime, mas calculado no PC em fp32 (ver discussão na doc da classe).
         */
        fun createFromQuantizedAsset(file: File): TextEmbeddingProvider {
            val prompts = InstrumentVocabulary.CANONICAL_PROMPTS
            require(prompts.size in 1..NUM_SLOTS) {
                "CANONICAL_PROMPTS deve ter entre 1 e $NUM_SLOTS prompts (tinha ${prompts.size})"
            }

            val expectedBytes = NUM_SLOTS * EMBEDDING_DIM
            val raw = file.readBytes()
            require(raw.size == expectedBytes) {
                "txt_feats.u8 inválido (${file.absolutePath}): ${raw.size} bytes, esperado $expectedBytes " +
                    "(${NUM_SLOTS}x$EMBEDDING_DIM uint8). Regenere com _gen_txt_feats.py."
            }

            val quantizedTxtFeats = ByteBuffer.allocateDirect(expectedBytes).order(ByteOrder.nativeOrder())
            quantizedTxtFeats.put(raw).rewind()
            Log.i(
                TAG,
                "txt_feats pronto do asset ${file.name}: $expectedBytes bytes (${prompts.size} prompts telhados em $NUM_SLOTS slots)"
            )
            return TextEmbeddingProvider(quantizedTxtFeats, IntArray(NUM_SLOTS) { slot -> slot % prompts.size })
        }

        fun create(env: OrtEnvironment, textEncoderSession: OrtSession, tokenizer: ClipTokenizer): TextEmbeddingProvider {
            val prompts = InstrumentVocabulary.CANONICAL_PROMPTS
            val numPrompts = prompts.size
            require(numPrompts in 1..NUM_SLOTS) {
                "CANONICAL_PROMPTS deve ter entre 1 e $NUM_SLOTS prompts (tinha $numPrompts)"
            }

            val slotToPromptIndex = IntArray(NUM_SLOTS) { slot -> slot % numPrompts }

            val tokensBytes = ByteBuffer.allocateDirect(NUM_SLOTS * CONTEXT_LENGTH * Int.SIZE_BYTES)
                .order(ByteOrder.nativeOrder())
            val tokensBuffer = tokensBytes.asIntBuffer()
            for (slot in 0 until NUM_SLOTS) {
                tokensBuffer.put(tokenizer.tokenize(prompts[slotToPromptIndex[slot]], CONTEXT_LENGTH))
            }
            tokensBuffer.rewind()

            val floatEmbeddings = FloatArray(NUM_SLOTS * EMBEDDING_DIM)
            OnnxTensor.createTensor(env, tokensBuffer, longArrayOf(NUM_SLOTS.toLong(), CONTEXT_LENGTH.toLong())).use { tokensTensor ->
                textEncoderSession.run(mapOf("tokens" to tokensTensor)).use { result ->
                    val txtFeatsTensor = result.get("txt_feats").get() as OnnxTensor
                    val quantized = txtFeatsTensor.byteBuffer
                    quantized.rewind()
                    for (i in floatEmbeddings.indices) {
                        val u8 = quantized.get().toInt() and 0xFF
                        floatEmbeddings[i] = (u8 - TEXT_ENCODER_OUTPUT_ZERO_POINT) * TEXT_ENCODER_OUTPUT_SCALE
                    }
                }
            }

            val quantizedTxtFeats = ByteBuffer.allocateDirect(NUM_SLOTS * EMBEDDING_DIM).order(ByteOrder.nativeOrder())
            for (value in floatEmbeddings) {
                val requantized = (value / DETECTOR_TXT_FEATS_SCALE + DETECTOR_TXT_FEATS_ZERO_POINT)
                    .roundToInt()
                    .coerceIn(0, 255)
                quantizedTxtFeats.put(requantized.toByte())
            }
            quantizedTxtFeats.rewind()

            Log.i(TAG, "txt_feats pronto: ${prompts.size} prompts canônicos telhados em $NUM_SLOTS slots")
            return TextEmbeddingProvider(quantizedTxtFeats, slotToPromptIndex)
        }
    }
}
