package com.rsiprevention.app.text

import org.junit.Assert.assertArrayEquals
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.File
import java.io.FileInputStream
import java.util.zip.GZIPInputStream

/**
 * Valida que [ClipTokenizer] produz exatamente os mesmos ids de vocabulário que
 * o tokenizador CLIP de referência em Python (`clip.simple_tokenizer.SimpleTokenizer`,
 * o mesmo usado pelo `qai_hub_models` no export do `text_encoder`).
 *
 * Os valores esperados abaixo foram gerados rodando o tokenizador Python real
 * (pacote `clip`, instalado no ambiente de export) sobre exatamente os mesmos
 * prompts de [com.rsiprevention.app.vocabulary.InstrumentVocabulary.CANONICAL_PROMPTS],
 * incluindo os tokens `<|startoftext|>` (49406) e `<|endoftext|>` (49407).
 * Qualquer divergência aqui indica que o tokenizador Kotlin produzirá embeddings
 * de texto incompatíveis com o grafo `text_encoder` exportado.
 */
class ClipTokenizerTest {

    private fun loadTokenizer(): ClipTokenizer {
        val vocabFile = File("src/main/assets/clip/bpe_simple_vocab_16e6.txt.gz")
        return FileInputStream(vocabFile).use { ClipTokenizer.loadFromGzipStream(it) }
    }

    private fun assertPromptMatches(tokenizer: ClipTokenizer, prompt: String, expected: IntArray) {
        val tokens = tokenizer.tokenize(prompt, contextLength = expected.size)
        assertArrayEquals("prompt='$prompt'", expected, tokens)
    }

    @Test
    fun `tokenizes canonical prompts exactly like the python reference`() {
        val tokenizer = loadTokenizer()

        assertPromptMatches(tokenizer, "surgical scissors", intArrayOf(49406, 16386, 30867, 49407))
        assertPromptMatches(tokenizer, "forceps", intArrayOf(49406, 12068, 814, 49407))
        assertPromptMatches(tokenizer, "hemostatic forceps", intArrayOf(49406, 651, 7034, 2647, 12068, 814, 49407))
        assertPromptMatches(tokenizer, "scalpel", intArrayOf(49406, 10859, 7006, 49407))
        assertPromptMatches(tokenizer, "needle holder", intArrayOf(49406, 19886, 7862, 49407))
        assertPromptMatches(tokenizer, "gauze", intArrayOf(49406, 5919, 1298, 49407))
        assertPromptMatches(tokenizer, "surgical clamp", intArrayOf(49406, 16386, 41501, 49407))
        assertPromptMatches(tokenizer, "retractor", intArrayOf(49406, 515, 14607, 49407))
        assertPromptMatches(tokenizer, "suction tube", intArrayOf(49406, 42786, 3308, 49407))
    }

    @Test
    fun `pads with zeros up to context length 77`() {
        val tokenizer = loadTokenizer()
        val tokens = tokenizer.tokenize("scalpel", contextLength = 77)

        assertArrayEquals(intArrayOf(49406, 10859, 7006, 49407), tokens.copyOfRange(0, 4))
        for (i in 4 until 77) {
            assert(tokens[i] == 0) { "posição $i deveria ser zero-padding, era ${tokens[i]}" }
        }
    }

    @Test
    fun `loads vocabulary from a plain text stream like the one aapt packages`() {
        // O AAPT remove a extensão `.gz` e descompacta o conteúdo ao empacotar
        // assets (dentro do APK o arquivo é `clip/bpe_simple_vocab_16e6.txt` em
        // texto puro — ver ClipTokenizer.load). Este teste garante que o parser
        // aceita o stream sem gzip, não apenas o `.gz` do repositório.
        val vocabFile = File("src/main/assets/clip/bpe_simple_vocab_16e6.txt.gz")
        val plainText = GZIPInputStream(FileInputStream(vocabFile)).use { it.readBytes() }

        val tokenizer = ByteArrayInputStream(plainText).use { ClipTokenizer.loadFromVocabStream(it) }
        assertPromptMatches(tokenizer, "scalpel", intArrayOf(49406, 10859, 7006, 49407))
    }
}
