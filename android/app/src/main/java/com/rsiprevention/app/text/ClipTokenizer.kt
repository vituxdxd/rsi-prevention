package com.rsiprevention.app.text

import android.content.Context
import android.util.Log
import java.io.BufferedInputStream
import java.io.InputStream
import java.util.regex.Pattern
import java.util.zip.GZIPInputStream

/**
 * Tokenizador CLIP BPE (Byte-Pair Encoding), reimplementação em Kotlin do
 * `SimpleTokenizer` original da OpenAI/Ultralytics (usado pelo `qai_hub_models`
 * para gerar os `tokens` de entrada do grafo `text_encoder`).
 *
 * Este tokenizador precisa produzir **exatamente** os mesmos ids de vocabulário
 * que o tokenizador Python usado no export — os ids são índices de uma tabela
 * de embeddings aprendida durante o treino do CLIP; qualquer divergência produz
 * embeddings de texto incorretos silenciosamente (sem erro, só detecção ruim).
 *
 * Simplificação deliberada e documentada: o `basic_clean` original usa `ftfy`
 * (correção de mojibake) + `html.unescape` duplo, pensado para texto arbitrário
 * da internet. Os prompts deste projeto são strings técnicas fixas em inglês
 * simples (ver [com.rsiprevention.app.vocabulary.InstrumentVocabulary]), então
 * `basicClean` aqui é apenas um `trim()` — não há mojibake nem entidades HTML
 * para corrigir nesse vocabulário controlado.
 */
class ClipTokenizer private constructor(
    private val encoder: Map<String, Int>,
    private val bpeRanks: Map<Pair<String, String>, Int>,
    private val byteEncoder: Map<Int, Char>,
    private val startOfTextId: Int,
    private val endOfTextId: Int
) {
    private val bpeCache = HashMap<String, String>()

    /**
     * Tokeniza [text] para um `IntArray` de tamanho fixo [contextLength],
     * incluindo `<|startoftext|>`/`<|endoftext|>` e padding com zeros — mesmo
     * formato exigido pela entrada `tokens` do grafo `text_encoder` (int32).
     */
    fun tokenize(text: String, contextLength: Int): IntArray {
        val ids = encodeToIds(text)
        val sequence = ArrayList<Int>(ids.size + 2)
        sequence.add(startOfTextId)
        sequence.addAll(ids)
        sequence.add(endOfTextId)

        val result = IntArray(contextLength) // já nasce zero-padded
        val effectiveLength = minOf(sequence.size, contextLength)
        for (i in 0 until effectiveLength) {
            result[i] = sequence[i]
        }
        if (sequence.size > contextLength) {
            result[contextLength - 1] = endOfTextId
        }
        return result
    }

    private fun encodeToIds(text: String): List<Int> {
        val cleaned = whitespaceClean(basicClean(text)).lowercase()
        val matcher = TOKEN_PATTERN.matcher(cleaned)
        val ids = mutableListOf<Int>()
        while (matcher.find()) {
            val rawToken = matcher.group()
            val byteEncoded = StringBuilder()
            for (b in rawToken.toByteArray(Charsets.UTF_8)) {
                val unsignedByte = b.toInt() and 0xFF
                byteEncoded.append(byteEncoder.getValue(unsignedByte))
            }
            for (piece in bpe(byteEncoded.toString()).split(' ')) {
                val id = encoder[piece]
                if (id != null) {
                    ids.add(id)
                } else {
                    Log.w(TAG, "Peça BPE sem id no vocabulário: '$piece' (token bruto: '$rawToken')")
                }
            }
        }
        return ids
    }

    /** Algoritmo BPE guloso padrão: mescla repetidamente o par adjacente de menor rank. */
    private fun bpe(token: String): String {
        bpeCache[token]?.let { return it }
        if (token.isEmpty()) return token

        var word: List<String> = if (token.length == 1) {
            listOf(token + END_OF_WORD)
        } else {
            token.dropLast(1).map { it.toString() } + (token.last() + END_OF_WORD)
        }

        var pairs = adjacentPairs(word)
        if (pairs.isEmpty()) {
            val result = token + END_OF_WORD
            bpeCache[token] = result
            return result
        }

        while (true) {
            val bigram = pairs.minByOrNull { bpeRanks[it] ?: Int.MAX_VALUE } ?: break
            if (!bpeRanks.containsKey(bigram)) break
            val (first, second) = bigram

            val newWord = mutableListOf<String>()
            var i = 0
            while (i < word.size) {
                val j = indexOfFrom(word, first, i)
                if (j == -1) {
                    newWord.addAll(word.subList(i, word.size))
                    break
                }
                newWord.addAll(word.subList(i, j))
                i = j

                if (word[i] == first && i < word.size - 1 && word[i + 1] == second) {
                    newWord.add(first + second)
                    i += 2
                } else {
                    newWord.add(word[i])
                    i += 1
                }
            }
            word = newWord
            if (word.size == 1) break
            pairs = adjacentPairs(word)
        }

        val result = word.joinToString(" ")
        bpeCache[token] = result
        return result
    }

    private fun indexOfFrom(list: List<String>, element: String, from: Int): Int {
        for (idx in from until list.size) {
            if (list[idx] == element) return idx
        }
        return -1
    }

    private fun adjacentPairs(word: List<String>): Set<Pair<String, String>> {
        val pairs = mutableSetOf<Pair<String, String>>()
        for (i in 0 until word.size - 1) {
            pairs.add(word[i] to word[i + 1])
        }
        return pairs
    }

    private fun basicClean(text: String): String = text.trim()

    private fun whitespaceClean(text: String): String =
        WHITESPACE_PATTERN.matcher(text).replaceAll(" ").trim()

    companion object {
        private const val TAG = "ClipTokenizer"
        private const val END_OF_WORD = "</w>"
        private const val VOCAB_ASSET_PATH = "clip/bpe_simple_vocab_16e6.txt"
        private const val VOCAB_ASSET_PATH_GZ = "clip/bpe_simple_vocab_16e6.txt.gz"

        /**
         * Número exato de linhas de merge usadas pelo tokenizador CLIP original a
         * partir deste arquivo `bpe_simple_vocab_16e6.txt.gz` (mesma constante de
         * `merges[1 : 49152-256-2+1]` no `simple_tokenizer.py` da OpenAI/Ultralytics).
         * Verificado byte a byte contra a cópia do pacote `clip` do Python: o
         * arquivo tem 262146 linhas no total, mas o vocabulário original usa
         * **apenas** as primeiras 48894 linhas de merge após o cabeçalho — usar
         * o arquivo inteiro produziria um vocabulário/ranking diferente e
         * incompatível com os pesos do CLIP.
         */
        private const val NUM_MERGES = 48894

        private val TOKEN_PATTERN: Pattern = Pattern.compile(
            "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+",
            Pattern.CASE_INSENSITIVE
        )
        private val WHITESPACE_PATTERN: Pattern = Pattern.compile("\\s+")

        /**
         * Deve ser chamado fora da thread principal (leitura + parsing do vocabulário).
         *
         * Atenção ao nome do arquivo: o AAPT **remove a extensão `.gz` e
         * descompacta o conteúdo** ao empacotar assets — dentro do APK o arquivo
         * vira `clip/bpe_simple_vocab_16e6.txt` (texto puro), embora no repositório
         * ele exista como `.txt.gz`. Por isso tentamos primeiro o nome sem `.gz`
         * (o real no aparelho) e caímos para o nome original; o conteúdo é lido
         * como gzip ou texto puro conforme os magic bytes do próprio arquivo
         * (ver [loadFromVocabStream]).
         */
        fun load(context: Context): ClipTokenizer {
            val stream = runCatching { context.assets.open(VOCAB_ASSET_PATH) }
                .getOrElse { context.assets.open(VOCAB_ASSET_PATH_GZ) }
            return stream.use { loadFromVocabStream(it) }
        }

        /**
         * Variante testável em JVM puro (sem `Context`/Robolectric): recebe
         * diretamente o stream do `.gz` do vocabulário BPE.
         */
        fun loadFromGzipStream(gzipInput: InputStream): ClipTokenizer = loadFromVocabStream(gzipInput)

        /**
         * Aceita o vocabulário tanto em gzip (arquivo do repositório) quanto em
         * texto puro (formato que o AAPT grava no APK ao remover o `.gz`).
         */
        fun loadFromVocabStream(input: InputStream): ClipTokenizer {
            val buffered = BufferedInputStream(input)
            buffered.mark(2)
            val magic = ByteArray(2)
            val magicRead = buffered.read(magic)
            buffered.reset()
            val isGzip = magicRead == 2 && magic[0] == 0x1F.toByte() && magic[1] == 0x8B.toByte()

            val vocabText = if (isGzip) {
                GZIPInputStream(buffered).use { gzip -> gzip.readBytes().toString(Charsets.UTF_8) }
            } else {
                buffered.use { plain -> plain.readBytes().toString(Charsets.UTF_8) }
            }
            val allLines = vocabText.split("\n")
            // allLines[0] é o cabeçalho ("#version: 0.2"); as linhas de merge
            // começam no índice 1.
            val mergeLines = allLines.drop(1).take(NUM_MERGES)
            val merges: List<Pair<String, String>> = mergeLines.map { line ->
                val parts = line.split(' ')
                require(parts.size == 2) { "Linha de merge malformada no vocabulário BPE: '$line'" }
                parts[0] to parts[1]
            }

            val byteEncoder = bytesToUnicode()
            val baseSymbols: List<String> = byteEncoder.values.map { it.toString() }

            val vocab = ArrayList<String>(baseSymbols.size * 2 + merges.size + 2)
            vocab.addAll(baseSymbols)
            vocab.addAll(baseSymbols.map { it + END_OF_WORD })
            merges.forEach { (a, b) -> vocab.add(a + b) }
            vocab.add("<|startoftext|>")
            vocab.add("<|endoftext|>")

            val encoder = HashMap<String, Int>(vocab.size * 2)
            vocab.forEachIndexed { index, symbol -> encoder[symbol] = index }

            val bpeRanks = HashMap<Pair<String, String>, Int>(merges.size * 2)
            merges.forEachIndexed { index, pair -> bpeRanks[pair] = index }

            val startOfTextId = encoder.getValue("<|startoftext|>")
            val endOfTextId = encoder.getValue("<|endoftext|>")

            Log.i(TAG, "CLIP tokenizer carregado: vocabulário=${encoder.size} merges=${merges.size}")

            return ClipTokenizer(encoder, bpeRanks, byteEncoder, startOfTextId, endOfTextId)
        }

        /**
         * Reimplementação exata de `bytes_to_unicode()` (OpenAI CLIP): mapeia
         * cada um dos 256 valores de byte para um caractere Unicode "visível",
         * na mesma ordem de inserção do algoritmo original — a ORDEM importa,
         * pois define a ordem (e portanto os ids) das primeiras 256 entradas
         * do vocabulário.
         */
        private fun bytesToUnicode(): Map<Int, Char> {
            val bs = mutableListOf<Int>()
            bs.addAll(33..126)   // '!'..'~'
            bs.addAll(161..172)  // '¡'..'¬'
            bs.addAll(174..255)  // '®'..'ÿ'

            val cs = bs.toMutableList()
            var n = 0
            for (b in 0..255) {
                if (b !in bs) {
                    bs.add(b)
                    cs.add(256 + n)
                    n++
                }
            }

            val map = LinkedHashMap<Int, Char>(256)
            for (i in bs.indices) {
                map[bs[i]] = cs[i].toChar()
            }
            return map
        }
    }
}
