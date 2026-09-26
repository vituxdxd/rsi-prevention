package com.rsiprevention.app.vocabulary

/**
 * Vocabulário exibido pelo app.
 *
 * Decisão de projeto: o modelo bruto distingue 5 instrumentos (`clamp`,
 * `needle_holder`, `scalpel`, `shear`, `tweezer`), mas para prevenção de
 * retenção inadvertida de corpos estranhos a subclasse **não importa** — o que
 * importa é detectar, rastrear e conferir qualquer instrumento deixado ou
 * retirado da mesa. A classificação fina errava muito (ex.: tesoura × pinça) e
 * gerava troca de rótulo na UI. O [com.rsiprevention.app.detection.Postprocessor]
 * colapsa as 5 pontuações em uma única classe [SINGLE_CLASS_NAME].
 *
 * Regra do projeto: texto exibido ao usuário sempre em Português do Brasil.
 */
object InstrumentVocabulary {

    /** Classe única do app. */
    const val SINGLE_CLASS_ID = 0
    const val SINGLE_CLASS_NAME = "instrumento_cirurgico"

    /**
     * Nomes brutos das 5 classes do modelo. Uso interno (votação de classe do
     * tracker); a exibição é sempre [SINGLE_CLASS_NAME].
     */
    val MODEL_RAW_CLASSES: List<String> = listOf(
        "clamp",
        "needle_holder",
        "scalpel",
        "shear",
        "tweezer"
    )

    /**
     * Rótulos pt-BR. Inclui também os nomes brutos das 5 classes do modelo
     * para que qualquer detecção/track antigo continue exibindo o rótulo único.
     */
    private val labelPtBr: Map<String, String> = mapOf(
        SINGLE_CLASS_NAME to "Instrumento Cirúrgico",
        "clamp" to "Instrumento Cirúrgico",
        "needle_holder" to "Instrumento Cirúrgico",
        "scalpel" to "Instrumento Cirúrgico",
        "shear" to "Instrumento Cirúrgico",
        "tweezer" to "Instrumento Cirúrgico"
    )

    /**
     * Traduz o nome de classe bruto retornado pelo modelo para pt-BR.
     * Se a classe não existir no mapa, registra o caso (log) e retorna o texto
     * original temporariamente, conforme regra do projeto.
     */
    fun displayLabel(modelClassName: String): String {
        val key = modelClassName.trim().lowercase()
        val translated = labelPtBr[key]
        if (translated == null) {
            android.util.Log.w(
                "InstrumentVocabulary",
                "Classe sem tradução pt-BR cadastrada: '$modelClassName'. Adicionar ao mapa labelPtBr."
            )
        }
        return translated ?: modelClassName
    }
}
