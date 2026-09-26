# Complemento obrigatório: rótulos das boxes em Português do Brasil

> **Nota histórica**: este documento é a especificação original da fase
> YOLO-World (multi-classe, com `text_encoder` CLIP). O projeto evoluiu para um
> detector YOLO26s de **classe única** ("Instrumento Cirúrgico") — ver
> `README.md`, seção "Classe única". A regra de UI em pt-BR continua valendo;
> o mapeamento de classes abaixo é referência para o caso de um futuro retorno
> a múltiplas classes.

Aplicar esta regra ao projeto do POCO F7 com YOLOv8s-worldv2 / QNN Hexagon V73:

## Regra de interface

Todo texto exibido ao usuário no overlay da câmera deve estar em **Português do Brasil (pt-BR)**. Isso inclui:

- Nome da classe acima/ao lado da bounding box
- Status do instrumento no overlay
- Lista de instrumentos detectados
- Alertas visuais e sonoros
- Botões, títulos, diálogos, mensagens de erro e tela de checagem final

O modelo YOLO-World deve continuar recebendo prompts técnicos em inglês, pois são os prompts de melhor compatibilidade com o modelo. A tradução é feita exclusivamente na camada de apresentação Kotlin, depois que o resultado do modelo é associado à classe/prompt original.

## Mapeamento obrigatório de classes

```kotlin
val labelPtBr = mapOf(
    "surgical scissors" to "Tesoura cirúrgica",
    "scissors" to "Tesoura",
    "forceps" to "Pinça",
    "surgical forceps" to "Pinça cirúrgica",
    "hemostatic forceps" to "Pinça hemostática",
    "clamp" to "Pinça clamp",
    "surgical clamp" to "Pinça cirúrgica",
    "scalpel" to "Bisturi",
    "surgical scalpel" to "Bisturi cirúrgico",
    "needle" to "Agulha",
    "surgical needle" to "Agulha cirúrgica",
    "needle holder" to "Porta-agulha",
    "needle driver" to "Porta-agulha",
    "gauze" to "Gaze",
    "surgical gauze" to "Gaze cirúrgica",
    "sponge" to "Compressa",
    "surgical sponge" to "Compressa cirúrgica",
    "surgical tray" to "Bandeja cirúrgica",
    "instrument tray" to "Bandeja de instrumentos",
    "retractor" to "Afastador",
    "surgical retractor" to "Afastador cirúrgico",
    "suction tube" to "Cânula de aspiração",
    "surgical suction" to "Aspirador cirúrgico"
)

fun displayLabel(modelClassName: String): String =
    labelPtBr[modelClassName.trim().lowercase()] ?: modelClassName
```

## Exibição das boxes

A box deve usar o rótulo já traduzido, nunca o prompt em inglês. Formato recomendado:

```text
Tesoura cirúrgica • 92%
```

Exemplo de construção do texto:

```kotlin
val label = displayLabel(detection.className)
val confidence = (detection.confidence * 100).toInt()
val boxText = "$label • $confidence%"
canvas.drawText(boxText, textX, textY, paint)
```

## Estados de tracking

Os estados também devem ser exibidos exclusivamente em pt-BR:

```kotlin
enum class InstrumentStatus(val displayName: String) {
    ON_TABLE("Na mesa"),
    REMOVED("Retirado da mesa"),
    RETURNED("Devolvido à mesa"),
    NOT_RETURNED("Não devolvido")
}
```

Mensagens obrigatoriamente em pt-BR:

```text
Monitoramento ativo
Instrumento detectado
Instrumento retirado da mesa
Instrumento devolvido à mesa
Checagem concluída
ATENÇÃO: instrumento não devolvido à mesa
NPU Hexagon ativa
```

## Regra para o agente de IA

- Prompts enviados para YOLO-World: inglês técnico
- Texto apresentado ao usuário: Português do Brasil
- Nunca mostrar `forceps`, `gauze`, `surgical scissors`, `clamp` ou qualquer label bruto em inglês no overlay
- Se o modelo retornar uma classe que não existe no mapa, usar temporariamente o texto original, registrar em log e adicionar a tradução pt-BR ao mapa na próxima revisão
