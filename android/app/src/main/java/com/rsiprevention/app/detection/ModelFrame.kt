package com.rsiprevention.app.detection

/**
 * Frame de câmera já no **espaço de entrada do modelo** (640×640, ver
 * [Preprocessor]): pixels ARGB linha a linha, sem escala.
 *
 * É a referência espacial correta para qualquer processamento que precise
 * relacionar pixels com as caixas devolvidas pelo detector — em particular o
 * template visual do [com.rsiprevention.app.tracking.Appearance], que antes
 * amostrava o bitmap "em pé" da câmera como se as coordenadas do modelo fossem
 * pixels dele (região errada quando o frame não era 640×640).
 *
 * O [pixels] é reutilizado a cada frame pelo [Preprocessor] (mesmo array) e só
 * é válido **durante o processamento síncrono do frame atual**, na mesma thread.
 */
class ModelFrame(
    val pixels: IntArray,
    val size: Int
)
