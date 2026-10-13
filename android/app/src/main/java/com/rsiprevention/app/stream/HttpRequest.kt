package com.rsiprevention.app.stream

/** Primeira linha de um request HTTP — o mínimo que o servidor de stream usa. */
data class HttpRequest(val method: String, val path: String)

/** Parser da linha de request (pura — testável em JVM). */
object HttpRequestParser {

    /**
     * @param requestLine ex.: `GET /frame.jpg?t=123 HTTP/1.1`
     * @return request parseado (path sem query/fragment) ou `null` se inválido.
     */
    fun parse(requestLine: String): HttpRequest? {
        val parts = requestLine.trim().split(' ')
        if (parts.size < 2) return null
        val method = parts[0].uppercase()
        val path = parts[1].substringBefore('?').substringBefore('#')
        if (path.isEmpty() || !path.startsWith('/')) return null
        return HttpRequest(method, path)
    }
}

/** Formatação do stream MJPEG (`multipart/x-mixed-replace`) — pura, testável. */
object MjpegFormat {

    const val BOUNDARY = "rsi-frame"
    const val CONTENT_TYPE = "multipart/x-mixed-replace; boundary=$BOUNDARY"

    val HEADER: ByteArray = (
        "HTTP/1.1 200 OK\r\n" +
            "Cache-Control: no-store\r\n" +
            "Pragma: no-cache\r\n" +
            "Connection: close\r\n" +
            "Content-Type: $CONTENT_TYPE\r\n\r\n"
        ).toByteArray(Charsets.US_ASCII)

    val PART_TRAILER: ByteArray = "\r\n".toByteArray(Charsets.US_ASCII)

    fun partHeader(jpegLength: Int): ByteArray = (
        "--$BOUNDARY\r\n" +
            "Content-Type: image/jpeg\r\n" +
            "Content-Length: $jpegLength\r\n\r\n"
        ).toByteArray(Charsets.US_ASCII)
}
