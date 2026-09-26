package com.rsiprevention.app.stream

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** Parser da linha de request HTTP e formatação do stream MJPEG. */
class HttpAndMjpegTest {

    @Test
    fun `parse extrai metodo e path sem query`() {
        val request = HttpRequestParser.parse("GET /frame.jpg?t=123 HTTP/1.1")

        assertEquals("GET", request?.method)
        assertEquals("/frame.jpg", request?.path)
    }

    @Test
    fun `parse normaliza o metodo para maiusculo`() {
        assertEquals("GET", HttpRequestParser.parse("get / HTTP/1.1")?.method)
    }

    @Test
    fun `parse rejeita linhas invalidas`() {
        assertNull(HttpRequestParser.parse(""))
        assertNull(HttpRequestParser.parse("GET"))
        assertNull(HttpRequestParser.parse("GET relative/path HTTP/1.1"))
    }

    @Test
    fun `part header do mjpeg tem boundary tamanho e terminador`() {
        val header = String(MjpegFormat.partHeader(1234), Charsets.US_ASCII)

        assertTrue(header.startsWith("--${MjpegFormat.BOUNDARY}\r\n"))
        assertTrue(header.contains("Content-Type: image/jpeg"))
        assertTrue(header.contains("Content-Length: 1234"))
        assertTrue(header.endsWith("\r\n\r\n"))
    }

    @Test
    fun `response header do mjpeg e multipart x-mixed-replace`() {
        val header = String(MjpegFormat.HEADER, Charsets.US_ASCII)

        assertTrue(header.startsWith("HTTP/1.1 200 OK"))
        assertTrue(header.contains("Content-Type: ${MjpegFormat.CONTENT_TYPE}"))
        assertTrue(header.endsWith("\r\n\r\n"))
    }
}
