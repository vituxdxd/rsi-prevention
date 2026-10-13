package com.rsiprevention.app.stream

import android.util.Log
import java.io.BufferedOutputStream
import java.io.ByteArrayOutputStream
import java.io.Closeable
import java.io.InputStream
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.atomic.AtomicInteger

/**
 * Servidor HTTP mínimo embutido (sem dependências) para a transmissão:
 *
 *  - `GET /`           página de visualização (pt-BR). No Android/Chrome usa
 *                      MJPEG nativo; no Safari do iPhone (que **não** suporta
 *                      `multipart/x-mixed-replace`) cai automaticamente para
 *                      polling em `/frame.jpg`.
 *  - `GET /stream`     stream MJPEG (`multipart/x-mixed-replace`) — 1 fila de
 *                      1 frame por cliente, descartando o antigo.
 *  - `GET /frame.jpg`  último frame JPEG (fallback iOS / teste rápido).
 *
 * Performance: `TCP_NODELAY` em todos os sockets (mata o Nagle — latência
 * mínima no hotspot), escrita em poucos `write` + `flush`, e o JPEG publicado
 * é o **mesmo array** para todos os clientes (sem cópia por cliente).
 *
 * O hotspot é criado manualmente pelo usuário (Configurações do Android); o
 * app só escuta em `0.0.0.0`, então basta a permissão `INTERNET`.
 */
class StreamServer(private val port: Int = DEFAULT_PORT) {

    companion object {
        const val DEFAULT_PORT = 8080
        private const val TAG = "StreamServer"
        private const val MAX_MJPEG_CLIENTS = 4
        private const val MAX_HEADER_BYTES = 4096
        private val WAKEUP = ByteArray(0)

        private val PAGE = """
            <!doctype html>
            <html lang="pt-BR">
            <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>RSI Prevention — Transmissão</title>
            <style>
              html,body{margin:0;height:100%;background:#000;color:#eee;font-family:sans-serif}
              #wrap{height:100%;display:flex;align-items:center;justify-content:center}
              #v{max-width:100%;max-height:100%}
              #status{position:fixed;top:8px;left:10px;font-size:13px;background:rgba(0,0,0,.55);padding:4px 8px;border-radius:6px}
            </style>
            </head>
            <body>
            <div id="wrap"><img id="v" alt="aguardando sinal"></div>
            <div id="status">conectando…</div>
            <script>
            var img=document.getElementById('v'), st=document.getElementById('status');
            var frames=0, last=performance.now();
            setInterval(function(){
              var now=performance.now();
              if(now-last>=1000){ st.textContent=(frames*1000/(now-last)).toFixed(1)+' fps'; frames=0; last=now; }
            },250);
            var isIOS=/iPad|iPhone|iPod/.test(navigator.userAgent);
            function poll(){
              var probe=new Image();
              probe.onload=function(){ img.src=probe.src; frames++; setTimeout(poll,25); };
              probe.onerror=function(){ st.textContent='sem sinal'; setTimeout(poll,500); };
              probe.src='/frame.jpg?t='+Date.now();
            }
            if(isIOS){ poll(); }
            else {
              img.onload=function(){ st.textContent='ao vivo (MJPEG)'; };
              img.onerror=function(){ st.textContent='sem sinal'; };
              img.src='/stream';
            }
            </script>
            </body>
            </html>
        """.trimIndent()
    }

    private class MjpegClient(val socket: Socket, val queue: LinkedBlockingQueue<ByteArray>) {
        @Volatile
        var closed = false
    }

    @Volatile
    private var latestJpeg: ByteArray? = null

    @Volatile
    private var running = false

    private var serverSocket: ServerSocket? = null
    private val clients = CopyOnWriteArrayList<MjpegClient>()
    private val clientCount = AtomicInteger()

    /** Abre a porta e começa a aceitar conexões (threads daemon). Idempotente. */
    fun start() {
        if (running) return
        val socket = ServerSocket(port).apply { reuseAddress = true }
        serverSocket = socket
        running = true
        Thread({ acceptLoop(socket) }, "rsi-accept").apply { isDaemon = true }.start()
        Log.i(TAG, "Transmissão ativa em http://0.0.0.0:$port")
    }

    /** Fecha o servidor e todos os clientes. Idempotente. */
    fun stop() {
        if (!running) return
        running = false
        closeQuietly(serverSocket)
        serverSocket = null
        for (client in clients) {
            client.closed = true
            client.queue.offer(WAKEUP)
            closeQuietly(client.socket)
        }
        clients.clear()
        clientCount.set(0)
        latestJpeg = null
        Log.i(TAG, "Transmissão encerrada")
    }

    /** Publica o JPEG mais recente para todos os clientes MJPEG (fila de 1). */
    fun publishFrame(jpeg: ByteArray) {
        latestJpeg = jpeg
        for (client in clients) {
            if (client.closed) continue
            client.queue.poll()
            client.queue.offer(jpeg)
        }
    }

    private fun acceptLoop(server: ServerSocket) {
        while (running) {
            val socket = try {
                server.accept()
            } catch (t: Throwable) {
                break
            }
            Thread({ handle(socket) }, "rsi-http").apply { isDaemon = true }.start()
        }
    }

    private fun handle(socket: Socket) {
        try {
            socket.tcpNoDelay = true
            val requestLine = readRequestLine(socket.getInputStream()) ?: return
            val request = HttpRequestParser.parse(requestLine)
            val output = BufferedOutputStream(socket.getOutputStream(), 64 * 1024)
            when {
                request == null || request.method != "GET" ->
                    respond(output, "400 Bad Request", "text/plain; charset=utf-8", "Bad Request".toByteArray())

                request.path == "/" ->
                    respond(output, "200 OK", "text/html; charset=utf-8", PAGE.toByteArray(Charsets.UTF_8))

                request.path == "/frame.jpg" ->
                    respondSnapshot(output)

                request.path == "/stream" ->
                    streamMjpeg(socket, output)

                else ->
                    respond(output, "404 Not Found", "text/plain; charset=utf-8", "Not Found".toByteArray())
            }
        } catch (t: Throwable) {
            // Cliente caiu no meio da resposta — ignora.
        } finally {
            closeQuietly(socket)
        }
    }

    /** Lê só a primeira linha do request (até `\r\n`), com teto de bytes. */
    private fun readRequestLine(input: InputStream): String? {
        val header = ByteArrayOutputStream()
        var last4 = 0
        while (header.size() < MAX_HEADER_BYTES) {
            val b = input.read()
            if (b < 0) break
            header.write(b)
            last4 = (last4 shl 8) or b
            if (last4 == 0x0D0A0D0A) break
        }
        if (header.size() == 0) return null
        return String(header.toByteArray(), Charsets.US_ASCII).lineSequence().firstOrNull()
    }

    private fun respond(output: BufferedOutputStream, status: String, contentType: String, body: ByteArray) {
        val header = (
            "HTTP/1.1 $status\r\n" +
                "Content-Type: $contentType\r\n" +
                "Content-Length: ${body.size}\r\n" +
                "Cache-Control: no-store\r\n" +
                "Connection: close\r\n\r\n"
            ).toByteArray(Charsets.US_ASCII)
        output.write(header)
        output.write(body)
        output.flush()
    }

    private fun respondSnapshot(output: BufferedOutputStream) {
        val jpeg = latestJpeg
        if (jpeg == null) {
            respond(
                output,
                "503 Service Unavailable",
                "text/plain; charset=utf-8",
                "Sem frame disponível ainda".toByteArray()
            )
        } else {
            respond(output, "200 OK", "image/jpeg", jpeg)
        }
    }

    private fun streamMjpeg(socket: Socket, output: BufferedOutputStream) {
        if (clientCount.get() >= MAX_MJPEG_CLIENTS) {
            respond(
                output,
                "503 Service Unavailable",
                "text/plain; charset=utf-8",
                "Limite de visualizadores atingido".toByteArray()
            )
            return
        }

        val client = MjpegClient(socket, LinkedBlockingQueue(1))
        output.write(MjpegFormat.HEADER)
        output.flush()
        clients.add(client)
        clientCount.incrementAndGet()

        try {
            while (running && !client.closed) {
                val jpeg = client.queue.take()
                if (client.closed || jpeg === WAKEUP) break
                output.write(MjpegFormat.partHeader(jpeg.size))
                output.write(jpeg)
                output.write(MjpegFormat.PART_TRAILER)
                output.flush()
            }
        } catch (t: Throwable) {
            // Desconectou — cai no finally.
        } finally {
            clients.remove(client)
            clientCount.decrementAndGet()
        }
    }

    private fun closeQuietly(closeable: Closeable?) {
        runCatching { closeable?.close() }
    }
}
