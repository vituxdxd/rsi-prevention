package com.rsiprevention.app.ui

import android.app.AlertDialog
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.view.Gravity
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import com.google.zxing.BarcodeFormat
import com.google.zxing.qrcode.QRCodeWriter
import com.rsiprevention.app.R

/**
 * Diálogo com o QR code da URL de transmissão — apontar a câmera do outro
 * aparelho abre a página no navegador sem digitar o endereço.
 *
 * O QR é gerado localmente pelo ZXing (só geração, sem câmera).

 * Gerado uma única vez, ao abrir o diálogo (custo desprezível).
 */
object QrCodeDialog {

    private const val QR_SIZE_PX = 640
    private const val PADDING_PX = 48

    fun show(context: Context, url: String) {
        val container = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(PADDING_PX, PADDING_PX, PADDING_PX, PADDING_PX)
        }

        runCatching { QrCodeFactory.create(url, QR_SIZE_PX) }
            .getOrNull()
            ?.let { qr ->
                container.addView(
                    ImageView(context).apply {
                        setImageBitmap(qr)
                        contentDescription = url
                    }
                )
            }

        container.addView(
            TextView(context).apply {
                text = url
                textSize = 14f
                gravity = Gravity.CENTER
                setPadding(0, PADDING_PX / 2, 0, 0)
            }
        )

        AlertDialog.Builder(context)
            .setTitle(R.string.stream_qr_title)
            .setView(container)
            .setPositiveButton(R.string.button_ok, null)
            .show()
    }
}

/** Geração pura do QR (ZXing) em um [Bitmap] — separada para facilitar manutenção. */
object QrCodeFactory {

    fun create(text: String, sizePx: Int): Bitmap {
        val matrix = QRCodeWriter().encode(text, BarcodeFormat.QR_CODE, sizePx, sizePx)
        val pixels = IntArray(sizePx * sizePx)
        var index = 0
        for (y in 0 until sizePx) {
            for (x in 0 until sizePx) {
                pixels[index++] = if (matrix[x, y]) Color.BLACK else Color.WHITE
            }
        }
        return Bitmap.createBitmap(pixels, sizePx, sizePx, Bitmap.Config.ARGB_8888)
    }
}
