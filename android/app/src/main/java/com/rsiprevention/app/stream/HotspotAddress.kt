package com.rsiprevention.app.stream

import java.net.Inet4Address
import java.net.NetworkInterface

/**
 * Descobre o endereço IPv4 local para exibir a URL do stream (ex.: no hotspot
 * do aparelho, normalmente `192.168.43.1` ou `192.168.187.1`).
 *
 * Prefere interfaces de access point (`ap*`, `swlan*`, `softap*`, `wlan1`);
 * se não achar, usa o primeiro endereço site-local disponível.
 */
object HotspotAddress {

    /** Fallback exibido quando nenhuma interface é encontrada (hotspot desligado). */
    const val FALLBACK_IP = "192.168.43.1"

    private val AP_PREFIXES = listOf("ap", "swlan", "softap", "wlan1")

    fun find(): String? = choose(collectCandidates())

    /** Escolha pura (testável): prefere interface de AP, senão a primeira. */
    fun choose(candidates: List<Pair<String, String>>): String? {
        val preferred = candidates.firstOrNull { (name, _) ->
            AP_PREFIXES.any { name.startsWith(it) }
        }
        return (preferred ?: candidates.firstOrNull())?.second
    }

    fun url(ip: String, port: Int): String = "http://$ip:$port"

    private fun collectCandidates(): List<Pair<String, String>> = try {
        NetworkInterface.getNetworkInterfaces().toList()
            .filter { it.isUp && !it.isLoopback }
            .flatMap { nic -> nic.inetAddresses.toList().map { nic.name to it } }
            .filter { (_, address) -> address is Inet4Address && address.isSiteLocalAddress }
            .map { (name, address) -> name to address.hostAddress.orEmpty() }
            .filter { (_, ip) -> ip.isNotEmpty() }
    } catch (t: Throwable) {
        emptyList()
    }
}
