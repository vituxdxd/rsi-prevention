package com.rsiprevention.app.stream

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/** Escolha do IP exibido para o stream: prefere interfaces de access point. */
class HotspotAddressTest {

    @Test
    fun `prefere interface de access point`() {
        val chosen = HotspotAddress.choose(
            listOf("wlan0" to "192.168.0.10", "ap0" to "192.168.43.1")
        )
        assertEquals("192.168.43.1", chosen)
    }

    @Test
    fun `wlan1 tambem e tratada como hotspot`() {
        val chosen = HotspotAddress.choose(
            listOf("wlan0" to "10.0.0.5", "wlan1" to "192.168.187.1")
        )
        assertEquals("192.168.187.1", chosen)
    }

    @Test
    fun `sem interface de ap usa o primeiro candidato`() {
        assertEquals("10.0.0.5", HotspotAddress.choose(listOf("eth0" to "10.0.0.5")))
    }

    @Test
    fun `sem candidatos devolve null`() {
        assertNull(HotspotAddress.choose(emptyList()))
    }

    @Test
    fun `url monta http com ip e porta`() {
        assertEquals("http://192.168.43.1:8080", HotspotAddress.url("192.168.43.1", 8080))
    }
}
