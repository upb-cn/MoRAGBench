package com.example.cli

import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class OverallMetricsTest {
    private val json = Json {
        prettyPrint = true
    }

    @Test
    fun explicitCpuThreadCountIsSerialized() {
        val metrics = OverallMetrics().apply {
            embeddingBackend = "cpu"
            embeddingIntraOpNumThreads = 1
        }

        val result = json.parseToJsonElement(json.encodeToString(metrics)).jsonObject

        assertEquals("cpu", result.getValue("embeddingBackend").jsonPrimitive.content)
        assertEquals("1", result.getValue("embeddingIntraOpNumThreads").jsonPrimitive.content)
    }

    @Test
    fun unspecifiedCpuThreadCountIsSerializedAsNull() {
        val metrics = OverallMetrics().apply {
            embeddingBackend = "cpu"
        }

        val result = json.parseToJsonElement(json.encodeToString(metrics)).jsonObject

        assertTrue(result.containsKey("embeddingIntraOpNumThreads"))
        assertEquals("null", result.getValue("embeddingIntraOpNumThreads").toString())
    }
}
