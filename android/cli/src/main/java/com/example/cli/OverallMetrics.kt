package com.example.cli
import com.example.cli.Progress.IndexSource
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File

@Serializable
class OverallMetrics {
    var benchmarkTimeMs: Long = 0L

    // These 2 are for Task only
    var embeddingInitTimeMs: Long = 0L
    var llmInitTimeMs: Long = 0L

    // This is for ANN only
    var testTimeMs: Long = 0L

    var indexSource: IndexSource? = null
}

fun writeMetricsToFile(metrics: OverallMetrics, file: File) {
    file.parentFile?.mkdirs()

    val json = Json {
        prettyPrint = true
    }

    val jsonString = json.encodeToString(metrics)
    file.writeText(jsonString)
}
