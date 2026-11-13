package com.example.cli
import com.example.cli.Progress.IndexSource
import kotlinx.serialization.EncodeDefault
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File

@OptIn(ExperimentalSerializationApi::class)
@Serializable
class OverallMetrics {
    var benchmarkTimeMs: Long = 0L

    // These fields are for Task benchmarks only.
    var embeddingInitTimeMs: Long = 0L
    var llmInitTimeMs: Long = 0L

    @EncodeDefault
    var embeddingBackend: String? = null

    // Explicit ONNX Runtime CPU intra-op thread setting; null means no CPU override.
    @EncodeDefault
    var embeddingIntraOpNumThreads: Int? = null

    // This is for ANN only
    var testTimeMs: Long = 0L

    var indexSource: IndexSource? = null

    // true for overall_resume_N.json (only the resumed questions)
    @EncodeDefault
    var resumed: Boolean = false

    // false if the run was stopped or failed before the last question
    @EncodeDefault
    var completed: Boolean = false
}

fun writeMetricsToFile(metrics: OverallMetrics, file: File) {
    file.parentFile?.mkdirs()

    val json = Json {
        prettyPrint = true
    }

    val jsonString = json.encodeToString(metrics)
    file.writeText(jsonString)
}
