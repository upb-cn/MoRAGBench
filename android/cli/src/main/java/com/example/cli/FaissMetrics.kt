package com.example.cli
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import com.example.faiss.IvfStats
import com.example.faiss.HnswStats
import com.example.faiss.FlatStats

@Serializable
class FaissMetrics {
    var initIndexTimeNs: Long = 0L
    var trainingTimeNs: Long = 0L
    var buildTimeMs: Long = 0L

    var loadTimeNs: Long = 0L
    var saveTimeNs: Long = 0L

    var chunkingTimeNs = RunningStats() // only for Task
    var embeddingTimeNs = RunningStats() // only for Task
    var addTimeNs = RunningStats()
    var readTimeNs = RunningStats() // Only for ANN

    var ivfStats: IvfStats? = null
    var hnswStats: HnswStats? = null
    var flatStats: FlatStats? = null

    fun calcAvg() {
        chunkingTimeNs.calcAvg()
        embeddingTimeNs.calcAvg()
        addTimeNs.calcAvg()
        readTimeNs.calcAvg()
    }
}

@Serializable
class RunningStats {
    var count: Long = 0
    var totalNs: Long = 0
    var minNs: Long? = null
    var maxNs: Long = 0
    var avgNs: Long = 0

    fun add(valueMs: Long) {
        count++
        totalNs += valueMs
        minNs = minNs?.let { minOf(it, valueMs) } ?: valueMs
        maxNs = maxOf(maxNs, valueMs)
    }

    fun calcAvg() {
        if (count == 0L) return
        avgNs = (totalNs.toDouble() / count).toLong()
    }
}

fun writeMetricsToFile(metrics: FaissMetrics, file: File) {
    file.parentFile?.mkdirs()

    val json = Json {
        prettyPrint = true
        encodeDefaults = true
    }

    val jsonString = json.encodeToString(metrics)
    file.writeText(jsonString)
}
