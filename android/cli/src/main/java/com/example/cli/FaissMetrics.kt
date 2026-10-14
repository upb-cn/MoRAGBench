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
    var numThreadsRequested: Int? = null   // from config (null = default)
    var numThreadsUsed: Int = 0            // what OpenMP reported
    var cpuPin: String? = null             // requested core type (null = none)
    var pinnedCores: List<Int> = emptyList()   // cores it resolved to
    var coreMaxFreqsKHz: List<Int> = emptyList() // phone's core layout
    var pinError: String? = null           // why pinning was skipped, if it was
    var cpusObservedDuringQueries: Map<Int, Int> = emptyMap() // core -> number of batches
    // Per batch: which core it ran on and that core's clock speed (MHz) right after it.
    // Shows whether the phone slowed down (lower MHz) during a run. -1 MHz = not readable.
    var queryBatchCores: List<Int> = emptyList()
    var queryBatchMHz: List<Int> = emptyList()
    var buildBatchMHz: List<Int> = emptyList()
    // Android priority group of the app ("top-app" = on screen and in use, gets the fast
    // cores; "foreground"/"background" = lower priority). Read from /proc/self/cpuset.
    var cpusetAtQueryStart: String? = null
    var cpusetAtQueryEnd: String? = null
    // Wall-clock time until the index is ready to search: create + pick training sample +
    // train + read vectors + add. The fair "setup cost" to compare methods (IVF's training is
    // NOT in buildTimeMs). 0 when the index was loaded from cache (see loadTimeNs instead).
    var totalSetupTimeMs: Long = 0L
    // App memory, in MB. FAISS keeps the index in native (C++) memory, so the change in
    // native heap is the index's memory cost. The dataset files are memory-mapped and are
    // NOT counted here. (hardware_metrics.json "memory" is the whole phone - don't use it
    // to compare methods.)
    var nativeHeapBeforeMB: Double = 0.0        // before the index is created
    var nativeHeapPeakDuringBuildMB: Double = 0.0 // highest seen while training/building
    var nativeHeapIndexReadyMB: Double = 0.0    // index built or loaded, ready to search
    var nativeHeapAfterQueriesMB: Double = 0.0
    var indexMemoryMB: Double = 0.0             // = IndexReady - Before
    var buildPeakExtraMB: Double = 0.0          // = PeakDuringBuild - Before (temporary cost)
    var indexFileMB: Double = 0.0               // size of the saved index file
    var appPssAfterQueriesMB: Double = 0.0      // whole app (Java + native + code), for reference
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

    var resumed: Boolean = false   // true for faiss_metrics_resume_N.json (only the resumed questions)
    var completed: Boolean = false // false if the run was stopped or failed before the last question

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
    var minNs: Long = 0      // 0 when count == 0
    var maxNs: Long = 0
    var avgNs: Long = 0

    fun add(valueNs: Long) {
        count++
        totalNs += valueNs
        minNs = if (count == 1L) valueNs else minOf(minNs, valueNs)
        maxNs = maxOf(maxNs, valueNs)
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
