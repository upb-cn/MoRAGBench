package com.example.faiss

import kotlinx.serialization.json.Json
import kotlinx.serialization.Serializable


@Serializable
data class IvfStats(
    val ntotal: Long? = null,
    val ivf_nq: Long? = null,
    val ivf_lists_scanned: Long? = null,
    val ivf_ndis: Long? = null,
    val ivf_search_time_ms: Double? = null,
    val ivf_quant_time_ms: Double? = null,
    val note: String? = null,   // set when stats are not available
    val error: String? = null,  // set when the native code reports an error
)

@Serializable
data class HnswStats(
    val ntotal: Long? = null,
    val nhops: Long? = null,
    val n1: Long? = null,
    val ndis: Long? = null,
    val note: String? = null,
    val error: String? = null,
)

@Serializable
data class FlatStats(
    val ntotal: Long? = null,
    val error: String? = null,
)

@Serializable
data class IvfPqStats(
    val ntotal: Long? = null,
    val nlist: Long? = null,
    val nprobe: Long? = null,
    val m: Long? = null,
    val nbits: Long? = null,
    val nonempty_lists: Long? = null,
    val ivf_nq: Long? = null,
    val ivf_ndis: Long? = null,
    val ivf_search_time_ms: Double? = null,
    val ivf_quant_time_ms: Double? = null,
)

@Serializable
data class BinaryStats(
    val ntotal: Long? = null,
    val type: String? = null,
    val nlist: Long? = null,
    val nprobe: Long? = null,
    val nonempty_lists: Long? = null,
)

private val json = Json {
    ignoreUnknownKeys = true
}

class FaissIndex {
    /**
     * Distance metric mapping to the native faiss metrics.
     */
    enum class DistanceMetric(private val value: Int) {
        L2(0),
        IP(1);

        fun getValue(): Int {
            return value
        }

        companion object {
            fun fromInt(value: Int): DistanceMetric {
                for (m in DistanceMetric.entries) {
                    if (m.value == value) return m
                }

                throw Exception("Invalid distance metric value: $value")
            }

            fun fromString(value: String): DistanceMetric {
                for (m in DistanceMetric.entries) {
                    if (m.name.equals(value, ignoreCase = true)) return m
                }

                throw Exception("Invalid distance metric value: $value")
            }
        }
    }

    /**
     * Track which index type is currently created (so we only report for it).
     */
    enum class IndexType {
        NONE,
        FLAT,
        IVF,
        HNSW,
        IVFPQ,
        BINARY_FLAT,
        BINARY_IVF
    }

    // --- track current index type locally in Kotlin ---
    @Volatile
    var currentIndexType: IndexType = IndexType.NONE
        private set

    /**
     * Flat config
     */
    class FlatConfig @JvmOverloads constructor(
        val metric: DistanceMetric = DistanceMetric.L2
    )


    /**
     * IVF config
     */
    class IvfConfig @JvmOverloads constructor(
        val nlist: Int = 100,
        val nprobe: Int = 10,
        val metric: DistanceMetric = DistanceMetric.L2
    )

    /**
     * IVFPQ config.
     *
     * @param nlist number of coarse clusters (same role as in IVF-Flat)
     * @param nprobe number of clusters searched at query time
     * @param m number of Product Quantization subquantizers. Must evenly divide the embedding dim.
     * @param nbits bits per subquantizer code (8 -> 256 centroids per subquantizer, the common default)
     * @param metric distance metric (L2 or IP)
     */
    class IvfPqConfig @JvmOverloads constructor(
        val nlist: Int = 100,
        val nprobe: Int = 10,
        val m: Int = 8,
        val nbits: Int = 8,
        val metric: DistanceMetric = DistanceMetric.L2
    )

    /**
     * HNSW config.
     *
     * @param M number of bi-directional links created for each new element (typical defaults: 16 - 48)
     * @param efConstruction internal construction-time parameter (higher -> better quality, slower)
     * @param efSearch runtime search parameter (higher -> more accurate, slower)
     * @param metric distance metric (L2 or IP)
     */
    class HnswConfig @JvmOverloads constructor(
        val M: Int = 32,
        val efConstruction: Int = 200,
        val efSearch: Int = 10,
        val metric: DistanceMetric = DistanceMetric.L2
    )

    /**
     * Binary IVF config.
     *
     * Binary indices always use Hamming distance (no metric choice — there's no
     * L2/IP distinction once vectors are packed bits), so unlike the other configs
     * this one has no `metric` field.
     *
     * @param nlist number of coarse clusters (same role as in IVF-Flat)
     * @param nprobe number of clusters searched at query time
     */
    class BinaryIvfConfig @JvmOverloads constructor(
        val nlist: Int = 100,
        val nprobe: Int = 10
    )

    /**
     * Query results container. JNI will construct this via constructor (long[], float[]).
     * Signature expected by native layer: ([J[F)V
     */
    class QueryResults (val labels: LongArray?, val distances: FloatArray?) {
        override fun toString(): String {
            if (labels == null || distances == null || labels.size != distances.size) {
                return "QueryResults(labels=${labels?.contentToString()}, distances=${distances?.contentToString()})"
            }

            val sb = StringBuilder("QueryResults[\n")
            for (i in labels.indices) {
                sb.append("  Label: ${labels[i]}, Distance: ${distances[i]}\n")
            }
            sb.append("]")

            return sb.toString()
        }
    }

    // --- Java-facing convenience methods (call into the native layer) ---
    /** Initialize a flat index (IndexFlatL2 or IndexFlatIP depending on metric; default L2)  */
    fun initFlat(dim: Int, flatConfig: FlatConfig) {
        initFlatNative(dim, flatConfig.metric.getValue())
        currentIndexType = IndexType.FLAT
    }

    /**
     * Initialize an IVFFlat index.
     * - dim: vector dimension
     * - config: IvfConfig containing nlist, nprobe, metric
     */
    fun initIvf(dim: Int, config: IvfConfig) {
        initIvfNative(dim, config.nlist, config.nprobe, config.metric.getValue())
        currentIndexType = IndexType.IVF
    }

    /**
     * Initialize an IVFPQ index (IndexIVFPQ). Requires `dim % config.m == 0`.
     * - dim: vector dimension
     * - config: IvfPqConfig containing nlist, nprobe, m, nbits, metric
     */
    fun initIvfPq(dim: Int, config: IvfPqConfig) {
        require(config.m > 0 && dim % config.m == 0) {
            "IVFPQ requires embedding dim ($dim) to be divisible by m (${config.m})"
        }
        initIvfPqNative(dim, config.nlist, config.nprobe, config.m, config.nbits, config.metric.getValue())
        currentIndexType = IndexType.IVFPQ
    }

    /**
     * Initialize an HNSW index (IndexHNSWFlat).
     * - dim: vector dimension
     * - config: HnswConfig containing M, efConstruction, efSearch, metric
     */
    fun initHnsw(dim: Int, config: HnswConfig) {
        initHnswNative(dim, config.M, config.efConstruction, config.efSearch, config.metric.getValue())
        currentIndexType = IndexType.HNSW
    }

    /**
     * Initialize a binary flat index (IndexBinaryFlat): exhaustive Hamming-distance search.
     * - dim: vector dimension in FLOATS (same value used for the float-index methods above).
     *   It is sign-thresholded down to one bit per dimension, so `dim` must be a multiple of 8.
     */
    fun initBinaryFlat(dim: Int) {
        require(dim > 0 && dim % 8 == 0) {
            "Binary index requires embedding dim ($dim) to be a positive multiple of 8"
        }
        initBinaryFlatNative(dim)
        currentIndexType = IndexType.BINARY_FLAT
    }

    /**
     * Initialize a binary IVF index (IndexBinaryIVF): IVF-style clustering over packed-bit
     * codes with Hamming distance.
     * - dim: vector dimension in FLOATS; must be a multiple of 8 (see initBinaryFlat).
     * - config: BinaryIvfConfig containing nlist, nprobe
     */
    fun initBinaryIvf(dim: Int, config: BinaryIvfConfig) {
        require(dim > 0 && dim % 8 == 0) {
            "Binary index requires embedding dim ($dim) to be a positive multiple of 8"
        }
        initBinaryIvfNative(dim, config.nlist, config.nprobe)
        currentIndexType = IndexType.BINARY_IVF
    }

    /**
     * Sign-threshold binarization: each float becomes 1 bit (1 if > 0, else 0), packed
     * 8-per-byte, MSB first. This is the standard cheap binarization used to approximate
     * cosine/L2 similarity with Hamming distance over embeddings.
     *
     * @param vectors flattened float array, length count * dim
     * @param count number of vectors
     * @param dim dimension per vector (must be a multiple of 8)
     */
    private fun binarize(vectors: FloatArray, count: Int, dim: Int): ByteArray {
        val codeSize = dim / 8
        val out = ByteArray(count * codeSize)
        for (v in 0 until count) {
            val vecOffset = v * dim
            val outOffset = v * codeSize
            for (byteIdx in 0 until codeSize) {
                var b = 0
                for (bit in 0 until 8) {
                    val f = vectors[vecOffset + byteIdx * 8 + bit]
                    if (f > 0f) {
                        b = b or (1 shl (7 - bit))
                    }
                }
                out[outOffset + byteIdx] = b.toByte()
            }
        }
        return out
    }

    private fun isBinary(): Boolean =
        currentIndexType == IndexType.BINARY_FLAT || currentIndexType == IndexType.BINARY_IVF

    /**
     * Train an IVF-family index (IVFFlat, IVFPQ, or Binary IVF) with trainingVectors
     * (flattened vector array: numVectors * dim). Only valid for IVF-family indexes —
     * native will check and return if not. For a binary IVF index, trainingVectors are
     * binarized here before being handed to the native layer.
     *
     * @param trainingVectors Float array length numVectors * dim
     * @param numVectors number of vectors in trainingVectors (native needs this)
     */
    fun trainIvf(trainingVectors: FloatArray?, numVectors: Int) {
        if (trainingVectors == null) {
            android.util.Log.e(TAG, "trainIvf(): null training vectors")
            throw IllegalArgumentException("Training vectors cannot be null")
        }

        try {
            if (currentIndexType == IndexType.BINARY_IVF) {
                val dim = trainingVectors.size / numVectors
                val packed = binarize(trainingVectors, numVectors, dim)
                trainBinaryIvfNative(packed, numVectors)
            } else {
                trainIvfNative(trainingVectors, numVectors)
            }
        } catch (e: Exception) {
            android.util.Log.e(TAG, "trainIvf(): native failed", e)
            throw IllegalStateException("Native FAISS train failed")
        }
    }

    fun query(queries: FloatArray, queryCount: Int,  k: Int): QueryResults {
        val results = if (isBinary()) {
            val dim = queries.size / queryCount
            val packed = binarize(queries, queryCount, dim)
            queryBinaryNative(packed, queryCount, k)
        } else {
            queryNative(queries, queryCount, k)
        }

        if (results == null || results.labels == null || results.distances == null) {
            android.util.Log.e(TAG, "query(): native search failed or returned null")
            throw IllegalStateException("Native FAISS query failed")
        }

        return results
    }

    /** Change IVF nprobe at runtime. Safe to call even if index is Flat (will affect IVF only).  */
    fun setNprobe(nprobe: Int) {
        setNprobeNative(nprobe)
    }

    /** Change HNSW efSearch at runtime (used after loading a cached index). */
    fun setEfSearch(efSearch: Int) {
        setEfSearchNative(efSearch)
    }

    fun getVector(id: Long): FloatArray {
        val vec = getVectorNative(id)
        if (vec == null) {
            android.util.Log.e(TAG, "getVector(): native failed to reconstruct id=$id")
            throw IllegalStateException("Native FAISS getVector failed for id=$id")
        }
        return vec
    }

    // add and return faiss ids
    fun add(vectors: FloatArray, batchSize: Int): LongArray {
        val ids = if (isBinary()) {
            val dim = vectors.size / batchSize
            val packed = binarize(vectors, batchSize, dim)
            addBinaryNative(packed, batchSize)
        } else {
            addNative(vectors, batchSize)
        }
        if (ids == null) {
            android.util.Log.e(TAG, "add(): native layer returned null (fatal add failure)")
            throw IllegalStateException("Native FAISS add failed")
        }

        return ids
    }

    /** Returns the number of vectors currently stored in the native FAISS index (ntotal). */
    fun getTotal(): Long {
        return try {
            getTotalNative()
        } catch (e: Exception) {
            android.util.Log.e(TAG, "getTotal(): native call failed", e)
            throw IllegalStateException("Could not read ntotal from native FAISS")
        }
    }

    /**
     * Persist current FAISS index to disk at 'path' (absolute path).
     * Example: context.filesDir.absolutePath + "/faiss.index"
     */
    fun saveTo(path: String) {
        try {
            // call native; native will throw if something goes wrong
            writeIndexNative(path)
        } catch (e: Exception) {
            // log for diagnostics
            android.util.Log.e("FAISS", "saveTo failed for path=$path", e)
            // Clear native resources to avoid inconsistent state
            try {
                clear()
            } catch (cle: Exception) {
                android.util.Log.e("FAISS", "Failed to clear index after save failure", cle)
            }
            // rethrow so caller can abort UI/workflow
            throw e
        }
    }

    /**
     * Load a FAISS serialized index file from 'path'. Returns true on success.
     * If the loaded file contains an IndexIDMap, native code will set next-id automatically.
     */
    fun loadFrom(path: String): Boolean {
        return try {
            val ok = readIndexNative(path)
            if (!ok) {
                // native returned false (e.g., read failed in non-exception path)
                android.util.Log.e("FAISS", "loadFrom: native readIndexNative returned false for $path")
                try { clear() } catch (e: Exception) { android.util.Log.e("FAISS", "clear failed", e) }
                false
            } else {
                // readIndexNative() has no matching Kotlin call site (unlike initFlat/initIvf/etc.)
                // to set currentIndexType, so add()/query() would otherwise wrongly take the
                // float-vector path for a freshly-loaded binary index. Sync it from native state.
                // (BINARY_FLAT is used as a generic "binary" marker here — add()/query() only
                // need to know binary-vs-float, not which binary variant; Flat-vs-IVF dispatch
                // for a loaded index is handled entirely in the native layer via dynamic_cast.)
                currentIndexType = if (isBinaryActiveNative()) IndexType.BINARY_FLAT else currentIndexType
                true
            }
        } catch (e: Exception) {
            // native threw an exception (e.g., not IndexIDMap)
            android.util.Log.e("FAISS", "loadFrom failed for path=$path", e)
            try { clear() } catch (ce: Exception) { android.util.Log.e("FAISS", "clear failed", ce) }
            false
        }
    }

    fun setMethod(method: String) {
        val normMethod = method.lowercase()
        when (normMethod) {
            "flat" -> {
                currentIndexType = IndexType.FLAT
            }
            "ivf" -> {
                currentIndexType = IndexType.IVF
            }
            "hnsw" -> {
                currentIndexType = IndexType.HNSW
            }
            "ivfpq" -> {
                currentIndexType = IndexType.IVFPQ
            }
            "binary", "binaryflat" -> {
                currentIndexType = IndexType.BINARY_FLAT
            }
            "binaryivf" -> {
                currentIndexType = IndexType.BINARY_IVF
            }
        }
    }

    fun getIVFDebug(): String {
        return try {
            val debugString = getIVFDebugNative()
            return debugString
        } catch (e: Exception) {
            "getIVFDebug error: ${e.message}"
        }
    }

    fun resetStats() {
        resetFaissStatsNative()
    }

    fun getIvfStats(): IvfStats {
        try {
            return json.decodeFromString(getIvfStatsNative())
        } catch (e: Exception) {
            throw Exception("getIvfStats error: ${e.message}")
        }
    }

    fun getHnswStats(): HnswStats {
        try {
            return json.decodeFromString(getHnswStatsNative())
        } catch (e: Exception) {
            throw Exception("getHnswStats error: ${e.message}")
        }
    }

    fun getFlatStats(): FlatStats {
        try {
            return json.decodeFromString(getFlatStatsNative())
        } catch (e: Exception) {
            throw Exception("getFlatStats error: ${e.message}")
        }
    }

    fun getIvfPqStats(): IvfPqStats {
        try {
            return json.decodeFromString(getIvfPqStatsNative())
        } catch (e: Exception) {
            throw Exception("getIvfPqStats error: ${e.message}")
        }
    }

    fun getBinaryStats(): BinaryStats {
        try {
            return json.decodeFromString(getBinaryStatsNative())
        } catch (e: Exception) {
            throw Exception("getBinaryStats error: ${e.message}")
        }
    }

    /**
     * Limit how many CPU threads FAISS (OpenMP) and OpenBLAS may use for train/add/search.
     * n <= 0 keeps the library default (all cores). Applies to this process.
     */
    fun setNumThreads(n: Int) {
        setNumThreadsNative(n)
    }

    /** Threads FAISS will actually use (as reported by OpenMP). */
    fun getNumThreads(): Int = getNumThreadsNative()

    /**
     * Pin FAISS work to these CPU cores (empty = no pinning). Returns 0 on success,
     * otherwise the Linux errno. Re-applied before every train/add/search.
     */
    fun setCpuAffinity(cores: IntArray): Int = setCpuAffinityNative(cores)

    /** Core the FAISS calling thread is on right now (after applying the pinning). */
    fun getCurrentCpu(): Int = getCurrentCpuNative()

    object CpuTopology {
        /** Max frequency (kHz) per core, read from sysfs; -1 if unreadable. */
        fun coreMaxFreqsKHz(): IntArray {
            val n = Runtime.getRuntime().availableProcessors()
            return IntArray(n) { i ->
                try {
                    java.io.File("/sys/devices/system/cpu/cpu$i/cpufreq/cpuinfo_max_freq").readText().trim().toInt()
                } catch (e: Exception) { -1 }
            }
        }

        /** Current clock speed of one core in MHz, or -1 if the phone does not allow reading it. */
        fun coreCurMHz(cpu: Int): Int = try {
            java.io.File("/sys/devices/system/cpu/cpu$cpu/cpufreq/scaling_cur_freq").readText().trim().toInt() / 1000
        } catch (e: Exception) { -1 }

        /**
         * Cores belonging to a core type: "fastest", "slowest", or "mid" (the largest
         * cluster that is neither fastest nor slowest). Empty if it cannot be resolved.
         */
        fun coresForPolicy(policy: String, freqs: IntArray = coreMaxFreqsKHz()): IntArray {
            if (freqs.any { it <= 0 }) return IntArray(0)
            val groups = freqs.indices.groupBy { freqs[it] }.toSortedMap(compareByDescending { it })
            val clusters = groups.values.toList()
            val chosen: List<Int> = when (policy.lowercase()) {
                "fastest" -> clusters.first()
                "slowest" -> clusters.last()
                "mid" -> if (clusters.size <= 2) clusters.last()
                         else clusters.subList(1, clusters.size - 1).maxByOrNull { it.size }!!
                else -> emptyList()
            }
            return chosen.toIntArray()
        }
    }

    fun clear() {
        clearNative()
        currentIndexType = IndexType.NONE
    }

    // --- native methods (implemented in C++) ---
    private external fun initFlatNative(dim: Int, metric: Int)
    private external fun initIvfNative(dim: Int, nlist: Int, nprobe: Int, metric: Int)
    private external fun initIvfPqNative(dim: Int, nlist: Int, nprobe: Int, m: Int, nbits: Int, metric: Int)
    private external fun initHnswNative(dim: Int, M: Int, efConstruction: Int, efSearch: Int, metric: Int)
    private external fun initBinaryFlatNative(dimBits: Int)
    private external fun initBinaryIvfNative(dimBits: Int, nlist: Int, nprobe: Int)
    private external fun trainIvfNative(trainingVectors: FloatArray?, numVectors: Int)
    private external fun trainBinaryIvfNative(packedTrainingVectors: ByteArray, numVectors: Int)
    private external fun addNative(vectors: FloatArray, batchSize: Int): LongArray?
    private external fun addBinaryNative(packedVectors: ByteArray, batchSize: Int): LongArray?
    private external fun queryNative(queries: FloatArray, queryCount: Int,  k: Int): QueryResults?
    private external fun queryBinaryNative(packedQueries: ByteArray, queryCount: Int, k: Int): QueryResults?
    private external fun getVectorNative(id: Long): FloatArray?
    private external fun getTotalNative(): Long
    private external fun setNprobeNative(nprobe: Int)
    private external fun setEfSearchNative(efSearch: Int)
    private external fun clearNative()
    private external fun setNumThreadsNative(n: Int)
    private external fun getNumThreadsNative(): Int
    private external fun setCpuAffinityNative(cores: IntArray): Int
    private external fun getCurrentCpuNative(): Int
    private external fun writeIndexNative(path: String)
    private external fun readIndexNative(path: String): Boolean
    private external fun isBinaryActiveNative(): Boolean
    private external fun getIVFDebugNative(): String
    private external fun resetFaissStatsNative()
    private external fun getIvfStatsNative(): String
    private external fun getHnswStatsNative(): String
    private external fun getFlatStatsNative(): String
    private external fun getIvfPqStatsNative(): String
    private external fun getBinaryStatsNative(): String


    companion object {
        init {
            System.loadLibrary("native-lib")
        }
        const val TAG = "FAISS"
    }
}