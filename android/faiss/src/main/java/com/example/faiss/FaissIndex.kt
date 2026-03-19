package com.example.faiss

import kotlinx.serialization.json.Json
import kotlinx.serialization.Serializable


@Serializable
data class IvfStats(
    val ntotal: Long? = null,
    val ivf_nq: Long? = null,
    val ivf_nlist: Long? = null,
    val ivf_ndis: Long? = null,
    val ivf_search_time_ms: Double? = null,
    val ivf_quant_time_ms: Double? = null,
)

@Serializable
data class HnswStats(
    val ntotal: Long? = null,
    val nhops: Long? = null,
    val n1: Long? = null,
    val n2: Long? = null,
    val ndis: Long? = null,
)

@Serializable
data class FlatStats(
    val ntotal: Long? = null,
    val distances_per_query: Long? = null
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
        HNSW
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
     * Initialize an HNSW index (IndexHNSWFlat).
     * - dim: vector dimension
     * - config: HnswConfig containing M, efConstruction, efSearch, metric
     */
    fun initHnsw(dim: Int, config: HnswConfig) {
        initHnswNative(dim, config.M, config.efConstruction, config.efSearch, config.metric.getValue())
        currentIndexType = IndexType.HNSW
    }

    /**
     * Train IVFFlat index with trainingVectors (flattened vector array: numVectors * dim).
     * Only valid for IVFFlat indexes — native will check and return if not IVFFlat.
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
            trainIvfNative(trainingVectors, numVectors)
        } catch (e: Exception) {
            android.util.Log.e(TAG, "trainIvf(): native failed", e)
            throw IllegalStateException("Native FAISS train failed")
        }
    }

    fun query(queries: FloatArray, queryCount: Int,  k: Int): QueryResults {
        val results = queryNative(queries, queryCount, k)

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
        val ids = addNative(vectors, batchSize)
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

    fun clear() {
        clearNative()
        currentIndexType = IndexType.NONE
    }

    // --- native methods (implemented in C++) ---
    private external fun initFlatNative(dim: Int, metric: Int)
    private external fun initIvfNative(dim: Int, nlist: Int, nprobe: Int, metric: Int)
    private external fun initHnswNative(dim: Int, M: Int, efConstruction: Int, efSearch: Int, metric: Int)
    private external fun trainIvfNative(trainingVectors: FloatArray?, numVectors: Int)
    private external fun addNative(vectors: FloatArray, batchSize: Int): LongArray?
    private external fun queryNative(queries: FloatArray, queryCount: Int,  k: Int): QueryResults?
    private external fun getVectorNative(id: Long): FloatArray?
    private external fun getTotalNative(): Long
    private external fun setNprobeNative(nprobe: Int)
    private external fun clearNative()
    private external fun writeIndexNative(path: String)
    private external fun readIndexNative(path: String): Boolean
    private external fun getIVFDebugNative(): String
    private external fun resetFaissStatsNative()
    private external fun getIvfStatsNative(): String
    private external fun getHnswStatsNative(): String
    private external fun getFlatStatsNative(): String


    companion object {
        init {
            System.loadLibrary("native-lib")
        }
        const val TAG = "FAISS"
    }
}