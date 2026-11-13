package com.example.cli

import ANNConfig
import ANNDataset
import TaskConfig
import DownstreamTask
import FaissInnerConfig
import kotlinx.serialization.json.Json
import java.security.MessageDigest

import kotlinx.serialization.Serializable

val CanonicalJson = Json {
    encodeDefaults = true
    explicitNulls = false
    prettyPrint = false
}

@Serializable
data class FaissIndexSignature(
    val faiss: FaissSignature,
    val task: TaskSignature,
    val embedding: EmbeddingSignature? = null,
    val chunker: ChunkerSignature? = null
)

@Serializable
data class FaissSignature(
    val method: String,
    val metric: String,
    val backend: String,
    val batchSize: Int,
    val config: FaissInnerConfig
)

@Serializable
data class EmbeddingSignature(
    val modelName: String,
    val dtype: String,
    val backend: String
)

@Serializable
data class ChunkerSignature(
    val method: String,
    val size: Int,
    val overlapEnabled: Boolean,
    val overlapSize: Int
)

@Serializable
data class TaskSignature(
    val name: String,
    val samplingMethod: String,
    val corpusLimit: Int?,
    val seed: Int,
    val limit: Int
)



/**
 * Only the settings that change the index itself. nprobe (IVF) and ef_search (HNSW) are
 * search-time settings: they don't change what is stored, so they are left out of the cache
 * key and re-applied after loading. Otherwise every nprobe/ef_search value rebuilds the
 * same index (about 12 minutes per HNSW build on glove_100).
 */
fun indexOnly(c: FaissInnerConfig): FaissInnerConfig = c.copy(nprobe = null, efSearch = null)

object CacheManager {
    fun buildTaskSignature(
        config: TaskConfig,
        task: DownstreamTask
    ): FaissIndexSignature {

        val faiss = config.ragPipeline.faiss
        val embedding = config.ragPipeline.embedding
        val chunker = embedding.chunker

        return FaissIndexSignature(
            faiss = FaissSignature(
                method = faiss.method,
                metric = faiss.metric,
                config = indexOnly(faiss.config),
                backend = faiss.backend,
                batchSize = faiss.batchSize
            ),
            embedding = EmbeddingSignature(
                modelName = embedding.modelName,
                dtype = embedding.dtype,
                backend = embedding.backend
            ),
            chunker = ChunkerSignature(
                method = chunker.method,
                size = chunker.size,
                overlapEnabled = chunker.overlapEnabled,
                overlapSize = chunker.overlapSize
            ),
            task = TaskSignature(
                name = task.name,
                samplingMethod = task.samplingMethod,
                corpusLimit = task.corpusLimit,
                seed = task.seed,
                limit = task.limit
            )
        )
    }

    fun buildANNSignature(
        config: ANNConfig,
        dataset: ANNDataset
    ): FaissIndexSignature {

        val faiss = config.faiss

        return FaissIndexSignature(
            faiss = FaissSignature(
                method = faiss.method,
                metric = faiss.metric,
                config = indexOnly(faiss.config),
                backend = faiss.backend,
                batchSize = faiss.batchSize
            ),
            task = TaskSignature(
                name = dataset.name,
                samplingMethod = dataset.samplingMethod,
                corpusLimit = null,
                seed = dataset.seed,
                limit = dataset.limit
            )
        )
    }

    fun hashSignature(signature: FaissIndexSignature): String {
        val json = CanonicalJson.encodeToString(
            FaissIndexSignature.serializer(),
            signature
        )

        val digest = MessageDigest
            .getInstance("SHA-256")
            .digest(json.toByteArray(Charsets.UTF_8))

        return digest.joinToString("") { "%02x".format(it) }
    }
}