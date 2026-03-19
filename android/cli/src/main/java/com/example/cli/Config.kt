import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class ANNConfig(
    @SerialName("ann_dataset")
    val annDataset: ANNDataset,
    val faiss: FaissConfig,
)

@Serializable
data class ANNDataset(
    val name: String,

    @SerialName("sampling_method")
    val samplingMethod: String,

    val seed: Int,
    val limit: Int,
)


@Serializable
data class TaskConfig(
    @SerialName("downstream_task")
    val downstreamTask: DownstreamTask,

    @SerialName("rag_pipeline")
    val ragPipeline: RagPipeline,

    @SerialName("hf_token")
    val hfToken: String? = null
)

@Serializable
data class DownstreamTask(
    val name: String,

    @SerialName("sampling_method")
    val samplingMethod: String,

    val seed: Int,
    val limit: Int,
)

@Serializable
data class RagPipeline(
    val embedding: EmbeddingConfig,
    val faiss: FaissConfig,
    val llm: LlmConfig
)

@Serializable
data class EmbeddingConfig(
    val backend: String,

    @SerialName("model_name")
    val modelName: String,

    val dtype: String,
    val chunker: ChunkerConfig
)

@Serializable
data class ChunkerConfig(
    val method: String,
    val size: Int,

    @SerialName("overlap_enabled")
    val overlapEnabled: Boolean,

    @SerialName("overlap_size")
    val overlapSize: Int
)

@Serializable
data class FaissConfig(
    val method: String,
    val backend: String,
    val metric: String,

    @SerialName("top_k")
    val topK: Int,

    @SerialName("batch_size")
    val batchSize: Int,

    @SerialName("use_cache")
    val useCache: Boolean,

    val config: FaissInnerConfig
)

@Serializable
data class FaissInnerConfig(
    val nprobe: Int? = null,
    val nlist: Int? = null,

    @SerialName("num_training_vectors")
    val numTrainingVectors: Int? = null,

    val m: Int? = null,

    @SerialName("ef_construction")
    val efConstruction: Int? = null,

    @SerialName("ef_search")
    val efSearch: Int? = null
)

@Serializable
data class LlmConfig(
    @SerialName("aug_method")
    val augMethod: String,

    val backend: String,

    @SerialName("model_name")
    val modelName: String,

    @SerialName("use_sampling")
    val useSampling: Boolean,

    val temp: Double,
    val dtype: String,

    @SerialName("top_p")
    val topP: Double,

    @SerialName("top_k")
    val topK: Int,

    @SerialName("system_prompt")
    val systemPrompt: String,

    @SerialName("repetition_penalty")
    val repetitionPenalty: Double,

    @SerialName("kv_window")
    val kvWindow: Int,

    @SerialName("prefill_chunk_size")
    val prefillChunkSize: Int,

    @SerialName("max_tokens")
    val maxTokens: Int,

    @SerialName("generate_until")
    val generateUntil: List<String>?,

    @SerialName("ignore_eos")
    val ignoreEos: Boolean,
)
