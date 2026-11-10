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

    @SerialName("corpus_limit")
    val corpusLimit: Int,

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
    val engine: String = "onnx",

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

    // defaults keep pre-field configs parsing and preserve the old behavior
    @SerialName("investigator_system_prompt")
    val investigatorSystemPrompt: String = "Follow the instructions exactly.",

    // Every LLM instruction lives here, client-editable (agentic_pipeline_config.json),
    // never in Agentic.kt. %s placeholders, in order:
    //   answerer_prompt: question, bridge-line block, evidence
    //   investigator_prompt: question, searches already tried, evidence so far
    @SerialName("answerer_prompt")
    val answererPrompt: String = """Answer the question using only the evidence below.

Question: %s
%sEvidence: %s

Write one line starting with REASON: first say which fact the question asks
for, then the passage facts in order that lead to that fact, making sure the
last fact is the requested one and not the other side of a relation.
Then one line starting with ANSWER: with only the shortest exact answer,
copied from the evidence, at most 5 words, with no extra or changed words.
Do not answer with an intermediate entity.
For yes/no questions answer exactly yes or no.
Only when no evidence supports the answer, answer: not in evidence.

Example:
REASON: the question asks who founded Kestrel Corp; passage 2 says <founder> did.
ANSWER: <short answer>""",

    @SerialName("investigator_prompt")
    val investigatorPrompt: String = """Check every link needed to answer, in order. If a link is missing, query
for it using the known entity. Never repeat the question or a tried query.

Question: "%s"
Searches already tried:
%s
Evidence so far:
%s

If all links are supported: ANSWERABLE: <answer>
Otherwise: QUERY: <search>
One line only.""",

    @SerialName("answerer_retry_prompt")
    val answererRetryPrompt: String =
        "\nKeep the REASON/ANSWER lines; the ANSWER line is one short phrase, no list",

    @SerialName("investigator_retry_prompt")
    val investigatorRetryPrompt: String =
        "\nExactly one line: QUERY: <search> or ANSWERABLE: <answer>.",

    @SerialName("bridge_line")
    val bridgeLine: String = "Intermediate fact (established): %s",

    @SerialName("force_complex_mandate")
    val forceComplexMandate: String =
        "\nAlways reply with QUERY: one more lookup for this question.",

    @SerialName("repetition_penalty")
    val repetitionPenalty: Double,

    @SerialName("kv_window")
    val kvWindow: Int,

    @SerialName("prefill_chunk_size")
    val prefillChunkSize: Int,

    @SerialName("max_tokens")
    val maxTokens: Int,

    // Prompt budget, independent of maxTokens above. Null means "derive it from
    // the KV window"
    @SerialName("max_prompt_tokens")
    val maxPromptTokens: Int? = null,

    @SerialName("generate_until")
    val generateUntil: List<String>?,

    @SerialName("ignore_eos")
    val ignoreEos: Boolean,
)
