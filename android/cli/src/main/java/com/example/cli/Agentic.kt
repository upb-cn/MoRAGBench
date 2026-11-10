package com.example.cli

import LlmConfig
import android.content.Context
import android.os.SystemClock
import com.example.cli.Progress.State
import com.example.faiss.EmbeddingDb
import com.example.faiss.FaissIndex
import com.example.onnxtok.EmbeddingGenerator
import com.example.onnxtok.EmbeddingModel
import com.example.shared.SupportedEmbeddingModels
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File

/**
 * On-device agentic RAG: retrieve once with the original question, then loop up
 * to 3 investigator rounds (sufficient evidence? -> ANSWERABLE, else next lookup
 * -> QUERY), each conditioned on the evidence read so far; answer once over the
 * unioned evidence with the investigator's key fact.
 */

// ── events ─────────────────────────────────────────────────────────────────────

@Serializable
data class AgentEvent(
    val seq: Long,
    val tMs: Long,
    val agent: String,
    val step: String,
    val detail: String = "",
    val output: String = "",
    val inputTokens: Int = 0,
    val genTokens: Int = 0,
    val ttftMs: Long = 0,
    val durationMs: Long = 0,
    val decodeTps: Double = 0.0
)

object AgentEventLog {
    /** Always encodes every field so clients see a stable event shape. */
    val json = Json { encodeDefaults = true }

    private val events = java.util.concurrent.CopyOnWriteArrayList<AgentEvent>()

    @Volatile
    private var startMs = SystemClock.elapsedRealtime()

    @Volatile
    var state: State = State.NOT_STARTED
        private set

    fun reset() {
        events.clear()
        startMs = SystemClock.elapsedRealtime()
        state = State.RUNNING
    }

    fun done() {
        state = if (state == State.RUNNING) State.COMPLETED else state
    }

    fun fail() {
        state = State.FAILED
    }

    fun emit(
        agent: String,
        step: String,
        detail: String = "",
        output: String = "",
        r: InferenceEngine.GenerationResult? = null
    ) {
        val e = AgentEvent(
            seq = events.size.toLong(),
            tMs = SystemClock.elapsedRealtime() - startMs,
            agent = agent,
            step = step,
            detail = detail,
            output = output,
            inputTokens = r?.inputTokens ?: 0,
            genTokens = r?.generatedTokens ?: 0,
            ttftMs = r?.ttftMs ?: 0,
            durationMs = r?.overallDurationMs ?: 0,
            decodeTps = r?.decodingSpeedTokensPerSec ?: 0.0
        )
        events.add(e)
        LiveLogger.info("[agent] ${e.agent}/${e.step} $detail")
    }

    fun since(n: Int): List<AgentEvent> = events.drop(n.coerceIn(0, events.size))

    fun count(): Int = events.size
}

// ── agent prompts ──────────────────────────────────────────────────────────────
// Prompt text lives in the client config (LlmConfig: answerer_prompt,
// investigator_prompt, retry prompts, bridge_line, force_complex_mandate) so it
// can be edited without rebuilding the app. This file only names the agents
// and carries their generation budgets.

data class Agent(val name: String, val maxTokens: Int = 96)

// One agent, one contract, up to 3 rounds: "is the evidence enough? if not,
// what next?" Replaced the closed-book SIMPLE/COMPLEX router (always SIMPLE on
// multi-hop sets), the blind decomposer (could not name the bridge entity) and
// the bridge extractor (failed its own example with NONE) — the loop handles
// the hop-2-needs-hop-1-answer dependency by construction: each round's query
// is written AFTER reading the previous round's evidence.
private val INVESTIGATOR = Agent("Evidence Investigator", 96)

private val REASONER = Agent("Answerer")

// internal for unit tests. Finds the contract tag anywhere in the reply: a
// truncated/mangled first word ("ery where is...") must not fail the parse.
internal fun parseReply(text: String): Pair<String, String>? {
    val lines = text.lineSequence().map { it.trim() }.filter { it.isNotEmpty() }.toList()
    for (tag in listOf("ANSWERABLE", "QUERY")) {
        for (line in lines) {
            val m = Regex("(?i)\\b$tag\\b\\s*:?\\s*(.*)").find(line) ?: continue
            return tag to m.groupValues[1].trim().trimStart(':').trim()
        }
    }
    return null
}

// internal for unit tests: a list-shaped reply means the model dumped chunk
// titles instead of one phrase.
internal fun isListy(text: String): Boolean {
    val lines = text.lineSequence().map { it.trim() }.filter { it.isNotEmpty() }.toList()
    return lines.size > 1 || lines.firstOrNull()?.matches(Regex("\\d+[.)].*")) == true
}

// ── retrieval (index + embeddings, loaded once and reused) ─────────────────────

/**
 * ponytail: one cached index, guarded by config signature. The cache path mirrors
 * TaskBenchmark.initFaissIndex; if that derivation changes there, change it here.
 * Ceiling: the whole FAISS index is held in memory, so a corpus that does not fit
 * alongside the LLM will OOM -- same ceiling the benchmark path already has.
 */
private class Retrieval(context: Context) {
    val index = FaissIndex()
    private val db: EmbeddingDb
    private val generator: EmbeddingGenerator
    private val k: Int
    private val queryPrefix: String

    init {
        val config = Parser(context).readTaskConfig()
        val task = config.downstreamTask
        k = config.ragPipeline.faiss.topK

        val hash = CacheManager.hashSignature(CacheManager.buildTaskSignature(config, task))
        val cacheDir = context.getExternalFilesDir(null)!!.resolve(Constants.TASK_CACHE_DIR)
        val indexPath = File(cacheDir, "faiss_$hash.index")
        if (!indexPath.exists()) {
            throw IllegalStateException(
                "No cached index at ${indexPath.absolutePath}. Run a benchmark once with " +
                    "\"use_cache\": true in faiss config to build one."
            )
        }
        index.loadFrom(indexPath.absolutePath)
        index.setMethod(config.ragPipeline.faiss.method)
        db = BenchmarkManager.getDB(hash, context)

        val ec = config.ragPipeline.embedding
        val model: EmbeddingModel = SupportedEmbeddingModels.findByName(ec.modelName)
        queryPrefix = model.queryPrefix
        val base = context.getExternalFilesDir(null)!!.resolve(Constants.TASK_BASE_DIR)
        val folder = File(base, "embedding/${ec.modelName}_${ec.dtype}")
        model.modelPath = File(folder, "model.onnx").canonicalPath
        model.tokenizerPath = File(folder, "tokenizer.json").canonicalPath
        generator = EmbeddingGenerator(context)
        kotlinx.coroutines.runBlocking { generator.initialize(model, backend = ec.backend) }
    }

    /** Returns retrieved chunk texts in FAISS rank order. */
    suspend fun retrieve(query: String): List<String> {
        val labels = index.query(generator.generate(queryPrefix + query), 1, k).labels ?: return emptyList()
        return labels.map { db.getByFaissId(it)?.chunkText }.filterNotNull()
    }

    fun close() {
        runCatching { index.clear() }
        runCatching { db.close() }
        runCatching { generator.close() }
    }
}

// ── orchestrator ───────────────────────────────────────────────────────────────

object Agentic {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    @Volatile
    private var retrieval: Retrieval? = null

    @Volatile
    private var retrievalKey: String? = null

    private fun configSignature(context: Context): String {
        val c = Parser(context).readTaskConfig()
        val l = c.ragPipeline.llm
        return "${l.modelName}|${l.dtype}|${l.backend}|${c.ragPipeline.embedding.modelName}|" +
            "${c.ragPipeline.embedding.dtype}|${c.ragPipeline.faiss.method}|${c.downstreamTask.name}|" +
            "${c.downstreamTask.corpusLimit}|${c.downstreamTask.limit}|${c.ragPipeline.faiss.topK}"
    }

    private fun retrievalFor(context: Context): Retrieval {
        val key = configSignature(context)
        if (retrieval == null || retrievalKey != key) {
            retrieval?.close()
            retrieval = Retrieval(context)
            retrievalKey = key
        }
        return retrieval!!
    }

    @Volatile
    private var running = false

    fun tryStart(context: Context, query: String, forceComplex: Boolean = false): Boolean {
        if (running) return false
        running = true
        AgentEventLog.reset()
        val app = context.applicationContext
        scope.launch {
            try {
                runQuery(app, query, forceComplex)
                AgentEventLog.done()
            } catch (e: Throwable) {
                AgentEventLog.emit("error", "error", e.message ?: e::class.java.simpleName)
                AgentEventLog.fail()
            } finally {
                running = false
            }
        }
        return true
    }

    private suspend fun runQuery(context: Context, query: String, forceComplex: Boolean) {
        val config = Parser(context).readTaskConfig()
        val llm = config.ragPipeline.llm
        val engine = BenchmarkManager.getInferenceEngine()

        // aug_method selects the pipeline so all arms come off one endpoint and
        // are scored under the same engine, model and system prompt:
        //   none          - closed book, no retrieval
        //   concatenation - retrieve top_k, answer once
        //   agentic       - retrieve -> investigator loop (max 3) -> answer
        val aug = llm.augMethod.lowercase()
        if (aug == "none") {
            runClosedBook(context, engine, query)
            return
        }
        if (aug != "agentic") {
            runSimple(context, engine, query)
            return
        }

        AgentEventLog.emit("system", "start", "agentic loop")
        val retrieval = retrievalFor(context)

        // one retrieval of the original question; every investigator round and
        // the answerer read this same accumulating evidence
        val t0 = System.nanoTime()
        var evidence = retrieval.retrieve(query)
        emitRetriever(query, evidence, t0)
        val seenQueries = mutableSetOf(queryKey(query))

        // investigator loop: max 3 rounds bounds cost (simple question exits at
        // round 1 = old router cost). forceComplex adds a round-1 mandate so
        // evals measure the chain without depending on routing luck.
        var findings: String? = null
        for (round in 1..3) {
            val mandate = if (forceComplex && round == 1)
                llm.forceComplexMandate else ""
            AgentEventLog.emit(INVESTIGATOR.name, "start", "round $round")
            val prompt = String.format(llm.investigatorPrompt, query,
                seenQueries.joinToString("\n"), renderEvidence(evidence)) + mandate
            var r = generate(context, engine, llm, INVESTIGATOR, prompt)
            var response = cleanModelText(r.response)
            var parsed = parseReply(response)
            if (parsed == null) {
                // one retry before giving up: truncation/mangling drops the
                // QUERY: prefix, and answering from junk evidence is worse
                AgentEventLog.emit(INVESTIGATOR.name, "fallback", "retrying malformed reply", response, r)
                r = generate(context, engine, llm, INVESTIGATOR,
                    prompt + llm.investigatorRetryPrompt)
                response = cleanModelText(r.response)
                parsed = parseReply(response)
            }
            if (parsed == null) {
                AgentEventLog.emit(INVESTIGATOR.name, "fallback", "using evidence after malformed reply", response, r)
                break
            }
            val (verb, rest) = parsed

            if (verb == "ANSWERABLE") {
                AgentEventLog.emit(INVESTIGATOR.name, "end", "sufficient", response, r)
                findings = rest.takeUnless { it.isEmpty() || it == "-" }
                break
            }
            if (rest.isEmpty()) break
            val key = queryKey(rest)
            if (key in seenQueries) {
                AgentEventLog.emit(INVESTIGATOR.name, "fallback", "stopping repeated search", response, r)
                break
            }
            seenQueries += key
            AgentEventLog.emit(INVESTIGATOR.name, "end", "round $round next query", response, r)
            AgentEventLog.emit(INVESTIGATOR.name, "item", "[round $round]", rest)
            val t1 = System.nanoTime()
            val more = retrieval.retrieve(rest)
            emitRetriever(rest, more, t1)
            val updated = (evidence + more).distinct()
            if (updated.size == evidence.size) {
                AgentEventLog.emit(INVESTIGATOR.name, "fallback", "no new evidence", response, r)
                break
            }
            evidence = updated
        }

        answer(context, engine, llm, query, evidence, findings)
    }

    // Closed-book arm: same reasoner call, zero retrieval. Exists so the
    // no-RAG baseline runs through this endpoint too and is scored under the
    // identical engine, model and system prompt as the RAG arms.
    private suspend fun runClosedBook(
        context: Context,
        engine: InferenceEngine,
        query: String
    ) {
        val llm = Parser(context).readTaskConfig().ragPipeline.llm
        AgentEventLog.emit("Retriever", "end", "skipped (aug_method=none)", query)
        AgentEventLog.emit(REASONER.name, "start", "closed-book, no documents")
        val final = generate(context, engine, llm, REASONER,
            String.format(llm.answererPrompt, query, "", "No retrieved evidence."))
        AgentEventLog.emit(REASONER.name, "end", "final answer", cleanModelText(final.response), final)
    }

    private fun renderEvidence(chunks: List<String>): String =
        chunks.mapIndexed { i, text -> "[${i + 1}] $text" }.joinToString("\n\n")

    private fun queryKey(text: String): String =
        text.lowercase().replace(Regex("[^\\p{L}\\p{N}]+"), " ").trim()

    private fun cleanModelText(text: String): String = text
        .replace(Regex("<\\|(?:im_start|im_end|endoftext)\\|>"), "")
        .replace(Regex("(?m)^assistant\\s*"), "")
        .trim()

    private fun emitRetriever(query: String, chunks: List<String>, t0: Long) {
        AgentEventLog.emit("Retriever", "context", query, renderEvidence(chunks))
        AgentEventLog.emit(
            "Retriever", "end",
            "${chunks.size} chunks in ${(System.nanoTime() - t0) / 1_000_000} ms",
            query
        )
    }

    private suspend fun answer(
        context: Context,
        engine: InferenceEngine,
        llm: LlmConfig,
        query: String,
        chunks: List<String>,
        bridge: String? = null
    ) {
        AgentEventLog.emit(REASONER.name, "start", "answering from ${chunks.size} passages")
        // bridgeLine is empty for the simple/closed-book arms, so their prompt
        // stays byte-identical to before the two-stage change
        val bridgeLine = bridge?.let { String.format(llm.bridgeLine, it) + "\n" } ?: ""
        val prompt = String.format(llm.answererPrompt, query, bridgeLine, renderEvidence(chunks))
        var final = generate(context, engine, llm, REASONER, prompt)
        var out = cleanModelText(final.response)
        // REASON/ANSWER is two lines by design; only a tag-less list gets rewritten
        if (isListy(out) && !out.contains(Regex("(?i)\\bANSWER\\s*:"))) {
            // small models dump chunk-title lists when the evidence misses the
            // question; one forced rewrite, then accept whatever comes back
            AgentEventLog.emit(REASONER.name, "fallback", "retrying list-shaped answer", out, final)
            final = generate(context, engine, llm, REASONER, prompt + llm.answererRetryPrompt)
            out = cleanModelText(final.response)
        }
        AgentEventLog.emit(REASONER.name, "end", "final answer", out, final)
    }

    private suspend fun runSimple(
        context: Context,
        engine: InferenceEngine,
        query: String
    ) {
        val llm = Parser(context).readTaskConfig().ragPipeline.llm
        val retrieval = retrievalFor(context)
        val t0 = System.nanoTime()
        val chunks = retrieval.retrieve(query)
        emitRetriever(query, chunks, t0)
        answer(context, engine, llm, query, chunks)
    }

    private suspend fun generate(
        context: Context,
        engine: InferenceEngine,
        llm: LlmConfig,
        agent: Agent,
        prompt: String
    ): InferenceEngine.GenerationResult {
        if (!engine.isLoaded()) {
            AgentEventLog.emit(agent.name, "start", "loading ${llm.modelName}")
            val modelDir = context.getExternalFilesDir(null)!!
                .resolve(Constants.LLM_DIR)
                .resolve("${llm.modelName}_${llm.dtype}")
            val init = engine.initialize(context, llm, modelDir)
            if (!init.success) {
                AgentEventLog.emit(agent.name, "error", init.error ?: "LLM init failed")
                throw IllegalStateException("${agent.name} failed: ${init.error ?: "LLM init failed"}")
            }
            BenchmarkManager.lastInferenceLoadTimeMs = init.loadTimeMs
        }

        // Hoisted so the exact string handed to the engine is recorded in the
        // trace's start event: any saved run then proves which config value the
        // LLM was given (config file -> readTaskConfig -> engine).
        val sysPrompt = if (agent.name == REASONER.name) llm.systemPrompt
        else llm.investigatorSystemPrompt
        // The start event records both halves sent to the model: output = the
        // full assembled user prompt (question + evidence + config template),
        // detail = the system prompt — so any saved jsonl shows exactly what
        // the LLM received, end to end.
        AgentEventLog.emit(agent.name, "start", "generating · systemPrompt=$sysPrompt", prompt)
        val r = engine.generate(
            InferenceEngine.GenerationRequest(
                prompt = prompt,
                // Both prompts come fresh from the config on every request: the
                // Answerer gets system_prompt (its contract, shared with the
                // benchmark's no-RAG baseline), the Evidence Investigator gets
                // investigator_system_prompt. Passing them explicitly bypasses
                // LiteRTEngine's `ifBlank { cfg.systemPrompt }` fallback, which
                // serves the value cached when the engine was loaded and thus
                // goes stale while the server process stays alive.
                systemPrompt = sysPrompt,
                maxTokens = agent.maxTokens,
                timeoutMs = 120_000
            )
        )
        if (r.status != InferenceEngine.Status.OK) {
            AgentEventLog.emit(agent.name, "error", r.error ?: r.status.name)
            // No silent continuation: a failed agent fails the whole run so the
            // client/eval sees a failure instead of scoring empty output.
            throw IllegalStateException("${agent.name} failed: ${r.error ?: r.status.name}")
        }
        return r
    }
}
