package com.example.moragbench

import android.app.ActivityManager
import android.app.AlertDialog
import android.content.Intent
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.example.onnxtok.EmbeddingGenerator
import com.example.faiss.FaissIndex
import com.example.local_llm.LocalLLM
import kotlinx.coroutines.*
import java.util.Locale
import android.os.SystemClock
import com.example.moragbench.ui.LoaderDialog
import com.example.faiss.EmbeddingDb
import com.example.faiss.EmbeddingRow
import com.example.faiss.PendingChunk
import com.example.local_llm.GenerationMetrics
import com.example.onnxtok.EmbeddingModel
import com.example.shared.Shared
import com.example.shared.SupportedEmbeddingModels

class MainActivity : AppCompatActivity() {

    private val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    private lateinit var recycler: RecyclerView
    private lateinit var input: EditText
    private lateinit var sendButton: ImageButton
    private lateinit var showPromptToggle: CheckBox
    private lateinit var speakButton: ImageButton

    // Icons
    private lateinit var settingsButton: ImageButton
    private lateinit var clearMessages: ImageButton
    private lateinit var loadModel: ImageButton

    private lateinit var adapter: ChatAdapter
    private val messages = mutableListOf<Message>()

    // RAG pieces (kept static behavior as requested)
    private lateinit var embeddingGenerator: EmbeddingGenerator
    private lateinit var embeddingModel: EmbeddingModel
    private lateinit var faissIndex: FaissIndex
    val faissIndexPath: String by lazy {
        filesDir.absolutePath + "/faiss.index"
    }
    private lateinit var llm: LocalLLM
    private lateinit var corpus: Map<String, List<String>>

    // EmbeddingDb
    private lateinit var db: EmbeddingDb

    // Generation state
    private var isGenerating = false
    private var lastAssistantText: String = ""

    // TTS
    private var tts: TextToSpeech? = null

    // Misc
    private var genStartMs: Long = 0L
    private var tokenCount: Int = 0
    private lateinit var loader: LoaderDialog
    private lateinit var helper: Helper
    private lateinit var settings: SettingsManager
    private var modelIsLoaded: Boolean = false


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        recycler = findViewById(R.id.chatRecycler)
        input = findViewById(R.id.inputField)
        sendButton = findViewById(R.id.sendButton)
        showPromptToggle = findViewById(R.id.showPromptToggle)
        speakButton = findViewById(R.id.speakButton)
        settingsButton = findViewById(R.id.settingsButton)
        clearMessages = findViewById(R.id.clearMessages)
        loadModel = findViewById(R.id.loadModel)

        adapter = ChatAdapter(messages)
        recycler.layoutManager = LinearLayoutManager(this).apply { stackFromEnd = true }
        recycler.adapter = adapter

        // TTS init (for “say the response”)
        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.language = Locale.US
            }
        }

        // Initialize the loader
        loader = LoaderDialog(this)

        // Initialize the helper
        helper = Helper(this)

        // Initialize the settings manager
        settings = SettingsManager(this)

        // Initialize FaissIndex
        faissIndex = FaissIndex()

        // Initialize EmbeddingDB
        db = EmbeddingDb(this)

        // Add RAM usage to action bar
         startRamUsageUpdater()

        // Add all click listeners
        sendButton.setOnClickListener {
            if (!modelIsLoaded) {
                helper.toast("Please load model first.")
                return@setOnClickListener
            }

            if (isGenerating) {
                // Stop generation
                llm.stop()
                isGenerating = false
                sendButton.setImageResource(android.R.drawable.ic_menu_send)
            } else {
                val query = input.text.toString().trim()
                if (query.isEmpty()) {
                    helper.toast("Please enter a query.")
                    return@setOnClickListener
                }
                startQuery(query)
                input.setText("")
            }
        }

        speakButton.setOnClickListener {
            if (lastAssistantText.isNotBlank()) {
                tts?.speak(lastAssistantText, TextToSpeech.QUEUE_FLUSH, null, "llm_say")
            } else {
                helper.toast("No assistant response to say.")
            }
        }

        settingsButton.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        clearMessages.setOnClickListener {
            AlertDialog.Builder(this)
                .setTitle("Confirm Clear Chat")
                .setMessage("Are you sure you want to clear the chat?")
                .setPositiveButton("Yes") { dialog,     _ ->
                    dialog.dismiss()

                    messages.clear()
                    adapter.notifyDataSetChanged()
                    lastAssistantText = ""
                }
                .setNegativeButton("Cancel") { dialog, _ ->
                    dialog.dismiss()
                }
                .show()
        }

        loadModel.setOnClickListener {
            // Prepare pipeline (embeddings, FAISS, LLM)
            scope.launch {
                sendButton.isEnabled = false

                // Initialize the embedding model
                loader.show("Initializing embeddings")
                embeddingModel = SupportedEmbeddingModels.findByName(settings.embedding.model)
                embeddingGenerator = EmbeddingGenerator(this@MainActivity)
                embeddingGenerator.initialize(embeddingModel, backend = settings.embedding.backend.lowercase())

                // Initialize FAISS
                loader.update("Initializing FAISS")
                initFaiss()

                // Initialize LLM
                loader.update("Initializing LLM")
                llm = initLLM()
                llm.initialize()

                loader.hide()
                sendButton.isEnabled = true

                modelIsLoaded = true
            }
        }
    }

    private fun startQuery(query: String) {
        // If not checked, Show user message in chat UI directly
        // else, need to wait for full prompt
        if (!showPromptToggle.isChecked) {
            addMessage(Message(content = query, isUser = true))
        }

        // Build static prompt context (top-k docs via FAISS)
        isGenerating = true
        sendButton.setImageResource(R.drawable.baseline_stop_24)

        scope.launch {
            try {
                val queryEmbedding = withContext(Dispatchers.Default) {
                    embeddingGenerator.generate(query)
                }
                val k = settings.faiss.topK
                val dim = embeddingModel.dim
                val buffer = FloatArray(dim)
                System.arraycopy(queryEmbedding, 0, buffer, 0, dim)
                val results = withContext(Dispatchers.IO) {
                    faissIndex.query(buffer, 1, k)
                }

                val labels = results.labels

                if (labels == null) {
                    throw Exception("Faiss returned no results!")
                }

                val topDocs = labels.map { db.getByFaissId(it)?.chunkText }
                // Prepare context
                val contextText = topDocs
                    .filterNotNull()
                    .mapIndexed { index, doc ->
                        "[${index + 1}] $doc"
                    }
                    .joinToString("\n\n")

                val systemPrompt = settings.llm.systemPrompt

                // Show user message in chat UI
                if (showPromptToggle.isChecked) {
                    val msgToAdd = """
                        |System prompt with retrieved documents:
                        |
                        |$systemPrompt
                        |
                        |Documents:
                        |$contextText
                        |
                        |User query:
                        |$query
                    """.trimMargin()
                    addMessage(Message(content = msgToAdd, isUser = true))
                }

                // Stream tokens to the UI; DO NOT append previous conversation to model.
                val acc = StringBuilder()
                // Start timing + token counting for this assistant generation
                genStartMs = SystemClock.elapsedRealtime()
                tokenCount = 0

                lastAssistantText = ""

                // Add a streaming assistant message with timestamp now
                addMessage(
                    Message(
                        content = "",
                        isUser = false,
                        isStreaming = true,
                        timeMillis = SystemClock.elapsedRealtime()
                    )
                )

                withContext(Dispatchers.Default) {
                    llm.generateStreaming(
                        inputText = query,
                        systemPrompt = systemPrompt,
                        contextText = contextText,
                        metrics = GenerationMetrics(), // Dummy, not used here
                        maxTokens = settings.llm.maxTokens,
                        onToken = { token ->
                            tokenCount += 1           // count tokens as they stream
                            acc.append(token)
                            scope.launch { updateStreamingMessage(acc.toString()) }
                        },
                        onComplete = {
                            val elapsedSec = (SystemClock.elapsedRealtime() - genStartMs) / 1000.0
                            val tps = if (elapsedSec > 0) tokenCount.toDouble() / elapsedSec else 0.0

                            scope.launch {
                                // finalize and attach metrics
                                finalizeStreamingMessageWithMetrics(
                                    finalText = acc.toString(),
                                    tokensPerSec = tps,
                                    durationSec = elapsedSec
                                )
                                lastAssistantText = acc.toString()
                                isGenerating = false
                                sendButton.setImageResource(android.R.drawable.ic_menu_send)
                            }
                        },
                        onError = { e ->
                            scope.launch {
                                finalizeStreamingMessage("Error: ${e.message}")
                                isGenerating = false
                                sendButton.setImageResource(android.R.drawable.ic_menu_send)
                            }
                        }
                    )
                }
            } catch (e: CancellationException) {
                finalizeStreamingMessage("Generation stopped.")
                isGenerating = false
                sendButton.setImageResource(android.R.drawable.ic_menu_send)
            } catch (t: Throwable) {
                finalizeStreamingMessage("Error: ${t.message}")
                isGenerating = false
                sendButton.setImageResource(android.R.drawable.ic_menu_send)
            }
        }
    }

    private suspend fun initFaiss() {
        // Try to load persisted FAISS index
        val faissLoaded = faissIndex.loadFrom(faissIndexPath)

        val dbCount = db.countEmbeddings()
        var faissCount = 0L
        if (faissLoaded) {
            faissCount = faissIndex.getTotal()
        }

        // Consistency decisions:
        when {
            faissLoaded && dbCount == faissCount -> {
                // Everything matches: good to go.
                helper.toast("Reading index from disk. Skip building")
                return
            }
            !faissLoaded && dbCount == 0L -> {
                // Both missing -> build fresh from corpus (existing path)
                buildIndexFromCorpus()
                return
            }
            else -> {
                // Any mismatch, or database present but FAISS missing, or counts differ -> wipe both and rebuild
                // Remove persisted files (FAISS file + DB) then rebuild
                Shared.safeDeleteFaissFile(faissIndexPath)
                db.clearAll()
                buildIndexFromCorpus()
                return
            }
        }
    }

    private suspend fun buildIndexFromCorpus() {
        helper.toast("Rebuilding FAISS index…")
        try {
            /* TODO: Currently, this reads the whole corpus into memory.
                     This is not ideal for large datasets.
                     Try to divide into batches */
            // Load & chunk
            corpus = helper.loadAndChunk(embeddingModel.tokenizerPath)

            val dim = embeddingModel.dim

            // Create new index based on settings
            // Read settings

            val distanceMetric = FaissIndex.DistanceMetric.fromString(settings.faiss.metric)
            val indexMethod = settings.faiss.indexMethod

            // We'll collect precomputed training vectors (for IVF) here if needed.
            // Key: Pair(fileName, chunkIndex) -> FloatArray vector
            val precomputedTrainingVectors = mutableMapOf<Pair<String, Int>, FloatArray>()

            when (indexMethod) {
                "Flat" -> {
                    // Init FlatIndex
                    val flatConfig = FaissIndex.FlatConfig(metric = distanceMetric)
                    faissIndex.initFlat(dim, flatConfig)
                }
                "IVF" -> {
                    // Read rest of the settings
                    val nprobe = settings.faiss.nprobe
                    var nlist = settings.faiss.nlist
                    var numTrainingVectors = settings.faiss.numTrainingVectors


                    /* TODO: This currently works, but when changing to read
                             corpus in batches, this needs to be revisited. */
                    val totalChunks = corpus.values.sumOf { it.size }
                    if (totalChunks == 0) {
                        throw Exception("Corpus is empty — cannot train IVF index")
                    }

                    // Adjust numTrainingVectors to be sensible
                    if (numTrainingVectors > totalChunks) {
                        // can't train with more vectors than available
                        numTrainingVectors = totalChunks
                        helper.toast("Number of training vectors adjusted to $numTrainingVectors")
                    }

                    // nlist should be <= total num of training vectors
                    if (nlist > numTrainingVectors) {
                        nlist = numTrainingVectors
                        helper.toast("Setting nlist to $nlist")
                    }

                    // Init IVFIndex
                    val config = FaissIndex.IvfConfig(nlist, nprobe, distanceMetric)
                    faissIndex.initIvf(dim, config)

                    // --- Prepare training vectors ---
                    loader.update("Training IVF index")

                    // Choose numTrainingVectors chunk positions evenly across the corpus for diversity
                    // Build a flat list of (fileName, chunkIndex) for deterministic selection
                    val flatChunks = mutableListOf<Pair<String, Int>>()
                    corpus.forEach { (fileName, chunks) ->
                        chunks.forEachIndexed { idx, _ ->
                            flatChunks.add(Pair(fileName, idx))
                        }
                    }

                    // If we need all chunks, just use them
                    val chosenPositions = if (numTrainingVectors == totalChunks) {
                        flatChunks
                    } else {
                        // Evenly sample indices across flatChunks
                        val step = totalChunks.toDouble() / numTrainingVectors.toDouble()
                        val chosen = mutableListOf<Pair<String, Int>>()
                        var pos = 0.0
                        for (i in 0 until numTrainingVectors) {
                            val idx = kotlin.math.floor(pos).toInt().coerceIn(0, totalChunks - 1)
                            chosen.add(flatChunks[idx])
                            pos += step
                        }
                        chosen
                    }

                    // Generate embeddings for chosen positions and store them
                    // This may be somewhat expensive but necessary for good IVF training data.
                    for ((fileName, chunkIdx) in chosenPositions) {
                        val chunkText = corpus[fileName]?.get(chunkIdx)
                            ?: throw Exception("Corrupted corpus: cannot find $fileName chunk $chunkIdx")
                        val vec = embeddingGenerator.generate(chunkText) // FloatArray expected
                        precomputedTrainingVectors[Pair(fileName, chunkIdx)] = vec
                    }

                    // Flatten selected vectors into single FloatArray (dim * numTrainingVectors)
                    val flattened = FloatArray(dim * numTrainingVectors)
                    var writePos = 0
                    for ((fileName, chunkIdx) in chosenPositions) {
                        val vec = precomputedTrainingVectors[Pair(fileName, chunkIdx)]
                            ?: throw Exception("Training vector missing for $fileName:$chunkIdx")
                        System.arraycopy(vec, 0, flattened, writePos, dim)
                        writePos += dim
                    }

                    // Train IVF
                    faissIndex.trainIvf(flattened, numTrainingVectors)
                }
                "HNSW" -> {
                    // Read rest of the settings
                    val m = settings.faiss.m
                    val efConstruction = settings.faiss.efConstruction
                    val efSearch = settings.faiss.efSearch

                    // Init Index
                    val hnswConfig = FaissIndex.HnswConfig(
                        M = m,
                        efConstruction = efConstruction,
                        efSearch = efSearch,
                        metric = distanceMetric
                    )
                    faissIndex.initHnsw(dim, hnswConfig)
                }
                else -> {
                    throw Exception("Unsupported index method: $indexMethod")
                }
            }

            // Reset stats
            faissIndex.resetStats()

            // Add all chunks to the index.
            // If a chunk was used for training, reuse the precomputed vector.
            val batchSize = settings.faiss.batchSize
            val pending = ArrayList<PendingChunk>(batchSize)
            var counter = 0
            for ((fileName, chunks) in corpus) {
                counter++
                chunks.forEachIndexed { index, chunk ->
                    // pass optional precomputed vector if available
                    val vec = precomputedTrainingVectors[Pair(fileName, index)]
                        ?: embeddingGenerator.generate(chunk)
                    pending.add(
                        PendingChunk(
                            docId = fileName,
                            chunkId = index,
                            chunkText = chunk,
                            model = settings.embedding.model,
                            vec = vec
                        )
                    )

                    // When batch full, or last chunk → flush to FAISS + DB
                    if (pending.size == batchSize || (counter == corpus.size && index == chunks.lastIndex)) {
                        addChunksAndPersist(pending)
                        pending.clear()
                    }
                }
            }

            // Atomic save: save to tmp then rename
            faissIndex.saveTo(faissIndexPath)
        } catch (e: Exception) {
            throw Exception("buildIndexFromCorpus failed: ${e.message}")
        }
    }

    private fun addChunksAndPersist(pending: List<PendingChunk>) {
        try {
            if (pending.isEmpty()) return

            val dim = embeddingModel.dim
            val batchSize = pending.size
            val buffer = FloatArray(batchSize * dim)

            // Build contiguous FAISS buffer
            var offset = 0
            for (p in pending) {
                System.arraycopy(p.vec, 0, buffer, offset, dim)
                offset += dim
            }

            val ids = faissIndex.add(buffer, batchSize)

            // persist mapping into SQLite DB synchronously
            val nowSec = SystemClock.elapsedRealtime() / 1000L
            for (i in pending.indices) {
                val p = pending[i]

                val row = EmbeddingRow(
                    faissId = ids[i],
                    docId = p.docId,
                    chunkId = p.chunkId,
                    chunkText = p.chunkText,
                    dim = dim,
                    model = p.model,
                    metaJson = null,
                    createdAt = nowSec
                )

                db.insertEmbedding(row)
            }
        } catch (e: Exception) {
            throw Exception("Add failed: ${e.message}")
        }
    }

    private fun updateStreamingMessage(text: String) {
        val idx = messages.indexOfLast { !it.isUser && it.isStreaming }
        if (idx >= 0) {
            messages[idx] = messages[idx].copy(content = text)
            adapter.notifyItemChanged(idx)
            recycler.scrollToPosition(messages.lastIndex)
        }
    }

    private fun finalizeStreamingMessage(text: String) {
        val idx = messages.indexOfLast { !it.isUser && it.isStreaming }
        if (idx >= 0) {
            messages[idx] = messages[idx].copy(content = text, isStreaming = false)
            adapter.notifyItemChanged(idx)
            recycler.scrollToPosition(messages.lastIndex)
        } else {
            addMessage(Message(content = text, isUser = false))
        }
    }

    private fun finalizeStreamingMessageWithMetrics(finalText: String, tokensPerSec: Double, durationSec: Double) {
        val idx = messages.indexOfLast { !it.isUser && it.isStreaming }
        if (idx >= 0) {
            val old = messages[idx]
            messages[idx] = old.copy(
                content = finalText,
                isStreaming = false,
                metrics = MessageMetrics(tokensPerSec, durationSec)
            )
            adapter.notifyItemChanged(idx)
            recycler.scrollToPosition(messages.lastIndex)
        } else {
            // Fallback: add a fresh assistant message with metrics
            addMessage(
                Message(
                    content = finalText,
                    isUser = false,
                    isStreaming = false,
                    metrics = MessageMetrics(tokensPerSec, durationSec)
                )
            )
        }
    }

    private fun addMessage(msg: Message) {
        messages.add(msg)
        adapter.notifyItemInserted(messages.lastIndex)
        recycler.scrollToPosition(messages.lastIndex)
    }

    private fun initLLM(): LocalLLM {
        val config = helper.getModelConfig()

        // Change model config based on settings
        config.backend = settings.llm.genBackend.lowercase()
        config.temperature = settings.llm.temp
        config.topP = settings.llm.topP
        config.topK = settings.llm.topK
        config.doSample = settings.llm.useSampling
        config.repetitionPenalty = settings.llm.repetitionPenalty
        config.defaultSystemPrompt = settings.llm.systemPrompt

        return LocalLLM(this, config)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        intent.let {
            if (it.getBooleanExtra("from_settings", false)) {
                // When changing settings, clear the DB and delete FAISS file and its metadata
                // TODO: Make smarter decisions here
                Shared.safeDeleteFaissFile(faissIndexPath)
                Shared.safeDeleteFaissFile("$faissIndexPath.meta")
                db.clearAll()

                // When return from settings, recreate this activity
                recreate()
            }
        }
    }

    private fun startRamUsageUpdater() {
        val ramTextView = findViewById<TextView>(R.id.ram_usage_live)
        scope.launch {
            while (isActive) {
                val am = getSystemService(ACTIVITY_SERVICE) as ActivityManager
                val mi = ActivityManager.MemoryInfo()
                am.getMemoryInfo(mi)
                val usedMb = (mi.totalMem - mi.availMem) / (1024 * 1024)
                ramTextView.text = "${usedMb}MB"
                delay(1000L)
            }
        }
    }

    fun freeResources() {
        tts?.shutdown()
        if (llm.isInitialized) llm.shutdown()
        if (this::embeddingGenerator.isInitialized) embeddingGenerator.close()
    }

    override fun onDestroy() {
        super.onDestroy()
        scope.cancel()
        freeResources()
    }
}
