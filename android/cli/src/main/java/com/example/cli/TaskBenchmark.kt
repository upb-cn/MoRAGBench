package com.example.cli

import DownstreamTask
import TaskConfig
import android.content.Context
import android.os.SystemClock
import com.example.cli.BenchmarkManager.getDB
import com.example.cli.CacheManager.buildTaskSignature
import com.example.cli.CacheManager.hashSignature
import com.example.cli.Progress.ProgressPhase
import com.example.cli.Progress.State
import com.example.cli.Progress.TaskProgress
import com.example.faiss.EmbeddingDb
import com.example.faiss.EmbeddingRow
import com.example.faiss.FaissIndex
import com.example.faiss.PendingChunk
import com.example.local_llm.GenerationMetrics
import com.example.local_llm.LocalLLM
import com.example.local_llm.PromptIntent
import com.example.local_llm.TokenizerSource
import com.example.onnxtok.EmbeddingGenerator
import com.example.onnxtok.EmbeddingModel
import com.example.onnxtok.ExternalTokenizerSource
import com.example.onnxtok.TextChunker
import com.example.shared.ModelPathOverrides
import com.example.shared.SupportedEmbeddingModels
import com.example.shared.SupportedLLMs
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonPrimitive
import java.io.File


class TaskBenchmark(private val context: Context) {
    private val benchmarkScope = CoroutineScope(
        SupervisorJob() + Dispatchers.Default
    )
    private var benchmarkJob: Job? = null

    private val taskConfig: TaskConfig
        get() {
            val parser = Parser(context)
            return parser.readTaskConfig()
        }

    private val parser: Parser
        get() {
            val parser = Parser(context)
            return parser
        }

    var progress: TaskProgress = TaskProgress()
    val overallMetrics: OverallMetrics = OverallMetrics()

    lateinit var textChunker: TextChunker

    fun getHash(task: DownstreamTask): String {
        val signature = buildTaskSignature(taskConfig, task)
        val hash = hashSignature(signature)

        return hash
    }

    fun stop() {
        benchmarkJob?.cancel()
        benchmarkJob = null
    }

    fun start() {
        val taskName = taskConfig.downstreamTask.name

        // Init Logger
        val logDir = context.getExternalFilesDir(null)!!
            .resolve(Constants.TASK_RESULTS_DIR)
            .resolve(taskName)
        FileLogger.init(logDir)

        // Check if progress is initialized
        val progressSnapshot = progress.getSnapshot()
        if (progressSnapshot != null && progressSnapshot.overallState == State.RUNNING) return

        // Record start time
        val benchmarkTimeStart = System.nanoTime()

        // ****** Read downstream tasks ******
        val taskFiles = parser.readDownstreamTaskFiles(taskName)

        benchmarkJob = benchmarkScope.launch {
            try {
                // Init progress
                val downstreamTask = taskConfig.downstreamTask
                progress.init(downstreamTask)

                // Start metrics engine
                val hardwareMetricsEngine = HardwareMetricsEngine(
                    context = context,
                    scope = benchmarkScope,
                    intervalMs = 20L
                )
                hardwareMetricsEngine.start()

                // Run the benchmark
                runBenchmark(taskFiles, downstreamTask)

                // Stop metrics engine
                val metricsResult = hardwareMetricsEngine.stop()

                // Update progress: Complete
                progress.complete()

                // Update overall metrics
                val benchmarkTimeEnd = System.nanoTime()
                overallMetrics.benchmarkTimeMs = (benchmarkTimeEnd - benchmarkTimeStart) / 1_000_000L
                overallMetrics.indexSource = progress.getIndexSource()

                // Write overall metrics to file
                val metricsFile = context.getExternalFilesDir(null)!!
                    .resolve(Constants.TASK_RESULTS_DIR)
                    .resolve(taskName)
                    .resolve(Constants.OVERALL_METRICS_FILE)
                writeMetricsToFile(overallMetrics, metricsFile)

                // Save hardware metrics JSON
                val hardwareMetricsFile = context.getExternalFilesDir(null)!!
                    .resolve(Constants.TASK_RESULTS_DIR)
                    .resolve(taskName)
                    .resolve(Constants.HARDWARE_METRICS_FILE)
                hardwareMetricsEngine.writeMetricsJson(metricsResult, hardwareMetricsFile)

            } catch (e: Exception) {
                // Update progress: Fail
                progress.fail(e.stackTraceToString())
            }
        }
    }

    private suspend fun runBenchmark(
        taskFiles: Map<String, JsonElement>,
        downstreamTask: DownstreamTask
    ) {
        // ****** Initialize embedding model ******
        LiveLogger.info("Initializing embedding model...")
        val embeddingInitStart = System.nanoTime()

        val embeddingConfig = taskConfig.ragPipeline.embedding
        val embeddingModel: EmbeddingModel = SupportedEmbeddingModels.findByName(embeddingConfig.modelName)

        // Append baseDir path to modelPath and tokenizerPath
        val baseDir = context.getExternalFilesDir(null)!!.resolve(Constants.TASK_BASE_DIR)
        embeddingModel.modelPath = File(baseDir, embeddingModel.modelPath).canonicalPath
        embeddingModel.tokenizerPath = File(baseDir, embeddingModel.tokenizerPath).canonicalPath

        val embeddingGenerator = EmbeddingGenerator(context)
        embeddingGenerator.initialize(embeddingModel, backend = embeddingConfig.backend)

        // Update overall metrics
        val embeddingInitEnd = System.nanoTime()
        overallMetrics.embeddingInitTimeMs = (embeddingInitEnd - embeddingInitStart) / 1_000_000L
        LiveLogger.info("Embedding model initialized in ${overallMetrics.embeddingInitTimeMs / 1000} s")

        // Set tokenizer path
        val embeddingTokenizerPath = parser.getTokenizerPath(taskConfig.ragPipeline.embedding.modelName)

        // ****** Initialize LLM ******
        LiveLogger.info("Initializing LLM...")
        val llmInitStart = System.nanoTime()

        val llmConfig = taskConfig.ragPipeline.llm

        // Prepare llm Dir
        val llmDir = context.getExternalFilesDir(null)!!
            .resolve(Constants.LLM_DIR)
            .resolve("${llmConfig.modelName}_${llmConfig.dtype}")

        // Get model and tokenizer paths
        val modelPath = llmDir.resolve("model.onnx").canonicalPath
        val tokenizerPath = llmDir.resolve("tokenizer.json").canonicalPath

        // Get model config
        val config = SupportedLLMs.findByName(
            context,
            "${llmConfig.modelName}-${llmConfig.dtype}",
            overrides = ModelPathOverrides(
                modelPath = modelPath,
                tokenizer = TokenizerSource.File(tokenizerPath),
            )
        )
        // Change some configurations based on config
        config.backend = llmConfig.backend
        config.temperature = llmConfig.temp.toFloat()
        config.topP = llmConfig.topP.toFloat()
        config.topK = llmConfig.topK
        config.doSample = llmConfig.useSampling
        config.repetitionPenalty = llmConfig.repetitionPenalty.toFloat()
        config.defaultSystemPrompt = llmConfig.systemPrompt
        config.kvWindow = llmConfig.kvWindow
        config.prefillChunkSize = llmConfig.prefillChunkSize

        // init llm
        val llm = LocalLLM(context, config)
        llm.initialize()

        // Update overall metrics
        val llmInitEnd = System.nanoTime()
        overallMetrics.llmInitTimeMs = (llmInitEnd - llmInitStart) / 1_000_000L
        LiveLogger.info("LLM initialized in ${overallMetrics.llmInitTimeMs / 1000} s")


        // ****** Init Faiss Index ******

        // Update progress: Start task
        progress.startTask()

        val documentCount = parser.countDocuments(downstreamTask.name)
        val (faissIndex, faissMetrics) = initFaissIndex(
            task = downstreamTask,
            documentCount = documentCount,
            documentStream = { action -> parser.forEachDocument(downstreamTask.name, action) },
            embeddingModel = embeddingModel,
            embeddingGenerator = embeddingGenerator,
            embeddingTokenizerPath = embeddingTokenizerPath
        )

        // Update progress: Change phase
        progress.setPhase(ProgressPhase.EVALUATING)

        // ****** Benchmark Task ******
        benchmarkTask(
            questions = taskFiles["questions"]!!.jsonArray,
            embeddingGenerator = embeddingGenerator,
            faissIndex = faissIndex,
            task = downstreamTask,
            llm = llm
        )

        // Add index stats based on type
        when (faissIndex.currentIndexType) {
            FaissIndex.IndexType.FLAT -> {
                faissMetrics.flatStats = faissIndex.getFlatStats()
            }
            FaissIndex.IndexType.HNSW -> {
                faissMetrics.hnswStats = faissIndex.getHnswStats()
            }
            FaissIndex.IndexType.IVF -> {
                faissMetrics.ivfStats = faissIndex.getIvfStats()
            }
            FaissIndex.IndexType.NONE -> {
                // Will never reach this
            }
        }

        // Save faissMetrics to JSON file
        val faissMetricsFile = context.getExternalFilesDir(null)!!
            .resolve(Constants.TASK_RESULTS_DIR)
            .resolve(downstreamTask.name)
            .resolve(Constants.FAISS_SETUP_FILE)
        writeMetricsToFile(faissMetrics, faissMetricsFile)

        // Update progress: End task
        progress.completeTask()

        // Close
        faissIndex.clear()
        llm.shutdown()
        embeddingGenerator.close()
    }


    private suspend fun initFaissIndex(
        task: DownstreamTask,
        documentCount: Int,
        documentStream: ((String, String) -> Unit) -> Unit,
        embeddingModel: EmbeddingModel,
        embeddingGenerator: EmbeddingGenerator,
        embeddingTokenizerPath: String
    ): Pair<FaissIndex, FaissMetrics> {
        // -------------------------------
        // READ/WRITE FAISS FROM/TO DISK
        // -------------------------------

        // The idea is to save the index for future use
        // In order to know that the file is valid, we
        // will use hashing. The important params that
        // might invalidate the index will be used in
        // the hash. Later when reading, if the hash
        // is different, then we rebuild the index

        // Create metrics instance
        val faissMetrics = FaissMetrics()

        // Update progress: Start Initializing Index
        progress.initIndex()

        // Init faissIndex instance
        val faissIndex = FaissIndex()

        // Retrieve cache dir. Make sure it exists
        val cacheDir = context.getExternalFilesDir(null)!!.resolve(Constants.TASK_CACHE_DIR)
        if (!cacheDir.exists()) {
            cacheDir.mkdirs()
        }

        // Build signature and hash
        val hash = getHash(task)

        // Derive index path
        val indexPath = File(cacheDir, "faiss_$hash.index")

        if (taskConfig.ragPipeline.faiss.useCache && indexPath.exists()) {
            // If index exists, load it from disk

            // Update progress: Set source to cache
            progress.setIndexSourceToCache(documentCount)

            // Load from disk and measure time
            val loadStart = System.nanoTime()
            faissIndex.loadFrom(indexPath.absolutePath)
            val loadEnd = System.nanoTime()
            faissMetrics.loadTimeNs = loadEnd - loadStart

            // Set index method
            // TODO: This is not ideal. It should be read from native file
            faissIndex.setMethod(taskConfig.ragPipeline.faiss.method)
        } else {
            // Else, build index from scratch

            // Update progress: Set source to scratch
            progress.setIndexSourceToScratch(documentCount)

            buildFaissIndex(
                faissIndex = faissIndex,
                embeddingModel = embeddingModel,
                embeddingTokenizerPath = embeddingTokenizerPath,
                documentCount = documentCount,
                documentStream = documentStream,
                embeddingGenerator = embeddingGenerator,
                task = task,
                faissMetrics = faissMetrics
            )

            // Update progress: Update status to saving
            progress.startSaving()

            // Save to cache and measure time
            val saveStart = System.nanoTime()
            faissIndex.saveTo(indexPath.absolutePath)
            val saveEnd = System.nanoTime()
            faissMetrics.saveTimeNs = saveEnd - saveStart
        }

        // Update progress: Complete Index Phase
        progress.completeIndex()

        // Trigger calculation of avg
        faissMetrics.calcAvg()

        return Pair(faissIndex, faissMetrics)
    }


    private suspend fun buildFaissIndex(
        embeddingModel: EmbeddingModel,
        documentCount: Int,
        documentStream: ((String, String) -> Unit) -> Unit,
        embeddingGenerator: EmbeddingGenerator,
        embeddingTokenizerPath: String,
        faissIndex: FaissIndex,
        task: DownstreamTask,
        faissMetrics: FaissMetrics
    ) {
        val faissConfig = taskConfig.ragPipeline.faiss
        val dim = embeddingModel.dim
        val distanceMetric = FaissIndex.DistanceMetric.fromString(faissConfig.metric)
        val indexMethod = faissConfig.method

        val hash = getHash(task)
        val db = getDB(hash, context)

        fun addChunksAndPersist(pending: List<PendingChunk>) {
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
        }

        // -------------------------------
        // INIT INDEX
        // -------------------------------

        // Initialize text chunker
        val chunkerConfig = taskConfig.ragPipeline.embedding.chunker
        val chunkingMethod = chunkerConfig.method
        val chunkSize = chunkerConfig.size
        val overlapEnabled = chunkerConfig.overlapEnabled
        val overlapSize = chunkerConfig.overlapSize

        textChunker = TextChunker(
            tokenizerSource = ExternalTokenizerSource(
                context,
                embeddingTokenizerPath
            ),
            chunkingMethod = chunkingMethod,
            chunkSize = chunkSize,
            overlap = if (overlapEnabled) overlapSize else 0
        )
        textChunker.initialize()

        // Start init time
        val initFaissStart = System.nanoTime()
        var initFaissEnd: Long
        try {
            when (indexMethod) {
                "flat" -> {
                    faissIndex.initFlat(
                        dim,
                        FaissIndex.FlatConfig(metric = distanceMetric)
                    )

                    // Record end time
                    initFaissEnd = System.nanoTime()
                }

                "hnsw" -> {
                    faissIndex.initHnsw(
                        dim,
                        FaissIndex.HnswConfig(
                            M = faissConfig.config.m!!,
                            efConstruction = faissConfig.config.efConstruction!!,
                            efSearch = faissConfig.config.efSearch!!,
                            metric = distanceMetric
                        )
                    )

                    // Record end time
                    initFaissEnd = System.nanoTime()
                }

                "ivf" -> {
                    val nprobe = faissConfig.config.nprobe
                    val nlist = faissConfig.config.nlist
                    var numTrainingVectors = faissConfig.config.numTrainingVectors!!

                    faissIndex.initIvf(
                        dim,
                        FaissIndex.IvfConfig(
                            nlist = nlist!!,
                            nprobe = nprobe!!,
                            metric = distanceMetric
                        )
                    )

                    // Record end time
                    initFaissEnd = System.nanoTime()

                    // -------------------------------
                    // COLLECT TRAINING  VECTORS
                    // -------------------------------

                    // Sampling algorithm: Reservoir Sampling
                    // Intuition:
                    //  Keep a buffer of size numTrainingVectors
                    //  Fill it initially
                    //  For each additional chunk:
                    //      Replace a random element with probability numTrainingVectors / seenChunks
                    //      This guarantees even sampling across the entire stream.

                    // In order to avoid vector recomputation later, we will do the following:
                    // - Generate embeddings once
                    // - Write them to a temp binary file
                    // - Use them twice, for train, and for adding

                    // Update progress: Training task
                    progress.startTraining()

                    EmbeddingsPool(dim).use { spool ->
                        val buildStart = System.nanoTime()

                        // STREAM DOCS → DISK
                        documentStream { docId, text ->
                            // TODO: I am currently increasing stats here
                            //       instead of when its actually added to
                            //       the index since most of the time will be
                            //       spend here. This is not very accurate.

                            // Chunk document and measure time
                            val chunkStart = System.nanoTime()
                            val chunks = runBlocking { textChunker.chunkText(text) }
                            val chunkEnd = System.nanoTime()
                            faissMetrics.chunkingTimeNs.add(chunkEnd - chunkStart)


                            // Update progress: Increment documents
                            progress.incrementDocuments()

                            chunks.forEachIndexed { chunkId, chunkText ->
                                // Generate embeddings for chunk and measure time
                                val embeddingStart = System.nanoTime()
                                val vec = runBlocking { embeddingGenerator.generate(chunkText) }
                                val embeddingEnd = System.nanoTime()
                                faissMetrics.embeddingTimeNs.add(embeddingEnd - embeddingStart)

                                // Add to embeddings pool (disk)
                                spool.append(vec, docId, chunkId, chunkText)
                            }

                            // Update progress: Increment chunks
                            progress.incrementChunks(chunks.size)
                        }

                        // Adjust numTrainingVectors to be sensible
                        if (numTrainingVectors > spool.recordCount) {
                            // can't train with more vectors than available
                            numTrainingVectors = spool.recordCount
                            LiveLogger.info("WARNING: Number of training vectors adjusted to $numTrainingVectors")
                        }

                        // TRAIN IVF
                        val trainingVectors = spool.sampleReservoir(numTrainingVectors, taskConfig.downstreamTask.seed)
                        val flat = FloatArray(trainingVectors.size * dim)

                        trainingVectors.forEachIndexed { i, v ->
                            System.arraycopy(v, 0, flat, i * dim, dim)
                        }

                        // Train index and measure time
                        val trainStart = System.nanoTime()
                        faissIndex.trainIvf(flat, trainingVectors.size)
                        val trainEnd = System.nanoTime()
                        faissMetrics.trainingTimeNs = trainEnd - trainStart

                        // Update progress. Start building phase
                        progress.startBuilding()

                        // ADD ALL VECTORS; Build index and measure time
                        val batchSize = faissConfig.batchSize
                        val pending = ArrayList<PendingChunk>(batchSize)
                        var counter = 0
                        spool.streamAll { record ->
                            counter++
                            pending.add(
                                PendingChunk(
                                    docId = record.docId,
                                    chunkId = record.chunkId,
                                    chunkText = record.chunkText,
                                    model = embeddingModel.modelPath,
                                    vec = record.vector
                                )
                            )

                            // When batch full → flush to FAISS + DB
                            if (pending.size == batchSize || counter == spool.recordCount) {
                                // Add chunk to faiss index and SQLite DB and measure time
                                val addStart = System.nanoTime()
                                addChunksAndPersist(pending)
                                val addEnd = System.nanoTime()
                                faissMetrics.addTimeNs.add(addEnd - addStart)
                                pending.clear()
                            }
                        }
                        val buildEnd = System.nanoTime()
                        faissMetrics.buildTimeMs = (buildEnd - buildStart) / 1_000_000L
                    }
                }

                else -> {
                    throw Exception("Unsupported index method: $indexMethod")
                }
            }

            // Set init time
            faissMetrics.initIndexTimeNs = initFaissEnd - initFaissStart


            faissIndex.resetStats()

            // -------------------------------
            // STREAM DOCUMENTS → INDEX
            // -------------------------------

            // Vectors for IVF are already added after training
            if (indexMethod != "ivf") {
                // Update progress. Start building phase
                progress.startBuilding()


                // Build Index and measure time
                val buildStart = System.nanoTime()
                var counter = 0
                val batchSize = faissConfig.batchSize
                val pending = ArrayList<PendingChunk>(batchSize)
                documentStream { docId, text ->
                    counter++

                    // Update progress: Increment documents
                    progress.incrementDocuments()

                    // Chunk document and measure time
                    val chunkStart = System.nanoTime()
                    val chunks = runBlocking { textChunker.chunkText(text) }
                    val chunkEnd = System.nanoTime()
                    faissMetrics.chunkingTimeNs.add(chunkEnd - chunkStart)

                    chunks.forEachIndexed { chunkIndex, chunkText ->
                        // Generate embeddings for chunk and measure time
                        val embeddingStart = System.nanoTime()
                        val vec = runBlocking { embeddingGenerator.generate(chunkText) }
                        val embeddingEnd = System.nanoTime()
                        faissMetrics.embeddingTimeNs.add(embeddingEnd - embeddingStart)

                        pending.add(
                            PendingChunk(
                                docId = docId,
                                chunkId = chunkIndex,
                                chunkText = chunkText,
                                model = embeddingModel.modelPath,
                                vec = vec
                            )
                        )

                        // When batch full → flush to FAISS + DB
                        if (pending.size == batchSize) {
                            // Add chunk to faiss index and SQLite DB and measure time
                            val addStart = System.nanoTime()
                            addChunksAndPersist(pending)
                            val addEnd = System.nanoTime()
                            faissMetrics.addTimeNs.add(addEnd - addStart)
                            pending.clear()
                        }
                    }
                    // Update progress: Increment chunks
                    progress.incrementChunks(chunks.size)
                }
                // Flush remaining chunks after all documents are processed
                if (pending.isNotEmpty()) {
                    val addStart = System.nanoTime()
                    addChunksAndPersist(pending)
                    val addEnd = System.nanoTime()
                    faissMetrics.addTimeNs.add(addEnd - addStart)
                    pending.clear()
                }
                val buildEnd = System.nanoTime()
                faissMetrics.buildTimeMs = (buildEnd - buildStart) / 1_000_000L
            }
        } finally {
            // Close
            textChunker.close()
            db.close()
        }

    }


    private suspend fun benchmarkTask(
        questions: JsonArray,
        embeddingGenerator: EmbeddingGenerator,
        faissIndex: FaissIndex,
        task: DownstreamTask,
        llm: LocalLLM
    ) {
        // Update progress: Start Evaluation
        progress.startEvaluation()

        val k = taskConfig.ragPipeline.faiss.topK
            val hash = getHash(task)
        val db = getDB(hash, context)

        val writer = JsonlWriter(
            file = context.getExternalFilesDir(null)!!
                .resolve(Constants.TASK_RESULTS_DIR)
                .resolve(task.name)
                .resolve(Constants.MAIN_RESULTS_FILE)
        )
        try {
            questions.forEachIndexed { index, question ->
                processQuestion(
                    questionIndex = index,
                    question = question.jsonPrimitive.content,
                    embeddingGenerator = embeddingGenerator,
                    faissIndex = faissIndex,
                    llm = llm,
                    db = db,
                    k = k,
                    writer = writer
                )
                // Update progress: Increment question
                progress.incrementQuestion()
            }

            // Update progress: Complete Evaluation
            progress.completeEvaluation()
        } finally {
            db.close()
            writer.close()
        }
    }

    private suspend fun processQuestion(
        questionIndex: Int,
        question: String,
        embeddingGenerator: EmbeddingGenerator,
        faissIndex: FaissIndex,
        llm: LocalLLM,
        k: Int,
        db: EmbeddingDb,
        writer: JsonlWriter
    ) {
        // Prepare response
        var systemPrompt = ""
        var contextText = ""
        val response = StringBuilder()

        // Create metrics object
        val metrics = GenerationMetrics()
        metrics.requestStartMs = SystemClock.elapsedRealtime()

        try {
            // Generate query embeddings and measure time
            val queryEmbeddingStart = System.nanoTime()
            val queryEmbedding = embeddingGenerator.generate(question)
            val queryEmbeddingEnd = System.nanoTime()
            metrics.queryEmbeddingsMs = (queryEmbeddingEnd - queryEmbeddingStart) / 1_000_000L

            // Prepare buffer
            val dim = queryEmbedding.size
            val buffer = FloatArray(dim)
            System.arraycopy(queryEmbedding, 0, buffer, 0, dim)

            // Retrieve top K chunks and measure time
            val retrieveDocsStart = System.nanoTime()
            val topKChunksVec = faissIndex.query(buffer, 1, k)
            val retrieveDocsEnd = System.nanoTime()
            metrics.retrieveTopKDocsNs = retrieveDocsEnd - retrieveDocsStart

            // Get top K chunk IDs
            val labels = topKChunksVec.labels

            if (labels == null) {
                throw Exception("Faiss returned no results!")
            }

            // Get topDocs text from DB
            val topDocs = labels.map { db.getByFaissId(it)?.chunkText }

            // Prepare context
            contextText = topDocs
                .filterNotNull()
                .mapIndexed { index, doc ->
                    "[${index + 1}] $doc"
                }
                .joinToString("\n\n")

            // Prepare system prompt
            val llmConfig = taskConfig.ragPipeline.llm

            systemPrompt = llmConfig.systemPrompt

            // Generate response
            suspendCancellableCoroutine { cont ->
                llm.generateStreaming(
                    inputText = question,
                    systemPrompt = systemPrompt,
                    contextText = contextText,
                    metrics = metrics,
                    generateUntil = llmConfig.generateUntil,
                    maxTokens = llmConfig.maxTokens,
                    ignoreEos = llmConfig.ignoreEos,
                    intent = PromptIntent.CHAT,
                    onToken = { token ->
                        response.append(token)
                    },
                    onComplete = { finalMetrics ->
                        val metricsResult = finalMetrics.toResult()

                        // Append response to output file
                        writer.append(
                            questionIndex = questionIndex,
                            question = question,
                            systemPrompt = systemPrompt,
                            contextText = contextText,
                            response = response.toString(),
                            error = null,
                            metrics = metricsResult
                        )

                        cont.resume(Unit) { cause, _, _ -> cont.cancel(cause) }
                    },
                    onError = { e ->
                        // Also write error to output file
                        writer.append(
                            questionIndex = questionIndex,
                            question = question,
                            systemPrompt = systemPrompt,
                            contextText = contextText,
                            response = null,
                            error = e.toString(),
                            metrics = null
                        )

                        cont.resume(Unit) { cause, _, _ -> cont.cancel(cause) }
                    }
                )

                cont.invokeOnCancellation {
                    llm.stop()
                }
            }
        } catch (e: Exception) {
            writer.append(
                questionIndex = questionIndex,
                question = question,
                systemPrompt = systemPrompt,
                contextText = contextText,
                response = null,
                error = e.message ?: "unknown error",
                metrics = null
            )
        }
    }
}