package com.example.cli

import ANNConfig
import ANNDataset
import android.content.Context
import com.example.cli.CacheManager.buildANNSignature
import com.example.cli.CacheManager.hashSignature
import com.example.cli.Progress.ANNProgress
import com.example.cli.Progress.ProgressPhase
import com.example.cli.Progress.State
import com.example.faiss.FaissIndex
import kotlinx.coroutines.*
import java.io.File
import kotlin.random.Random

class ANNBenchmark(private val context: Context) {

    private val benchmarkScope = CoroutineScope(
        SupervisorJob() + Dispatchers.Default
    )
    private var benchmarkJob: Job? = null

    private val annConfig: ANNConfig
        get() {
            val parser = Parser(context)
            return parser.readAnnConfig()
        }

    var progress: ANNProgress = ANNProgress()
    val overallMetrics: OverallMetrics = OverallMetrics()

    fun getHash(dataset: ANNDataset): String {
        val signature = buildANNSignature(annConfig, dataset)
        val hash = hashSignature(signature)

        return hash
    }

    fun stop() {
        benchmarkJob?.cancel()
        benchmarkJob = null
        benchmarkScope.cancel()
    }

    fun start() {
        val datasetName = annConfig.annDataset.name

        // Init Logger
        val logDir = context.getExternalFilesDir(null)!!
            .resolve(Constants.ANN_RESULTS_DIR)
            .resolve(datasetName)
        FileLogger.init(logDir)

        // Check if progress is initialized
        val progressSnapshot = progress.getSnapshot()
        if (progressSnapshot != null && progressSnapshot.overallState == State.RUNNING) return

        // Record start time
        val benchmarkTimeStart = System.nanoTime()

        // ****** Read ANN dataset ******
        val vectorsDir = context.getExternalFilesDir(null)!!
            .resolve(Constants.ANN_DATASET_DIR)
            .resolve(datasetName)
        val store = VectorStore(vectorsDir)

        benchmarkJob = benchmarkScope.launch {
            try {
                // Init progress
                val annDataset = annConfig.annDataset
                progress.init(annDataset)

                // Start metrics engine
                val hardwareMetricsEngine = HardwareMetricsEngine(
                    context = context,
                    scope = benchmarkScope,
                    intervalMs = 20L
                )
                hardwareMetricsEngine.start()

                runBenchmark(annDataset, store)

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
                    .resolve(Constants.ANN_RESULTS_DIR)
                    .resolve(datasetName)
                    .resolve(Constants.OVERALL_METRICS_FILE)
                writeMetricsToFile(overallMetrics, metricsFile)

                // Save hardware metrics JSON
                val hardwareMetricsFile = context.getExternalFilesDir(null)!!
                    .resolve(Constants.ANN_RESULTS_DIR)
                    .resolve(datasetName)
                    .resolve(Constants.HARDWARE_METRICS_FILE)
                hardwareMetricsEngine.writeMetricsJson(metricsResult, hardwareMetricsFile)

            } catch (e: Exception) {
                // Update progress: Fail
                progress.fail(e.stackTraceToString())
            } finally {
                store.close()
            }
        }
    }

    private fun runBenchmark(dataset: ANNDataset, store: VectorStore) {
        // Update progress: Start dataset
        progress.startDataset()

        val (faissIndex, faissMetrics) = initFaissIndex(
            dataset = dataset,
            store = store
        )

        // Update progress: Change phase
        progress.setPhase(ProgressPhase.EVALUATING)

        // ****** Benchmark Task ******
        benchmarkTask(
            store = store,
            faissIndex = faissIndex,
            dataset = dataset
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
            .resolve(Constants.ANN_RESULTS_DIR)
            .resolve(dataset.name)
            .resolve(Constants.FAISS_SETUP_FILE)
        writeMetricsToFile(faissMetrics, faissMetricsFile)

        // Update progress: End dataset
        progress.completeDataset()

        // Clear faiss index
        faissIndex.clear()
    }

    private fun initFaissIndex(
        dataset: ANNDataset,
        store: VectorStore,
    ): Pair<FaissIndex, FaissMetrics> {
        // -------------------------------
        // READ/WRITE FAISS FROM/TO DISK
        // -------------------------------

        // Create metrics instance
        val faissMetrics = FaissMetrics()

        // Update progress: Start Initializing Index
        progress.initIndex()

        // Init faissIndex instance
        val faissIndex = FaissIndex()

        // Retrieve cache dir. Make sure it exists
        val cacheDir = context.getExternalFilesDir(null)!!.resolve(Constants.ANN_CACHE_DIR)
        if (!cacheDir.exists()) {
            cacheDir.mkdirs()
        }

        // Build signature and hash
        val hash = getHash(dataset)

        // Derive index path
        val indexPath = File(cacheDir, "faiss_$hash.index")

        if (annConfig.faiss.useCache && indexPath.exists()) {
            // If index exists, load it from disk

            // Update progress: Set source to cache
            progress.setIndexSourceToCache(store.trainCount)

            // Load from disk and measure time
            val loadStart = System.nanoTime()
            faissIndex.loadFrom(indexPath.absolutePath)
            val loadEnd = System.nanoTime()
            faissMetrics.loadTimeNs = loadEnd - loadStart

            // Set index method
            // TODO: This is not ideal. It should be read from native file
            faissIndex.setMethod(annConfig.faiss.method)
        } else {
            // Else, build index from scratch

            // Update progress: Set source to scratch
            progress.setIndexSourceToScratch(store.trainCount)

            buildFaissIndex(
                faissIndex = faissIndex,
                store = store,
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

    private fun buildFaissIndex(
        store: VectorStore,
        faissIndex: FaissIndex,
        faissMetrics: FaissMetrics
    ) {
        val faissConfig = annConfig.faiss
        val dim = store.dim
        val distanceMetric = FaissIndex.DistanceMetric.fromString(faissConfig.metric)
        val indexMethod = faissConfig.method

        // -------------------------------
        // INIT INDEX
        // -------------------------------

        // Start init time
        val initFaissStart = System.nanoTime()
        var initFaissEnd: Long

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
                // START TRAINING
                // -------------------------------

                // Update progress: Training task
                progress.startTraining()

                // Clamp training count to a sensible value
                if (numTrainingVectors > store.trainCount) {
                    numTrainingVectors = store.trainCount
                    LiveLogger.info("WARNING: Number of training vectors adjusted to $numTrainingVectors")
                }

                // RNG with fixed seed for reproducibility
                val rng = Random(annConfig.annDataset.seed)

                // Sample unique indices
                val sampledIndices = IntArray(numTrainingVectors)
                val used = HashSet<Int>(numTrainingVectors)

                var i = 0
                while (i < numTrainingVectors) {
                    val idx = rng.nextInt(store.trainCount)
                    if (used.add(idx)) {
                        sampledIndices[i] = idx
                        i++
                    }
                }

                // Flatten sampled vectors
                val flat = FloatArray(numTrainingVectors * dim)

                sampledIndices.forEachIndexed { j, idx ->
                    store.readTrainVectorInto(idx, flat, j * dim)
                }

                // Train index and measure time
                val trainStart = System.nanoTime()
                faissIndex.trainIvf(flat, numTrainingVectors)
                val trainEnd = System.nanoTime()
                faissMetrics.trainingTimeNs = trainEnd - trainStart
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

        // Update progress. Start building phase
        progress.startBuilding()

        // Build Index and measure time
        val buildStart = System.nanoTime()

        val batchSize = annConfig.faiss.batchSize
        val batchBuffer = FloatArray(batchSize * dim)
        val tmp = FloatArray(dim)
        var cursor = 0
        while (cursor < store.trainCount) {
            val currentBatch = minOf(batchSize, store.trainCount - cursor)

            // Update progress: Increment documents
            if (cursor % batchSize == 0) progress.incrementTrainVectors(currentBatch)

            // Fill one big contiguous buffer
            var offset = 0
            for (i in 0 until currentBatch) {
                // Convert vec to FloatArray
                val readStart = System.nanoTime()
                store.readTrainVector(cursor + i, tmp)
                val readEnd = System.nanoTime()
                faissMetrics.readTimeNs.add(readEnd - readStart)

                System.arraycopy(tmp, 0, batchBuffer, offset, dim)
                offset += dim
            }

            // Add vector to faiss index and measure time
            val addStart = System.nanoTime()
            faissIndex.add(batchBuffer, currentBatch)
            val addEnd = System.nanoTime()
            faissMetrics.addTimeNs.add(addEnd - addStart)

            cursor += currentBatch
        }


        val buildEnd = System.nanoTime()
        faissMetrics.buildTimeMs = (buildEnd - buildStart) / 1_000_000L
    }

    private fun benchmarkTask(
        store: VectorStore,
        faissIndex: FaissIndex,
        dataset: ANNDataset
    ) {
        // Update progress: Start Evaluation
        progress.startEvaluation()

        val k = annConfig.faiss.topK

        // Prepare writer
        val resultsFile = context.getExternalFilesDir(null)!!
            .resolve(Constants.ANN_RESULTS_DIR)
            .resolve(dataset.name)
            .resolve(Constants.ANN_RESULTS_FILE)
        val jsonWriter = ResultJsonStreamWriter(
            file = resultsFile,
            prettyPrint = false,
            autoFlush = true
        )

        try {
            val testTimeStart = System.nanoTime()
            val dim = store.dim
            val batchSize = annConfig.faiss.batchSize
            val buffer = FloatArray(batchSize * dim)
            val tmp = FloatArray(dim)

            var cursor = 0
            // Loop over test queries + ground truth
            while (cursor < store.testCount) {
                val currentBatch = minOf(batchSize, store.testCount - cursor)

                // Update progress: Increment question
                if (cursor % batchSize == 0) progress.incrementTestVector(currentBatch)

                // Fill contiguous query buffer
                var offset = 0
                for (i in 0 until currentBatch) {
                    store.readTestVector(cursor + i, tmp)
                    System.arraycopy(tmp, 0, buffer, offset, dim)
                    offset += dim
                }

                // Retrieve top K neighbors and measure time
                val queryStart = System.nanoTime()
                val results = faissIndex.query(buffer, currentBatch, k)
                val queryEnd = System.nanoTime()

                val labels = results.labels
                val distances = results.distances

                // Time per query (so metrics stay meaningful)
                val timePerQuery = (queryEnd - queryStart) / currentBatch

                // Split back per query results
                for (i in 0 until currentBatch) {
                    val base = i * k

                    val neighbors = labels?.copyOfRange(base, base + k)
                    val dists = distances?.copyOfRange(base, base + k)

                    jsonWriter.write(
                        id = cursor + i,
                        processingTimeNs = timePerQuery,
                        neighbors = neighbors,
                        distances = dists
                    )
                }

                cursor += currentBatch
            }
            val testTimeEnd = System.nanoTime()
            overallMetrics.testTimeMs = (testTimeEnd - testTimeStart) / 1_000_000L

            // Update progress: Complete Evaluation
            progress.completeEvaluation()
        } finally {
            jsonWriter.close()
        }
    }
}