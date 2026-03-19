package com.example.local_llm

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import android.os.SystemClock
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.EnumSet
import kotlin.math.exp
import kotlin.random.Random

class OnnxModel(private val context: Context, private val config: ModelConfig) {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession = initializeModel()

    // Initialize ONNX session from asset model path
    private fun initializeModel(): OrtSession {
        val modelFile = loadModelFile(config.modelPath)
        val opts = OrtSession.SessionOptions()
        // opts.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO)

        // Graph optimizations
        try {
            // Use the highest optimization level available
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)

            // How many operations can run in parallel.
            // Thread tuning: leave at least one core free
            val cpuCores = Runtime.getRuntime().availableProcessors()
            val threadsToUse = (cpuCores - 1).coerceAtLeast(1)
            try { opts.setInterOpNumThreads(1) } catch (_: Exception) {}
            try { opts.setIntraOpNumThreads(threadsToUse) } catch (_: Exception) {}

            if (config.backend == "nnapi") {
                val flags = EnumSet.of(NNAPIFlags.USE_FP16) // Will be ignored if unsupported
                opts.addNnapi(flags)
            }

            if (config.backend == "xnnpack") {
                 val numThreads = threadsToUse.toString()
                opts.addXnnpack(mapOf("intra_op_num_threads" to numThreads))
            }

            val createdSession = env.createSession(modelFile.absolutePath, opts)
            return createdSession
        } catch (e: Exception) {
            throw RuntimeException("Failed to create an ONNX Runtime session for provider: ${config.backend}", e)
        }
    }


    // Copy model file from assets to internal storage (required by ONNX runtime)
    private fun loadModelFile(filePath: String): File {
        val file = File(filePath)

        // Case 1: already a real file on disk
        if (file.isAbsolute && file.exists()) {
            return file
        }

        // Case 2: asset → copy to internal storage
        val assetManager = context.applicationContext.assets
        val inputStream = assetManager.open(filePath)

        val outFile = File(context.filesDir, filePath)
        outFile.parentFile?.mkdirs()

        inputStream.use { input ->
            FileOutputStream(outFile).use { output ->
                input.copyTo(output)
            }
        }

        return outFile
    }

    private fun softmaxTopK(
        logits: FloatArray,
        size: Int,
        probs: FloatArray
    ) {
        var max = Float.NEGATIVE_INFINITY
        for (i in 0 until size) {
            if (logits[i] > max) max = logits[i]
        }

        var sum = 0f
        for (i in 0 until size) {
            val e = exp((logits[i] - max).toDouble()).toFloat()
            probs[i] = e
            sum += e
        }

        val invSum = 1f / sum
        for (i in 0 until size) {
            probs[i] *= invSum
        }
    }

    // top-p (nucleus) filter: returns logits copy where tokens outside nucleus are set to -Inf
    private fun applyTopP(
        probs: FloatArray,
        indices: IntArray,
        size: Int,
        topP: Float
    ): Int {
        if (topP >= 1.0f) return size

        // sort topK by probability descending (K is small)
        for (i in 0 until size - 1) {
            for (j in i + 1 until size) {
                if (probs[j] > probs[i]) {
                    val tp = probs[i]; probs[i] = probs[j]; probs[j] = tp
                    val ti = indices[i]; indices[i] = indices[j]; indices[j] = ti
                }
            }
        }

        var cumsum = 0f
        var newSize = 0
        for (i in 0 until size) {
            cumsum += probs[i]
            newSize++
            if (cumsum >= topP) break
        }
        return newSize
    }

    private fun selectTopKHeap(
        logits: FloatArray,
        k: Int,
        outIndices: IntArray,
        outLogits: FloatArray
    ): Int {

        var size = 0

        for (i in logits.indices) {
            val v = logits[i]

            if (size < k) {
                outIndices[size] = i
                outLogits[size] = v
                size++

                // build heap bottom-up
                var j = size - 1
                while (j > 0) {
                    val parent = (j - 1) / 2
                    if (outLogits[parent] <= outLogits[j]) break

                    val tmpL = outLogits[parent]
                    val tmpI = outIndices[parent]

                    outLogits[parent] = outLogits[j]
                    outIndices[parent] = outIndices[j]

                    outLogits[j] = tmpL
                    outIndices[j] = tmpI

                    j = parent
                }
            } else if (v > outLogits[0]) {
                // replace root
                outIndices[0] = i
                outLogits[0] = v

                // heapify down
                var j = 0
                while (true) {
                    val left = 2 * j + 1
                    val right = left + 1
                    var smallest = j

                    if (left < k && outLogits[left] < outLogits[smallest]) {
                        smallest = left
                    }
                    if (right < k && outLogits[right] < outLogits[smallest]) {
                        smallest = right
                    }
                    if (smallest == j) break

                    val tmpL = outLogits[j]
                    val tmpI = outIndices[j]

                    outLogits[j] = outLogits[smallest]
                    outIndices[j] = outIndices[smallest]

                    outLogits[smallest] = tmpL
                    outIndices[smallest] = tmpI

                    j = smallest
                }
            }
        }

        return size
    }


    private fun applyRepetitionPenaltyInPlace(
        logits: FloatArray,
        buffers: SamplerBuffers,
        penalty: Float
    ) {
        if (penalty == 1.0f) return

        for (i in 0 until buffers.seenCount) {
            val token = buffers.seenTokens[i]
            val v = logits[token]
            logits[token] =
                if (v < 0f) v * penalty else v / penalty
        }
    }

    private fun sampleFromProbs(
        probs: FloatArray,
        indices: IntArray,
        size: Int,
        rng: Random
    ): Int {
        val r = rng.nextFloat()
        var cumsum = 0f
        for (i in 0 until size) {
            cumsum += probs[i]
            if (r <= cumsum) {
                return indices[i]
            }
        }
        return indices[size - 1]
    }

    private fun sampleToken(
        logits: FloatArray,
        buffers: SamplerBuffers,
        temperature: Float,
        topK: Int,
        topP: Float,
        repetitionPenalty: Float,
        rng: Random
    ): Int {

        // repetition penalty only on seen tokens
        applyRepetitionPenaltyInPlace(logits, buffers, repetitionPenalty)

        // temperature
        if (temperature != 1.0f) {
            val invT = 1f / temperature
            for (i in logits.indices) logits[i] *= invT
        }

        val size = selectTopKHeap(
            logits,
            topK,
            buffers.topKIndices,
            buffers.topKLogits
        )

        softmaxTopK(
            buffers.topKLogits,
            size,
            buffers.topKProbs
        )

        val finalSize = applyTopP(
            buffers.topKProbs,
            buffers.topKIndices,
            size,
            topP
        )

        return sampleFromProbs(
            buffers.topKProbs,
            buffers.topKIndices,
            finalSize,
            rng
        )
    }

    private fun readLogitsIntoBuffer(
        src: FloatBuffer,
        dst: FloatArray,
        offset: Int,
        size: Int
    ) {
        src.position(offset)
        src.get(dst, 0, size)
    }

    fun initEmptyPastKV(): MutableMap<String, OnnxTensor> {
        val map = mutableMapOf<String, OnnxTensor>()
        repeat(config.numLayers) { layer ->
            listOf("key", "value").forEach { kv ->
                map["past_key_values.$layer.$kv"] =
                    OnnxTensor.createTensor(
                        env,
                        FloatBuffer.allocate(0),
                        longArrayOf(
                            1,
                            config.numKvHeads.toLong(),
                            0,
                            config.headDim.toLong()
                        )
                    )
            }
        }
        return map
    }

    fun currentKvLength(past: Map<String, OnnxTensor>): Int {
        val t = past.values.first()
        val shape = (t.info as TensorInfo).shape
        return shape[2].toInt()
    }

    fun slicePastKV(
        tensor: OnnxTensor,
        keepLast: Int
    ): OnnxTensor {
        val info = tensor.info as TensorInfo
        val shape = info.shape
        val seqLen = shape[2].toInt()

        if (seqLen <= keepLast) {
            val copy = FloatArray(shape[1].toInt() * seqLen * shape[3].toInt())
            tensor.floatBuffer.position(0)
            tensor.floatBuffer.get(copy)
            return OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(copy),
                longArrayOf(1, shape[1], shape[2], shape[3])
            )
        }

        val heads = shape[1].toInt()
        val headDim = shape[3].toInt()
        val start = seqLen - keepLast

        val src = tensor.floatBuffer
        val dst = FloatArray(heads * keepLast * headDim)

        var di = 0
        for (h in 0 until heads)
            for (t in start until seqLen) {
                val off = (h * seqLen + t) * headDim
                src.position(off)
                src.get(dst, di, headDim)
                di += headDim
            }

        return OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(dst),
            longArrayOf(1, heads.toLong(), keepLast.toLong(), headDim.toLong())
        )
    }

    fun prefill(
        inputIds: IntArray,
        past: MutableMap<String, OnnxTensor>
    ): Long {
        var pos = 0L
        var offset = 0

        while (offset < inputIds.size) {
            val end = minOf(offset + config.prefillChunkSize, inputIds.size)
            val chunk = inputIds.sliceArray(offset until end)
            val seqLen = chunk.size

            val inputTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(chunk.map { it.toLong() }.toLongArray()),
                longArrayOf(1, seqLen.toLong())
            )

            val posTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(LongArray(seqLen) { pos + it }),
                longArrayOf(1, seqLen.toLong())
            )

            val pastKvLen = currentKvLength(past)
            val attnLen = minOf(pastKvLen + seqLen, config.kvWindow)

            val attnTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(LongArray(attnLen) { 1L }),
                longArrayOf(1, attnLen.toLong())
            )

            val inputs = mutableMapOf(
                "input_ids" to inputTensor,
                "position_ids" to posTensor,
                "attention_mask" to attnTensor
            ).apply { putAll(past) }

            val outputs = session.run(inputs)

            // skip logits, update KV only
            for (i in 1 until outputs.size()) {
                val tensor = outputs[i] as OnnxTensor
                val layer = (i - 1) / 2
                val kv = if ((i - 1) % 2 == 0) "key" else "value"
                val name = "past_key_values.$layer.$kv"

                val sliced = slicePastKV(tensor, config.kvWindow)
                past[name] = sliced
            }

            // No need to close other tensors, as they will all be closed by this
            outputs.close()

            pos += seqLen
            offset = end
        }

        return pos
    }

    fun decodeStreaming(
        past: MutableMap<String, OnnxTensor>,
        startPosition: Long,
        initialToken: Int,
        shouldStop: () -> Boolean,
        onTokenGenerated: (Int) -> Unit,
        doSample: Boolean,
        temperature: Float,
        topK: Int,
        topP: Float,
        repetitionPenalty: Float
    ) {
        var position = startPosition
        var lastToken = initialToken

        val vocabSize = config.vocabSize
        val samplerBuffers = SamplerBuffers(vocabSize, maxOf(topK, vocabSize))
        val rng = Random.Default

        while (true) {
            if (shouldStop()) return

            val inputTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(longArrayOf(lastToken.toLong())),
                longArrayOf(1, 1)
            )

            val posTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(longArrayOf(position)),
                longArrayOf(1, 1)
            )

            val pastKvLen = currentKvLength(past)
            val attnLen = pastKvLen + 1

            val attnTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(LongArray(attnLen) { 1L }),
                longArrayOf(1, attnLen.toLong())
            )

            val inputs = mutableMapOf(
                "input_ids" to inputTensor,
                "position_ids" to posTensor,
                "attention_mask" to attnTensor
            ).apply { putAll(past) }

            val outputs = session.run(inputs)

            val logitsTensor = outputs[0] as OnnxTensor
            readLogitsIntoBuffer(
                logitsTensor.floatBuffer,
                samplerBuffers.logitsScratch,
                0,
                vocabSize
            )

            val nextToken =
                if (doSample)
                    sampleToken(
                        samplerBuffers.logitsScratch,
                        samplerBuffers,
                        temperature,
                        topK,
                        topP,
                        repetitionPenalty,
                        rng
                    )
                else
                    samplerBuffers.logitsScratch.indices.maxBy {
                        samplerBuffers.logitsScratch[it]
                    }

            onTokenGenerated(nextToken)

            if (samplerBuffers.tokenFreq[nextToken] == 0) {
                if (samplerBuffers.seenCount < samplerBuffers.seenTokens.size) {
                    samplerBuffers.seenTokens[samplerBuffers.seenCount++] = nextToken
                }
            }
            samplerBuffers.tokenFreq[nextToken]++

            lastToken = nextToken
            position++

            for (i in 1 until outputs.size()) {
                val tensor = outputs[i] as OnnxTensor
                val layer = (i - 1) / 2
                val kv = if ((i - 1) % 2 == 0) "key" else "value"
                val name = "past_key_values.$layer.$kv"

                val sliced = slicePastKV(tensor, config.kvWindow)
                past[name] = sliced
            }


            // No need to close other tensors, as they will all be closed by this
            outputs.close()
        }
    }

    fun runInferenceStreamingWithPastKV(
        inputIds: IntArray,
        shouldStop: () -> Boolean,
        doSample: Boolean,
        temperature: Float,
        topK: Int,
        topP: Float,
        repetitionPenalty: Float,
        metrics: GenerationMetrics,
        onTokenGenerated: (Int) -> Unit,
    ) {
        val past = initEmptyPastKV()

        try {
            val startPos = prefill(
                inputIds = inputIds,
                past = past
            )

            // Update metrics. Log prefill end time
            metrics.prefillEndMs = SystemClock.elapsedRealtime()

            val lastPromptToken = inputIds.last()
            var firstTokenSeen = false

            decodeStreaming(
                past = past,
                startPosition = startPos,
                initialToken = lastPromptToken,
                shouldStop = shouldStop,
                onTokenGenerated = { token ->
                    // Update metrics. Log first token time
                    val now = SystemClock.elapsedRealtime()

                    if (!firstTokenSeen) {
                        metrics.firstTokenMs = now
                        firstTokenSeen = true
                    }

                    // Also log token generation time (TBOT) and number
                    // of generated tokens
                    metrics.tokenTimestampsMs.add(now)
                    metrics.generatedTokens++

                    onTokenGenerated(token)
                },
                doSample = doSample,
                temperature = temperature,
                topK = topK,
                topP = topP,
                repetitionPenalty = repetitionPenalty
            )
        } finally {
            // Update metrics. Log generation end time
            metrics.generationEndMs = SystemClock.elapsedRealtime()

            // Free resources
            past.values.forEach { it.close() }
        }
    }

    fun close() {
        try { session.close() } catch (_: Exception) {}
        try { env.close() } catch (_: Exception) {}
    }

}

class SamplerBuffers(
    vocabSize: Int,
    maxTopK: Int
) {
    val logitsScratch = FloatArray(vocabSize)

    // top-k working buffers
    val topKIndices = IntArray(maxTopK)
    val topKLogits = FloatArray(maxTopK)
    val topKProbs = FloatArray(maxTopK)

    // repetition tracking
    val tokenFreq = IntArray(vocabSize)
    val seenTokens = IntArray(1024) // grows if needed
    var seenCount = 0
}
