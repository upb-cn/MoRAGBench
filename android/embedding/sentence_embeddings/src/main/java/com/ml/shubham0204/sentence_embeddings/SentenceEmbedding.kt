package com.ml.shubham0204.sentence_embeddings

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.LongBuffer
import java.util.EnumSet
import kotlin.math.max
import kotlin.math.sqrt

class SentenceEmbedding {
    private lateinit var hfTokenizer: HFTokenizer
    private lateinit var ortEnvironment: OrtEnvironment
    private lateinit var ortSession: OrtSession
    private var useTokenTypeIds: Boolean = false
    private var outputTensorName: String = ""
    private var normalizeEmbedding: Boolean = false

    suspend fun init(
        modelFilepath: String,
        tokenizerBytes: ByteArray,
        useTokenTypeIds: Boolean,
        outputTensorName: String,
        backend: String, // Should be one of "cpu", "xnnpack", "nnapi"
        normalizeEmbeddings: Boolean,
    ) = withContext(Dispatchers.IO) {

        hfTokenizer = HFTokenizer(tokenizerBytes)
        ortEnvironment = OrtEnvironment.getEnvironment()

        val options = OrtSession.SessionOptions()
        // options.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE)

        when (backend) {
            "cpu" -> {
                // nothing, CPU is default
            }

            "xnnpack" -> {
                val numThreads = Runtime.getRuntime().availableProcessors().toString()
                options.addXnnpack(mapOf("intra_op_num_threads" to numThreads))
            }

            "nnapi" -> {
                options.addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
            }
        }
        ortSession = ortEnvironment.createSession(modelFilepath, options)

        this@SentenceEmbedding.useTokenTypeIds = useTokenTypeIds
        this@SentenceEmbedding.outputTensorName = outputTensorName
        this@SentenceEmbedding.normalizeEmbedding = normalizeEmbeddings
    }

    suspend fun encode(sentence: String): FloatArray =
        withContext(Dispatchers.IO) {

            val result = hfTokenizer.tokenize(sentence)

            // Keep references so we can always close them
            val tensors = mutableListOf<OnnxTensor>()

            try {
                val inputTensorMap = HashMap<String, OnnxTensor>(3)

                fun tensor(name: String, data: LongArray): OnnxTensor {
                    return OnnxTensor.createTensor(
                        ortEnvironment,
                        LongBuffer.wrap(data),
                        longArrayOf(1, data.size.toLong())
                    ).also {
                        tensors += it
                        inputTensorMap[name] = it
                    }
                }

                tensor("input_ids", result.ids)
                tensor("attention_mask", result.attentionMask)

                if (useTokenTypeIds) {
                    tensor("token_type_ids", result.tokenTypeIds)
                }

                ortSession.run(inputTensorMap).use { outputs ->
                    val outputValue = outputs[0].value

                    val pooledEmbedding: FloatArray = when (outputValue) {
                        is Array<*> -> when {
                            // [1, seq_len, hidden]
                            outputValue.isNotEmpty()
                                    && outputValue[0] is Array<*>
                                    && (outputValue[0] as Array<*>)[0] is FloatArray -> {
                                val tokenEmbeddings3D =
                                    outputValue as Array<Array<FloatArray>>
                                meanPooling(tokenEmbeddings3D[0], result.attentionMask)
                            }

                            // [1, hidden]
                            outputValue.isNotEmpty()
                                    && outputValue[0] is FloatArray -> {
                                (outputValue as Array<FloatArray>)[0]
                            }

                            else -> error("Unexpected output shape: ${outputValue::class}")
                        }

                        else -> error("Unexpected output type: ${outputValue::class}")
                    }

                    if (normalizeEmbedding) normalize(pooledEmbedding) else pooledEmbedding
                }
            } finally {
                // ALWAYS free native memory
                tensors.forEach { it.close() }
            }
        }


    private fun meanPooling(
        tokenEmbeddings: Array<FloatArray>,
        attentionMask: LongArray,
    ): FloatArray {
        var pooledEmbeddings = FloatArray(tokenEmbeddings[0].size) { 0f }
        var validTokenCount = 0

        tokenEmbeddings
            .filterIndexed { index, _ -> attentionMask[index] == 1L }
            .forEachIndexed { index, token ->
                validTokenCount++
                token.forEachIndexed { j, value ->
                    pooledEmbeddings[j] += value
                }
            }

        // Avoid division by zero
        val divisor = max(validTokenCount, 1)
        pooledEmbeddings = pooledEmbeddings.map { it / divisor }.toFloatArray()

        return pooledEmbeddings
    }

    // Function to normalize embeddings
    private fun normalize(embeddings: FloatArray): FloatArray {
        // Calculate the L2 norm (Euclidean norm)
        val norm = sqrt(embeddings.sumOf { it * it.toDouble() }).toFloat()
        // Normalize each embedding by dividing by the norm
        return embeddings.map { it / norm }.toFloatArray()
    }

    fun close() {
        ortSession.close()
        ortEnvironment.close()
        hfTokenizer.close()
    }
}
