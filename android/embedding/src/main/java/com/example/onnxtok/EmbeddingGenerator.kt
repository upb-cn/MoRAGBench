package com.example.onnxtok

import android.content.Context
import com.ml.shubham0204.sentence_embeddings.SentenceEmbedding
import kotlinx.coroutines.*
import java.io.File

/**
 * EmbeddingGenerator provides a simple interface to generate sentence embeddings
 * using an ONNX Runtime model and tokenizer assets.
 */

class EmbeddingGenerator(private val context: Context) : AutoCloseable {

    private var sentenceEmbedding: SentenceEmbedding? = null
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    /**
     * Initializes the ONNX model and tokenizer.
     * Must be called before calling [generate].
     */
    suspend fun initialize(
        embeddingModel: EmbeddingModel,
        backend: String = "cpu", // Should be one of "cpu", "xnnpack", "nnapi"
        normalizeEmbeddings: Boolean = true,
    ) = withContext(Dispatchers.IO) {

        // Read model and tokenizer
        val modelFile = resolveModelFile(context, embeddingModel.modelPath)
        val tokenizerBytes = loadTokenizerBytes(context, embeddingModel.tokenizerPath)

        val se = SentenceEmbedding()
        se.init(
            modelFilepath = modelFile.absolutePath,
            tokenizerBytes = tokenizerBytes,
            useTokenTypeIds = embeddingModel.useTokenTypeIds,
            outputTensorName = embeddingModel.outputTensorName,
            backend = backend,
            normalizeEmbeddings = normalizeEmbeddings
        )
        sentenceEmbedding = se
    }

    private fun isAbsolutePath(path: String): Boolean {
        return path.startsWith("/")
    }

    private fun resolveModelFile(context: Context, path: String): File {
        return if (isAbsolutePath(path)) {
            File(path)
        } else {
            // Copy model from assets to cache (ONNX requires a file path)
            val outFile = File(context.cacheDir, path.substringAfterLast('/'))
            if (!outFile.exists()) {
                context.assets.open(path).use { input ->
                    outFile.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
            }
            outFile
        }
    }

    private fun loadTokenizerBytes(context: Context, path: String): ByteArray {
        return if (isAbsolutePath(path)) {
            File(path).readBytes()
        } else {
            context.assets.open(path).readBytes()
        }
    }

    /**
     * Encodes a given text and returns its embedding vector as FloatArray.
     * [initialize] must be called before using this.
     */
    suspend fun generate(text: String): FloatArray = withContext(Dispatchers.IO) {
        val se = sentenceEmbedding
            ?: throw IllegalStateException("EmbeddingGenerator not initialized. Call initialize() first.")
        se.encode(text)
    }

    /**
     * Releases ONNX resources.
     */
    override fun close() {
        sentenceEmbedding?.close()
        scope.cancel()
    }
}
