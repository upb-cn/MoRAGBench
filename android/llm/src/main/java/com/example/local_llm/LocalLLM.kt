package com.example.local_llm

import android.content.Context
import android.util.Log
import kotlinx.coroutines.*
import java.io.File

class LocalLLM(private val context: Context, private val config: ModelConfig) {

    private val inferenceScope = CoroutineScope(Dispatchers.IO)
    private var inferenceJob: Job? = null
    private var stopGeneration: Boolean = false

    private lateinit var tokenizer: TokenizerBridge
    private lateinit var onnxModel: OnnxModel
    private lateinit var promptBuilder: PromptBuilder

    var isInitialized = false
        private set

    /**
     * Initializes tokenizer and model. Must be called before generateStreaming().
     */
    private fun resolveTokenizerSource(
        path: String
    ): TokenizerSource {
        val file = File(path)

        return if (file.isAbsolute && file.exists()) {
            TokenizerSource.File(file.absolutePath)
        } else {
            TokenizerSource.Assets(path)
        }
    }

    suspend fun initialize() {
        withContext(Dispatchers.IO) {
            val tokenizerSource = resolveTokenizerSource(config.tokenizerPath)
            tokenizer = TokenizerBridge(context, tokenizerSource)
            promptBuilder = PromptBuilder(tokenizer, config)
            onnxModel = OnnxModel(context, config)
            isInitialized = true
        }
    }

    fun generateStreaming(
        inputText: String,
        systemPrompt: String,
        contextText: String,
        metrics: GenerationMetrics,
        onToken: (String) -> Unit,
        onComplete: (GenerationMetrics) -> Unit,
        onError: (Throwable) -> Unit,
        maxTokens: Int = 512,
        generateUntil: List<String>? = null,
        ignoreEos: Boolean = true,
        intent: PromptIntent = PromptIntent.CHAT
    ) {
        if (!isInitialized) {
            onError(IllegalStateException("LocalLLM not initialized. Call initialize() first."))
            return
        }

        // Restart flag
        stopGeneration = false

        // Read params from config
        val doSample: Boolean = config.doSample
        val temperature: Float = config.temperature
        val topP: Float = config.topP
        val topK: Int = config.topK
        val repetitionPenalty: Float = config.repetitionPenalty


        val userPrompt = """
            $inputText
            
            Documents:
            $contextText
        """.trimIndent()

        val messages = mutableListOf(
            Message(userPrompt, isUser = true)
        )

        val inputIds = promptBuilder.buildPromptTokens(messages, intent, systemPrompt, maxTokens)

        // Update metrics. Log number of input tokens
        metrics.inputTokens = inputIds.size

        inferenceJob = inferenceScope.launch {
            try {
                var tokenCounter = 0
                // Using this because some "until" values are more than one token
                val generatedText = StringBuilder()

                onnxModel.runInferenceStreamingWithPastKV(
                    inputIds = inputIds,
                    shouldStop = { stopGeneration },
                    doSample = doSample,
                    topK = topK,
                    topP = topP,
                    temperature = temperature,
                    repetitionPenalty = repetitionPenalty,
                    metrics = metrics,
                    onTokenGenerated = { tokenId ->
                        // 1) EOS handling
                        if ((!ignoreEos) && tokenId in config.eosTokenIds) {
                            stop()
                            return@runInferenceStreamingWithPastKV
                        }

                        val tokenStr = tokenizer.decode(intArrayOf(tokenId), skipSpecialTokens=true)

                        // Skip initial prefix tokens (Qwen3 quirk)
                        if (!(config.modelName.equals("Qwen3", ignoreCase = true) && tokenCounter < 4)) {
                            onToken(tokenStr)
                            generatedText.append(tokenStr)
                        }

                        tokenCounter++

                        // 2) max_tokens
                        if (tokenCounter >= maxTokens) {
                            stop()
                            return@runInferenceStreamingWithPastKV
                        }

                        // 3) generate_until
                        if (generateUntil != null) {
                            for (stopSeq in generateUntil) {
                                if (generatedText.endsWith(stopSeq)) {
                                    stop()
                                    return@runInferenceStreamingWithPastKV
                                }
                            }
                        }
                    }
                )

                withContext(Dispatchers.Main) { onComplete(metrics) }

            } catch (e: Exception) {
                withContext(Dispatchers.Main) { onError(e) }
            }
        }
    }

    /**
     * Stops current inference.
     */
    fun stop() {
        stopGeneration = true
    }

    /**
     * Frees model resources and cancels all coroutines.
     */
    fun shutdown() {
        stop()

        try {
            if (this::onnxModel.isInitialized) {
                onnxModel.close()
            }
        } catch (e: Exception) {
            Log.e("LocalLLM", "Error closing ONNX model", e)
        }

        inferenceScope.cancel()
        inferenceJob?.cancel()
    }
}
