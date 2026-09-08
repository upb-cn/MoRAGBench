package com.example.cli

import LlmConfig
import android.content.Context
import android.os.Debug
import android.os.SystemClock
import com.example.local_llm.GenerationMetrics
import com.example.local_llm.LocalLLM
import com.example.local_llm.PromptIntent
import com.example.local_llm.TokenizerSource
import com.example.shared.ModelPathOverrides
import com.example.shared.SupportedLLMs
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withTimeout
import java.io.File
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

class OnnxEngine : InferenceEngine {

    private val mutex = Mutex()
    private var llm: LocalLLM? = null
    private var llmConfig: LlmConfig? = null

    override fun isLoaded(): Boolean = llm != null

    override suspend fun initialize(
        context: Context,
        llmConfig: LlmConfig,
        modelDir: File,
        initTimeoutMs: Long
    ): InferenceEngine.InitResult = mutex.withLock {
        if (llm != null) {
            return@withLock InferenceEngine.InitResult(success = true, loadTimeMs = 0L)
        }

        val modelPath = modelDir.resolve("model.onnx").canonicalPath
        val tokenizerPath = modelDir.resolve("tokenizer.json").canonicalPath

        val config = SupportedLLMs.findByName(
            context,
            "${llmConfig.modelName}-${llmConfig.dtype}",
            overrides = ModelPathOverrides(
                modelPath = modelPath,
                tokenizer = TokenizerSource.File(tokenizerPath),
            )
        )
        config.backend = llmConfig.backend
        config.temperature = llmConfig.temp.toFloat()
        config.topP = llmConfig.topP.toFloat()
        config.topK = llmConfig.topK
        config.doSample = llmConfig.useSampling
        config.repetitionPenalty = llmConfig.repetitionPenalty.toFloat()
        config.defaultSystemPrompt = llmConfig.systemPrompt
        config.kvWindow = llmConfig.kvWindow
        config.prefillChunkSize = llmConfig.prefillChunkSize

        val newLlm = LocalLLM(context, config)
        val loadStart = SystemClock.elapsedRealtime()
        try {
            withTimeout(initTimeoutMs) { newLlm.initialize() }
        } catch (e: TimeoutCancellationException) {
            newLlm.shutdown()
            return@withLock InferenceEngine.InitResult(
                success = false,
                loadTimeMs = SystemClock.elapsedRealtime() - loadStart,
                error = "LLM init timed out after ${initTimeoutMs}ms"
            )
        } catch (e: Exception) {
            newLlm.shutdown()
            return@withLock InferenceEngine.InitResult(
                success = false,
                loadTimeMs = SystemClock.elapsedRealtime() - loadStart,
                error = "LLM init failed: ${e.message}"
            )
        }

        llm = newLlm
        this.llmConfig = llmConfig
        InferenceEngine.InitResult(
            success = true,
            loadTimeMs = SystemClock.elapsedRealtime() - loadStart
        )
    }

    override suspend fun generate(
        request: InferenceEngine.GenerationRequest
    ): InferenceEngine.GenerationResult {
        val current = llm
        if (current == null) {
            return InferenceEngine.GenerationResult(
                status = InferenceEngine.Status.FAILED,
                error = "Engine not initialized. Call initialize() first."
            )
        }

        return mutex.withLock {
            val result = StringBuilder()
            val metrics = GenerationMetrics()
            metrics.requestStartMs = SystemClock.elapsedRealtime()
            val nativeBefore = Debug.getNativeHeapAllocatedSize()

            var timedOut = false
            var failure: String? = null
            try {
                withTimeout(request.timeoutMs) {
                    suspendCancellableCoroutine { cont ->
                        current.generateStreaming(
                            inputText = request.prompt,
                            systemPrompt = request.systemPrompt,
                            contextText = request.contextText,
                            metrics = metrics,
                            maxTokens = request.maxTokens,
                            generateUntil = llmConfig?.generateUntil,
                            ignoreEos = llmConfig?.ignoreEos ?: false,
                            intent = PromptIntent.CHAT,
                            onToken = { token -> result.append(token) },
                            onComplete = { cont.resume(Unit) },
                            onError = { e -> cont.resumeWithException(e) }
                        )
                        cont.invokeOnCancellation { current.stop() }
                    }
                }
            } catch (e: TimeoutCancellationException) {
                timedOut = true
                failure = "Generation timed out after ${request.timeoutMs}ms"
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                failure = e.message ?: e::class.java.simpleName
            }

            val metricsResult = metrics.toResult()
            val status = when {
                timedOut -> InferenceEngine.Status.TIMEOUT
                failure != null -> InferenceEngine.Status.FAILED
                else -> InferenceEngine.Status.OK
            }

            InferenceEngine.GenerationResult(
                status = status,
                response = result.toString(),
                inputTokens = metricsResult.inputTokens,
                generatedTokens = metricsResult.generatedTokens,
                ttftMs = metricsResult.ttftMs,
                tbtMs = metricsResult.tbt,
                tokenTimestampsMs = metrics.tokenTimestampsMs.toList(),
                overallDurationMs = maxOf(metricsResult.overallDurationMs, 0L),
                decodingSpeedTokensPerSec = metricsResult.decodingSpeedTokensPerSec,
                peakMemBytes = maxOf(nativeBefore, Debug.getNativeHeapAllocatedSize()),
                error = failure
            )
        }
    }

    override suspend fun close() {
        mutex.withLock {
            llm?.shutdown()
            llm = null
            llmConfig = null
        }
    }
}
