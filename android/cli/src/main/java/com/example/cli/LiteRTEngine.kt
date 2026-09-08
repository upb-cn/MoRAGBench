package com.example.cli

import LlmConfig
import android.content.Context
import android.os.Debug
import android.os.SystemClock
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.SamplerConfig
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import java.io.File

@OptIn(ExperimentalApi::class)
class LiteRTEngine : InferenceEngine {

    private val mutex = Mutex()
    private var engine: Engine? = null
    private var llmConfig: LlmConfig? = null

    override fun isLoaded(): Boolean = engine?.isInitialized() == true

    override suspend fun initialize(
        context: Context,
        llmConfig: LlmConfig,
        modelDir: File,
        initTimeoutMs: Long
    ): InferenceEngine.InitResult = mutex.withLock {
        if (engine != null) {
            return@withLock InferenceEngine.InitResult(success = true, loadTimeMs = 0L)
        }

        val modelFile = resolveModelFile(context, modelDir)
            ?: return@withLock InferenceEngine.InitResult(
                success = false,
                loadTimeMs = 0L,
                error = "LiteRT LM model not found: expected $LITE_RT_MODEL_FILENAME " +
                    "under ${modelDir.absolutePath} or ${context.getExternalFilesDir(null)?.absolutePath}"
            )

        val config = EngineConfig(
            modelPath = modelFile.absolutePath,
            backend = mapBackend(llmConfig.backend)
        )
        val newEngine = Engine(config)
        val loadStart = SystemClock.elapsedRealtime()
        try {
            withContext(Dispatchers.IO) {
                withTimeout(initTimeoutMs) { newEngine.initialize() }
            }
        } catch (e: TimeoutCancellationException) {
            runCatching { newEngine.close() }
            return@withLock InferenceEngine.InitResult(
                success = false,
                loadTimeMs = SystemClock.elapsedRealtime() - loadStart,
                error = "LiteRT init timed out after ${initTimeoutMs}ms"
            )
        } catch (e: CancellationException) {
            runCatching { newEngine.close() }
            throw e
        } catch (e: Exception) {
            runCatching { newEngine.close() }
            return@withLock InferenceEngine.InitResult(
                success = false,
                loadTimeMs = SystemClock.elapsedRealtime() - loadStart,
                error = "LiteRT init failed: ${e.message}"
            )
        }

        engine = newEngine
        this.llmConfig = llmConfig
        InferenceEngine.InitResult(
            success = true,
            loadTimeMs = SystemClock.elapsedRealtime() - loadStart
        )
    }

    override suspend fun generate(
        request: InferenceEngine.GenerationRequest
    ): InferenceEngine.GenerationResult {
        val current = engine
        if (current == null) {
            return InferenceEngine.GenerationResult(
                status = InferenceEngine.Status.FAILED,
                error = "Engine not initialized. Call initialize() first."
            )
        }

        return mutex.withLock {
            val result = StringBuilder()
            val tokenTimestampsMs = mutableListOf<Long>()
            val requestStart = SystemClock.elapsedRealtime()
            val nativeBefore = Debug.getNativeHeapAllocatedSize()

            var timedOut = false
            var failure: String? = null
            var tokensBefore = 0
            var tokensAfter: Int? = null
            val conversation = try {
                withContext(Dispatchers.IO) {
                    current.createConversation(buildConversationConfig(request))
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                failure = "Conversation create failed: ${e.message ?: e::class.java.simpleName}"
                null
            }

            if (conversation != null) {
                tokensBefore = runCatching { conversation.getTokenCount() }.getOrNull() ?: 0
                try {
                    withTimeout(request.timeoutMs) {
                        withContext(Dispatchers.IO) {
                            conversation.sendMessageAsync(request.prompt).collect { message ->
                                tokenTimestampsMs.add(SystemClock.elapsedRealtime())
                                result.append(message.contents.toString())
                            }
                        }
                    }
                    tokensAfter = runCatching { conversation.getTokenCount() }.getOrNull()
                } catch (e: TimeoutCancellationException) {
                    timedOut = true
                    failure = "Generation timed out after ${request.timeoutMs}ms"
                    runCatching { conversation.cancelProcess() }
                } catch (e: CancellationException) {
                    runCatching { conversation.close() }
                    throw e
                } catch (e: Exception) {
                    failure = e.message ?: e::class.java.simpleName
                } finally {
                    runCatching { conversation.close() }
                }
            }

            val status = when {
                timedOut -> InferenceEngine.Status.TIMEOUT
                failure != null -> InferenceEngine.Status.FAILED
                else -> InferenceEngine.Status.OK
            }

            val generatedTokens = tokenTimestampsMs.size
            val inputTokens = if (tokensAfter != null)
                maxOf(0, tokensAfter - tokensBefore - generatedTokens)
            else 0
            val overallDurationMs = SystemClock.elapsedRealtime() - requestStart
            val ttftMs = tokenTimestampsMs.firstOrNull()?.let { it - requestStart } ?: 0L
            val tbtMs = tokenTimestampsMs.zipWithNext { a, b -> b - a }
            val decodeDurationMs = tokenTimestampsMs.lastOrNull()
                ?.let { it - tokenTimestampsMs.first() } ?: 0L

            InferenceEngine.GenerationResult(
                status = status,
                response = result.toString(),
                inputTokens = inputTokens,
                generatedTokens = generatedTokens,
                ttftMs = ttftMs,
                tbtMs = tbtMs,
                tokenTimestampsMs = tokenTimestampsMs,
                overallDurationMs = overallDurationMs,
                decodingSpeedTokensPerSec = if (decodeDurationMs > 0 && generatedTokens > 1)
                    (generatedTokens - 1).toDouble() / (decodeDurationMs / 1000.0) else 0.0,
                peakMemBytes = maxOf(nativeBefore, Debug.getNativeHeapAllocatedSize()),
                error = failure
            )
        }
    }

    override suspend fun close() {
        mutex.withLock {
            engine?.close()
            engine = null
            llmConfig = null
        }
    }

    private fun buildConversationConfig(
        request: InferenceEngine.GenerationRequest
    ): ConversationConfig {
        val cfg = llmConfig
        val systemPrompt = request.systemPrompt.ifBlank { cfg?.systemPrompt.orEmpty() }
        val samplerConfig = if (cfg?.useSampling == true) {
            SamplerConfig(
                topK = cfg.topK,
                topP = cfg.topP,
                temperature = cfg.temp
            )
        } else {
            null
        }
        return ConversationConfig(
            systemInstruction = Contents.of(systemPrompt),
            samplerConfig = samplerConfig,
            maxOutputToken = request.maxTokens
        )
    }

    private fun resolveModelFile(context: Context, modelDir: File): File? {
        val inModelDir = modelDir.resolve(LITE_RT_MODEL_FILENAME)
        if (inModelDir.exists()) {
            return inModelDir
        }
        val inExternalRoot = context.getExternalFilesDir(null)
            ?.resolve(LITE_RT_MODEL_FILENAME)
        if (inExternalRoot != null && inExternalRoot.exists()) {
            return inExternalRoot
        }
        return null
    }

    private fun mapBackend(name: String): Backend {
        return when (name.lowercase()) {
            "gpu" -> Backend.GPU()
            "npu", "nnapi" -> Backend.NPU()
            else -> Backend.CPU()
        }
    }

    companion object {
        const val LITE_RT_MODEL_FILENAME =
            "Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.litertlm"
    }
}
