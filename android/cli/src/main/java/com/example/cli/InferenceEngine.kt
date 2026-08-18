package com.example.cli

import LlmConfig
import android.content.Context
import java.io.File

interface InferenceEngine {

    enum class Status { OK, FAILED, TIMEOUT }

    data class InitResult(
        val success: Boolean,
        val loadTimeMs: Long,
        val error: String? = null
    )

    data class GenerationRequest(
        val prompt: String,
        val systemPrompt: String = "",
        val contextText: String = "",
        val maxTokens: Int = 256,
        val timeoutMs: Long = 120_000
    )

    data class GenerationResult(
        val status: Status,
        val response: String = "",
        val inputTokens: Int = 0,
        val generatedTokens: Int = 0,
        val ttftMs: Long = 0L,
        val tbtMs: List<Long> = emptyList(),
        val tokenTimestampsMs: List<Long> = emptyList(),
        val overallDurationMs: Long = 0L,
        val decodingSpeedTokensPerSec: Double = 0.0,
        val peakMemBytes: Long = 0L,
        val error: String? = null
    )

    fun isLoaded(): Boolean

    suspend fun initialize(
        context: Context,
        llmConfig: LlmConfig,
        modelDir: File,
        initTimeoutMs: Long = 120_000
    ): InitResult

    suspend fun generate(request: GenerationRequest): GenerationResult

    suspend fun close()

    companion object {
        fun create(engine: String): InferenceEngine {
            return when (engine.lowercase()) {
                "onnx" -> OnnxEngine()
                "litert" -> LiteRTEngine()
                else -> throw IllegalArgumentException("Unknown inference engine: $engine")
            }
        }
    }
}
