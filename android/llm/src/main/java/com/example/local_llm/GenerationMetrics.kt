package com.example.local_llm

import kotlin.Long

data class GenerationMetrics(
    // counts
    var inputTokens: Int = 0,
    var generatedTokens: Int = 0,

    // timestamps (ms)
    var requestStartMs: Long = 0L,
    var prefillEndMs: Long = 0L,
    var firstTokenMs: Long = 0L,
    var generationEndMs: Long = 0L,
    var queryEmbeddingsMs: Long = 0L,
    var retrieveTopKDocsNs: Long = 0L,

    // per-token decode timestamps
    val tokenTimestampsMs: MutableList<Long> = mutableListOf()
) {
    val overallDurationMs: Long
        get() = generationEndMs - requestStartMs

    val ttftMs: Long
        get() = firstTokenMs - requestStartMs

    val decodingDurationMs: Long
        get() = generationEndMs - prefillEndMs

    val decodingSpeedTokensPerSec: Double
        get() =
            if (generatedTokens > 0 && decodingDurationMs > 0)
                generatedTokens * 1000.0 / decodingDurationMs
            else 0.0

    val tbt: List<Long>
        get() =
            tokenTimestampsMs.zipWithNext { a, b -> b - a }

    fun toResult(): GenerationMetricsResult {
        return GenerationMetricsResult(
            inputTokens = inputTokens,
            generatedTokens = generatedTokens,
            prefillTimeMs = prefillEndMs - requestStartMs,

            ttftMs = ttftMs,
            overallDurationMs = overallDurationMs,

            decodingDurationMs = decodingDurationMs,
            decodingSpeedTokensPerSec = decodingSpeedTokensPerSec,

            queryEmbeddingsMs = queryEmbeddingsMs,
            retrieveTopKDocsNs = retrieveTopKDocsNs,

            tbt = tbt.toList()
        )
    }
}

data class GenerationMetricsResult(
    val inputTokens: Int,
    val generatedTokens: Int,
    val prefillTimeMs: Long,

    val ttftMs: Long,
    val overallDurationMs: Long,

    val decodingDurationMs: Long,
    val decodingSpeedTokensPerSec: Double,

    val queryEmbeddingsMs: Long,
    val retrieveTopKDocsNs: Long,

    val tbt: List<Long>
)