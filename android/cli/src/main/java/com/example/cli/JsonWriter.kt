package com.example.cli

import com.example.local_llm.GenerationMetricsResult
import java.io.BufferedWriter
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.OutputStreamWriter

class JsonlWriter(file: File) {

    // Resolve file path
    private val writer: BufferedWriter
    private val lock = Any()

    fun String.jsonEscape(): String =
        buildString {
            for (c in this@jsonEscape) {
                when (c) {
                    '\\' -> append("\\\\")
                    '"'  -> append("\\\"")
                    '\n' -> append("\\n")
                    '\r' -> append("\\r")
                    '\t' -> append("\\t")
                    else -> append(c)
                }
            }
        }

    init {
        // Ensure parent directories exist
        val parent = file.parentFile
        if (parent != null && !parent.exists()) {
            if (!parent.mkdirs()) {
                throw IOException("Failed to create directories: ${parent.absolutePath}")
            }
        }

        // Delete previous file if exists
        if (file.exists()) {
            file.delete()
        }

        // Open writer in append mode
        writer = BufferedWriter(
            OutputStreamWriter(
                FileOutputStream(file, true),
                Charsets.UTF_8
            )
        )
    }

    fun append(
        questionIndex: Int,
        question: String,
        systemPrompt: String,
        contextText: String,
        response: String?,
        error: String?,
        metrics: GenerationMetricsResult?
    ) {
        val json = buildString {
            append("{")
            append("\"index\":$questionIndex,")
            append("\"question\":\"${question.jsonEscape()}\",")
            append("\"system_prompt\":\"${systemPrompt.jsonEscape()}\",")
            append("\"context_text\":\"${contextText.jsonEscape()}\",")

            append("\"response\":")
            if (response == null) append("null")
            else append("\"${response.jsonEscape()}\"")
            append(",")

            append("\"error\":")
            if (error == null) append("null")
            else append("\"${error.jsonEscape()}\"")
            append(",")

            append("\"metrics\":")
            if (metrics == null) append("null")
            else append(metrics.toJson())

            append("}")
        }

        appendToFile(json)
    }

    private fun appendToFile(json: String) {
        synchronized(lock) {
            writer.write(json)
            writer.newLine()
            writer.flush()
        }
    }

    private fun GenerationMetricsResult.toJson(): String =
        buildString {
            append("{")
            append("\"input_tokens\":$inputTokens,")
            append("\"generated_tokens\":$generatedTokens,")
            append("\"ttft_ms\":$ttftMs,")
            append("\"prefill_ms\":$prefillTimeMs,")
            append("\"overall_duration_ms\":$overallDurationMs,")
            append("\"decoding_duration_ms\":$decodingDurationMs,")
            append("\"decoding_speed_toks_per_sec\":$decodingSpeedTokensPerSec,")
            append("\"query_embeddings_ms\":$queryEmbeddingsMs,")
            append("\"retrieve_top_k_docs_ns\":$retrieveTopKDocsNs,")
            append("\"tbt\":")
            append(tbt.joinToString(prefix = "[", postfix = "]"))
            append("}")
        }

    fun close() {
        synchronized(lock) {
            writer.close()
        }
    }
}