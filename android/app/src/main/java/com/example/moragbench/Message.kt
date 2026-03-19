package com.example.moragbench

data class Message(
    val content: String,
    val isUser: Boolean,
    val isStreaming: Boolean = false,
    val timeMillis: Long = System.nanoTime() * 1_000_000,
    val metrics: MessageMetrics? = null // assistant-only when finished
)
data class MessageMetrics(
    val tokensPerSec: Double,
    val durationSec: Double
)