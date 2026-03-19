package com.example.local_llm

// Token IDs used to mark system/user/assistant roles and boundaries in prompts
data class RoleTokenIds(
    val systemStart: List<Int>,     // Tokens prepended before system prompt
    val userStart: List<Int>,       // Tokens prepended before user message
    val assistantStart: List<Int>,  // Tokens prepended before model/assistant response
    val endToken: Int               // Token appended at the end of each role block
)

// Main configuration class for defining model behavior, structure, and prompting format
data class ModelConfig(
    val modelName: String,                       // Display name of the model (e.g., "Qwen2_5")
    val modelPath: String = "llm/qwen2.5-0.5B_int8/model.onnx",        // File path to the ONNX model inside assets
    val tokenizerPath: String = "llm/qwen2.5-0.5B_int8/tokenizer.json",        // File path to the ONNX model inside assets
    var backend: String = "cpu",          // Execution backend: CPU, XNNPACK, or NNAPI

    // Generation params              // How to construct the input prompt
    val eosTokenIds: Set<Int>,                   // Set of token IDs that signal end-of-sequence
    var temperature: Float = 0.8f,
    var topP: Float = 0.95f,
    var topK: Int = 0,
    var doSample: Boolean = true,                // if false, greedy decoding
    var repetitionPenalty: Float = 1.0f,
    var defaultSystemPrompt: String = "You are a helpful assistant. Use the following retrieved documents to answer the user's query",             // Fallback system prompt if none is provided
    var kvWindow: Int = 2048, // KV slicing window
    var prefillChunkSize: Int = 512, // Prefill chunk size (per iteration)

    // Model meta
    val numLayers: Int,                           // Number of transformer layers
    val numKvHeads: Int,                          // Number of key/value heads (can be < attention heads)
    val headDim: Int,                             // Dimensionality of each attention head
    val batchSize: Int,                           // Number of inputs processed together (usually 1)
    val roleTokenIds: RoleTokenIds,               // Tokens for prompt structure and role separation
    val scalarPosId: Boolean = false,             // Enables scalar-style position IDs (used by Qwen3)
    val IsThinkingModeAvailable: Boolean = false, // Enables toggle for "thinking mode" (Qwen3-specific)
    val vocabSize: Int,                           // Size of the tokenizer vocabulary
)
