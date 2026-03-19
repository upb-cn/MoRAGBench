package com.example.shared

import android.content.Context
import com.example.local_llm.ModelConfig
import com.example.local_llm.RoleTokenIds
import com.example.local_llm.TokenizerBridge
import com.example.local_llm.TokenizerSource
import com.example.onnxtok.EmbeddingModel

data class ModelPathOverrides(
    val tokenizer: TokenizerSource? = null,
    val modelPath: String? = null
)

private fun normalizeModelKey(name: String): String {
    return name
        .replace("-Instruct", "", ignoreCase = true)
        .lowercase()
        .trim()
}

object SupportedLLMs {

    // Map each model name to its builder function
    private val builders: Map<String, (Context, ModelPathOverrides?) -> ModelConfig> =
        mapOf(
            normalizeModelKey("Qwen2.5-0.5B-Instruct-Int8") to
                    { ctx, o -> buildQwen05B(ctx, "int8", o) },

            normalizeModelKey("Qwen2.5-0.5B-Instruct-Q4") to
                    { ctx, o -> buildQwen05B(ctx, "q4", o) },

            normalizeModelKey("Qwen2.5-0.5B-Instruct") to
                    { ctx, o -> buildQwen05B(ctx, null, o) },

            normalizeModelKey("Qwen2.5-0.5B-Instruct-Float16") to
                    { ctx, o -> buildQwen05B(ctx, null, o) },

            normalizeModelKey("Qwen2.5-1.5B-Instruct-Int8") to
                    { ctx, o -> buildQwen15B(ctx, o) }
        )

    fun getAll(context: Context): List<ModelConfig> {
        return builders.values.map { buildFn -> buildFn(context, null) }
    }

    fun findByName(
        context: Context,
        name: String,
        overrides: ModelPathOverrides? = null
    ): ModelConfig {
        val buildFn = builders[normalizeModelKey(name)]
            ?: error("LLM model \"$name\" is not supported")

        return buildFn(context, overrides)
    }

    private fun buildQwen05B(
        context: Context,
        type: String?,
        overrides: ModelPathOverrides?
    ): ModelConfig {

        var modelName = "qwen2.5-0.5B"
        if (type != null) {
            modelName += "_$type"
        }
        val defaultTokenizerPath = "llm/$modelName/tokenizer.json"


        val tokenizerSource =
            overrides?.tokenizer
                ?: TokenizerSource.Assets(defaultTokenizerPath)
        val tokenizer = TokenizerBridge(context, tokenizerSource)

        val effectiveTokenizerPath = when (tokenizerSource) {
            is TokenizerSource.Assets -> tokenizerSource.assetPath
            is TokenizerSource.File -> tokenizerSource.absolutePath
        }

        val roles = RoleTokenIds(
            systemStart = listOf(
                tokenizer.getTokenId("<|im_start|>"),
                tokenizer.getTokenId("system"),
                tokenizer.getTokenId("Ċ")
            ),
            userStart = listOf(
                tokenizer.getTokenId("<|im_start|>"),
                tokenizer.getTokenId("user"),
                tokenizer.getTokenId("Ċ")
            ),
            assistantStart = listOf(
                tokenizer.getTokenId("<|im_start|>"),
                tokenizer.getTokenId("assistant"),
                tokenizer.getTokenId("Ċ")
            ),
            endToken = tokenizer.getTokenId("<|im_end|>")
        )

        return ModelConfig(
            modelName = "Qwen2.5-0.5B-Instruct",
            modelPath = overrides?.modelPath
                ?: "llm/$modelName/model.onnx",
            tokenizerPath = effectiveTokenizerPath,
            eosTokenIds = setOf(151643, 151645),
            numLayers = 24,
            numKvHeads = 2,
            headDim = 64,
            batchSize = 1,
            defaultSystemPrompt = "You are Qwen, a helpful assistant.",
            roleTokenIds = roles,
            scalarPosId = false,
            vocabSize = 151936
        )
    }

    private fun buildQwen15B(context: Context, overrides: ModelPathOverrides?): ModelConfig {
        val defaultTokenizerPath = "llm/qwen2.5-1.5B_int8/tokenizer.json"

        val tokenizerSource =
            overrides?.tokenizer
                ?: TokenizerSource.Assets(defaultTokenizerPath)
        val tokenizer = TokenizerBridge(context, tokenizerSource)

        val effectiveTokenizerPath = when (tokenizerSource) {
            is TokenizerSource.Assets -> tokenizerSource.assetPath
            is TokenizerSource.File -> tokenizerSource.absolutePath
        }

        val roles = RoleTokenIds(
            systemStart = listOf(
                tokenizer.getTokenId("<|im_start|>"),
                tokenizer.getTokenId("system"),
                tokenizer.getTokenId("Ċ")
            ),
            userStart = listOf(
                tokenizer.getTokenId("<|im_start|>"),
                tokenizer.getTokenId("user"),
                tokenizer.getTokenId("Ċ")
            ),
            assistantStart = listOf(
                tokenizer.getTokenId("<|im_start|>"),
                tokenizer.getTokenId("assistant"),
                tokenizer.getTokenId("Ċ")
            ),
            endToken = tokenizer.getTokenId("<|im_end|>")
        )

        return ModelConfig(
            modelName = "Qwen2.5-1.5B-Instruct",
            modelPath = overrides?.modelPath
                ?: "llm/qwen2.5-1.5B_int8/model.onnx",
            tokenizerPath = effectiveTokenizerPath,
            eosTokenIds = setOf(151643, 151645),
            numLayers = 28,
            numKvHeads = 2,
            headDim = 128,
            batchSize = 1,
            defaultSystemPrompt = "You are Qwen, a helpful assistant.",
            roleTokenIds = roles,
            scalarPosId = false,
            vocabSize = 151936,
        )
    }
}


object SupportedEmbeddingModels {

    val models: List<EmbeddingModel> = listOf(
        EmbeddingModel(
            modelPath = "embedding/all-minilm-l6-v2/model.onnx",
            tokenizerPath = "embedding/all-minilm-l6-v2/tokenizer.json",
            useTokenTypeIds = true,
            outputTensorName = "sentence_embedding",
            dim = 384
        ),
        EmbeddingModel(
            modelPath = "embedding/all-minilm-l12-v2/model.onnx",
            tokenizerPath = "embedding/all-minilm-l12-v2/tokenizer.json",
            useTokenTypeIds = true,
            outputTensorName = "sentence_embedding",
            dim = 384
        ),
    )

    fun getAll(): List<EmbeddingModel> = models.map { it.copy() }

    fun findByName(name: String): EmbeddingModel {
        return models.find { it.modelPath.contains(name, ignoreCase = true) }
            ?.copy()
            ?: error("Embedding model \"$name\" is not supported.")
    }
}
