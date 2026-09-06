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
                    { ctx, o -> buildQwen15B(ctx, o) },

            normalizeModelKey("Llama-3.2-1B-Instruct-Q4") to
                    { ctx, o -> buildLlama32_1B(ctx, "q4", o) },
            normalizeModelKey("SmolLM2-1.7B-Instruct-Q4") to
                    { ctx, o -> buildSmolLM2_1_7B(ctx, "q4", o) },
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
            modelFamily = "qwen",
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
            modelFamily = "qwen",
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

    //new model Llama 3.2
    private fun buildLlama32_1B(context: Context, dtype: String, overrides: ModelPathOverrides?): ModelConfig {
        val modelFolder = "llm/llama-3.2-1B_$dtype"
        val defaultTokenizerPath = "$modelFolder/tokenizer.json"

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
                tokenizer.getTokenId("<|start_header_id|>"),
                tokenizer.getTokenId("system"),
                tokenizer.getTokenId("<|end_header_id|>")
            ),
            userStart = listOf(
                tokenizer.getTokenId("<|start_header_id|>"),
                tokenizer.getTokenId("user"),
                tokenizer.getTokenId("<|end_header_id|>")
            ),
            assistantStart = listOf(
                tokenizer.getTokenId("<|start_header_id|>"),
                tokenizer.getTokenId("assistant"),
                tokenizer.getTokenId("<|end_header_id|>")
            ),
            endToken = tokenizer.getTokenId("<|eot_id|>")
        )

        val MODEL_SIDECAR_BY_DTYPE = mapOf(
            "float32" to listOf("${modelFolder}/model.onnx_data", "${modelFolder}/model.onnx_data_1", "${modelFolder}/model.onnx_data_2"),
            "float16" to listOf("${modelFolder}/model_fp16.onnx_data", "${modelFolder}/model_fp16.onnx_data_1"),
            "int8" to emptyList(),
            "uint8" to emptyList(),
            "bnb4" to emptyList(),
            "q4" to listOf("${modelFolder}/model_q4.onnx_data"),
            "q4f16" to listOf("${modelFolder}/model_q4f16.onnx_data"),
        )
        val USE_POSITION_IDS_BY_DTYPE = mapOf(
            "float32" to true,
            "float16" to true,
            "int8" to true,
            "uint8" to true,
            "bnb4" to true,
            "q4" to false,
            "q4f16" to false,
        )
        return ModelConfig(
            modelName = "Llama-3.2-1B-Instruct",
            modelFamily = "llama",
            modelPath = overrides?.modelPath
                ?: "$modelFolder/model.onnx",
            sidecarPaths = MODEL_SIDECAR_BY_DTYPE[dtype] ?: emptyList(),
            tokenizerPath = effectiveTokenizerPath,
            eosTokenIds = setOf(128001, 128009),
            numLayers = 16,
            numKvHeads = 8,
            headDim = 64,
            batchSize = 1,
            defaultSystemPrompt = "You are a helpful assistant.",
            roleTokenIds = roles,
            scalarPosId = false,
            usePositionIds = USE_POSITION_IDS_BY_DTYPE[dtype] ?: false,
            vocabSize = 128256
        )
    }
    // SmolLM2 1.7B — ChatML format (same as Qwen)
    private fun buildSmolLM2_1_7B(context: Context, dtype: String, overrides: ModelPathOverrides?): ModelConfig {
        val modelFolder = "llm/smollm2-1.7B_$dtype"
        val defaultTokenizerPath = "$modelFolder/tokenizer.json"

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
            modelName = "SmolLM2-1.7B-Instruct",
            modelFamily = "qwen",
            modelPath = overrides?.modelPath
                ?: "$modelFolder/model.onnx",
            sidecarPaths = emptyList(),
            tokenizerPath = effectiveTokenizerPath,
            eosTokenIds = setOf(tokenizer.getTokenId("<|im_end|>")),
            numLayers = 24,
            numKvHeads = 32,
            headDim = 64,
            batchSize = 1,
            defaultSystemPrompt = "You are a helpful assistant.",
            roleTokenIds = roles,
            scalarPosId = false,
            usePositionIds = true,
            vocabSize = 49152
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
        EmbeddingModel(
            modelPath = "embedding/embeddinggemma/model.onnx",
            tokenizerPath = "embedding/embeddinggemma/tokenizer.json",
            useTokenTypeIds = false,
            outputTensorName = "sentence_embedding",
            dim = 768
        ),
    )

    fun getAll(): List<EmbeddingModel> = models.map { it.copy() }

    fun findByName(name: String): EmbeddingModel {
        return models.find { it.modelPath.contains(name, ignoreCase = true) }
            ?.copy()
            ?: error("Embedding model \"$name\" is not supported.")
    }
}
