package com.example.onnxtok

enum class EmbeddingPoolingStrategy {
    MEAN,
    CLS,
}

data class EmbeddingModel(
    var modelPath: String = "embedding/all-minilm-l6-v2/model.onnx",
    var tokenizerPath: String = "embedding/all-minilm-l6-v2/tokenizer.json",
    val useTokenTypeIds: Boolean = true,
    val outputTensorName: String = "sentence_embedding",
    val dim: Int = 384,
    val poolingStrategy: EmbeddingPoolingStrategy = EmbeddingPoolingStrategy.MEAN,
    val queryPrefix: String = "",
    val intraOpNumThreads: Int? = null,
)
