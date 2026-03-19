package com.example.onnxtok

data class EmbeddingModel(
    var modelPath: String = "embedding/all-minilm-l6-v2/model.onnx",
    var tokenizerPath: String = "embedding/all-minilm-l6-v2/tokenizer.json",
    val useTokenTypeIds: Boolean = true,
    val outputTensorName: String = "sentence_embedding",
    val dim: Int = 384,
)