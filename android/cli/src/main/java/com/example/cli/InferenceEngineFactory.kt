package com.example.cli

object InferenceEngineFactory {

    fun create(engine: String): InferenceEngine {
        return when (engine.lowercase()) {
            "onnx" -> OnnxEngine()
            "litert" -> LiteRTEngine()
            else -> throw IllegalArgumentException("Unknown inference engine: $engine")
        }
    }
}
