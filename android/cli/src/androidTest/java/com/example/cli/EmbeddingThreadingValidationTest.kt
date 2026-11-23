package com.example.cli

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.onnxtok.EmbeddingGenerator
import com.example.shared.SupportedEmbeddingModels
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class EmbeddingThreadingValidationTest {

    @Test
    fun compareDefaultAndSingleThreadEmbeddings() = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val baseDir = File(context.filesDir, "embedding_threading_models")
        val outputDir = File(context.getExternalFilesDir(null), "embedding_threading_validation")
        check(outputDir.exists() || outputDir.mkdirs())

        val sentence = "The quick brown fox jumps over the lazy dog."

        val models = listOf(
            Triple("bge-base-en-v1.5", "float32", 768),
            Triple("all-minilm-l6-v2", "int8", 384)
        )

        for ((name, dtype, expectedDim) in models) {
            val modelFile = File(
                baseDir,
                if (name == "bge-base-en-v1.5") "bge-base-model.onnx" else "minilm-model.onnx"
            )
            val tokenizerFile = File(
                baseDir,
                if (name == "bge-base-en-v1.5") "bge-base-tokenizer.json" else "minilm-tokenizer.json"
            )

            check(modelFile.isFile) { "Missing model: $modelFile" }
            check(tokenizerFile.isFile) { "Missing tokenizer: $tokenizerFile" }

            val original = SupportedEmbeddingModels.findByName(name)
            val inputText = original.queryPrefix + sentence

            for ((threadLabel, threadCount) in listOf(
                "default" to null,
                "single" to 1
            )) {
                val config = original.copy(
                    modelPath = modelFile.absolutePath,
                    tokenizerPath = tokenizerFile.absolutePath,
                    intraOpNumThreads = threadCount
                )

                EmbeddingGenerator(context).use { generator ->
                    generator.initialize(
                        embeddingModel = config,
                        backend = "cpu",
                        normalizeEmbeddings = true
                    )

                    repeat(3) { repetition ->
                        val vector = generator.generate(inputText)
                        assertEquals(expectedDim, vector.size)

                        val result = JSONObject().apply {
                            put("model", name)
                            put("dtype", dtype)
                            put("threading", threadLabel)
                            put("intra_op_threads", threadCount ?: JSONObject.NULL)
                            put("repetition", repetition)
                            put("input_text", inputText)
                            put("pooling", original.poolingStrategy.name)
                            put("normalized", true)
                            put("embedding", JSONArray(vector.map { it.toDouble() }))
                        }

                        File(
                            outputDir,
                            "${name}_${dtype}_${threadLabel}_${repetition}.json"
                        ).writeText(result.toString())
                    }
                }
            }
        }
    }
}
