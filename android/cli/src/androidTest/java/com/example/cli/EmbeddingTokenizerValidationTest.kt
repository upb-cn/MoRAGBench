package com.example.cli

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.ml.shubham0204.sentence_embeddings.HFTokenizer
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class EmbeddingTokenizerValidationTest {

    @Test
    fun captureMiniLMTokenization() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val tokenizerFile = File(
            context.filesDir,
            "embedding_threading_models/minilm-tokenizer.json"
        )
        check(tokenizerFile.isFile) { "Missing tokenizer: $tokenizerFile" }

        val text = "The quick brown fox jumps over the lazy dog."
        val tokenizer = HFTokenizer(tokenizerFile.readBytes())

        try {
            val tokens = tokenizer.tokenize(text)
            val result = JSONObject().apply {
                put("model", "all-minilm-l6-v2")
                put("input_text", text)
                put("ids", JSONArray(tokens.ids.toList()))
                put("attention_mask", JSONArray(tokens.attentionMask.toList()))
                put("token_type_ids", JSONArray(tokens.tokenTypeIds.toList()))
            }

            val outputDir = File(
                context.getExternalFilesDir(null),
                "embedding_tokenizer_validation"
            )
            check(outputDir.exists() || outputDir.mkdirs())

            File(outputDir, "minilm_android_tokens.json").writeText(result.toString())
        } finally {
            tokenizer.close()
        }
    }
}
