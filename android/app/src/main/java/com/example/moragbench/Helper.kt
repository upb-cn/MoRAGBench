package com.example.moragbench

import android.content.Context
import android.widget.EditText
import android.widget.Toast
import com.example.local_llm.ModelConfig
import com.example.onnxtok.AssetTokenizerSource
import com.example.onnxtok.TextChunker
import com.example.shared.SupportedLLMs
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

class Helper(private val context: Context) {
    private val settings = SettingsManager(context)
    // General functions
    fun toast(msg: String) = Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
    fun parseInt(editText: EditText): Int = editText.text.toString().toIntOrNull() ?: 0
    fun parseFloat(editText: EditText): Float = editText.text.toString().toFloatOrNull() ?: 0f


    // LLM functions
    fun getModelFamily(selectedModel: String): String {
        return when {
            selectedModel.lowercase().contains("qwen2.5") -> "Qwen2.5"
            else -> {
                throw Error("The model \"$selectedModel\" is not yet supported")
            }
        }
    }

    fun getModelConfig(): ModelConfig {
        val selectedModel = settings.llm.model

        val config = SupportedLLMs.findByName(context, selectedModel)

        return config
    }

    // Chunker functions
    suspend fun loadAndChunk(tokenizerAssetName: String): Map<String, List<String>> = withContext(Dispatchers.IO) {
        // Read settings
        val chunkingMethod = settings.chunker.method
        val chunkSize = settings.chunker.size
        val overlapEnabled = settings.chunker.overlapEnabled
        val overlapSize = settings.chunker.overlapSize


        val chunker = TextChunker(
            tokenizerSource = AssetTokenizerSource(
                context,
                tokenizerAssetName
            ),
            chunkingMethod = chunkingMethod,
            chunkSize = chunkSize,
            overlap = if (overlapEnabled) overlapSize else 0
        )
        chunker.initialize()

        val assetFiles = context.assets.list("input_files") ?: emptyArray()
        val tempFiles = assetFiles.map { name ->
            val tmp = File(context.cacheDir, name)
            context.assets.open("input_files/$name").use { it.copyTo(tmp.outputStream()) }
            tmp
        }

        val chunks = chunker.chunkFiles(tempFiles)
        chunker.close()
        tempFiles.forEach { it.delete() }
        chunks
    }
}