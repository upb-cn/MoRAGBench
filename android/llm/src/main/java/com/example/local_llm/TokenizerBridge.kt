package com.example.local_llm

import android.content.Context
import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader

sealed class TokenizerSource {
    data class Assets(val assetPath: String) : TokenizerSource()
    data class File(val absolutePath: String) : TokenizerSource()
}

class TokenizerBridge(context: Context, source: TokenizerSource) {

    init {
        System.loadLibrary("qwen_tokenizer")

        val json = when (source) {
            is TokenizerSource.Assets -> readFromAssets(context, source.assetPath)
            is TokenizerSource.File -> readFromFile(source.absolutePath)
        }

        val success = initFromJson(json)
        check(success) { "Failed to init tokenizer" }
    }

    // --- JNI native functions ---
    private external fun initFromJson(json: String): Boolean
    external fun encode(text: String, addSpecialTokens: Boolean = false): IntArray
    external fun decode(ids: IntArray, skipSpecialTokens: Boolean = false): String
    external fun getTokenId(token: String): Int

    // --- helper to load JSON from assets ---
    private fun readFromAssets(context: Context, assetPath: String): String {
        return context.applicationContext.assets
            .open(assetPath)
            .use { BufferedReader(InputStreamReader(it)).readText() }
    }

    private fun readFromFile(path: String): String {
        return File(path).bufferedReader().use { it.readText() }
    }
}
