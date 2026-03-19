package com.example.moragbench

import android.content.Context
import android.content.SharedPreferences
import androidx.core.content.edit

object SettingsDefaults {
    object Chunker {
        const val METHOD = "token"
        const val SIZE = 256
        const val OVERLAP_ENABLED = true
        const val OVERLAP_SIZE = 50
    }

    object Embedding {
        const val BACKEND = "CPU"
        const val MODEL = "all-minilm-l6-v2"
    }

    object Faiss {
        const val INDEX_BACKEND = "CPU"
        const val SEARCH_BACKEND = "CPU"
        const val METRIC = "L2"
        const val TOPK = 3
        const val BATCH_SIZE = 1000
        const val INDEX_METHOD = "Flat"
        const val NPROBE = 10
        const val NLIST = 100
        const val NUM_TRAINING_VECTORS = 10000
        const val M = 16
        const val EF_CONSTRUCTION = 200
        const val EF_SEARCH = 50
    }

    object LLM {
        const val AUG_METHOD = "Concatenation"
        const val GEN_BACKEND = "CPU"
        const val MODEL = "Qwen2.5-0.5B-Instruct"
        const val USE_SAMPLING = true
        const val REPETITION_PENALTY = 1f
        const val TEMP = 0.8f
        const val TOP_P = 0.95f
        const val TOP_K = 0
        const val MAX_TOKENS = 512
        const val SYSTEM_PROMPT = "You are a helpful assistant."
    }
}

class SettingsManager(context: Context) {
    private val sp: SharedPreferences =
        context.getSharedPreferences("rag_settings", Context.MODE_PRIVATE)

    val chunker = ChunkerConfig(sp)
    val embedding = EmbeddingConfig(sp)
    val faiss = FaissConfig(sp)
    val llm = LlmConfig(sp)

    fun resetToDefaults() {
        sp.edit { clear() }
    }
}

// ---------- Section Classes ----------
class ChunkerConfig(private val sp: SharedPreferences) {

    var method: String
        get() = sp.getString("chunk_method", SettingsDefaults.Chunker.METHOD)!!
        set(v) = sp.edit { putString("chunk_method", v) }

    var size: Int
        get() = sp.getInt("chunk_size", SettingsDefaults.Chunker.SIZE)
        set(v) = sp.edit { putInt("chunk_size", v) }

    var overlapEnabled: Boolean
        get() = sp.getBoolean("chunk_overlap_enabled", SettingsDefaults.Chunker.OVERLAP_ENABLED)
        set(v) = sp.edit { putBoolean("chunk_overlap_enabled", v) }

    var overlapSize: Int
        get() = sp.getInt("chunk_overlap_size", SettingsDefaults.Chunker.OVERLAP_SIZE)
        set(v) = sp.edit { putInt("chunk_overlap_size", v) }
}

class EmbeddingConfig(private val sp: SharedPreferences) {
    var backend: String
        get() = sp.getString("embed_backend", SettingsDefaults.Embedding.BACKEND)!!
        set(v) = sp.edit { putString("embed_backend", v) }

    var model: String
        get() = sp.getString("embed_model", SettingsDefaults.Embedding.MODEL)!!
        set(v) = sp.edit { putString("embed_model", v) }
}

class FaissConfig(private val sp: SharedPreferences) {
    var indexBackend: String
        get() = sp.getString("faiss_index_backend", SettingsDefaults.Faiss.INDEX_BACKEND)!!
        set(v) = sp.edit { putString("faiss_index_backend", v) }

    var searchBackend: String
        get() = sp.getString("faiss_search_backend", SettingsDefaults.Faiss.SEARCH_BACKEND)!!
        set(v) = sp.edit { putString("faiss_search_backend", v) }

    var metric: String
        get() = sp.getString("faiss_metric", SettingsDefaults.Faiss.METRIC)!!
        set(v) = sp.edit { putString("faiss_metric", v) }

    var topK: Int
        get() = sp.getInt("faiss_topk", SettingsDefaults.Faiss.TOPK)
        set(v) = sp.edit { putInt("faiss_topk", v) }

    var batchSize: Int
        get() = sp.getInt("faiss_batchSize", SettingsDefaults.Faiss.BATCH_SIZE)
        set(v) = sp.edit { putInt("faiss_batchSize", v) }

    var indexMethod: String
        get() = sp.getString("faiss_index_method", SettingsDefaults.Faiss.INDEX_METHOD)!!
        set(v) = sp.edit { putString("faiss_index_method", v) }

    var nprobe: Int
        get() = sp.getInt("faiss_nprobe", SettingsDefaults.Faiss.NPROBE)
        set(v) = sp.edit { putInt("faiss_nprobe", v) }

    var nlist: Int
        get() = sp.getInt("faiss_nlist", SettingsDefaults.Faiss.NLIST)
        set(v) = sp.edit { putInt("faiss_nlist", v) }

    var numTrainingVectors: Int
        get() = sp.getInt("faiss_num_training_vectors", SettingsDefaults.Faiss.NUM_TRAINING_VECTORS)
        set(v) = sp.edit { putInt("faiss_num_training_vectors", v) }

    var m: Int
        get() = sp.getInt("faiss_m", SettingsDefaults.Faiss.M)
        set(v) = sp.edit { putInt("faiss_m", v) }

    var efConstruction: Int
        get() = sp.getInt("faiss_ef_construction", SettingsDefaults.Faiss.EF_CONSTRUCTION)
        set(v) = sp.edit { putInt("faiss_ef_construction", v) }

    var efSearch: Int
        get() = sp.getInt("faiss_ef_search", SettingsDefaults.Faiss.EF_SEARCH)
        set(v) = sp.edit { putInt("faiss_ef_search", v) }
}

class LlmConfig(private val sp: SharedPreferences) {
    var augMethod: String
        get() = sp.getString("llm_aug_method", SettingsDefaults.LLM.AUG_METHOD)!!
        set(v) = sp.edit { putString("llm_aug_method", v) }

    var genBackend: String
        get() = sp.getString("llm_gen_backend", SettingsDefaults.LLM.GEN_BACKEND)!!
        set(v) = sp.edit { putString("llm_gen_backend", v) }

    var model: String
        get() = sp.getString("llm_model", SettingsDefaults.LLM.MODEL)!!
        set(v) = sp.edit { putString("llm_model", v) }

    var useSampling: Boolean
        get() = sp.getBoolean("llm_use_sampling", SettingsDefaults.LLM.USE_SAMPLING)
        set(v) = sp.edit { putBoolean("llm_use_sampling", v) }

    var repetitionPenalty: Float
        get() = sp.getFloat("llm_repetition_penalty", SettingsDefaults.LLM.REPETITION_PENALTY)
        set(v) = sp.edit { putFloat("llm_repetition_penalty", v) }

    var temp: Float
        get() = sp.getFloat("llm_temp", SettingsDefaults.LLM.TEMP)
        set(v) = sp.edit { putFloat("llm_temp", v) }

    var topP: Float
        get() = sp.getFloat("llm_topp", SettingsDefaults.LLM.TOP_P)
        set(v) = sp.edit { putFloat("llm_topp", v) }

    var topK: Int
        get() = sp.getInt("llm_topk", SettingsDefaults.LLM.TOP_K)
        set(v) = sp.edit { putInt("llm_topk", v) }

    var maxTokens: Int
        get() = sp.getInt("llm_max_tokens", SettingsDefaults.LLM.MAX_TOKENS)
        set(v) = sp.edit { putInt("llm_max_tokens", v) }

    var systemPrompt: String
        get() = sp.getString("llm_system_prompt", SettingsDefaults.LLM.SYSTEM_PROMPT)!!
        set(v) = sp.edit { putString("llm_system_prompt", v) }
}