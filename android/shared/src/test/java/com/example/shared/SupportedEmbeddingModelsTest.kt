package com.example.shared

import org.junit.Assert.assertEquals
import org.junit.Test

class SupportedEmbeddingModelsTest {
    @Test
    fun bgeSmallUsesSingleThreadedInference() {
        val model = SupportedEmbeddingModels.findByName("bge-small-en-v1.5")
        assertEquals(1, model.intraOpNumThreads)
    }

    @Test
    fun bgeBaseUsesSingleThreadedInference() {
        val model = SupportedEmbeddingModels.findByName("bge-base-en-v1.5")
        assertEquals(1, model.intraOpNumThreads)
    }
}
