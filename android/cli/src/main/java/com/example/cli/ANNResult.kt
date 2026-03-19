package com.example.cli

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.BufferedWriter
import java.io.Closeable
import java.io.File
import java.io.IOException

@Serializable
data class ResultItem(
    val id: Int,
    val retrievalLatencyNs: Long,
    val neighbors: LongArray?,
    val distances: FloatArray?
)

class ResultJsonStreamWriter(
    file: File,
    prettyPrint: Boolean = false,
    private val autoFlush: Boolean = true
): Closeable {

    private val writer: BufferedWriter
    private val json: Json = Json {
        this.prettyPrint = prettyPrint
        encodeDefaults = true
    }

    private var firstItem = true
    private var closed = false

    init {
        val parent = file.parentFile
        if (parent != null && !parent.exists()) {
            if (!parent.mkdirs()) {
                throw IOException("Failed to create directories: ${parent.absolutePath}")
            }
        }

        // Delete previous file if exists
        if (file.exists()) {
            file.delete()
        }

        // Write JSON header once
        writer = file.bufferedWriter()
        writer.write("{\"items\":[")
        writer.flush()
    }

    /**
     * Writes a single ResultItem into the JSON array.
     */
    fun write(item: ResultItem) {
        check(!closed) { "Writer already closed" }

        if (!firstItem) {
            writer.write(",")
        }
        firstItem = false

        val jsonString = json.encodeToString(ResultItem.serializer(), item)
        writer.write(jsonString)

        if (autoFlush) {
            writer.flush()
        }
    }

    fun write(
        id: Int,
        processingTimeNs: Long,
        neighbors: LongArray?,
        distances: FloatArray?
    ) {
        write(
            ResultItem(
                id = id,
                retrievalLatencyNs = processingTimeNs,
                neighbors = neighbors,
                distances = distances
            )
        )
    }

    /**
     * Must be called exactly once.
     * Closes the JSON array and file.
     */
    override fun close() {
        if (closed) return

        writer.write("]}")
        writer.flush()
        writer.close()
        closed = true
    }
}