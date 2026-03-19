package com.example.cli

import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.Closeable
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.EOFException
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import kotlin.random.Random

data class PoolRecord(
    val vector: FloatArray,
    val docId: String,
    val chunkId: Int,
    val chunkText: String
)
class EmbeddingsPool(
    private val dim: Int
) : Closeable {

    private val file = File.createTempFile("embedding_spool", ".bin")
    private val out = DataOutputStream(
        BufferedOutputStream(FileOutputStream(file))
    )

    var recordCount = 0
        private set

    fun append(
        vector: FloatArray,
        docId: String,
        chunkId: Int,
        chunkText: String
    ) {
        require(vector.size == dim)

        // vector
        for (v in vector) out.writeFloat(v)

        // metadata
        out.writeUTF(docId)
        out.writeInt(chunkId)
        out.writeUTF(chunkText)

        recordCount++
    }

    fun sampleReservoir(k: Int, seed: Int = 42): List<FloatArray> {
        out.flush()

        val rand = Random(seed)
        val reservoir = ArrayList<FloatArray>(k)

        DataInputStream(
            BufferedInputStream(FileInputStream(file))
        ).use { input ->

            var seen = 0

            try {
                while (true) {
                    // read vector
                    val vec = FloatArray(dim) { input.readFloat() }

                    // skip metadata
                    input.readUTF()
                    input.readInt()
                    input.readUTF()

                    seen++
                    if (reservoir.size < k) {
                        reservoir.add(vec)
                    } else {
                        val j = rand.nextInt(seen)
                        if (j < k) reservoir[j] = vec
                    }
                }
            } catch (e: EOFException) {
                // normal termination
            }
        }

        return reservoir
    }

    fun streamAll(consumer: (PoolRecord) -> Unit) {
        out.flush()

        DataInputStream(
            BufferedInputStream(FileInputStream(file))
        ).use { input ->
            try {
                while (true) {
                    // --- atomic record read ---
                    val vec = FloatArray(dim) {
                        input.readFloat()
                    }

                    val docId = input.readUTF()
                    val chunkId = input.readInt()
                    val chunkText = input.readUTF()

                    consumer(
                        PoolRecord(
                            vector = vec,
                            docId = docId,
                            chunkId = chunkId,
                            chunkText = chunkText
                        )
                    )
                }
            } catch (e: EOFException) {
                // expected termination
            }
        }
    }

    override fun close() {
        out.close()
        file.delete()
    }
}
