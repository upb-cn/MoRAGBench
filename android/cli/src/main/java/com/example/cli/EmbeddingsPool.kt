package com.example.cli

import com.example.faiss.FaissIndex
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

    /**
     * Reservoir-samples k vectors from the spool (pass 1 — tracks only file-
     * record indices, never vectors), then rereads the file in order and streams
     * the selected vectors into a native training buffer via FaissIndex, in
     * batches (pass 2). Java-side memory is one batch array, an IntArray of k
     * indices and a HashSet of those k indices, never the full k*dim sample.
     * On success the caller must call faissIndex.trainIvfFromBuffer(k) and then
     * faissIndex.freeTrainBuffer(); on failure this function frees the buffer itself.
     *
     * NOTE: vectors are written to the native buffer in ASCENDING FILE ORDER,
     * not in reservoir-slot order. This is intentional: it preserves the fast
     * batched write path (one writeToBuffer call per full batch, not per vector).
     * FAISS's IndexIVF::train() treats its input as an unordered sample for
     * k-means clustering, so output order has no effect on training
     * correctness — only the SET of selected vectors matters, not their
     * position.
     */
    fun sampleReservoirToNativeBuffer(
        faissIndex: FaissIndex,
        k: Int,
        dim: Int,
        seed: Int = 42,
        batchSize: Int = 2000
    ) {
        // The native buffer is addressed with Int offsets: k * dim overflows Int for very
        // large samples, so compute it as a Long and fail with a clear message.
        val totalFloats = k.toLong() * dim
        require(totalFloats <= Int.MAX_VALUE) {
            "Training sample too large: $k vectors * dim $dim = $totalFloats floats " +
                    "(limit ${Int.MAX_VALUE}). Lower num_training_vectors."
        }
        out.flush()

        // ---- Pass 1: select which record indices go into the reservoir ----
        val rand = Random(seed)
        val selectedIndices = IntArray(k)
        var reservoirSize = 0

        DataInputStream(
            BufferedInputStream(FileInputStream(file))
        ).use { input ->
            var seen = 0
            try {
                while (true) {
                    repeat(dim) { input.readFloat() } // skip vector — position only, this pass
                    input.readUTF()
                    input.readInt()
                    input.readUTF()

                    if (reservoirSize < k) {
                        selectedIndices[reservoirSize] = seen
                        reservoirSize++
                    } else {
                        val j = rand.nextInt(seen + 1)
                        if (j < k) selectedIndices[j] = seen
                    }
                    seen++
                }
            } catch (e: EOFException) {
                // normal termination
            }
        }

        if (reservoirSize < k) {
            throw IllegalStateException(
                "Requested $k training vectors but pool only has $reservoirSize records"
            )
        }

        val selectedSet = selectedIndices.toHashSet()

        // ---- Pass 2: reread the file in order, stream selected vectors to native ----
        val allocOk = faissIndex.allocTrainBuffer(totalFloats.toInt())
        if (!allocOk) {
            throw IllegalStateException("Failed to allocate native training buffer for $k vectors")
        }

        try {
            val batch = FloatArray(batchSize * dim)
            var batchFillVectors = 0
            var writtenVectors = 0
            val scratch = FloatArray(dim)

            fun flushBatch() {
                if (batchFillVectors == 0) return
                val writeOk = faissIndex.writeToBuffer(
                    batch,
                    writtenVectors * dim,
                    batchFillVectors * dim
                )
                if (!writeOk) {
                    throw IllegalStateException(
                        "Native buffer write failed at writtenVectors=$writtenVectors"
                    )
                }
                writtenVectors += batchFillVectors
                batchFillVectors = 0
            }

            DataInputStream(
                BufferedInputStream(FileInputStream(file))
            ).use { input ->
                var idx = 0
                try {
                    while (true) {
                        if (idx in selectedSet) {
                            val offset = batchFillVectors * dim
                            for (i in 0 until dim) {
                                batch[offset + i] = input.readFloat()
                            }
                            batchFillVectors++
                            if (batchFillVectors == batchSize) flushBatch()
                        } else {
                            for (i in 0 until dim) {
                                scratch[i] = input.readFloat()
                            }
                        }

                        input.readUTF()
                        input.readInt()
                        input.readUTF()

                        idx++
                    }
                } catch (e: EOFException) {
                    // normal termination
                }
            }

            flushBatch() // remaining partial batch

        } catch (e: Exception) {
            faissIndex.freeTrainBuffer()
            throw e
        }
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
