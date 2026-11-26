package com.example.cli

import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.nio.*
import java.nio.channels.FileChannel

class VectorStore(private val baseDir: File) {

    // ---------- metadata ----------
    val trainCount: Int
    val testCount: Int
    val dim: Int
    val k: Int

    private val meta: JSONObject

    // ---------- mapped buffers ----------
    private val trainChunks: List<FloatBuffer>
    private val vectorsPerChunk: Int
    private val testBuf: FloatBuffer
    private val neighborsBuf: IntBuffer
    private val distancesBuf: FloatBuffer
    private val mappedBuffers = mutableListOf<ByteBuffer>()

    init {
        meta = JSONObject(File(baseDir, "meta.json").readText())

        fun shape(name: String): Pair<Int, Int> {
            val arr = meta.getJSONObject(name).getJSONArray("shape")
            return arr.getInt(0) to arr.getInt(1)
        }

        val (trainN, trainDim) = shape("train")
        val (testN, testDim) = shape("test")
        val (_, kVal) = shape("neighbors")

        require(trainDim == testDim) { "train/test dim mismatch" }

        trainCount = trainN
        testCount = testN
        dim = trainDim
        k = kVal

        vectorsPerChunk = ((1L shl 30) / (dim * 4L)).toInt()
        trainChunks = mapFloatChunks(metaPath("train"), dim, vectorsPerChunk)


        testBuf = mapFloat(metaPath("test"))
        neighborsBuf = mapInt(metaPath("neighbors"))
        distancesBuf = mapFloat(metaPath("distances"))
    }

    // ---------- public API (fast) ----------

    fun readTrainVector(i: Int, dst: FloatArray) = readTrainVectorInto(i, dst, 0)

    fun readTrainVectorInto(i: Int, dst: FloatArray, dstOffset: Int) {
        val buf = trainChunks[i / vectorsPerChunk]
        buf.position((i % vectorsPerChunk) * dim)
        buf.get(dst, dstOffset, dim)
    }

    /** Copy test vector i into dst (length >= dim). */
    fun readTestVector(i: Int, dst: FloatArray) {
        val start = i * dim
        testBuf.position(start)
        testBuf.get(dst, 0, dim)
    }


    // ---------- internal helpers ----------

    private fun metaPath(name: String): File =
        File(baseDir, meta.getJSONObject(name).getString("path"))

    private fun mapFloat(file: File): FloatBuffer =
        mapFile(file).asFloatBuffer()

    private fun mapInt(file: File): IntBuffer =
        mapFile(file).asIntBuffer()

    private fun mapFile(file: File): ByteBuffer =
        FileInputStream(file).channel.use { ch ->
            ch.map(FileChannel.MapMode.READ_ONLY, 0, ch.size())
        }.order(ByteOrder.LITTLE_ENDIAN).also { buf ->
            mappedBuffers += buf
        }

    private fun mapFloatChunks(file: File, dim: Int, vectorsPerChunk: Int): List<FloatBuffer> {
        val chunkBytes = vectorsPerChunk * dim * 4L
        val chunks = mutableListOf<FloatBuffer>()
        FileInputStream(file).channel.use { ch ->
            val total = ch.size()
            var offset = 0L
            while (offset < total) {
                val len = minOf(chunkBytes, total - offset)
                val bb = ch.map(FileChannel.MapMode.READ_ONLY, offset, len)
                    .order(ByteOrder.LITTLE_ENDIAN)
                mappedBuffers += bb
                chunks += bb.asFloatBuffer()
                offset += len
            }
        }
        return chunks
    }

    fun close() {
        mappedBuffers.forEach { unmap(it) }
        mappedBuffers.clear()
    }

    private fun unmap(buffer: ByteBuffer) {
        if (buffer is MappedByteBuffer) {
            try {
                val unsafeClass = Class.forName("sun.misc.Unsafe")
                val field = unsafeClass.getDeclaredField("theUnsafe")
                field.isAccessible = true
                val unsafe = field.get(null)
                val method = unsafeClass.getMethod("invokeCleaner", ByteBuffer::class.java)
                method.invoke(unsafe, buffer)
            } catch (e: Exception) {
                // fallback: rely on GC
            }
        }
    }
}
