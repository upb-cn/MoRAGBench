package com.example.faiss

import android.content.ContentValues
import android.content.Context
import android.database.Cursor
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import androidx.core.database.sqlite.transaction

data class EmbeddingRow(
    val faissId: Long,
    val docId: String?,
    val chunkId: Int?,
    val chunkText: String?,
    val dim: Int?,
    val model: String?,
    val metaJson: String?,
    val createdAt: Long?
)

data class PendingChunk(
    val docId: String,
    val chunkId: Int,
    val chunkText: String,
    val model: String,
    val vec: FloatArray
)

class EmbeddingDb(context: Context, dbName: String = "embeddings.db") :
    SQLiteOpenHelper(context, dbName, null, DATABASE_VERSION) {

    companion object {
        private const val DATABASE_VERSION = 1
        private const val TABLE_EMBEDDINGS = "embeddings"
    }

    override fun onCreate(db: SQLiteDatabase) {
        val create = """
            CREATE TABLE $TABLE_EMBEDDINGS (
              faiss_id INTEGER PRIMARY KEY,
              doc_id TEXT,
              chunk_id INTEGER,
              chunk_text TEXT,
              dim INTEGER,
              model TEXT,
              meta JSON,
              created_at INTEGER
            );
        """.trimIndent()
        db.execSQL(create)
        db.execSQL("CREATE INDEX IF NOT EXISTS idx_embeddings_doc ON $TABLE_EMBEDDINGS(doc_id);")
    }

    fun clearAll() {
        val db = writableDatabase
        db.transaction {
            execSQL("DELETE FROM $TABLE_EMBEDDINGS")
        }
    }

    fun countEmbeddings(): Long {
        val db = readableDatabase
        var c: Cursor? = null
        try {
            c = db.rawQuery("SELECT COUNT(1) FROM $TABLE_EMBEDDINGS", null)
            if (c.moveToFirst()) {
                return c.getLong(0)
            }
        } finally {
            c?.close()
        }
        return 0L
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        // Handle migrations as needed
        // For now, simple approach: drop and recreate (data-lossy) — change this for production
        db.execSQL("DROP TABLE IF EXISTS $TABLE_EMBEDDINGS")
        onCreate(db)
    }

    fun insertEmbedding(row: EmbeddingRow) {
        val db = writableDatabase
        val cv = ContentValues().apply {
            put("faiss_id", row.faissId)
            put("doc_id", row.docId)
            row.chunkId?.let { put("chunk_id", it) }
            put("chunk_text", row.chunkText)
            row.dim?.let { put("dim", it) }
            put("model", row.model)
            put("meta", row.metaJson)
            row.createdAt?.let { put("created_at", it) }
        }
        db.insertWithOnConflict(TABLE_EMBEDDINGS, null, cv, SQLiteDatabase.CONFLICT_REPLACE)
    }

    fun getByFaissId(faissId: Long): EmbeddingRow? {
        val db = readableDatabase
        var c: Cursor? = null
        try {
            c = db.query(
                TABLE_EMBEDDINGS,
                arrayOf("faiss_id","doc_id","chunk_id","chunk_text","dim","model","meta","created_at"),
                "faiss_id = ?",
                arrayOf(faissId.toString()),
                null, null, null
            )
            if (c.moveToFirst()) {
                return EmbeddingRow(
                    faissId = c.getLong(0),
                    docId = c.getString(1),
                    chunkId = if (!c.isNull(2)) c.getInt(2) else null,
                    chunkText = c.getString(3),
                    dim = if (!c.isNull(4)) c.getInt(4) else null,
                    model = c.getString(5),
                    metaJson = c.getString(6),
                    createdAt = if (!c.isNull(7)) c.getLong(7) else null
                )
            }
        } finally {
            c?.close()
        }
        return null
    }

    /**
     * Bulk insert convenience for batch operations.
     * rows: list of EmbeddingRow; wraps in transaction.
     */
    fun bulkInsert(rows: List<EmbeddingRow>) {
        val db = writableDatabase
        db.beginTransaction()
        try {
            for (r in rows) {
                insertEmbedding(r)
            }
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
    }
}
