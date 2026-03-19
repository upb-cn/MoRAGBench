package com.example.shared

import android.util.Log
import java.io.File

object Shared {
    fun safeDeleteFaissFile(path: String) {
        try {
            val f = File(path)
            if (f.exists()) f.delete()
            // also delete possible tmps
            val tmp = File("$path.tmp")
            if (tmp.exists()) tmp.delete()
        } catch (e: Exception) {
            Log.w("MyDB", "Failed to delete faiss files: ${e.message}")
        }
    }
}