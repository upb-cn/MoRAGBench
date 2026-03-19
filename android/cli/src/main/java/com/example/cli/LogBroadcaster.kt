package com.example.cli

import com.example.cli.FileLogger.prepareLine
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object LiveLogger {

    fun info(message: String) {
        // Write to file
        FileLogger.log(message)

        // logcat
        val line = prepareLine(message)
        println(line)
    }
}

object FileLogger {

    private lateinit var logFile: File
    private val lock = Any()

    private val formatter =
        SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)

    fun init(logDir: File) {
        logDir.mkdirs()

        logFile = File(logDir, Constants.LOG_FILE)

        // Delete previous file if exist
        if (logFile.exists()) logFile.delete()
    }

    fun prepareLine(message: String): String {
        val ts = formatter.format(Date())
        val line = "$ts $message\n"

        return line
    }

    fun log(message: String) {
        val line = prepareLine(message)

        synchronized(lock) {
            FileWriter(logFile, true).use {
                it.write(line)
            }
        }
    }
}