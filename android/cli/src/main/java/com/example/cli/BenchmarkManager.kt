package com.example.cli
import android.annotation.SuppressLint
import android.app.ActivityManager
import android.content.Context
import android.os.Debug
import com.example.faiss.EmbeddingDb
import com.example.cli.Progress.State

object BenchmarkManager {
    private lateinit var appContext: Context

    var testType: String = "task"

    @SuppressLint("StaticFieldLeak")
    var taskBenchmark: TaskBenchmark? = null
    @SuppressLint("StaticFieldLeak")
    var annBenchmark: ANNBenchmark? = null


    fun init(context: Context) {
        // Use application context to avoid leaks
        appContext = context.applicationContext
    }

    fun startBenchmarkAsync(testType: String): String? {
        this.testType = testType

        when (testType) {
            "task" -> {
                val progressSnapshot = taskBenchmark?.progress?.getSnapshot()
                if (progressSnapshot != null && progressSnapshot.overallState == State.RUNNING) {
                    return "Test already running"
                }

                taskBenchmark = TaskBenchmark(appContext).also {
                    it.start()
                }

                return "Test started successfully"
            }
            "ann" -> {
                val progressSnapshot = annBenchmark?.progress?.getSnapshot()
                if (progressSnapshot != null && progressSnapshot.overallState == State.RUNNING) {
                    return "Test already running"
                }

                annBenchmark = ANNBenchmark(appContext).also {
                    it.start()
                }

                return "Test started successfully"
            }
            else -> {
                return "Unsupported test type: $testType"
            }
        }
    }

    fun stopBenchmark() {
        when (testType) {
            "task" -> {
                taskBenchmark?.stop()
                taskBenchmark = null
            }
            "ann" -> {
                annBenchmark?.stop()
                annBenchmark = null
            }
        }
    }

    fun logMemory(msg: String) {
        val runtime = Runtime.getRuntime()

        // --- Java heap ---
        val heapUsedKb = (runtime.totalMemory() - runtime.freeMemory()) / 1024
        val heapTotalKb = runtime.totalMemory() / 1024
        val heapMaxKb = runtime.maxMemory() / 1024

        val am = appContext.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager

        val pid = android.os.Process.myPid()
        val pssInfo = am.getProcessMemoryInfo(intArrayOf(pid))[0]
        val totalPssKb = pssInfo.totalPss

        // --- Detailed breakdown ---
        val debugInfo = Debug.MemoryInfo()
        Debug.getMemoryInfo(debugInfo)

        // --- System-wide memory ---
        val sysInfo = ActivityManager.MemoryInfo()
        am.getMemoryInfo(sysInfo)

        LiveLogger.info(
            """
        ===== MEMORY SNAPSHOT (BG) =====
        Msg:     $msg
        Heap:    used=${heapUsedKb}KB  total=${heapTotalKb}KB  max=${heapMaxKb}KB
        PSS:     total=${totalPssKb}KB
        Native:  private=${debugInfo.nativePrivateDirty}KB
        Dalvik:  private=${debugInfo.dalvikPrivateDirty}KB
        Other:   private=${debugInfo.otherPrivateDirty}KB
        SysAvail:${sysInfo.availMem / 1024}KB
        LowMem:  ${sysInfo.lowMemory}
        =================================
        """.trimIndent()
        )
    }

    // ********* Helper functions **********
    fun getDB(hash: String, context: Context): EmbeddingDb {
        return EmbeddingDb(context = context, dbName = hash)
    }
}
