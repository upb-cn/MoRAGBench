package com.example.cli

import android.app.Activity
import android.graphics.Color
import android.os.Build
import android.os.Bundle
import android.view.Gravity
import android.view.WindowManager
import android.widget.TextView

/**
 * A plain full-screen page shown while a benchmark runs.
 *
 * Why: without a visible screen the app only runs as a background service, and Android
 * then treats it as low priority - it is kept off the fastest core and often placed on
 * the slow energy-saving cores (3-4x slower). With this page in front, Android treats the
 * app as the one the user is actively using ("top app"), which gives it the fast cores.
 * It does no work itself; the benchmark still runs in ServerService.
 *
 * Started by the Python client (adb shell am start) when MORAG_FOREGROUND_UI=1.
 */
class BenchmarkActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1) {
            setShowWhenLocked(true)
            setTurnScreenOn(true)
        }
        setContentView(TextView(this).apply {
            text = "MoRAGBench\nbenchmark running\n\nkeep this screen open"
            textSize = 22f
            gravity = Gravity.CENTER
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.BLACK)
        })
    }
}
