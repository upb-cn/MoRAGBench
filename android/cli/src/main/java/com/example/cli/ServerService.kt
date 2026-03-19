package com.example.cli

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.util.Log
import java.io.File

class ServerService : Service() {
    private val TAG = "ServerService"
    private val CHANNEL_ID = "dev_server_channel"
    private val NOTIF_ID = 456
    private var server: HttpServer? = null

    override fun onCreate() {
        super.onCreate()

        BenchmarkManager.init(applicationContext)

        Log.i(TAG, "onCreate: starting foreground")
        startForegroundNotification()   // must call quickly

        Thread {
            try {
                server = HttpServer(Constants.PORT)
                server?.start()
            } catch (t: Throwable) {
                Log.e(TAG, "Failed to start HTTP server", t)
                // Write stacktrace to file for offline inspection
                try {
                    val f = File(filesDir, "server_start_error.txt")
                    f.writeText(Log.getStackTraceString(t))
                } catch (ignored: Exception) {}
                // Stop self to avoid repeated restart attempts
                stopSelf()
            }
        }.start()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.i(TAG, "onStartCommand id=$startId")
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        Log.i(TAG, "onDestroy: stopping server")
        try {
            server?.stop()
            BenchmarkManager.stopBenchmark()
        } catch (t: Throwable) {
            Log.e(TAG, "stop failed", t)
        }
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun startForegroundNotification() {
        val nm = getSystemService(NotificationManager::class.java)
        val ch = NotificationChannel(CHANNEL_ID, "Dev Server", NotificationManager.IMPORTANCE_LOW)
        nm.createNotificationChannel(ch)
        val notifBuilder =
            Notification.Builder(this, CHANNEL_ID)
        val notif: Notification = notifBuilder
            .setContentTitle("Dev HTTP server")
            .setContentText("Listening for bench requests")
            .setSmallIcon(android.R.drawable.ic_menu_info_details)
            .build()
        startForeground(NOTIF_ID, notif)
        Log.i(TAG, "startForeground called")
    }
}
