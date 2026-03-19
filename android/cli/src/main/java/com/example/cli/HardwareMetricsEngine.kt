package com.example.cli

import android.app.ActivityManager
import kotlinx.serialization.Serializable
import java.io.File
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import kotlinx.coroutines.*
import kotlinx.serialization.json.Json
import kotlin.math.abs

@Serializable
data class MetricsResult(
    val memory: MemoryMetrics,
    val power: PowerMetrics?
)

@Serializable
data class MemoryMetrics(
    val beforeMB: Long,
    val meanMB: Long,
    val peakMB: Long
)

@Serializable
data class PowerMetrics(
    val currentMa: PowerStats,
    val voltageV: PowerStats,
    val powerMw: PowerStats,
    val temperatureC: PowerStats
)

@Serializable
data class PowerStats(
    val mean: Float,
    val peak: Float
)

class FloatStats {
    private var sum = 0.0
    private var count = 0
    var peak = 0f

    fun add(v: Float) {
        sum += v
        count++
        if (v > peak) peak = v
    }

    fun mean(): Float = if (count == 0) 0f else (sum / count).toFloat()
}

class LongStats {
    private var sum = 0L
    private var count = 0
    var peak = 0L

    fun add(v: Long) {
        sum += v
        count++
        if (v > peak) peak = v
    }

    fun mean(): Long = if (count == 0) 0L else sum / count
}

object MemorySampler {
    fun sample(context: Context): Long {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo()
        am.getMemoryInfo(mi)
        val usedMb = (mi.totalMem - mi.availMem) / (1024 * 1024)

        return usedMb
    }
}

class PowerSampler(private val context: Context) {
    private val batteryManager = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager

    private val currentStats = FloatStats()
    private val voltageStats = FloatStats()
    private val powerStats = FloatStats()
    private val temperatureStats = FloatStats()

    /**
     * Sample current power consumption and update internal statistics.
     * @return Current power consumption in milliwatts (mW), or null if measurement failed
     */
    fun sample(): Float? {
        try {
            // Get current in microamperes (μA)
            val currentNowMicroAmps = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
            val currentAvgRaw = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_AVERAGE)

            // Some devices return negative values when discharging
            val currentMa = abs(currentNowMicroAmps) / 1000f

            // Get voltage and temperature from battery intent
            val intentFilter = IntentFilter(Intent.ACTION_BATTERY_CHANGED)
            val batteryStatus = context.registerReceiver(null, intentFilter)

            // Get voltage in millivolts (mV)
            val voltageMilliVolts = batteryStatus?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 0) ?: 0
            val voltageV = voltageMilliVolts / 1000f

            // Get temperature in deci-degrees Celsius (e.g., 251 = 25.1°C)
            val temperatureDeciCelsius = batteryStatus?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0
            val temperatureC = temperatureDeciCelsius / 10f

            // Calculate power: P = I × V (in milliwatts)
            val powerMw = currentMa * voltageV

            // Update statistics
            currentStats.add(currentMa)
            voltageStats.add(voltageV)
            powerStats.add(powerMw)
            temperatureStats.add(temperatureC)

            return powerMw

        } catch (e: Exception) {
            // Return null on error (device may not support battery stats)
            return null
        }
    }

    /**
     * Get summary statistics for all metrics
     */
    fun getStats(): PowerSamplerStats {
        return PowerSamplerStats(
            currentMa = PowerStats(currentStats.mean(), currentStats.peak),
            voltageV = PowerStats(voltageStats.mean(), voltageStats.peak),
            powerMw = PowerStats(powerStats.mean(), powerStats.peak),
            temperatureC = PowerStats(temperatureStats.mean(), temperatureStats.peak)
        )
    }

    data class PowerSamplerStats(
        val currentMa: PowerStats,
        val voltageV: PowerStats,
        val powerMw: PowerStats,
        val temperatureC: PowerStats
    )
}

class HardwareMetricsEngine(
    private val context: Context,
    private val scope: CoroutineScope,
    private val intervalMs: Long = 100L
) {
    private val memoryStats = LongStats()
    private val powerStats = FloatStats()

    val powerSampler = PowerSampler(context)

    private var job: Job? = null
    private var memoryBefore: Long = 0


    init {
        // Measure memory before
        memoryBefore = MemorySampler.sample(context)
    }

    fun start() {
        job = scope.launch(Dispatchers.Default) {
            while (isActive) {
                try {
                    // Memory
                    memoryStats.add(MemorySampler.sample(context))

                    // Power
                    powerSampler.sample()?.let { powerStats.add(it) }

                } catch (_: Throwable) {
                    // swallow errors to avoid affecting benchmark
                }

                delay(intervalMs)
            }
        }
    }

    fun stop(): MetricsResult {
        job?.cancel()
        scope.cancel()

        val samplerStats = powerSampler.getStats()

        val powerMetrics =
            if (powerStats.mean() > 0f)
                PowerMetrics(
                    currentMa = PowerStats(samplerStats.currentMa.mean, samplerStats.currentMa.peak),
                    voltageV = PowerStats(samplerStats.voltageV.mean, samplerStats.voltageV.peak),
                    powerMw = PowerStats(samplerStats.powerMw.mean, samplerStats.powerMw.peak),
                    temperatureC = PowerStats(samplerStats.temperatureC.mean, samplerStats.temperatureC.peak)
                )
            else null

        return MetricsResult(
            memory = MemoryMetrics(
                beforeMB = memoryBefore,
                meanMB = memoryStats.mean(),
                peakMB = memoryStats.peak
            ),
            power = powerMetrics
        )
    }

    fun writeMetricsJson(metrics: MetricsResult, file: File) {
        val json = Json { prettyPrint = true }
        file.parentFile?.mkdirs()
        file.writeText(json.encodeToString(MetricsResult.serializer(), metrics))
    }
}