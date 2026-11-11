package com.example.cli

import kotlinx.serialization.EncodeDefault
import kotlinx.serialization.ExperimentalSerializationApi
import android.app.ActivityManager
import kotlinx.serialization.Serializable
import java.io.File
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.Process
import android.os.SystemClock
import kotlinx.coroutines.*
import kotlinx.serialization.json.Json
import kotlin.math.abs

@OptIn(ExperimentalSerializationApi::class)
@Serializable
data class MetricsResult(
    val memory: MemoryMetrics,
    val power: PowerMetrics?,
    val cpu: CpuMetrics?,
    val gpu: GpuMetrics?,
    // true for hardware_metrics_resume_N.json (only the resumed questions)
    @EncodeDefault
    val resumed: Boolean = false,
    // false if the run was stopped or failed before the last question
    @EncodeDefault
    val completed: Boolean = false
)

@Serializable
data class MemoryMetrics(
    val beforeMB: Long,
    val meanMB: Long,
    val peakMB: Long
)

@Serializable
enum class CurrentMeasurementSource {
    // Standard Android BATTERY_PROPERTY_CURRENT_NOW, converted from µA to mA.
    CURRENT_NOW,

    // Standard Android BATTERY_PROPERTY_CURRENT_AVERAGE, converted from µA to mA.
    CURRENT_AVERAGE,

    // OPPO CPH2791 CURRENT_NOW, validated to behave as mA rather than the
    // standard Android µA contract.
    OPPO_CPH2791_CURRENT_NOW_MA,

    // OnePlus CPH2653 current properties could not be validated reliably, so
    // current- and power-derived values are intentionally suppressed.
    ONEPLUS_CPH2653_UNSUPPORTED,

    // Both CURRENT_NOW and CURRENT_AVERAGE contributed samples during the run.
    MIXED
}

@Serializable
data class PowerMetrics(
    val currentMa: FloatMetricStats?,
    val currentSource: CurrentMeasurementSource?,
    val voltageV: FloatMetricStats?,
    val powerMw: FloatMetricStats?,
    val temperatureC: FloatMetricStats?,
    // True if the device was connected to any external power source during sampling.
    val isPlugged: Boolean,
    // True if Android reported the battery status as charging or full during sampling.
    val isCharging: Boolean,
    val chargeCounter: ChargeCounterMetrics?
)

@Serializable
data class FloatMetricStats(
    val mean: Float,
    val peak: Float
)

@Serializable
enum class ChargeCounterInterpretation {
    // The values describe the net change in battery charge during sampling.
    // They may include charging from an external power source and therefore do
    // not necessarily represent the benchmark's total device energy consumption.
    NET_BATTERY_CHANGE
}

@Serializable
data class ChargeCounterMetrics(
    // Describes how the charge-counter fields should be interpreted.
    val interpretation: ChargeCounterInterpretation,

    // First valid BATTERY_PROPERTY_CHARGE_COUNTER value reported by Android, in microamp-hours (µAh).
    val startMicroAh: Int,

    // Last valid BATTERY_PROPERTY_CHARGE_COUNTER value reported by Android, in microamp-hours (µAh).
    val endMicroAh: Int,

    // Net charge change in microamp-hours (µAh): startMicroAh - endMicroAh.
    // Positive means net battery discharge; negative means net battery charge.
    val consumedMicroAh: Int,

    // Elapsed monotonic time in milliseconds (ms) between the first and last valid samples.
    val durationMs: Long,

    // Estimated battery-side discharged energy in joules:
    // consumedMicroAh / 1,000,000 * 3600 * meanVoltageV.
    // Null when the battery gained charge, voltage is unavailable or implausible,
    // or the charge-counter units cannot be validated.
    val estimatedEnergyJ: Float?,

    // Interval-average battery current in milliamperes (mA), derived from charge-counter change:
    // consumedMicroAh * 3600 / durationMs.
    // Positive means net discharge; negative means net charge.
    val intervalAverageBatteryCurrentMa: Float?,

    // Interval-average battery-side power in milliwatts (mW):
    // intervalAverageBatteryCurrentMa * meanVoltageV.
    // Positive means net battery discharge; negative means net battery charge.
    val intervalAverageBatteryPowerMw: Float?
)

@Serializable
data class CpuMetrics(
    val processUsagePercent: FloatMetricStats,
    // Peak is not the peak per-core CPU utilization.
    // It is the per-core average from the sample where total process CPU utilization was highest.
    val processUsagePercentPerCore: FloatMetricStats,
    val availableProcessors: Int
)

@Serializable
data class GpuMetrics(
    val usagePercent: FloatMetricStats,
    val frequencyMHz: FloatMetricStats?
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

    fun hasSamples(): Boolean = count > 0
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

class CpuSampler {
    private var previousCpuTimeMs: Long? = null
    private var previousWallTimeMs: Long? = null

    fun sample(): Float? {
        val currentCpuTimeMs = Process.getElapsedCpuTime()
        val currentWallTimeMs = SystemClock.elapsedRealtime()

        val lastCpuTimeMs = previousCpuTimeMs
        val lastWallTimeMs = previousWallTimeMs

        previousCpuTimeMs = currentCpuTimeMs
        previousWallTimeMs = currentWallTimeMs

        if (lastCpuTimeMs == null || lastWallTimeMs == null) {
            return null
        }

        val cpuDeltaMs = currentCpuTimeMs - lastCpuTimeMs
        val wallDeltaMs = currentWallTimeMs - lastWallTimeMs

        if (wallDeltaMs <= 0L) {
            return null
        }

        return (cpuDeltaMs.toFloat() / wallDeltaMs.toFloat()) * 100f
    }
}

data class GpuSample(
    val usagePercent: Float?,
    val frequencyMHz: Float?
)

class GpuSampler {
    private enum class GpuPlatform {
        QUALCOMM_ADRENO,
        MEDIATEK,
        UNKNOWN
    }

    private val gpuPlatform = detectGpuPlatform()

    private val qualcommGpuBusyPercentagePaths = listOf(
        "/sys/kernel/gpu/gpu_busy",
        "/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage",
        "/sys/devices/platform/soc/3d00000.qcom,kgsl-3d0/kgsl/kgsl-3d0/gpu_busy_percentage"
    )

    private val qualcommGpuFrequencyMHzPaths = listOf(
        "/sys/class/kgsl/kgsl-3d0/clock_mhz",
        "/sys/devices/platform/soc/3d00000.qcom,kgsl-3d0/kgsl/kgsl-3d0/clock_mhz"
    )

    private fun detectGpuPlatform(): GpuPlatform {
        val hardwareName = Build.HARDWARE.lowercase()

        return when {
            hardwareName.contains("qcom") -> GpuPlatform.QUALCOMM_ADRENO
            hardwareName.startsWith("mt") -> GpuPlatform.MEDIATEK
            else -> GpuPlatform.UNKNOWN
        }
    }

    fun sample(): GpuSample {
        return when (gpuPlatform) {
            GpuPlatform.QUALCOMM_ADRENO ->
                GpuSample(
                    usagePercent = sampleUsagePercent(),
                    frequencyMHz = sampleFrequencyMHz()
                )

            // GPU metrics are currently unsupported on MediaTek devices.
            GpuPlatform.MEDIATEK ->
                GpuSample(
                    usagePercent = null,
                    frequencyMHz = null
                )

            GpuPlatform.UNKNOWN ->
                GpuSample(
                    usagePercent = null,
                    frequencyMHz = null
                )
        }
    }

    private fun sampleUsagePercent(): Float? {
        return qualcommGpuBusyPercentagePaths
            .asSequence()
            .mapNotNull { path -> readFirstLineOrNull(path) }
            .mapNotNull { rawValue ->
                rawValue
                    .replace("%", "")
                    .trim()
                    .toFloatOrNull()
                    ?.takeIf { it in 0f..100f }
            }
            .firstOrNull()
    }

    private fun sampleFrequencyMHz(): Float? {
        return qualcommGpuFrequencyMHzPaths
            .asSequence()
            .mapNotNull { path -> readFirstLineOrNull(path) }
            .mapNotNull { rawValue ->
                rawValue
                    .trim()
                    .toFloatOrNull()
                    ?.takeIf { it.isFinite() && it >= 0f }
            }
            .firstOrNull()
    }

    private fun readFirstLineOrNull(path: String): String? {
        return try {
            File(path).useLines { lines -> lines.firstOrNull() }
        } catch (_: Exception) {
            null
        }
    }
}

class PowerSampler(private val context: Context) {
    private val batteryManager = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
    private val isOppo =
        android.os.Build.MANUFACTURER.equals("OPPO", ignoreCase = true)
    private val isOppoCph2791 =
        isOppo && android.os.Build.MODEL.equals("CPH2791", ignoreCase = true)
    private val isOnePlusCph2653 =
        android.os.Build.MANUFACTURER.equals("OnePlus", ignoreCase = true) &&
            android.os.Build.MODEL.equals("CPH2653", ignoreCase = true)

    private val currentStats = FloatStats()
    private val voltageStats = FloatStats()
    private val powerStats = FloatStats()
    private val temperatureStats = FloatStats()

    private var isPlugged = false
    private var isCharging = false
    private var usedCurrentNow = false
    private var usedCurrentAverage = false
    private var usedOppoCph2791CurrentNowNormalization = false
    private var encounteredUnsupportedOnePlusCurrent = false
    private var startChargeCounterMicroAh: Int? = null
    private var endChargeCounterMicroAh: Int? = null
    private var startChargeCounterTimeMs: Long? = null
    private var endChargeCounterTimeMs: Long? = null

    /**
     * Sample current power consumption and update internal statistics.
     * @return Current power consumption in milliwatts (mW), or null if measurement failed
     */
    fun sample(): Float? {
        try {
            // Get battery current in microamperes (μA). Some devices return 0 or Int.MIN_VALUE
            // when a current reading is unavailable, so prefer CURRENT_NOW and fall back to CURRENT_AVERAGE.
            val chargeCounterMicroAh = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)
            if (chargeCounterMicroAh != Int.MIN_VALUE && chargeCounterMicroAh > 0) {
                val sampleTimeMs = SystemClock.elapsedRealtime()
                if (startChargeCounterMicroAh == null) {
                    startChargeCounterMicroAh = chargeCounterMicroAh
                    startChargeCounterTimeMs = sampleTimeMs
                }
                endChargeCounterMicroAh = chargeCounterMicroAh
                endChargeCounterTimeMs = sampleTimeMs
            }

            val currentNowRaw =
                batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
            val currentAverageRaw =
                batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_AVERAGE)

            // Keep the source together with the raw reading so normalization is decided
            // independently for every sample.
            val hasCurrentNow =
                currentNowRaw != 0 && currentNowRaw != Int.MIN_VALUE
            val hasCurrentAverage =
                currentAverageRaw != 0 && currentAverageRaw != Int.MIN_VALUE

            // OnePlus CPH2653 exposes battery-current properties with
            // unvalidated vendor semantics. CURRENT_NOW does not follow the
            // standard microamp contract, and CURRENT_AVERAGE has not been
            // independently validated. Do not publish fabricated current or
            // power values.
            val currentReading =
                when {
                    isOnePlusCph2653 && (hasCurrentNow || hasCurrentAverage) -> {
                        encounteredUnsupportedOnePlusCurrent = true
                        null
                    }
                    hasCurrentNow -> {
                        usedCurrentNow = true
                        if (isOppoCph2791) {
                            usedOppoCph2791CurrentNowNormalization = true
                        }
                        currentNowRaw to true
                    }
                    hasCurrentAverage -> {
                        usedCurrentAverage = true
                        currentAverageRaw to false
                    }
                    else ->
                        null
                }

            // Android specifies battery current properties in microamperes. On the
            // validated OPPO CPH2791, CURRENT_NOW behaves as milliamperes instead.
            // CURRENT_AVERAGE keeps the standard conversion; it was unsupported on
            // the tested device and returned Int.MIN_VALUE.
            val signedCurrentMa =
                currentReading?.let { (rawCurrent, fromCurrentNow) ->
                    if (isOppoCph2791 && fromCurrentNow)
                        rawCurrent.toFloat()
                    else
                        rawCurrent / 1000f
                }

            // Preserve sign during normalization, then use magnitude for
            // battery-side current and power reporting.
            val currentMa = signedCurrentMa?.let(::abs)

            // Read battery-broadcast metrics independently of current availability.
            val intentFilter = IntentFilter(Intent.ACTION_BATTERY_CHANGED)
            val batteryStatus = context.registerReceiver(null, intentFilter)

            val pluggedStatus = batteryStatus?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0
            val batteryState = batteryStatus?.getIntExtra(BatteryManager.EXTRA_STATUS, -1) ?: -1
            isPlugged = isPlugged || pluggedStatus != 0
            isCharging = isCharging ||
                batteryState == BatteryManager.BATTERY_STATUS_CHARGING ||
                batteryState == BatteryManager.BATTERY_STATUS_FULL

            // Android normally reports battery voltage in millivolts.
            // Some OPPO firmware reports a truncated standard value such as 4 while exposing
            // the usable millivolt value through the vendor-specific battery_now_voltage_type extra.
            val standardVoltageMilliVolts =
                batteryStatus?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 0) ?: 0
            val oppoVoltageMilliVolts =
                batteryStatus?.getIntExtra("battery_now_voltage_type", 0) ?: 0

            val voltageMilliVolts =
                when {
                    standardVoltageMilliVolts > 1000 ->
                        standardVoltageMilliVolts
                    isOppoCph2791 && oppoVoltageMilliVolts > 1000 ->
                        oppoVoltageMilliVolts
                    else ->
                        null
                }

            val voltageV = voltageMilliVolts?.div(1000f)

            // Android reports battery temperature in tenths of a degree Celsius.
            val temperatureC =
                if (batteryStatus?.hasExtra(BatteryManager.EXTRA_TEMPERATURE) == true)
                    batteryStatus.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) / 10f
                else
                    null

            currentMa?.let { currentStats.add(it) }
            voltageV?.let { voltageStats.add(it) }
            temperatureC?.let { temperatureStats.add(it) }

            // Power is available only when both current and voltage are available.
            val powerMw =
                if (currentMa != null && voltageV != null)
                    currentMa * voltageV
                else
                    null

            powerMw?.let { powerStats.add(it) }

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
            currentMa =
                if (currentStats.hasSamples())
                    FloatMetricStats(currentStats.mean(), currentStats.peak)
                else
                    null,
            currentSource =
                when {
                    usedCurrentNow && usedCurrentAverage ->
                        CurrentMeasurementSource.MIXED
                    usedOppoCph2791CurrentNowNormalization ->
                        CurrentMeasurementSource.OPPO_CPH2791_CURRENT_NOW_MA
                    encounteredUnsupportedOnePlusCurrent ->
                        CurrentMeasurementSource.ONEPLUS_CPH2653_UNSUPPORTED
                    usedCurrentNow ->
                        CurrentMeasurementSource.CURRENT_NOW
                    usedCurrentAverage ->
                        CurrentMeasurementSource.CURRENT_AVERAGE
                    else ->
                        null
                },
            voltageV =
                if (voltageStats.hasSamples())
                    FloatMetricStats(voltageStats.mean(), voltageStats.peak)
                else
                    null,
            powerMw =
                if (powerStats.hasSamples())
                    FloatMetricStats(powerStats.mean(), powerStats.peak)
                else
                    null,
            temperatureC =
                if (temperatureStats.hasSamples())
                    FloatMetricStats(temperatureStats.mean(), temperatureStats.peak)
                else
                    null,
            isPlugged = isPlugged,
            isCharging = isCharging,
            chargeCounter = buildChargeCounterMetrics()
        )
    }

    private fun buildChargeCounterMetrics(): ChargeCounterMetrics? {
        // Charge-counter units on OnePlus CPH2653 could not be validated.
        // Suppress the block rather than labeling vendor-specific raw values as
        // microamp-hours or deriving misleading current, power, and energy values.
        if (isOnePlusCph2653) {
            return null
        }

        val start = startChargeCounterMicroAh ?: return null
        val end = endChargeCounterMicroAh ?: return null
        val startTimeMs = startChargeCounterTimeMs ?: return null
        val endTimeMs = endChargeCounterTimeMs ?: return null

        val consumed = start - end
        val durationMs = endTimeMs - startTimeMs

        val voltageMean =
            if (voltageStats.hasSamples())
                voltageStats.mean()
            else
                null

        val estimatedEnergyJ =
            if (consumed > 0 && voltageMean != null && voltageMean in 3.0f..5.0f)
                (consumed / 1_000_000f) * 3600f * voltageMean
            else
                null

        val intervalAverageBatteryCurrentMa =
            if (durationMs > 0L)
                consumed.toFloat() * 3600f / durationMs.toFloat()
            else
                null

        val intervalAverageBatteryPowerMw =
            if (intervalAverageBatteryCurrentMa != null && voltageMean != null)
                intervalAverageBatteryCurrentMa * voltageMean
            else
                null

        return ChargeCounterMetrics(
            interpretation = ChargeCounterInterpretation.NET_BATTERY_CHANGE,
            startMicroAh = start,
            endMicroAh = end,
            consumedMicroAh = consumed,
            durationMs = durationMs,
            estimatedEnergyJ = estimatedEnergyJ,
            intervalAverageBatteryCurrentMa = intervalAverageBatteryCurrentMa,
            intervalAverageBatteryPowerMw = intervalAverageBatteryPowerMw
        )
    }

    data class PowerSamplerStats(
        val currentMa: FloatMetricStats?,
        val currentSource: CurrentMeasurementSource?,
        val voltageV: FloatMetricStats?,
        val powerMw: FloatMetricStats?,
        val temperatureC: FloatMetricStats?,
        val isPlugged: Boolean,
        val isCharging: Boolean,
        val chargeCounter: ChargeCounterMetrics?
    )
}

class HardwareMetricsEngine(
    private val context: Context,
    private val scope: CoroutineScope,
    private val intervalMs: Long = 100L
) {
    private val memoryStats = LongStats()
    private val cpuStats = FloatStats()
    private val gpuUsageStats = FloatStats()
    private val gpuFrequencyStats = FloatStats()

    val powerSampler = PowerSampler(context)
    private val cpuSampler = CpuSampler()
    private val gpuSampler = GpuSampler()

    private var job: Job? = null
    private var memoryBefore: Long = 0


    init {
        // Measure memory before
        memoryBefore = MemorySampler.sample(context)
    }

    fun start() {
        cpuSampler.sample()

        job = scope.launch(Dispatchers.Default) {
            while (isActive) {
                try {
                    // Memory
                    memoryStats.add(MemorySampler.sample(context))

                    // CPU
                    cpuSampler.sample()?.let { cpuStats.add(it) }

                    // GPU
                    val gpuSample = gpuSampler.sample()
                    gpuSample.usagePercent?.let { gpuUsageStats.add(it) }
                    gpuSample.frequencyMHz?.let { gpuFrequencyStats.add(it) }

                    // Power
                    powerSampler.sample()

                } catch (_: Throwable) {
                    // swallow errors to avoid affecting benchmark
                }

                delay(intervalMs)
            }
        }
    }

    fun stop(): MetricsResult {
        // Only cancel the sampling job. The provided scope is owned by the benchmark caller.
        job?.cancel()

        memoryStats.add(MemorySampler.sample(context))
        cpuSampler.sample()?.let { cpuStats.add(it) }
        val gpuSample = gpuSampler.sample()
        gpuSample.usagePercent?.let { gpuUsageStats.add(it) }
        gpuSample.frequencyMHz?.let { gpuFrequencyStats.add(it) }
        powerSampler.sample()

        val samplerStats = powerSampler.getStats()

        val chargeCounterMetrics = samplerStats.chargeCounter

        val hasPowerMetrics =
            samplerStats.currentMa != null ||
                samplerStats.voltageV != null ||
                samplerStats.powerMw != null ||
                samplerStats.temperatureC != null ||
                chargeCounterMetrics != null

        val powerMetrics =
            if (hasPowerMetrics)
                PowerMetrics(
                    currentMa = samplerStats.currentMa,
                    currentSource = samplerStats.currentSource,
                    voltageV = samplerStats.voltageV,
                    powerMw = samplerStats.powerMw,
                    temperatureC = samplerStats.temperatureC,
                    isPlugged = samplerStats.isPlugged,
                    isCharging = samplerStats.isCharging,
                    chargeCounter = chargeCounterMetrics
                )
            else
                null

        val availableProcessors = Runtime.getRuntime().availableProcessors()
        val cpuMetrics =
            if (cpuStats.hasSamples())
                CpuMetrics(
                    processUsagePercent = FloatMetricStats(cpuStats.mean(), cpuStats.peak),
                    processUsagePercentPerCore = FloatMetricStats(
                        cpuStats.mean() / availableProcessors,
                        cpuStats.peak / availableProcessors
                    ),
                    availableProcessors = availableProcessors
                )
            else null

        val gpuMetrics =
            if (gpuUsageStats.hasSamples())
                GpuMetrics(
                    usagePercent = FloatMetricStats(gpuUsageStats.mean(), gpuUsageStats.peak),
                    frequencyMHz =
                        if (gpuFrequencyStats.hasSamples())
                            FloatMetricStats(gpuFrequencyStats.mean(), gpuFrequencyStats.peak)
                        else null
                )
            else null

        return MetricsResult(
            memory = MemoryMetrics(
                beforeMB = memoryBefore,
                meanMB = memoryStats.mean(),
                peakMB = memoryStats.peak
            ),
            power = powerMetrics,
            cpu = cpuMetrics,
            gpu = gpuMetrics
        )
    }

    fun writeMetricsJson(metrics: MetricsResult, file: File) {
        val json = Json { prettyPrint = true }
        file.parentFile?.mkdirs()
        file.writeText(json.encodeToString(MetricsResult.serializer(), metrics))
    }
}