package com.example.cli

import android.content.Context
import com.example.cli.Progress.TaskProgress
import com.example.cli.Progress.ANNProgress
import fi.iki.elonen.NanoHTTPD
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import org.json.JSONArray
import org.json.JSONObject

class HttpServer(private val context: Context, port: Int = Constants.PORT): NanoHTTPD(port) {
    override fun serve(session: IHTTPSession): Response {
        val uri = session.uri
        try {
            when (uri) {
                "/ping" -> {
                    val resp = JSONObject()
                    resp.put("status", "ok")
                    resp.put("onnx_loaded", BenchmarkManager.isLoaded("onnx"))
                    resp.put("litert_loaded", BenchmarkManager.isLoaded("litert"))
                    return newFixedLengthResponse(Response.Status.OK, "application/json", resp.toString())
                }
                "/status" -> {
                    val json = Json {
                        prettyPrint = true
                        encodeDefaults = true
                        explicitNulls = true
                    }

                    val taskSnapshot = BenchmarkManager.taskBenchmark?.progress?.getSnapshot()
                    val annSnapshot = BenchmarkManager.annBenchmark?.progress?.getSnapshot()

                    if (taskSnapshot == null && annSnapshot == null) {
                        return newFixedLengthResponse(
                            Response.Status.NO_CONTENT,
                            "application/json",
                            ""
                        )
                    } else {
                        val currentTestType = BenchmarkManager.testType
                        val jsonString = if (currentTestType == "task") {
                            json.encodeToString(
                                TaskProgress.ProgressSnapshot.serializer(),
                                taskSnapshot!!
                            )
                        } else {
                            json.encodeToString(
                                ANNProgress.ProgressSnapshot.serializer(),
                                annSnapshot!!
                            )
                        }

                        return newFixedLengthResponse(
                            Response.Status.OK,
                            "application/json",
                            jsonString
                        )
                    }
                }
                "/start_benchmark" -> {
                    if (session.method != Method.POST) {
                        return newFixedLengthResponse(
                            Response.Status.METHOD_NOT_ALLOWED,
                            "text/plain",
                            "Use POST"
                        )
                    }

                    // Parse body into a map
                    val body = HashMap<String, String>()
                    session.parseBody(body)

                    // Raw body is stored under "postData"
                    val rawJson = body["postData"] ?: ""

                    // Parse JSON
                    val json = JSONObject(rawJson)
                    val testType = json.optString("test_type", "task")
                    val resume = json.optBoolean("resume", false)
                    val testStatus = BenchmarkManager.startBenchmarkAsync(testType,resume)

                    return newFixedLengthResponse(
                        Response.Status.OK,
                        "application/json",
                        """{"status":$testStatus,"test_type":"$testType"}"""
                    )
                }
                "/prepare_dirs" -> {
                    if (session.method != Method.POST) {
                        return newFixedLengthResponse(
                            Response.Status.METHOD_NOT_ALLOWED,
                            "text/plain",
                            "Use POST"
                        )
                    }

                    // Read POST body
                    val body = HashMap<String, String>()
                    session.parseBody(body)
                    val rawJson = body["postData"] ?: ""

                    val req = JSONObject(rawJson)
                    val dirsArray = req.optJSONArray("dirs") ?: JSONArray()

                    val base = context.getExternalFilesDir(null)
                        ?: throw IllegalStateException("External files dir unavailable")
                    val basePath = base.canonicalPath

                    val created = JSONArray()
                    for (i in 0 until dirsArray.length()) {
                        val relative = dirsArray.getString(i)
                        val target = base.resolve(relative)

                        // Guard against path traversal escaping the sandbox
                        if (!target.canonicalPath.startsWith(basePath)) {
                            throw SecurityException("Illegal path outside sandbox: $relative")
                        }

                        target.mkdirs()
                        created.put(target.absolutePath)
                    }

                    val resp = JSONObject()
                    resp.put("created", created)
                    resp.put("base", base.absolutePath)

                    return newFixedLengthResponse(
                        Response.Status.OK,
                        "application/json",
                        resp.toString()
                    )
                }
                "/generate" -> return handleGenerate(session)
                "/generate_litert" -> return handleGenerate(session)
            }
            return newFixedLengthResponse(Response.Status.NOT_FOUND, "text/plain", "Not found")
        } catch (e: Exception) {
            val j = JSONObject()
            j.put("error", e.message)
            return newFixedLengthResponse(Response.Status.INTERNAL_ERROR, "application/json", j.toString())
        }
    }

    private fun handleGenerate(session: IHTTPSession): Response {
        if (session.method != Method.POST) {
            return newFixedLengthResponse(
                Response.Status.METHOD_NOT_ALLOWED,
                "text/plain",
                "Use POST"
            )
        }

        val body = HashMap<String, String>()
        session.parseBody(body)
        val rawJson = body["postData"] ?: ""
        val json = JSONObject(rawJson)
        val prompt = json.optString("prompt", "")
        val maxTokens = json.optInt("max_tokens", 512)
        val systemPrompt = json.optString("system_prompt", "")
        val timeoutMs = json.optLong("timeout_ms", 120_000L)

        if (prompt.isBlank()) {
            return newFixedLengthResponse(
                Response.Status.BAD_REQUEST,
                "application/json",
                """{"error":"prompt is required"}"""
            )
        }

        val taskConfig = Parser(context).readTaskConfig()
        val llmConfig = taskConfig.ragPipeline.llm
        val engine = BenchmarkManager.getInferenceEngine()

        if (!engine.isLoaded()) {
            val modelDir = context.getExternalFilesDir(null)!!
                .resolve(Constants.LLM_DIR)
                .resolve("${llmConfig.modelName}_${llmConfig.dtype}")
            val init = runBlocking { engine.initialize(context, llmConfig, modelDir) }
            if (!init.success) {
                val err = JSONObject()
                err.put("error", init.error ?: "LLM init failed")
                return newFixedLengthResponse(
                    Response.Status.INTERNAL_ERROR,
                    "application/json",
                    err.toString()
                )
            }
            BenchmarkManager.lastInferenceLoadTimeMs = init.loadTimeMs
        }

        val result = runBlocking {
            engine.generate(
                InferenceEngine.GenerationRequest(
                    prompt = prompt,
                    systemPrompt = systemPrompt,
                    maxTokens = maxTokens,
                    timeoutMs = timeoutMs
                )
            )
        }

        val resp = JSONObject()
        resp.put("response", result.response)
        val metricsObj = JSONObject()
        metricsObj.put("status", result.status.name)
        metricsObj.put("input_tokens", result.inputTokens)
        metricsObj.put("generated_tokens", result.generatedTokens)
        metricsObj.put("ttft_ms", result.ttftMs)
        metricsObj.put("load_time_ms", BenchmarkManager.lastInferenceLoadTimeMs)
        metricsObj.put("overall_duration_ms", result.overallDurationMs)
        metricsObj.put("decoding_speed_tokens_per_sec", result.decodingSpeedTokensPerSec)
        if (result.error != null) {
            metricsObj.put("error", result.error)
        }
        resp.put("metrics", metricsObj)

        return newFixedLengthResponse(
            Response.Status.OK,
            "application/json",
            resp.toString()
        )
    }
}
