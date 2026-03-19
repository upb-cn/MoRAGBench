package com.example.cli
import ANNConfig
import TaskConfig
import android.content.Context
import java.io.File
import kotlinx.serialization.json.*

class Parser(private val context: Context) {

    fun readDownstreamTaskFiles(taskName: String): Map<String, JsonElement> {

        val taskDir = context.getExternalFilesDir(null)
            ?.resolve(Constants.DOWNSTREAM_TASK_DIR)
            ?.resolve(taskName)

        val documentsFile = File(taskDir, "documents.json")
        val questionsFile = File(taskDir, "questions.json")

        val results = mutableMapOf<String, JsonElement>()
        results["documents"] = Json.parseToJsonElement(documentsFile.readText())
        results["questions"] = Json.parseToJsonElement(questionsFile.readText())

        return results
    }

    fun readTaskConfig(): TaskConfig {
        val configFile = context.getExternalFilesDir(null)!!.resolve(Constants.TASK_BASE_DIR).resolve(
            Constants.CONFIG_FILE)
        val jsonString = configFile.readText(Charsets.UTF_8)
        val config = Json.decodeFromString<TaskConfig>(jsonString)
        return config
    }

    fun readAnnConfig(): ANNConfig {
        val configFile = context.getExternalFilesDir(null)!!.resolve(Constants.ANN_BASE_DIR).resolve(
            Constants.CONFIG_FILE)
        val jsonString = configFile.readText(Charsets.UTF_8)
        val config = Json.decodeFromString<ANNConfig>(jsonString)
        return config
    }

    fun getTokenizerPath(modelName: String): String {
        return context.getExternalFilesDir(null)!!
            .resolve(Constants.EMBEDDING_DIR)
            .resolve(modelName)
            .resolve("tokenizer.json")
            .absolutePath

    }


}
