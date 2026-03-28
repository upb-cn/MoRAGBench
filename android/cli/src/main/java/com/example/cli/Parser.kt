package com.example.cli
import ANNConfig
import TaskConfig
import android.content.Context
import android.util.JsonReader
import java.io.File
import kotlinx.serialization.json.*

class Parser(private val context: Context) {

    fun readDownstreamTaskFiles(taskName: String): Map<String, JsonElement> {

        val taskDir = context.getExternalFilesDir(null)
            ?.resolve(Constants.DOWNSTREAM_TASK_DIR)
            ?.resolve(taskName)

        val questionsFile = File(taskDir, "questions.json")

        val results = mutableMapOf<String, JsonElement>()
        results["questions"] = Json.parseToJsonElement(questionsFile.readText())

        return results
    }

    private fun getDocumentsFile(taskName: String): File {
        return context.getExternalFilesDir(null)!!
            .resolve(Constants.DOWNSTREAM_TASK_DIR)
            .resolve(taskName)
            .resolve("documents.json")
    }

    fun countDocuments(taskName: String): Int {
        var count = 0
        JsonReader(getDocumentsFile(taskName).reader()).use { reader ->
            reader.beginObject()
            while (reader.hasNext()) {
                reader.nextName()
                reader.skipValue()
                count++
            }
            reader.endObject()
        }
        return count
    }

    fun forEachDocument(taskName: String, action: (String, String) -> Unit) {
        JsonReader(getDocumentsFile(taskName).reader()).use { reader ->
            reader.beginObject()
            while (reader.hasNext()) {
                val docId = reader.nextName()
                val text = reader.nextString()
                action(docId, text)
            }
            reader.endObject()
        }
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
