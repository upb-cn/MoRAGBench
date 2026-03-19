package com.example.moragbench

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity

object SupportedOptions {
    object Chunker {
        val METHODS = arrayOf("Token", "Word", "Character")
    }

    object Embedding {
        val BACKENDS = arrayOf("CPU", "XNNPACK", "NNAPI")
        val MODELS = arrayOf("all-minilm-l6-v2", "all-minilm-l12-v2")
    }

    object FAISS {
        val INDEX_BACKENDS = arrayOf("CPU") // Currently, only CPU is supported due to cross-compile
        val INDEX_METHODS = arrayOf("Flat", "IVF", "HNSW")
        val SEARCH_BACKENDS = arrayOf("CPU") // Same as above
        val DISTANCE_METRICS = arrayOf("L2", "IP")
    }

    object LLM {
        val AUG_METHODS = arrayOf("Concatenation")
        val GEN_BACKENDS = arrayOf("CPU", "XNNPACK", "NNAPI")
        val MODELS = arrayOf("Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B-Instruct-Int8", "Qwen2.5-0.5B-Instruct-Q4", "Qwen2.5-1.5B-Instruct-Int8")
    }
}

class SettingsActivity : AppCompatActivity() {

    private lateinit var settings: SettingsManager
    private lateinit var helper: Helper

    // Visibility groups for FAISS
    private lateinit var grpFaissIVF: LinearLayout
    private lateinit var grpFaissHNSW: LinearLayout

    // Visibility group for LLM sampling
    private lateinit var llSamplingGroup: LinearLayout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)
        findViewById<ImageButton>(R.id.backButton).setOnClickListener {
            finish()
        }

        settings = SettingsManager(this)
        helper = Helper(this)


        // ----- Text Chunker -----
        val spChunkMethod: Spinner = findViewById(R.id.spChunkMethod)
        val etChunkSize: EditText = findViewById(R.id.etChunkSize)
        val swChunkOverlap: Switch = findViewById(R.id.swChunkOverlap)
        val etChunkOverlap: EditText = findViewById(R.id.etChunkOverlap)

        // ----- Embedding -----
        val spEmbedBackend: Spinner = findViewById(R.id.spEmbedBackend)
        val spEmbedModel: Spinner = findViewById(R.id.spEmbedModel)

        // ----- FAISS -----
        val spFaissIndexBackend: Spinner = findViewById(R.id.spFaissIndexBackend)
        val spFaissSearchBackend: Spinner = findViewById(R.id.spFaissSearchBackend)
        val spFaissMetric: Spinner = findViewById(R.id.spFaissMetric)
        val etFaissTopK: EditText = findViewById(R.id.etFaissTopK)
        val etFaissBatchSize: EditText = findViewById(R.id.etFaissBatchSize)
        val spFaissIndexMethod: Spinner = findViewById(R.id.spFaissIndexMethod)

        // Group visibility toggling
        grpFaissIVF = findViewById(R.id.grpFaissIVF)
        grpFaissHNSW = findViewById(R.id.grpFaissHNSW)

        val etFaissNprobe: EditText = findViewById(R.id.etFaissNprobe)
        val etFaissNlist: EditText = findViewById(R.id.etFaissNlist)
        val etFaissNumTrainVectors: EditText = findViewById(R.id.etFaissNumTrainVectors)

        val etFaissM: EditText = findViewById(R.id.etFaissM)
        val etFaissEfConstruction: EditText = findViewById(R.id.etFaissEfConstruction)
        val etFaissEfSearch: EditText = findViewById(R.id.etFaissEfSearch)

        spFaissIndexMethod.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(
                parent: AdapterView<*>?,
                view: View?,
                position: Int,
                id: Long
            ) {
                val selected = parent?.getItemAtPosition(position)?.toString() ?: "Flat"
                when (selected) {
                    "Flat" -> showFlat()
                    "IVF" -> showIVF()
                    "HNSW" -> showHNSW()
                    else -> showFlat() // fallback
                }
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {
                showFlat()
            }
        }


        // ----- LLM -----
        val spLlmAugMethod: Spinner = findViewById(R.id.spLlmAugMethod)
        val spLlmGenBackend: Spinner = findViewById(R.id.spLlmGenBackend)
        val spLlmModel: Spinner = findViewById(R.id.spLlmModel)
        val cbLlmUseSampling: CheckBox = findViewById(R.id.cbLlmUseSampling)
        val etLlmMaxTok: EditText = findViewById(R.id.etLlmMaxTok)
        val etLlmRepetition: EditText = findViewById(R.id.etLlmRepetition)
        val etLlmTemp: EditText = findViewById(R.id.etLlmTemp)
        val etLlmTopP: EditText = findViewById(R.id.etLlmTopP)
        val etLlmTopK: EditText = findViewById(R.id.etLlmTopK)
        val etLlmSystem: EditText = findViewById(R.id.etLlmSystem)

        // Group visibility toggling
        llSamplingGroup = findViewById(R.id.llSamplingGroup)

        // toggle sampling fields
        cbLlmUseSampling.setOnCheckedChangeListener { _, checked ->
            llSamplingGroup.visibility = if (checked) View.VISIBLE else View.GONE
        }


        val btnSave: Button = findViewById(R.id.btnSave)
        val btnReset: Button = findViewById(R.id.btnResetDefaults)

        // Utility extension for Spinner population
        fun Spinner.load(items: Array<String>) {
            adapter = ArrayAdapter(this@SettingsActivity, android.R.layout.simple_spinner_dropdown_item, items)
        }

        // --- Populate spinner values ---
        spChunkMethod.load(SupportedOptions.Chunker.METHODS)
        spEmbedBackend.load(SupportedOptions.Embedding.BACKENDS)
        spEmbedModel.load(SupportedOptions.Embedding.MODELS)
        spFaissIndexBackend.load(SupportedOptions.FAISS.INDEX_BACKENDS)
        spFaissIndexMethod.load(SupportedOptions.FAISS.INDEX_METHODS)
        spFaissSearchBackend.load(SupportedOptions.FAISS.SEARCH_BACKENDS)
        spFaissMetric.load(SupportedOptions.FAISS.DISTANCE_METRICS)
        spLlmAugMethod.load(SupportedOptions.LLM.AUG_METHODS)
        spLlmGenBackend.load(SupportedOptions.LLM.GEN_BACKENDS)
        spLlmModel.load(SupportedOptions.LLM.MODELS)

        // --- Load current settings ---
        val chunk = settings.chunker
        spChunkMethod.setSelection(indexOf(spChunkMethod, chunk.method))
        etChunkSize.setText(chunk.size.toString())
        swChunkOverlap.isChecked = chunk.overlapEnabled
        etChunkOverlap.setText(chunk.overlapSize.toString())
        updateOverlapEnabled(swChunkOverlap, etChunkOverlap)

        swChunkOverlap.setOnCheckedChangeListener { _, _ ->
            updateOverlapEnabled(swChunkOverlap, etChunkOverlap)
        }

        val embed = settings.embedding
        spEmbedBackend.setSelection(indexOf(spEmbedBackend, embed.backend))
        spEmbedModel.setSelection(indexOf(spEmbedModel, embed.model))

        val faiss = settings.faiss
        spFaissIndexBackend.setSelection(indexOf(spFaissIndexBackend, faiss.indexBackend))
        spFaissSearchBackend.setSelection(indexOf(spFaissSearchBackend, faiss.searchBackend))
        spFaissMetric.setSelection(indexOf(spFaissMetric, faiss.metric))
        etFaissTopK.setText(faiss.topK.toString())
        etFaissBatchSize.setText(faiss.batchSize.toString())
        spFaissIndexMethod.setSelection(indexOf(spFaissIndexMethod, faiss.indexMethod))
        etFaissNprobe.setText(faiss.nprobe.toString())
        etFaissNlist.setText(faiss.nlist.toString())
        etFaissNumTrainVectors.setText(faiss.numTrainingVectors.toString())
        etFaissM.setText(faiss.m.toString())
        etFaissEfConstruction.setText(faiss.efConstruction.toString())
        etFaissEfSearch.setText(faiss.efSearch.toString())

        val llm = settings.llm
        spLlmAugMethod.setSelection(indexOf(spLlmAugMethod, llm.augMethod))
        spLlmGenBackend.setSelection(indexOf(spLlmGenBackend, llm.genBackend))
        spLlmModel.setSelection(indexOf(spLlmModel, llm.model))

        cbLlmUseSampling.isChecked = llm.useSampling
        etLlmRepetition.setText(llm.repetitionPenalty.toString())
        etLlmTemp.setText(llm.temp.toString())
        etLlmTopP.setText(llm.topP.toString())
        etLlmTopK.setText(llm.topK.toString())
        etLlmMaxTok.setText(llm.maxTokens.toString())
        etLlmSystem.setText(llm.systemPrompt)

        // --- Save button ---
        btnSave.setOnClickListener {

            // Sanity checks before saving
            // 1. Check that all EditText are not empty
            listOf(
                etChunkSize,
                etChunkOverlap,
                etFaissTopK,
                etFaissBatchSize,
                etLlmMaxTok
            ).forEach {
                if (it.getIntOrNullWithError() == null) {
                    return@setOnClickListener
                }
            }

            listOf(
                etLlmRepetition
            ).forEach {
                if (it.getFloatOrNullWithError() == null) {
                    return@setOnClickListener
                }
            }

            if (etLlmSystem.getStringOrNullWithError() == null) {
                return@setOnClickListener
            }

            // 2. Special handling for FAISS config params
            val selected = spFaissIndexMethod.selectedItem.toString()
            when (selected) {
                "IVF" -> {
                    val nprobe = etFaissNprobe.getIntOrNullWithError()
                    val nlist = etFaissNlist.getIntOrNullWithError()
                    val numTrain = etFaissNumTrainVectors.getIntOrNullWithError()
                    if (nprobe == null || nlist == null || numTrain == null) {
                        return@setOnClickListener
                    }
                }
                "HNSW" -> {
                    val m = etFaissM.getIntOrNullWithError()
                    val efC = etFaissEfConstruction.getIntOrNullWithError()
                    val efS = etFaissEfSearch.getIntOrNullWithError()
                    if (m == null || efC == null || efS == null) {
                        return@setOnClickListener
                    }
                }
            }


            // 3. Overlap size must be <= chunk size
            val overlapValue = helper.parseInt(etChunkOverlap)
            val chunkSizeValue = helper.parseInt(etChunkSize)
            if (swChunkOverlap.isChecked && overlapValue > chunkSizeValue) {
                // Mark the overlap field as invalid
                etChunkOverlap.error = "Overlap size must be <= chunk size"
                return@setOnClickListener
            }

            // 4. nprobe must be <= nlist (you cannot probe more clusters than exist).
            if (helper.parseInt(etFaissNprobe) > helper.parseInt(etFaissNlist)) {
                etFaissNprobe.error = "nprobe must be <= nlist"
                return@setOnClickListener
            }

            // 5. LLM options check
            // a. if doSample is true, check the params if they are valid
            if (cbLlmUseSampling.isChecked) {
                val temp = etLlmTemp.getFloatOrNullWithError()
                val topP = etLlmTopP.getFloatOrNullWithError()
                val topL = etFaissNlist.getIntOrNullWithError()
                val topK = etLlmTopK.getIntOrNullWithError()
                if (temp == null || topP == null || topL == null || topK == null) {
                    return@setOnClickListener
                }

                // topP should be less than 1
                if (topP > 1) {
                    etLlmTopP.error = "Top-P should be less than 1"
                    return@setOnClickListener
                }

                // topK should be > 0
                if (topK <= 0) {
                    etLlmTopK.error = "Top-K should be > 0"
                    return@setOnClickListener
                }
            }

            // b. Repetition penalty should be > 1
            if (helper.parseFloat(etLlmRepetition) < 1) {
                etLlmRepetition.error = "Repetition penalty should be > 1"
                return@setOnClickListener
            }


            chunk.method = spChunkMethod.selectedItem.toString()
            chunk.size = etChunkSize.text.toString().toIntOrNull() ?: chunk.size
            chunk.overlapEnabled = swChunkOverlap.isChecked
            chunk.overlapSize = etChunkOverlap.text.toString().toIntOrNull() ?: chunk.overlapSize

            embed.backend = spEmbedBackend.selectedItem.toString()
            embed.model = spEmbedModel.selectedItem.toString()

            faiss.indexBackend = spFaissIndexBackend.selectedItem.toString()
            faiss.searchBackend = spFaissSearchBackend.selectedItem.toString()
            faiss.metric = spFaissMetric.selectedItem.toString()
            faiss.topK = etFaissTopK.text.toString().toIntOrNull() ?: faiss.topK
            faiss.batchSize = etFaissBatchSize.text.toString().toIntOrNull() ?: faiss.batchSize
            faiss.indexMethod = spFaissIndexMethod.selectedItem.toString()
            faiss.nprobe = etFaissNprobe.text.toString().toIntOrNull() ?: faiss.nprobe
            faiss.nlist = etFaissNlist.text.toString().toIntOrNull() ?: faiss.nlist
            faiss.numTrainingVectors = etFaissNumTrainVectors.text.toString().toIntOrNull() ?: faiss.numTrainingVectors
            faiss.m = etFaissM.text.toString().toIntOrNull() ?: faiss.m
            faiss.efConstruction = etFaissEfConstruction.text.toString().toIntOrNull() ?: faiss.efConstruction
            faiss.efSearch = etFaissEfSearch.text.toString().toIntOrNull() ?: faiss.efSearch


            llm.augMethod = spLlmAugMethod.selectedItem.toString()
            llm.genBackend = spLlmGenBackend.selectedItem.toString()
            llm.model = spLlmModel.selectedItem.toString()
            llm.useSampling = cbLlmUseSampling.isChecked
            llm.repetitionPenalty = etLlmRepetition.text.toString().toFloatOrNull() ?: llm.repetitionPenalty
            llm.temp = etLlmTemp.text.toString().toFloatOrNull() ?: llm.temp
            llm.topP = etLlmTopP.text.toString().toFloatOrNull() ?: llm.topP
            llm.topK = etLlmTopK.text.toString().toIntOrNull() ?: llm.topK
            llm.maxTokens = etLlmMaxTok.text.toString().toIntOrNull() ?: llm.maxTokens
            llm.systemPrompt = etLlmSystem.text.toString()

            Toast.makeText(this, "Settings saved successfully", Toast.LENGTH_SHORT).show()

           goToMainActivity()
        }

        // Reset button
        btnReset.setOnClickListener {
            val dialog = androidx.appcompat.app.AlertDialog.Builder(this)
                .setTitle("Reset All Settings")
                .setMessage("Are you sure you want to restore all settings to their default values?")
                .setPositiveButton("Reset") { _, _ ->
                    // Reset SharedPreferences
                    settings.resetToDefaults()

                    helper.toast("Settings restored to defaults")

                    goToMainActivity()
                }
                .setNegativeButton("Cancel", null)
                .create()

            dialog.show()
        }
    }

    private fun EditText.getStringOrNullWithError(): String? {
        val s = this.text?.toString()?.trim()
        if (s.isNullOrEmpty()) {
            this.error = "This field is required"
            return null
        }

        return s
    }

    private fun EditText.getIntOrNullWithError(): Int? {
        val s = this.text?.toString()?.trim()
        if (s.isNullOrEmpty()) {
            this.error = "This field is required"
            return null
        }
        val v = s.toIntOrNull()
        if (v == null) {
            this.error = "This field must be an integer"
            return null
        }
        if (v <= 0) {
            this.error = "This field must be > 0"
            return null
        }
        return v
    }

    private fun EditText.getFloatOrNullWithError(): Float? {
        val s = this.text?.toString()?.trim()
        if (s.isNullOrEmpty()) {
            this.error = "This field is required"
            return null
        }
        val v = s.toFloatOrNull()
        if (v == null) {
            this.error = "This field must be a float"
            return null
        }
        if (v <= 0) {
            this.error = "This field must be > 0"
            return null
        }
        return v
    }

    private fun showFlat() {
        grpFaissIVF.visibility = View.GONE
        grpFaissHNSW.visibility = View.GONE
    }

    private fun showIVF() {
        grpFaissIVF.visibility = View.VISIBLE
        grpFaissHNSW.visibility = View.GONE
    }

    private fun showHNSW() {
        grpFaissIVF.visibility = View.GONE
        grpFaissHNSW.visibility = View.VISIBLE
    }

    fun goToMainActivity() {
        // Go to MainActivity
        val intent = Intent(this, MainActivity::class.java)
        // Flag to alert MainActivity that settings were changed
        intent.putExtra("from_settings", true)
        intent.addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
        startActivity(intent)
        finish()
    }

    private fun indexOf(spinner: Spinner, value: String): Int {
        for (i in 0 until spinner.adapter.count) {
            if (spinner.adapter.getItem(i)?.toString() == value) return i
        }
        return 0
    }

    private fun updateOverlapEnabled(switch: Switch, overlapField: EditText) {
        overlapField.isEnabled = switch.isChecked
        if (!switch.isChecked) overlapField.setText("0")
    }
}
