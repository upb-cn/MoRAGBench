package com.example.cli.Progress

import ANNDataset
import com.example.cli.LiveLogger
import java.time.Instant
import kotlinx.serialization.Serializable

@Serializable
class ANNProgress() {

    @Serializable
    data class ProgressSnapshot(
        val overallState: State,
        val currentPhase: ProgressPhase,
        val dataset: ANNDatasetProgress,
        @Serializable(with = InstantSerializer::class)
        val startedAt: Instant,
        @Serializable(with = InstantSerializer::class)
        val completedAt: Instant? = null,
        val lastError: String? = null,
    )

    @Serializable
    data class ANNDatasetProgress(
        val dataset: ANNDataset,
        val state: State,
        val indexProgress: IndexProgress,
        val evaluationProgress: EvaluationProgress
    )

    @Serializable
    data class IndexProgress(
        val state: State,
        val phase: IndexPhase,
        val source: IndexSource = IndexSource.SCRATCH,
        val trainTotal: Int? = null,
        val vectorsProcessed: Int = 0
    )

    @Serializable
    data class EvaluationProgress(
        val state: State,
        val testTotal: Int?,
        val vectorsProcessed: Int = 0
    )

    @Volatile
    private lateinit var snapshot: ProgressSnapshot

    /* -------------------- READ -------------------- */

    fun getSnapshot(): ProgressSnapshot? {
        if (!::snapshot.isInitialized) {
            return null
        }

        return snapshot
    }

    /* -------------------- CORE UPDATE -------------------- */

    @Synchronized
    private fun update(block: ProgressSnapshot.() -> ProgressSnapshot) {
        snapshot = snapshot.block()
    }

    /* -------------------- GLOBAL UPDATES -------------------- */

    fun init(dataset: ANNDataset) {
        LiveLogger.info("Starting benchmark")
        snapshot = ProgressSnapshot(
            overallState = State.RUNNING,
            currentPhase = ProgressPhase.INITIALIZING,
            startedAt = Instant.now(),
            dataset = ANNDatasetProgress(
                dataset = dataset,
                state = State.NOT_STARTED,
                indexProgress = IndexProgress(
                    state = State.NOT_STARTED,
                    phase = IndexPhase.NOT_STARTED
                ),
                evaluationProgress = EvaluationProgress(
                    state = State.NOT_STARTED,
                    testTotal = dataset.limit
                )
            )
        )
    }

    fun setPhase(phase: ProgressPhase) {
        LiveLogger.info("Moved to phase: $phase")
        update {
            copy(currentPhase = phase)
        }
    }

    fun complete() {
        LiveLogger.info("Benchmark completed")
        update {
            copy(
                overallState = State.COMPLETED,
                currentPhase = ProgressPhase.COMPLETED,
                completedAt = Instant.now()
            )
        }
    }

    fun fail(error: String) {
        LiveLogger.info("Benchmark failed with error: $error")
        update {
            copy(
                overallState = State.FAILED,
                lastError = error
            )
        }
    }

    /* -------------------- TASK-LEVEL UPDATE -------------------- */

    fun updateDataset(
        block: ANNDatasetProgress.() -> ANNDatasetProgress
    ) {
        update {
            copy(
                dataset = dataset.block()
            )
        }
    }

    /* -------------------- DATASET STATE -------------------- */

    fun startDataset() {
        updateDataset {
            LiveLogger.info("******* Starting benchmark for dataset ${dataset.name} *******")
            copy(state = State.RUNNING)
        }
    }

    fun completeDataset() {
        updateDataset {
            LiveLogger.info("******* Benchmark for dataset ${dataset.name} is completed *******")
            copy(state = State.COMPLETED)
        }
    }

    /* -------------------- INDEX UPDATES -------------------- */

    fun initIndex() {
        LiveLogger.info("Initializing FAISS index")
        updateDataset {
            copy(
                indexProgress = indexProgress.copy(
                    state = State.RUNNING,
                    phase = IndexPhase.INITIALIZING
                )
            )
        }
    }

    fun getIndexSource(): IndexSource {
        return snapshot.dataset.indexProgress.source
    }

    fun setIndexSourceToCache(trainTotal: Int) {
        LiveLogger.info("Reading FAISS index from cache")
        updateDataset {
            copy(
                indexProgress = indexProgress.copy(
                    source = IndexSource.CACHE,
                    phase = IndexPhase.LOADING,
                    trainTotal = trainTotal
                )
            )
        }
    }

    fun setIndexSourceToScratch(trainTotal: Int) {
        LiveLogger.info("Building FAISS index from scratch for $trainTotal vectors")
        updateDataset {
            copy(
                indexProgress = indexProgress.copy(
                    source = IndexSource.SCRATCH,
                    trainTotal = trainTotal
                )
            )
        }
    }

    fun startTraining() {
        LiveLogger.info("Start training IVF index")
        updateDataset {
            copy(
                indexProgress = indexProgress.copy(
                    phase = IndexPhase.TRAINING
                )
            )
        }
    }

    fun startBuilding() {
        LiveLogger.info("Building index...")
        updateDataset {
            copy(
                indexProgress = indexProgress.copy(
                    phase = IndexPhase.BUILDING
                )
            )
        }
    }

    fun incrementTrainVectors(i: Int) {
        updateDataset {
            val idx = indexProgress
            LiveLogger.info("Processing vector: ${idx.vectorsProcessed + i}/${idx.trainTotal}")
            copy(
                indexProgress = idx.copy(
                    vectorsProcessed = idx.vectorsProcessed + i,
                )
            )
        }
    }

    fun startSaving() {
        LiveLogger.info("Saving index to disk")
        updateDataset {
            copy(
                indexProgress = indexProgress.copy(
                    phase = IndexPhase.SAVING
                )
            )
        }
    }

    fun completeIndex() {
        LiveLogger.info("Building index completed")
        updateDataset {
            copy(
                indexProgress = indexProgress.copy(
                    state = State.COMPLETED,
                    phase = IndexPhase.COMPLETED,
                    vectorsProcessed = indexProgress.trainTotal!!
                )
            )
        }
    }

    /* -------------------- EVALUATION UPDATES -------------------- */

    fun startEvaluation() {
        LiveLogger.info("Starting evaluation")
        updateDataset {
            copy(
                evaluationProgress = evaluationProgress.copy(
                    state = State.RUNNING
                )
            )
        }
    }

    fun incrementTestVector(i: Int) =
        updateDataset {
            val ev = evaluationProgress
            LiveLogger.info("Processing vector: ${ev.vectorsProcessed + i}/${ev.testTotal}")
            copy(
                evaluationProgress = ev.copy(
                    vectorsProcessed = ev.vectorsProcessed + i
                )
            )
        }

    fun completeEvaluation() {
        LiveLogger.info("Evaluation completed")
        updateDataset {
            copy(
                evaluationProgress = evaluationProgress.copy(
                    state = State.COMPLETED,
                    vectorsProcessed = evaluationProgress.testTotal!!
                )
            )
        }
    }
}