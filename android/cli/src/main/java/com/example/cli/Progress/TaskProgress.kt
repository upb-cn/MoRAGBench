package com.example.cli.Progress

import DownstreamTask
import com.example.cli.LiveLogger
import java.time.Instant
import kotlinx.serialization.Serializable

@Serializable
class TaskProgress() {

    @Serializable
    data class ProgressSnapshot(
        val overallState: State,
        val currentPhase: ProgressPhase,
        val task: DownstreamTaskProgress,
        @Serializable(with = InstantSerializer::class)
        val startedAt: Instant,
        @Serializable(with = InstantSerializer::class)
        val completedAt: Instant? = null,
        val lastError: String? = null,
    )

    @Serializable
    data class DownstreamTaskProgress(
        val task: DownstreamTask,
        val state: State,
        val indexProgress: IndexProgress,
        val evaluationProgress: EvaluationProgress
    )

    @Serializable
    data class IndexProgress(
        val state: State,
        val phase: IndexPhase,
        val source: IndexSource = IndexSource.SCRATCH,
        val documentsTotal: Int? = null,
        val documentsProcessed: Int = 0,
        val chunksProcessed: Int = 0
    )

    @Serializable
    data class EvaluationProgress(
        val state: State,
        val questionsTotal: Int?,
        val questionsProcessed: Int = 0
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

    fun init(task: DownstreamTask) {
        LiveLogger.info("Starting benchmark")
        snapshot = ProgressSnapshot(
            overallState = State.RUNNING,
            currentPhase = ProgressPhase.INITIALIZING,
            startedAt = Instant.now(),
            task = DownstreamTaskProgress(
                    task = task,
                    state = State.NOT_STARTED,
                    indexProgress = IndexProgress(
                        state = State.NOT_STARTED,
                        phase = IndexPhase.NOT_STARTED
                    ),
                    evaluationProgress = EvaluationProgress(
                        state = State.NOT_STARTED,
                        questionsTotal = task.limit
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

    fun updateTask(
        block: DownstreamTaskProgress.() -> DownstreamTaskProgress
    ) {
        update {
            copy(
                task = task.block()
            )
        }
    }

    /* -------------------- TASK STATE -------------------- */

    fun startTask() {
        updateTask {
            LiveLogger.info("******* Starting benchmark for task ${task.name} *******")
            copy(state = State.RUNNING)
        }
    }

    fun completeTask() {
        updateTask {
            LiveLogger.info("******* Benchmark for task ${task.name} is completed *******")
            copy(state = State.COMPLETED)
        }
    }

    /* -------------------- INDEX UPDATES -------------------- */

    fun initIndex() {
        LiveLogger.info("Initializing FAISS index")
        updateTask {
            copy(
                indexProgress = indexProgress.copy(
                    state = State.RUNNING,
                    phase = IndexPhase.INITIALIZING
                )
            )
        }
    }

    fun getIndexSource(): IndexSource {
        return snapshot.task.indexProgress.source
    }

    fun setIndexSourceToCache(documentsTotal: Int) {
        LiveLogger.info("Reading FAISS index from cache")
        updateTask {
            copy(
                indexProgress = indexProgress.copy(
                    source = IndexSource.CACHE,
                    phase = IndexPhase.LOADING,
                    documentsTotal = documentsTotal
                )
            )
        }
    }

    fun setIndexSourceToScratch(documentsTotal: Int) {
        LiveLogger.info("Building FAISS index from scratch for $documentsTotal documents")
        updateTask {
            copy(
                indexProgress = indexProgress.copy(
                    source = IndexSource.SCRATCH,
                    documentsTotal = documentsTotal
                )
            )
        }
    }

    fun startTraining() {
        LiveLogger.info("Start training IVF index")
        updateTask {
            copy(
                indexProgress = indexProgress.copy(
                    phase = IndexPhase.TRAINING
                )
            )
        }
    }

    fun startBuilding() {
        LiveLogger.info("Building index...")
        updateTask {
            copy(
                indexProgress = indexProgress.copy(
                    phase = IndexPhase.BUILDING
                )
            )
        }
    }

    fun incrementDocuments() {
        updateTask {
            val idx = indexProgress
            LiveLogger.info("Processing documents: ${idx.documentsProcessed + 1}/${idx.documentsTotal}")
            copy(
                indexProgress = idx.copy(
                    documentsProcessed = idx.documentsProcessed + 1,
                )
            )
        }
    }

    fun incrementChunks(i: Int) =
        updateTask {
            val idx = indexProgress
             LiveLogger.info("Processed chunks: ${idx.chunksProcessed + i}")
            copy(
                indexProgress = idx.copy(
                    chunksProcessed = idx.chunksProcessed + i
                )
            )
        }

    fun startSaving() {
        LiveLogger.info("Saving index to disk")
        updateTask {
            copy(
                indexProgress = indexProgress.copy(
                    phase = IndexPhase.SAVING
                )
            )
        }
    }

    fun completeIndex() {
        LiveLogger.info("Building index completed")
        updateTask {
            copy(
                indexProgress = indexProgress.copy(
                    state = State.COMPLETED,
                    phase = IndexPhase.COMPLETED
                )
            )
        }
    }

    /* -------------------- EVALUATION UPDATES -------------------- */

    fun startEvaluation() {
        LiveLogger.info("Starting evaluation")
        updateTask {
            copy(
                evaluationProgress = evaluationProgress.copy(
                    state = State.RUNNING
                )
            )
        }
    }

    fun incrementQuestion() =
        updateTask {
            val ev = evaluationProgress
            LiveLogger.info("Processing questions: ${ev.questionsProcessed + 1}/${ev.questionsTotal}")
            copy(
                evaluationProgress = ev.copy(
                    questionsProcessed = ev.questionsProcessed + 1
                )
            )
        }

    fun completeEvaluation() {
        LiveLogger.info("Evaluation completed")
        updateTask {
            copy(
                evaluationProgress = evaluationProgress.copy(
                    state = State.COMPLETED
                )
            )
        }
    }

}