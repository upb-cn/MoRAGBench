package com.example.cli.Progress

import kotlinx.serialization.KSerializer
import kotlinx.serialization.Serializable
import kotlinx.serialization.descriptors.PrimitiveKind
import kotlinx.serialization.descriptors.PrimitiveSerialDescriptor
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import java.time.Instant

@Serializable
enum class ProgressPhase {
    INITIALIZING,
    EVALUATING,
    COMPLETED
}

@Serializable
enum class State {
    NOT_STARTED,
    RUNNING,
    COMPLETED,
    FAILED
}

@Serializable
enum class IndexSource {
    CACHE,
    SCRATCH
}

@Serializable
enum class IndexPhase {
    NOT_STARTED,
    INITIALIZING,
    TRAINING,        // IVF only,
    BUILDING,
    SAVING,
    LOADING,
    COMPLETED
}


// kotlinx.serialization does not support java.time.Instant out of the box.
// Custom serializer.
object InstantSerializer : KSerializer<Instant> {
    override val descriptor: SerialDescriptor =
        PrimitiveSerialDescriptor("Instant", PrimitiveKind.STRING)

    override fun serialize(encoder: Encoder, value: Instant) {
        encoder.encodeString(value.toString()) // ISO-8601
    }

    override fun deserialize(decoder: Decoder): Instant {
        return Instant.parse(decoder.decodeString())
    }
}