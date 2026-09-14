package com.fisherwiki.core.infer

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.fisherwiki.core.pack.ModelSpec
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer

/**
 * On-device classifier over ONNX Runtime.
 *
 * This class is the reason `:core` is a plain JVM module. The
 * `ai.onnxruntime` Java API is byte-for-byte the same on the desktop artefact
 * and on `onnxruntime-android`, so this exact code, loading the exact `.onnx`
 * file that will ship in a pack, is exercised by the desktop test suite. A
 * preprocessing or output-parsing mistake surfaces in CI rather than on a
 * riverbank with no signal.
 *
 * **Everything here is local.** There is no code path that opens a socket. The
 * model file comes from a verified pack on local storage, inference runs in
 * process, and the result never leaves the device.
 */
class OnnxClassifier private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    val spec: ModelSpec,
    private val inputName: String,
    private val logitsName: String,
    private val embeddingName: String?,
) : Closeable {

    /** Raw model output for one image. */
    data class Output(
        val logits: FloatArray,
        val embedding: FloatArray?,
        val millis: Long,
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is Output) return false
            return logits.contentEquals(other.logits) &&
                (embedding?.contentEquals(other.embedding ?: FloatArray(0)) ?: (other.embedding == null))
        }

        override fun hashCode(): Int =
            31 * logits.contentHashCode() + (embedding?.contentHashCode() ?: 0)
    }

    companion object {
        /**
         * Open a model file.
         *
         * @param threads intra-op threads. Phones throttle hard: 4 is a good
         *   default on a big.LITTLE SoC, and using every core makes the device
         *   hot and the result *slower*, not faster.
         * @param useNnapi enable the NNAPI execution provider on Android. Off
         *   by default because NNAPI quality varies wildly across vendors and a
         *   silently-wrong delegate is worse than a slower correct one; the
         *   benchmark screen lets a user turn it on and compare.
         */
        fun open(
            modelFile: File,
            spec: ModelSpec,
            threads: Int = 4,
            useNnapi: Boolean = false,
            useXnnpack: Boolean = true,
        ): OnnxClassifier {
            require(modelFile.isFile) { "model not found: ${modelFile.path}" }
            val env = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(threads)
                setInterOpNumThreads(1)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                if (useXnnpack) {
                    runCatching { addXnnpack(mapOf("intra_op_num_threads" to threads.toString())) }
                }
                if (useNnapi) {
                    runCatching { addNnapi() }
                }
            }
            val session = env.createSession(modelFile.absolutePath, opts)

            val inputName = session.inputNames.firstOrNull()
                ?: error("model has no inputs")
            val outputNames = session.outputNames.toList()
            val logitsName = outputNames.firstOrNull { it == spec.outputName }
                ?: outputNames.firstOrNull()
                ?: error("model has no outputs")
            val embeddingName = spec.embeddingName?.takeIf { it in outputNames }

            return OnnxClassifier(env, session, spec, inputName, logitsName, embeddingName)
        }
    }

    /**
     * Run one already-preprocessed NCHW tensor.
     *
     * @param chw float array of length `3 * size * size`, as produced by
     *   [Preprocessor.toTensor].
     */
    fun run(chw: FloatArray): Output {
        val s = spec.inputSize
        require(chw.size == 3 * s * s) {
            "expected ${3 * s * s} floats for ${s}x$s input, got ${chw.size}"
        }
        val shape = longArrayOf(1, 3, s.toLong(), s.toLong())
        val started = System.nanoTime()
        OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), shape).use { input ->
            session.run(mapOf(inputName to input)).use { result ->
                val logits = readVector(result, logitsName)
                require(logits.size == spec.numClasses) {
                    "model returned ${logits.size} logits but the manifest declares " +
                        "${spec.numClasses} classes; pack and model disagree"
                }
                val embedding = embeddingName?.let { readVector(result, it) }
                val millis = (System.nanoTime() - started) / 1_000_000
                return Output(logits, embedding, millis)
            }
        }
    }

    /**
     * Run a batch of preprocessed tensors in one session call.
     *
     * Used by Expert ID, where several photographs of the same fish are fused.
     * Batching matters on a phone: per-call overhead dominates for small models,
     * so four images in one call is markedly cheaper than four calls.
     */
    fun runBatch(tensors: List<FloatArray>): List<Output> {
        if (tensors.isEmpty()) return emptyList()
        if (tensors.size == 1) return listOf(run(tensors[0]))
        val s = spec.inputSize
        val per = 3 * s * s
        val flat = FloatArray(tensors.size * per)
        tensors.forEachIndexed { i, t ->
            require(t.size == per) { "tensor $i has ${t.size} floats, expected $per" }
            System.arraycopy(t, 0, flat, i * per, per)
        }
        val shape = longArrayOf(tensors.size.toLong(), 3, s.toLong(), s.toLong())
        val started = System.nanoTime()
        OnnxTensor.createTensor(env, FloatBuffer.wrap(flat), shape).use { input ->
            session.run(mapOf(inputName to input)).use { result ->
                val logits = readMatrix(result, logitsName, tensors.size, spec.numClasses)
                val embeddings = embeddingName?.let { name ->
                    val v = result.get(name).orElse(null)?.value
                    @Suppress("UNCHECKED_CAST")
                    (v as? Array<FloatArray>)
                }
                val millis = (System.nanoTime() - started) / 1_000_000
                return logits.mapIndexed { i, row ->
                    Output(row, embeddings?.getOrNull(i), millis / tensors.size)
                }
            }
        }
    }

    private fun readVector(result: OrtSession.Result, name: String): FloatArray {
        val value = result.get(name).orElseThrow {
            IllegalStateException("model output '$name' not found")
        }.value
        return when (value) {
            is Array<*> -> {
                val first = value.firstOrNull()
                    ?: error("output '$name' is an empty batch")
                (first as? FloatArray)
                    ?: error("output '$name' is not float32")
            }
            is FloatArray -> value
            else -> error("output '$name' has unexpected type ${value?.javaClass}")
        }
    }

    private fun readMatrix(
        result: OrtSession.Result,
        name: String,
        rows: Int,
        cols: Int,
    ): List<FloatArray> {
        val value = result.get(name).orElseThrow {
            IllegalStateException("model output '$name' not found")
        }.value
        @Suppress("UNCHECKED_CAST")
        val arr = value as? Array<FloatArray>
            ?: error("output '$name' is not a 2-D float tensor")
        require(arr.size == rows) { "expected $rows rows from '$name', got ${arr.size}" }
        require(arr.all { it.size == cols }) { "ragged output from '$name'" }
        return arr.toList()
    }

    /** Input/output description, surfaced in the diagnostics screen. */
    fun describe(): String = buildString {
        append("inputs: ")
        append(session.inputNames.joinToString())
        append("  outputs: ")
        append(session.outputNames.joinToString())
        append("  classes: ${spec.numClasses}")
        append("  size: ${spec.inputSize}")
        append("  quant: ${spec.quantization}")
    }

    override fun close() {
        runCatching { session.close() }
        // OrtEnvironment is a process-wide singleton; do not close it here.
    }
}
