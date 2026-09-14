package com.fisherwiki.cli

import com.fisherwiki.core.db.JdbcSqliteDriver
import com.fisherwiki.core.engine.IdentificationEngine
import com.fisherwiki.core.model.Certainty
import com.fisherwiki.core.pack.InstalledPack
import com.fisherwiki.core.pack.PackVerifier
import com.fisherwiki.core.rank.CandidateRanker
import java.io.File
import java.nio.file.Files
import kotlin.system.exitProcess

/**
 * Manual desktop identification, for testing without an Android device.
 *
 * This runs the exact same [IdentificationEngine] the phone runs: the same
 * `:core` jar, the same ONNX Runtime Java API, the same [PackVerifier]
 * checking the same real `.fwpack` file, the same [com.fisherwiki.core.infer.Preprocessor],
 * [com.fisherwiki.core.infer.Calibration] and [CandidateRanker]. Nothing here
 * reimplements or bypasses any of that; this file is I/O and text formatting
 * around calls into unmodified production code. See [Report] for how the
 * "why" behind a result is derived from real calibration data rather than
 * invented for this tool.
 */

private val IMAGE_EXTENSIONS = setOf("jpg", "jpeg", "png")

/**
 * Convenience default for this development machine, following the same
 * "personal path, used only if present" convention already established
 * throughout this project (see e.g. `E:/fisherwiki-cache` in `ml/train.py`).
 * A fresh clone without this exact layout simply has to pass `--pack`.
 */
internal const val DEFAULT_PACK_PATH = """D:\fisherwiki-data\artifacts\packs\global_v1-v1.fwpack"""

internal data class CliArgs(
    val target: File,
    val packPath: File,
    val lat: Double?,
    val lon: Double?,
    val threads: Int,
    val topK: Int,
    val recursive: Boolean,
    val language: String,
)

/**
 * Outcome of parsing the command line, mirroring the project's own
 * [PackVerifier.Result] idiom rather than calling `exitProcess` from inside
 * the parser. That keeps the actual parsing logic - flag/value consumption,
 * numeric validation, the `--lat`/`--lon` pairing rule - callable and
 * assertable from a test without killing the test JVM.
 */
internal sealed class ArgsResult {
    data class Ok(val args: CliArgs) : ArgsResult()

    /** `--help` was requested, or no arguments were given at all. Exit 0. */
    object ShowHelp : ArgsResult()

    /** Malformed input. The caller prints [message], then usage, then exits 2. */
    data class Error(val message: String) : ArgsResult()
}

fun main(rawArgs: Array<String>) {
    val args = when (val result = parseArgs(rawArgs)) {
        is ArgsResult.Ok -> result.args
        is ArgsResult.ShowHelp -> { printUsage(); exitProcess(0) }
        is ArgsResult.Error -> {
            System.err.println("error: ${result.message}")
            System.err.println()
            printUsage()
            exitProcess(2)
        }
    }

    if (!args.packPath.isFile) {
        System.err.println("error: pack not found: ${args.packPath.path}")
        exitProcess(2)
    }
    if (!args.target.exists()) {
        System.err.println("error: not found: ${args.target.path}")
        exitProcess(2)
    }
    val images = discoverImages(args.target, args.recursive)
    if (images.isEmpty()) {
        System.err.println(
            "error: no .jpg/.jpeg/.png files found under ${args.target.path}" +
                if (args.target.isDirectory && !args.recursive) " (pass --recursive to look in subfolders)" else ""
        )
        exitProcess(2)
    }

    println(banner())

    val staging = Files.createTempDirectory("fisherwiki-cli-pack-").toFile()
    try {
        val pack = openVerifiedPack(args.packPath, staging)

        IdentificationEngine.open(
            pack = pack,
            driverFactory = { path -> JdbcSqliteDriver(path) },
            language = args.language,
            threads = args.threads,
        ).use { engine ->
            printPackInfo(engine, pack)

            val location = if (args.lat != null && args.lon != null) {
                CandidateRanker.Location(args.lat, args.lon)
            } else null
            println(
                "geo prior: " + if (location != null) {
                    "applied (lat=${args.lat}, lon=${args.lon})"
                } else {
                    "not applied - no --lat/--lon given; ranking is purely visual"
                }
            )

            val outcomes = mutableListOf<Certainty>()
            var errors = 0
            var totalInferenceMillis = 0L
            var totalWallMillis = 0L

            for ((idx, file) in images.withIndex()) {
                println()
                println("=".repeat(72))
                println("[${idx + 1}/${images.size}] ${file.path}")
                println("=".repeat(72))

                val t0 = System.nanoTime()
                val decoded = try {
                    ImageDecoder.decode(file)
                } catch (e: ImageDecoder.DecodeException) {
                    println("  ERROR: ${e.message}")
                    errors++
                    continue
                }
                val decodeMillis = (System.nanoTime() - t0) / 1_000_000

                val identification = engine.identify(
                    decoded.pixels, decoded.width, decoded.height, location, args.topK
                )
                val totalMillis = (System.nanoTime() - t0) / 1_000_000

                printResult(engine, identification, pack.manifest.model.calibration, decodeMillis, totalMillis)

                outcomes += identification.certainty
                totalInferenceMillis += identification.inferenceMillis
                totalWallMillis += totalMillis
            }

            printSummary(images.size, outcomes, errors, totalInferenceMillis, totalWallMillis)
        }
    } finally {
        staging.deleteRecursively()
    }
}

// --------------------------------------------------------------- pack

private fun openVerifiedPack(archive: File, stagingDir: File): InstalledPack {
    println("verifying pack: ${archive.path}")
    return when (val result = PackVerifier.verifyAndExtract(archive, stagingDir)) {
        is PackVerifier.Result.Ok -> result.pack
        is PackVerifier.Result.Rejected -> {
            System.err.println("error: pack rejected: ${result.reason}")
            result.cause?.let { System.err.println("  caused by: $it") }
            exitProcess(3)
        }
    }
}

private fun printPackInfo(engine: IdentificationEngine, pack: InstalledPack) {
    println(
        "pack: ${pack.id} v${pack.version}  \"${pack.manifest.displayName}\"  " +
            "built ${pack.manifest.builtAt}"
    )
    val corpus = pack.manifest.corpus
    println(
        "  corpus: ${corpus.name}  policy=${corpus.licensePolicy}  " +
            "commercial_safe=${pack.commercialSafe}  " +
            "images=${corpus.imageCount}  classes=${corpus.classCount}"
    )
    println("  model: ${engine.describeModel()}")
    val cal = pack.manifest.model.calibration
    println(
        "  calibration: temperature=%.4f  unknown_threshold=%.2f  margin_threshold=%.2f  entropy_threshold=%.2f"
            .format(cal.temperature, cal.unknownThreshold, cal.marginThreshold, cal.entropyThreshold)
    )
}

// -------------------------------------------------------------- image

internal fun discoverImages(target: File, recursive: Boolean): List<File> {
    if (target.isFile) {
        return if (target.extension.lowercase() in IMAGE_EXTENSIONS) listOf(target) else emptyList()
    }
    val candidates = if (recursive) {
        target.walkTopDown().asSequence()
    } else {
        target.listFiles()?.asSequence() ?: emptySequence()
    }
    return candidates
        .filter { it.isFile && it.extension.lowercase() in IMAGE_EXTENSIONS }
        .sortedBy { it.path }
        .toList()
}

// ------------------------------------------------------------- output

private fun printResult(
    engine: IdentificationEngine,
    id: com.fisherwiki.core.model.Identification,
    calibration: com.fisherwiki.core.pack.CalibrationSpec,
    decodeMillis: Long,
    totalMillis: Long,
) {
    val winner = id.best ?: id.coarseFallback
    println("  decision:     ${Report.decisionLabel(id.certainty)}  (certainty=${id.certainty})")
    println("  scientific:   ${winner?.taxon?.scientificName ?: "-"}")
    println("  common name:  ${winner?.taxon?.commonName ?: "-"}")
    println(
        "  confidence:   " + (winner?.let { "%.1f%%".format(it.probability * 100) } ?: "-")
    )
    println(
        "  margin:       %.4f   normalised entropy: %.4f"
            .format(id.margin, id.normalizedEntropy)
    )
    println("  reason:       ${Report.explain(id, calibration)}")
    println(
        "  latency:      ${id.inferenceMillis} ms model  |  $decodeMillis ms decode  |  $totalMillis ms total"
    )

    val hazards = engine.hazards(id)
    if (hazards.isNotEmpty()) {
        println("  safety:")
        for (h in hazards) {
            val basis = if (h.isForBestCandidate) "candidate" else "known look-alike"
            println("    [${h.warning.severity}] ${h.scientificName} ($basis): ${h.warning.summary}")
        }
    }

    println("  top ${id.ranked.size} candidates:")
    for ((i, c) in id.ranked.withIndex()) {
        println(Report.candidateLine(i + 1, c))
    }
}

private fun printSummary(
    total: Int,
    outcomes: List<Certainty>,
    errors: Int,
    totalInferenceMillis: Long,
    totalWallMillis: Long,
) {
    println()
    println("=".repeat(72))
    println("summary  ($total image${if (total == 1) "" else "s"})")
    println("=".repeat(72))
    val byCertainty = outcomes.groupingBy { it }.eachCount()
    println("  SPECIES (confident):  ${byCertainty[Certainty.CONFIDENT] ?: 0}")
    println("  SPECIES (ambiguous):  ${byCertainty[Certainty.AMBIGUOUS] ?: 0}")
    println("  GENUS:                ${byCertainty[Certainty.COARSE_ONLY] ?: 0}")
    println("  UNCERTAIN:            ${byCertainty[Certainty.UNKNOWN] ?: 0}")
    println("  decode/read errors:   $errors")
    if (outcomes.isNotEmpty()) {
        println(
            "  mean inference (model):  %.1f ms".format(totalInferenceMillis.toDouble() / outcomes.size)
        )
        println(
            "  mean total per image:    %.1f ms".format(totalWallMillis.toDouble() / outcomes.size)
        )
    }
}

private fun banner(): String = buildString {
    appendLine("FisherWiki desktop identification CLI")
    val vmName = System.getProperty("java.vm.name", "?")
    val version = System.getProperty("java.version", "?")
    val vendor = System.getProperty("java.vendor", "?")
    appendLine("JVM: $vmName $version ($vendor)")
    if (vmName.contains("JBR", ignoreCase = true) || vendor.contains("JetBrains", ignoreCase = true)) {
        appendLine(
            "WARNING: this looks like the JetBrains Runtime, which fails to load ONNX " +
                "Runtime's native library on Windows. Run via `:cli:run` (pins Temurin 25), " +
                "or launch this jar directly with a Temurin 25 java.exe."
        )
    }
}

// --------------------------------------------------------------- args

/** Thrown only within [parseArgs], to unwind a deeply nested `?: fail(...)`
 * straight to one `catch` that turns it into [ArgsResult.Error]. Never
 * escapes [parseArgs] itself. */
private class ArgParseException(message: String) : Exception(message)

private fun fail(message: String): Nothing = throw ArgParseException(message)

private fun nextValue(args: Array<String>, flagIndex: Int, flag: String): String =
    args.getOrNull(flagIndex + 1) ?: fail("$flag needs a value")

internal fun parseArgs(rawArgs: Array<String>): ArgsResult {
    if (rawArgs.isEmpty() || rawArgs.any { it == "--help" || it == "-h" }) {
        return ArgsResult.ShowHelp
    }

    return try {
        var packPath: String? = null
        var lat: Double? = null
        var lon: Double? = null
        var threads = 4
        var topK = 5
        var recursive = false
        var language = "en"
        val positional = mutableListOf<String>()

        // Explicit index arithmetic per branch, rather than a clever i++
        // folded into the value-lookup call: a flag that takes a value
        // consumes two tokens, everything else consumes one, and that should
        // be readable straight off the branch rather than inferred from an
        // increment's side effect three tokens away.
        var i = 0
        while (i < rawArgs.size) {
            val token = rawArgs[i]
            when (token) {
                "--pack" -> { packPath = nextValue(rawArgs, i, token); i += 2 }
                "--lat" -> {
                    lat = nextValue(rawArgs, i, token).toDoubleOrNull() ?: fail("$token needs a number")
                    i += 2
                }
                "--lon" -> {
                    lon = nextValue(rawArgs, i, token).toDoubleOrNull() ?: fail("$token needs a number")
                    i += 2
                }
                "--threads" -> {
                    threads = nextValue(rawArgs, i, token).toIntOrNull() ?: fail("$token needs an integer")
                    i += 2
                }
                "--topk" -> {
                    topK = nextValue(rawArgs, i, token).toIntOrNull() ?: fail("$token needs an integer")
                    i += 2
                }
                "--lang" -> { language = nextValue(rawArgs, i, token); i += 2 }
                "--recursive" -> { recursive = true; i += 1 }
                else -> { positional += token; i += 1 }
            }
        }

        if (positional.size != 1) {
            fail("expected exactly one image file or folder argument, got ${positional.size}: $positional")
        }
        if ((lat == null) != (lon == null)) {
            fail("--lat and --lon must be given together")
        }

        val resolvedPackPath = packPath ?: run {
            if (File(DEFAULT_PACK_PATH).isFile) {
                DEFAULT_PACK_PATH
            } else {
                fail(
                    "--pack is required (no pack path given, and the default " +
                        "$DEFAULT_PACK_PATH does not exist on this machine)"
                )
            }
        }

        ArgsResult.Ok(
            CliArgs(
                target = File(positional[0]),
                packPath = File(resolvedPackPath),
                lat = lat,
                lon = lon,
                threads = threads,
                topK = topK,
                recursive = recursive,
                language = language,
            )
        )
    } catch (e: ArgParseException) {
        ArgsResult.Error(e.message ?: "invalid arguments")
    }
}

private fun printUsage() {
    println(
        """
        |FisherWiki desktop identification CLI
        |
        |Runs the real production :core inference engine - preprocessing,
        |calibration, taxonomy resolution, open-set rejection, all unmodified -
        |against a real .fwpack, so it can be exercised without an Android device.
        |
        |Usage:
        |  <run command> [options] <image-file-or-folder>
        |
        |Options:
        |  --pack <path>    Path to a .fwpack.
        |                   Default (if present): $DEFAULT_PACK_PATH
        |  --lat <deg>      Latitude, for the geographic prior (optional)
        |  --lon <deg>      Longitude (requires --lat)
        |  --threads <n>    ONNX intra-op threads (default: 4, same as the app)
        |  --topk <n>       Candidates to list (default: 5)
        |  --lang <bcp47>   Species database display language (default: en)
        |  --recursive      Recurse into subfolders when given a folder
        |  --help, -h       Show this message
        |
        |Examples (PowerShell, from the app/ directory):
        |  .\gradlew.bat :cli:run --args="'D:\photos\pike.jpg'"
        |  .\gradlew.bat :cli:run --args="'D:\photos\lake_trip' --recursive"
        |  .\gradlew.bat :cli:run --args="'D:\photos\p.jpg' --pack 'C:\packs\europe_v1.fwpack' --lat 52.1 --lon 5.2"
        """.trimMargin()
    )
}
