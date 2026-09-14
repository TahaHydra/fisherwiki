package com.fisherwiki.app

import android.app.Application
import com.fisherwiki.app.data.CatchLog
import com.fisherwiki.app.data.PackManager
import com.fisherwiki.app.data.Settings
import com.fisherwiki.app.platform.AndroidSqliteDriver
import com.fisherwiki.core.engine.IdentificationEngine

/**
 * Application-scoped wiring.
 *
 * There is no dependency-injection framework here on purpose: the object graph
 * is four things deep, and a reader auditing the privacy claims should be able
 * to see the entire graph in one screen rather than tracing annotations.
 *
 * Note what is *not* constructed: no analytics client, no crash reporter, no
 * ad SDK, no remote config. The dependency list in `android/build.gradle.kts`
 * contains none of them either, so the INTERNET permission can only be
 * exercised by the explicit pack-download path.
 */
class FisherWikiApplication : Application() {

    lateinit var packs: PackManager
        private set
    lateinit var catchLog: CatchLog
        private set
    lateinit var settings: Settings
        private set

    /** Lazily opened for the active pack; null when no pack is installed. */
    @Volatile
    private var engine: IdentificationEngine? = null
    private val engineLock = Any()

    override fun onCreate() {
        super.onCreate()
        packs = PackManager(this)
        catchLog = CatchLog(this)
        settings = Settings(this)
    }

    /**
     * The identification engine for the active pack.
     *
     * Opening it loads an ONNX session and reads ~2,000 taxon rows, so it is
     * cached. Returns null when no usable pack is installed, which the UI
     * presents as "install a pack" rather than as an error.
     */
    fun engine(): IdentificationEngine? {
        engine?.let { return it }
        synchronized(engineLock) {
            engine?.let { return it }
            val pack = packs.active() ?: return null
            val created = runCatching {
                IdentificationEngine.open(
                    pack = pack,
                    driverFactory = { path -> AndroidSqliteDriver(path) },
                    language = settings.language,
                    threads = settings.inferenceThreads,
                    useNnapi = settings.useNnapi,
                )
            }.getOrNull()
            engine = created
            return created
        }
    }

    /** Drop the cached engine, e.g. after installing or removing a pack. */
    fun resetEngine() {
        synchronized(engineLock) {
            engine?.close()
            engine = null
        }
    }

    override fun onTerminate() {
        resetEngine()
        catchLog.close()
        super.onTerminate()
    }
}
