package com.fisherwiki.app.data

import android.content.Context
import android.content.SharedPreferences

/**
 * Local settings.
 *
 * Every default is chosen so that a user who never opens this screen gets the
 * most private behaviour. Nothing here is read from or written to a network.
 */
class Settings(context: Context) {

    private val prefs: SharedPreferences =
        context.getSharedPreferences("fisherwiki_settings", Context.MODE_PRIVATE)

    /** BCP-47 language for common names. */
    var language: String
        get() = prefs.getString(KEY_LANGUAGE, null)
            ?: java.util.Locale.getDefault().language.ifBlank { "en" }
        set(v) = prefs.edit().putString(KEY_LANGUAGE, v).apply()

    /**
     * Whether to use location to rank candidates.
     *
     * Defaults to **false**. The app asks once, in context, when the user first
     * takes a photograph - not on first launch, when they have no idea why it
     * would help. Identification works fully without it.
     */
    var useLocation: Boolean
        get() = prefs.getBoolean(KEY_USE_LOCATION, false)
        set(v) = prefs.edit().putBoolean(KEY_USE_LOCATION, v).apply()

    /** Whether to store coordinates with a saved catch. Independent of ranking. */
    var storeLocationWithCatches: Boolean
        get() = prefs.getBoolean(KEY_STORE_LOCATION, false)
        set(v) = prefs.edit().putBoolean(KEY_STORE_LOCATION, v).apply()

    /**
     * Inference threads. Four is a deliberate default: on a big.LITTLE phone,
     * using every core makes the device hot and throttle, which is slower than
     * using four.
     */
    var inferenceThreads: Int
        get() = prefs.getInt(KEY_THREADS, 4).coerceIn(1, 8)
        set(v) = prefs.edit().putInt(KEY_THREADS, v.coerceIn(1, 8)).apply()

    /**
     * NNAPI execution provider. Off by default: delegate quality varies widely
     * across vendors, and a silently-wrong delegate is worse than a slower
     * correct one. The benchmark screen lets a user turn it on and compare.
     */
    var useNnapi: Boolean
        get() = prefs.getBoolean(KEY_NNAPI, false)
        set(v) = prefs.edit().putBoolean(KEY_NNAPI, v).apply()

    /** Show the coarse (genus/family) fallback when species is uncertain. */
    var showCoarseFallback: Boolean
        get() = prefs.getBoolean(KEY_COARSE, true)
        set(v) = prefs.edit().putBoolean(KEY_COARSE, v).apply()

    /** Units. Metric default; the UI offers imperial. */
    var useImperialUnits: Boolean
        get() = prefs.getBoolean(KEY_IMPERIAL, false)
        set(v) = prefs.edit().putBoolean(KEY_IMPERIAL, v).apply()

    /** Whether the location rationale has been shown at least once. */
    var locationRationaleShown: Boolean
        get() = prefs.getBoolean(KEY_LOC_RATIONALE, false)
        set(v) = prefs.edit().putBoolean(KEY_LOC_RATIONALE, v).apply()

    companion object {
        private const val KEY_LANGUAGE = "language"
        private const val KEY_USE_LOCATION = "use_location"
        private const val KEY_STORE_LOCATION = "store_location"
        private const val KEY_THREADS = "inference_threads"
        private const val KEY_NNAPI = "use_nnapi"
        private const val KEY_COARSE = "show_coarse_fallback"
        private const val KEY_IMPERIAL = "imperial_units"
        private const val KEY_LOC_RATIONALE = "location_rationale_shown"
    }
}
