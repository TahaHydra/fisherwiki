package com.fisherwiki.app.ui

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.fisherwiki.app.FisherWikiApplication
import com.fisherwiki.app.data.CatchLog
import com.fisherwiki.app.platform.ImageLoading
import com.fisherwiki.core.db.SpeciesRepository
import com.fisherwiki.core.model.Identification
import com.fisherwiki.core.rank.CandidateRanker
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Drives the identify flow.
 *
 * Inference runs on [Dispatchers.Default], never on the main thread: a
 * MobileNetV3 forward pass is tens of milliseconds on a good phone and can be
 * several hundred on a cheap one, which would drop frames and, worse, make the
 * app feel like it is doing something over a network when it is not.
 */
class IdentifyViewModel(app: Application) : AndroidViewModel(app) {

    sealed class State {
        object Idle : State()
        object NoPack : State()
        data class Working(val stage: String) : State()
        data class Done(
            val identification: Identification,
            val detail: SpeciesRepository.SpeciesDetail?,
            val photoPaths: List<String>,
            /**
             * Warnings across *all* candidates, not just the winner. A
             * venomous species sitting second in the list is exactly the case
             * the user needs told about, and it is a measured failure mode of
             * this model rather than a hypothetical one.
             */
            val hazards: List<SpeciesRepository.CandidateHazard> = emptyList(),
        ) : State()
        data class Failed(val message: String) : State()
    }

    private val _state = MutableStateFlow<State>(State.Idle)
    val state: StateFlow<State> = _state.asStateFlow()

    private val _photos = MutableStateFlow<List<Uri>>(emptyList())
    val photos: StateFlow<List<Uri>> = _photos.asStateFlow()

    private val appCtx get() = getApplication<FisherWikiApplication>()

    fun addPhoto(uri: Uri) {
        _photos.value = _photos.value + uri
    }

    fun clearPhotos() {
        _photos.value = emptyList()
        _state.value = State.Idle
    }

    /**
     * @param location caller passes null unless the user enabled location *and*
     *   granted the permission. The engine treats null as "rank visually only".
     */
    fun identify(location: CandidateRanker.Location? = null) {
        val uris = _photos.value
        if (uris.isEmpty()) return

        viewModelScope.launch {
            _state.value = State.Working("Reading photo")
            val engine = appCtx.engine()
            if (engine == null) {
                _state.value = State.NoPack
                return@launch
            }

            val result = withContext(Dispatchers.Default) {
                runCatching {
                    val decoded = uris.mapNotNull {
                        ImageLoading.fromUri(appCtx, it)
                    }
                    if (decoded.isEmpty()) error("Could not read the selected photograph")

                    val identification = if (decoded.size == 1) {
                        val d = decoded.first()
                        engine.identify(d.pixels, d.width, d.height, location)
                    } else {
                        engine.identifyMulti(decoded.map { it.asPhoto() }, location)
                    }

                    val taxonId = identification.best?.taxon?.id
                        ?: identification.coarseFallback?.taxon?.id
                    val detail = taxonId?.takeIf { it > 0 }?.let { engine.detail(it) }
                    Triple(identification, detail, engine.hazards(identification))
                }
            }

            result.fold(
                onSuccess = { (identification, detail, hazards) ->
                    _state.value = State.Done(
                        identification, detail, uris.map { it.toString() }, hazards
                    )
                },
                onFailure = {
                    _state.value = State.Failed(it.message ?: "Identification failed")
                },
            )
        }
    }

    /**
     * Persist the catch, copying photographs into app-private storage.
     *
     * Copying rather than referencing the gallery URI matters: a gallery entry
     * the user later deletes would leave a catch record pointing at nothing,
     * and a content URI permission does not survive a reboot.
     */
    fun save(
        identification: Identification,
        sourceUris: List<String>,
        latitude: Double? = null,
        longitude: Double? = null,
        /** Set when the user saved via "Not right?" and picked the actual
         * species themselves, rather than "Save catch" on the model's own
         * answer. (taxonId, scientificName). */
        correction: Pair<Long, String>? = null,
        onSaved: (Long) -> Unit = {},
    ) {
        viewModelScope.launch(Dispatchers.IO) {
            val dir = File(appCtx.filesDir, "catches").apply { mkdirs() }
            val stored = sourceUris.mapIndexedNotNull { i, s -> copyIntoCatchStorage(dir, s, i) }
            val record = CatchLog.from(
                identification, stored,
                latitude.takeIf { appCtx.settings.storeLocationWithCatches },
                longitude.takeIf { appCtx.settings.storeLocationWithCatches },
                correction = correction,
            )
            val id = appCtx.catchLog.insert(record)
            withContext(Dispatchers.Main) { onSaved(id) }
        }
    }

    /**
     * Copy one source image into app-private storage, honestly.
     *
     * Two bugs this fixes:
     *
     * 1. `openInputStream(uri)?.use { ... }` returning null (the URI's
     *    permission was revoked, the provider is gone, whatever the reason)
     *    used to fall through to returning `out.absolutePath` anyway - the
     *    `.use` block simply never ran, `out` was never created, and the
     *    catch log ended up pointing at a file that was never written.
     * 2. Every file was named `*.jpg` regardless of what was actually picked.
     *    A PNG or WebP image copied byte-for-byte under a `.jpg` name is
     *    still PNG/WebP bytes; anything that opens the file by its extension
     *    (including, eventually, a future export or share feature) would get
     *    it wrong. The real MIME type from the content resolver picks the
     *    honest extension, with `.jpg` only as a genuine last resort when the
     *    provider does not say.
     */
    private fun copyIntoCatchStorage(dir: File, sourceUri: String, index: Int): String? {
        val uri = Uri.parse(sourceUri)
        val mime = appCtx.contentResolver.getType(uri)
        val extension = EXTENSION_BY_MIME[mime] ?: "jpg"
        val out = File(dir, "${System.currentTimeMillis()}_$index.$extension")
        val wrote = runCatching {
            appCtx.contentResolver.openInputStream(uri)?.use { input ->
                out.outputStream().use { input.copyTo(it) }
                true
            } ?: false
        }.getOrDefault(false)
        if (!wrote || out.length() == 0L) {
            out.delete() // no partial or phantom file left in the catch log's photo directory
            return null
        }
        return out.absolutePath
    }

    /** Record a user correction against a saved catch. */
    fun correct(catchId: Long, taxonId: Long, name: String) {
        viewModelScope.launch(Dispatchers.IO) {
            appCtx.catchLog.correct(catchId, taxonId, name)
        }
    }

    fun searchSpecies(query: String) =
        appCtx.engine()?.search(query, limit = 40).orEmpty()

    private companion object {
        /** Every format the gallery picker and camera can realistically hand
         * back (`ActivityResultContracts.PickVisualMedia.ImageOnly`). Not
         * exhaustive by design: an unrecognised type falls back to `.jpg`
         * in [copyIntoCatchStorage] rather than growing this table forever. */
        val EXTENSION_BY_MIME = mapOf(
            "image/jpeg" to "jpg",
            "image/png" to "png",
            "image/webp" to "webp",
            "image/heic" to "heic",
            "image/heif" to "heif",
            "image/gif" to "gif",
        )
    }
}
