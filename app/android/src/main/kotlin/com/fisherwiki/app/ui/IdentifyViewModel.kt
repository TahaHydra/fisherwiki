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
        onSaved: (Long) -> Unit = {},
    ) {
        viewModelScope.launch(Dispatchers.IO) {
            val dir = File(appCtx.filesDir, "catches").apply { mkdirs() }
            val stored = sourceUris.mapIndexedNotNull { i, s ->
                runCatching {
                    val out = File(dir, "${System.currentTimeMillis()}_$i.jpg")
                    appCtx.contentResolver.openInputStream(Uri.parse(s))?.use { input ->
                        out.outputStream().use { input.copyTo(it) }
                    }
                    out.absolutePath
                }.getOrNull()
            }
            val record = CatchLog.from(
                identification, stored,
                latitude.takeIf { appCtx.settings.storeLocationWithCatches },
                longitude.takeIf { appCtx.settings.storeLocationWithCatches },
            )
            val id = appCtx.catchLog.insert(record)
            withContext(Dispatchers.Main) { onSaved(id) }
        }
    }

    /** Record a user correction against a saved catch. */
    fun correct(catchId: Long, taxonId: Long, name: String) {
        viewModelScope.launch(Dispatchers.IO) {
            appCtx.catchLog.correct(catchId, taxonId, name)
        }
    }

    fun searchSpecies(query: String) =
        appCtx.engine()?.search(query, limit = 40).orEmpty()
}
