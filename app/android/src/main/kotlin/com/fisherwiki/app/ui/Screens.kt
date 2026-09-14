package com.fisherwiki.app.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.clickable
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.PhotoCamera
import androidx.compose.material.icons.filled.PhotoLibrary
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Divider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import com.fisherwiki.app.BuildConfig
import com.fisherwiki.app.FisherWikiApplication
import com.fisherwiki.app.data.PackManager
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

private fun app(ctx: android.content.Context) = ctx.applicationContext as FisherWikiApplication

/**
 * A fresh, writable destination for [ActivityResultContracts.TakePicture],
 * exposed to the system camera app through [FileProvider] - the same
 * mechanism `installFromUri`/export already trust for content:// access, and
 * the reason `file_paths.xml` has a `camera_capture` cache path scoped
 * separately from catch photos: this file is not a catch yet, and is only
 * ever promoted to one through the ordinary save() path, same as a gallery
 * pick.
 */
private fun newCameraCaptureUri(ctx: android.content.Context): Uri {
    val dir = File(ctx.cacheDir, "camera").apply { mkdirs() }
    val file = File(dir, "capture_${System.currentTimeMillis()}.jpg")
    return FileProvider.getUriForFile(ctx, "${BuildConfig.APPLICATION_ID}.fileprovider", file)
}

/**
 * A cached fix, not a fresh one - this app never calls `requestLocationUpdates`
 * and never will, because a ranking hint does not justify the battery cost or
 * the privacy posture of continuous location. `getLastKnownLocation` returns
 * whatever the OS already has cached from other apps/services, which is
 * `null` more often on a device that has not used location recently; that is
 * an acceptable, honest trade for "works without it".
 *
 * Caller must already hold ACCESS_COARSE_LOCATION (or FINE) - this does not
 * check, because by the time it is called permission has just been confirmed
 * either via ContextCompat.checkSelfPermission or the request callback.
 */
@Suppress("MissingPermission")
private fun lastKnownLocation(
    ctx: android.content.Context,
): com.fisherwiki.core.rank.CandidateRanker.Location? {
    val lm = ctx.getSystemService(android.content.Context.LOCATION_SERVICE)
        as? android.location.LocationManager ?: return null
    val providers = listOf(
        android.location.LocationManager.GPS_PROVIDER,
        android.location.LocationManager.NETWORK_PROVIDER,
        android.location.LocationManager.PASSIVE_PROVIDER,
    )
    for (provider in providers) {
        val loc = runCatching { lm.getLastKnownLocation(provider) }.getOrNull()
        if (loc != null) {
            return com.fisherwiki.core.rank.CandidateRanker.Location(
                loc.latitude, loc.longitude
            )
        }
    }
    return null
}

@Composable
fun HomeScreen(
    onIdentify: () -> Unit,
    onOpenPacks: () -> Unit,
    onOpenSpecies: (Long) -> Unit,
) {
    val ctx = LocalContext.current
    val a = app(ctx)
    val pack = remember { a.packs.active() }
    val catches = remember { a.catchLog.count() }

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text("FisherWiki", style = MaterialTheme.typography.headlineLarge)
        Text(
            "Identify fish without a signal.",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        if (pack == null) {
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp)) {
                    Text("No species pack installed",
                        style = MaterialTheme.typography.titleMedium)
                    Spacer(Modifier.height(8.dp))
                    Text(
                        "A pack contains the recognition model and the species "
                            + "database. Install one while you have a connection; "
                            + "after that everything works offline.",
                        style = MaterialTheme.typography.bodyMedium,
                    )
                    Spacer(Modifier.height(12.dp))
                    Button(onClick = onOpenPacks) { Text("Install a pack") }
                }
            }
        } else {
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp)) {
                    Text(pack.manifest.displayName,
                        style = MaterialTheme.typography.titleMedium)
                    Text(
                        "${pack.manifest.model.numClasses} species • "
                            + "pack v${pack.version}",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    if (!pack.commercialSafe) {
                        Spacer(Modifier.height(8.dp))
                        Text(
                            "Research build: trained on non-commercial media. "
                                + "Not for distribution.",
                            style = MaterialTheme.typography.bodyMedium,
                            color = CertaintyColors.ambiguous,
                        )
                    }
                    Spacer(Modifier.height(12.dp))
                    Button(onClick = onIdentify, modifier = Modifier.fillMaxWidth()) {
                        Icon(Icons.Default.PhotoCamera, null, Modifier.size(18.dp))
                        Spacer(Modifier.width(8.dp))
                        Text("Identify a fish")
                    }
                }
            }
        }

        Text("$catches saved catches", style = MaterialTheme.typography.bodyLarge)

        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp)) {
                Text("Private by design", style = MaterialTheme.typography.titleMedium)
                Spacer(Modifier.height(6.dp))
                Text(
                    "No account. Recognition runs on this phone. Your photographs "
                        + "and locations are never uploaded.",
                    style = MaterialTheme.typography.bodyMedium,
                )
            }
        }
    }
}

@Composable
fun IdentifyScreen(
    vm: IdentifyViewModel,
    onOpenSpecies: (Long) -> Unit,
    onOpenPacks: () -> Unit,
) {
    val state by vm.state.collectAsState()
    val photos by vm.photos.collectAsState()
    var showCorrection by remember { mutableStateOf(false) }
    val ctx = LocalContext.current

    val pickImage = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri -> uri?.let { vm.addPhoto(it) } }

    // TakePicture's callback reports only success/failure, not which Uri -
    // it was always going to be the one we handed it when launching, so that
    // has to be remembered across the round trip to the camera app and back.
    var pendingCaptureUri by remember { mutableStateOf<Uri?>(null) }
    val takePicture = rememberLauncherForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { captured ->
        val uri = pendingCaptureUri
        pendingCaptureUri = null
        if (uri == null) return@rememberLauncherForActivityResult
        if (captured) {
            vm.addPhoto(uri)
        } else {
            // Cancelled from the camera app. It may have already created an
            // empty file at the destination we gave it; nothing else will
            // ever clean that up, and it is not a catch photo.
            runCatching { ctx.contentResolver.delete(uri, null, null) }
        }
    }
    val requestCameraPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        // Denied: no-op, deliberately. The button stays where it was; the
        // system's own "don't ask again" / settings flow handles the rest,
        // the same as it would for any other app.
        if (granted) {
            val uri = newCameraCaptureUri(ctx)
            pendingCaptureUri = uri
            takePicture.launch(uri)
        }
    }

    fun startCameraCapture() {
        val alreadyGranted = androidx.core.content.ContextCompat.checkSelfPermission(
            ctx, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
        if (alreadyGranted) {
            val uri = newCameraCaptureUri(ctx)
            pendingCaptureUri = uri
            takePicture.launch(uri)
        } else {
            requestCameraPermission.launch(Manifest.permission.CAMERA)
        }
    }

    // Location is opt-in (Settings.useLocation, default false) and coarse is
    // deliberately all this ever asks for: it is a ranking hint, bounded and
    // attenuated by GeoPrior so it can never zero a species out (see its
    // kdoc), not a feature that needs metre-level precision. Denial is not an
    // error path - identify() runs with location = null exactly as it always
    // has, which is the whole point of "identification works fully without
    // it" being true rather than aspirational.
    val requestLocationPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        vm.identify(if (granted) lastKnownLocation(ctx) else null)
    }
    // docs/PRIVACY.md §4 promises the app "explains what it does with it" when
    // it first asks for location - that is this dialog, not the bare system
    // permission prompt, which says nothing about *why*. Shown once
    // (locationRationaleShown), so turning the setting back on after an
    // earlier denial goes straight to the system prompt rather than nagging.
    var showLocationRationale by remember { mutableStateOf(false) }

    fun identifyNow() {
        val settings = app(ctx).settings
        if (!settings.useLocation) {
            vm.identify()
            return
        }
        val alreadyGranted = androidx.core.content.ContextCompat.checkSelfPermission(
            ctx, Manifest.permission.ACCESS_COARSE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED
        when {
            alreadyGranted -> vm.identify(lastKnownLocation(ctx))
            !settings.locationRationaleShown -> showLocationRationale = true
            else -> requestLocationPermission.launch(Manifest.permission.ACCESS_COARSE_LOCATION)
        }
    }

    if (showLocationRationale) {
        androidx.compose.material3.AlertDialog(
            onDismissRequest = {
                showLocationRationale = false
                app(ctx).settings.locationRationaleShown = true
                vm.identify()
            },
            title = { Text("Use your location?") },
            text = {
                Text(
                    "Knowing roughly where you are lets the app favour species " +
                        "actually found nearby. It never overrides what the photo " +
                        "shows, works fine if you say no, and your location is " +
                        "never sent anywhere - it only ever stays on this phone."
                )
            },
            confirmButton = {
                Button(onClick = {
                    showLocationRationale = false
                    app(ctx).settings.locationRationaleShown = true
                    requestLocationPermission.launch(Manifest.permission.ACCESS_COARSE_LOCATION)
                }) { Text("Allow") }
            },
            dismissButton = {
                OutlinedButton(onClick = {
                    showLocationRationale = false
                    app(ctx).settings.locationRationaleShown = true
                    vm.identify()
                }) { Text("Not now") }
            },
        )
    }

    Column(Modifier.fillMaxSize()) {
        when (val s = state) {
            is IdentifyViewModel.State.Done -> {
                ResultScreen(
                    identification = s.identification,
                    detail = s.detail,
                    onOpenSpecies = onOpenSpecies,
                    onCorrect = { showCorrection = true },
                    onSave = {
                        vm.save(s.identification, s.photoPaths)
                        vm.clearPhotos()
                    },
                    modifier = Modifier.weight(1f),
                    hazards = s.hazards,
                )

                if (showCorrection) {
                    CorrectionSheet(
                        search = vm::searchSpecies,
                        onPick = { taxon ->
                            // The model's own answer is preserved by
                            // CatchLog.from regardless - see its kdoc - so this
                            // saves once, correctly, rather than saving the
                            // model's guess now and correcting a second write
                            // later.
                            vm.save(
                                s.identification, s.photoPaths,
                                correction = taxon.id to taxon.scientificName,
                            )
                            vm.clearPhotos()
                            showCorrection = false
                        },
                        onDismiss = { showCorrection = false },
                    )
                }
            }

            is IdentifyViewModel.State.Working -> Box(
                Modifier.fillMaxSize(), Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CircularProgressIndicator()
                    Spacer(Modifier.height(12.dp))
                    Text(s.stage)
                }
            }

            is IdentifyViewModel.State.NoPack -> Box(
                Modifier.fillMaxSize().padding(24.dp), Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("No pack installed",
                        style = MaterialTheme.typography.titleLarge)
                    Spacer(Modifier.height(8.dp))
                    Text(
                        "Install a species pack to identify fish offline.",
                        style = MaterialTheme.typography.bodyMedium,
                    )
                    Spacer(Modifier.height(16.dp))
                    Button(onClick = onOpenPacks) { Text("Open packs") }
                }
            }

            is IdentifyViewModel.State.Failed -> Box(
                Modifier.fillMaxSize().padding(24.dp), Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("Could not identify", style = MaterialTheme.typography.titleLarge)
                    Spacer(Modifier.height(8.dp))
                    Text(s.message, style = MaterialTheme.typography.bodyMedium)
                    Spacer(Modifier.height(16.dp))
                    OutlinedButton(onClick = { vm.clearPhotos() }) { Text("Try again") }
                }
            }

            IdentifyViewModel.State.Idle -> Column(
                Modifier.fillMaxSize().padding(24.dp),
                verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Text("Photograph the fish",
                    style = MaterialTheme.typography.headlineMedium)
                Spacer(Modifier.height(8.dp))
                Text(
                    "A side-on view of the whole fish works best. For a tricky "
                        + "one, add a close-up of the head or the tail and the "
                        + "app will combine them.",
                    style = MaterialTheme.typography.bodyMedium,
                )
                Spacer(Modifier.height(24.dp))
                Button(
                    onClick = { startCameraCapture() },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Icon(Icons.Default.PhotoCamera, null, Modifier.size(18.dp))
                    Spacer(Modifier.width(8.dp))
                    Text("Take a photo")
                }
                Spacer(Modifier.height(8.dp))
                OutlinedButton(
                    onClick = {
                        pickImage.launch(
                            androidx.activity.result.PickVisualMediaRequest(
                                ActivityResultContracts.PickVisualMedia.ImageOnly
                            )
                        )
                    },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Icon(Icons.Default.PhotoLibrary, null, Modifier.size(18.dp))
                    Spacer(Modifier.width(8.dp))
                    Text("Choose a photo")
                }
                if (photos.isNotEmpty()) {
                    Spacer(Modifier.height(16.dp))
                    Text("${photos.size} photo(s) selected")
                    Spacer(Modifier.height(8.dp))
                    Button(onClick = { identifyNow() }, Modifier.fillMaxWidth()) {
                        Text("Identify")
                    }
                }
            }
        }
    }
}

@Composable
private fun Box(
    modifier: Modifier,
    alignment: Alignment,
    content: @Composable () -> Unit,
) = androidx.compose.foundation.layout.Box(modifier, alignment) { content() }

@Composable
fun SpeciesDetailScreen(taxonId: Long?, onOpenSpecies: (Long) -> Unit) {
    val ctx = LocalContext.current
    val detail = remember(taxonId) {
        taxonId?.let { app(ctx).engine()?.detail(it) }
    }
    if (detail == null) {
        Box(Modifier.fillMaxSize(), Alignment.Center) { Text("Species not found") }
        return
    }
    LazyColumn(
        Modifier.fillMaxSize(),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(20.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        item {
            Column {
                Text(detail.displayName, style = MaterialTheme.typography.headlineLarge)
                Text(
                    detail.taxon.scientificName,
                    style = MaterialTheme.typography.bodyLarge,
                    fontStyle = FontStyle.Italic,
                )
                if (detail.displayNameIsFallback) {
                    Text(
                        "Common name shown in ${detail.displayNameLanguage}; "
                            + "no name is available in your language.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
        if (detail.commonNames.size > 1) {
            item {
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp)) {
                        Text("Also known as",
                            style = MaterialTheme.typography.titleMedium)
                        Spacer(Modifier.height(6.dp))
                        for (n in detail.commonNames.take(12)) {
                            Text("${n.name}  (${n.lang})",
                                style = MaterialTheme.typography.bodyMedium)
                        }
                    }
                }
            }
        }
        if (detail.synonyms.isNotEmpty()) {
            item {
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp)) {
                        Text("Former scientific names",
                            style = MaterialTheme.typography.titleMedium)
                        Spacer(Modifier.height(6.dp))
                        Text(
                            detail.synonyms.take(10).joinToString(", "),
                            style = MaterialTheme.typography.bodyMedium,
                            fontStyle = FontStyle.Italic,
                        )
                    }
                }
            }
        }
        item {
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp)) {
                    Text("Sources", style = MaterialTheme.typography.titleMedium)
                    Spacer(Modifier.height(6.dp))
                    for (s in detail.sources) {
                        Text(s.title, style = MaterialTheme.typography.bodyLarge)
                        Text(
                            "${s.license} • retrieved ${s.retrievedOn}",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Spacer(Modifier.height(6.dp))
                    }
                }
            }
        }
    }
}

@Composable
fun PacksScreen() {
    val ctx = LocalContext.current
    val a = app(ctx)
    var refresh by remember { mutableStateOf(0) }
    val packs = remember(refresh) { a.packs.list() }
    var message by remember { mutableStateOf<String?>(null) }

    val pickPack = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            when (val r = a.packs.installFromUri(uri)) {
                is PackManager.Result.Installed -> {
                    a.resetEngine()
                    message = "Installed ${r.pack.manifest.displayName} " +
                        "v${r.pack.version}"
                    refresh++
                }
                is PackManager.Result.Rejected -> message = "Rejected: ${r.reason}"
            }
        }
    }

    LazyColumn(
        Modifier.fillMaxSize(),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(20.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        item {
            Column {
                Text("Offline packs", style = MaterialTheme.typography.headlineMedium)
                Spacer(Modifier.height(6.dp))
                Text(
                    "Each pack carries a recognition model and a species database "
                        + "for a region. Everything works offline once installed.",
                    style = MaterialTheme.typography.bodyMedium,
                )
            }
        }
        item {
            OutlinedButton(
                onClick = { pickPack.launch(arrayOf("*/*")) },
                modifier = Modifier.fillMaxWidth(),
            ) { Text("Install from file (.fwpack)") }
        }
        message?.let { item { Text(it, style = MaterialTheme.typography.bodyMedium) } }

        items(packs) { entry ->
            when (entry) {
                is PackManager.LoadedPack.Ok -> {
                    val p = entry.pack
                    Card(Modifier.fillMaxWidth()) {
                        Column(Modifier.padding(16.dp)) {
                            Text(p.manifest.displayName,
                                style = MaterialTheme.typography.titleMedium)
                            Text(
                                "v${p.version} • ${p.manifest.model.numClasses} species "
                                    + "• ${p.manifest.model.architecture} "
                                    + "(${p.manifest.model.quantization})",
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                            Text(
                                "Corpus ${p.manifest.corpus.name} • "
                                    + "${p.manifest.corpus.licensePolicy}"
                                    + if (p.commercialSafe) "" else " (research only)",
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                            Spacer(Modifier.height(10.dp))
                            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                OutlinedButton(onClick = {
                                    val problems = a.packs.verifyInstalled(p)
                                    message = if (problems.isEmpty())
                                        "${p.id}: integrity verified"
                                    else "${p.id}: ${problems.joinToString()}"
                                }) { Text("Verify") }
                                OutlinedButton(onClick = {
                                    a.packs.remove(p.id, p.version)
                                    a.resetEngine()
                                    refresh++
                                }) { Text("Remove") }
                            }
                        }
                    }
                }
                is PackManager.LoadedPack.Broken -> {
                    Card(Modifier.fillMaxWidth()) {
                        Column(Modifier.padding(16.dp)) {
                            Text("Unreadable pack: ${entry.dirName}",
                                style = MaterialTheme.typography.titleMedium)
                            Text(entry.reason,
                                style = MaterialTheme.typography.bodyMedium)
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun CatchLogScreen(onOpenSpecies: (Long) -> Unit) {
    val ctx = LocalContext.current
    val a = app(ctx)
    var refresh by remember { mutableStateOf(0) }
    val catches = remember(refresh) { a.catchLog.all() }
    val fmt = remember { SimpleDateFormat("d MMM yyyy, HH:mm", Locale.getDefault()) }

    if (catches.isEmpty()) {
        Box(Modifier.fillMaxSize().padding(24.dp), Alignment.Center) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text("No catches yet", style = MaterialTheme.typography.titleLarge)
                Spacer(Modifier.height(8.dp))
                Text(
                    "Saved catches stay on this phone.",
                    style = MaterialTheme.typography.bodyMedium,
                )
            }
        }
        return
    }

    LazyColumn(
        Modifier.fillMaxSize(),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(16.dp),
    ) {
        items(catches) { c ->
            // A genus-level catch (CandidateRanker's coarse fallback) stores a
            // synthetic negative id - `-1L - genusIndex`, never a real
            // fw_taxon_id - because there is no taxon row for "some kind of
            // Sebastes" to open. Passing it to onOpenSpecies used to navigate
            // straight to a "Species not found" screen; a real taxon id is
            // always positive, so that is the correct and complete guard.
            val hasSpeciesPage = (c.identifiedTaxonId ?: -1L) > 0L
            Column(
                Modifier
                    .fillMaxWidth()
                    .clickable {
                        if (hasSpeciesPage) {
                            c.identifiedTaxonId?.let(onOpenSpecies)
                        } else {
                            android.widget.Toast.makeText(
                                ctx,
                                "Identified to genus level only - no single species page for this catch",
                                android.widget.Toast.LENGTH_SHORT,
                            ).show()
                        }
                    }
                    .padding(vertical = 12.dp)
            ) {
                Text(
                    (c.displayName ?: "Unidentified") +
                        if (!hasSpeciesPage && c.displayName != null) " (genus)" else "",
                    style = MaterialTheme.typography.bodyLarge,
                )
                Text(
                    fmt.format(Date(c.createdAt)) +
                        (c.confidence?.let { " • ${Math.round(it * 100)}%" } ?: "") +
                        (if (c.wasCorrected) " • corrected by you" else ""),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Divider(color = MaterialTheme.colorScheme.surfaceVariant)
        }
    }
}

@Composable
fun SettingsScreen(onOpenAbout: () -> Unit) {
    val ctx = LocalContext.current
    val a = app(ctx)
    var useLocation by remember { mutableStateOf(a.settings.useLocation) }
    var storeLocation by remember { mutableStateOf(a.settings.storeLocationWithCatches) }
    var nnapi by remember { mutableStateOf(a.settings.useNnapi) }
    var imperial by remember { mutableStateOf(a.settings.useImperialUnits) }

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Text("Settings", style = MaterialTheme.typography.headlineMedium)
        Spacer(Modifier.height(8.dp))

        SettingRow(
            title = "Use location to rank candidates",
            subtitle = "Improves ranking where species ranges differ. "
                + "Identification works without it, and your location never "
                + "leaves the phone.",
            checked = useLocation,
            onChange = { useLocation = it; a.settings.useLocation = it },
        )
        SettingRow(
            title = "Save location with catches",
            subtitle = "Off by default. Exports ask before including coordinates.",
            checked = storeLocation,
            onChange = { storeLocation = it; a.settings.storeLocationWithCatches = it },
        )
        SettingRow(
            title = "Use NNAPI acceleration",
            subtitle = "Off by default: hardware delegates vary in quality "
                + "between phones. Turn on and compare if you like.",
            checked = nnapi,
            onChange = { nnapi = it; a.settings.useNnapi = it; a.resetEngine() },
        )
        SettingRow(
            title = "Imperial units",
            subtitle = "Inches and pounds instead of centimetres and grams.",
            checked = imperial,
            onChange = { imperial = it; a.settings.useImperialUnits = it },
        )

        Spacer(Modifier.height(16.dp))
        OutlinedButton(onClick = {
            val json = a.catchLog.exportJson(includeLocation = false)
            val send = Intent(Intent.ACTION_SEND).apply {
                type = "application/json"
                putExtra(Intent.EXTRA_TEXT, json)
            }
            ctx.startActivity(Intent.createChooser(send, "Export catch log"))
        }, modifier = Modifier.fillMaxWidth()) {
            Text("Export catch log (without locations)")
        }
        Spacer(Modifier.height(8.dp))
        OutlinedButton(onClick = onOpenAbout, modifier = Modifier.fillMaxWidth()) {
            Text("About, data sources and licences")
        }
    }
}

@Composable
private fun SettingRow(
    title: String,
    subtitle: String,
    checked: Boolean,
    onChange: (Boolean) -> Unit,
) {
    Row(
        Modifier.fillMaxWidth().padding(vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.bodyLarge)
            Text(
                subtitle,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Spacer(Modifier.width(12.dp))
        Switch(checked = checked, onCheckedChange = onChange)
    }
    Divider(color = MaterialTheme.colorScheme.surfaceVariant)
}

@Composable
fun AboutScreen() {
    val ctx = LocalContext.current
    val a = app(ctx)
    val sources = remember { a.engine()?.sources().orEmpty() }
    val pack = remember { a.packs.active() }

    LazyColumn(
        Modifier.fillMaxSize(),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(20.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        item {
            Column {
                Text("About FisherWiki",
                    style = MaterialTheme.typography.headlineMedium)
                Spacer(Modifier.height(8.dp))
                Text(
                    "Fish identification that runs entirely on your phone. "
                        + "No account, no cloud inference, no telemetry, no "
                        + "advertising. Photographs and locations never leave "
                        + "the device unless you explicitly export them.",
                    style = MaterialTheme.typography.bodyMedium,
                )
            }
        }
        pack?.let { p ->
            item {
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp)) {
                        Text("Installed model",
                            style = MaterialTheme.typography.titleMedium)
                        Spacer(Modifier.height(6.dp))
                        Text("${p.manifest.model.architecture} • "
                            + "${p.manifest.model.numClasses} species • "
                            + "${p.manifest.model.quantization}",
                            style = MaterialTheme.typography.bodyMedium)
                        Text("Trained on ${p.manifest.corpus.imageCount} images "
                            + "under the ${p.manifest.corpus.licensePolicy} "
                            + "licence policy",
                            style = MaterialTheme.typography.bodyMedium)
                        p.manifest.corpus.codeCommit?.let {
                            Text("Build ${it.take(10)}",
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                        p.manifest.model.calibration.expectedCalibrationError?.let {
                            Text(
                                "Calibration error ${"%.3f".format(it)} - how closely "
                                    + "the confidence percentages match reality.",
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                }
            }
        }
        item { Text("Data sources", style = MaterialTheme.typography.titleMedium) }
        items(sources) { s ->
            Column(Modifier.fillMaxWidth().padding(vertical = 8.dp)) {
                Text(s.title, style = MaterialTheme.typography.bodyLarge)
                Text(s.citation, style = MaterialTheme.typography.bodyMedium)
                Text(
                    "${s.license}  ${s.url.orEmpty()}",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Divider(color = MaterialTheme.colorScheme.surfaceVariant)
        }
        item {
            Text(
                "Photographs used for training are individually credited in the "
                    + "ATTRIBUTIONS.csv file inside each pack.",
                style = MaterialTheme.typography.bodyMedium,
            )
        }
    }
}
