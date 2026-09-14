package com.fisherwiki.app.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Divider
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.unit.dp
import com.fisherwiki.core.db.SpeciesRepository
import com.fisherwiki.core.model.Candidate
import com.fisherwiki.core.model.Certainty
import com.fisherwiki.core.model.Identification

/**
 * The identification result.
 *
 * This screen is where the project's central principle either holds or fails,
 * so its layout is driven by [Identification.certainty] rather than by a
 * percentage:
 *
 * * [Certainty.CONFIDENT] - name the species, large, with alternatives below.
 * * [Certainty.AMBIGUOUS] - name it, but lead with the comparison against the
 *   runner-up, because the user is the one who can settle it by looking.
 * * [Certainty.COARSE_ONLY] - name the **genus**, and say plainly that the
 *   species could not be determined. No species is shown as the answer.
 * * [Certainty.UNKNOWN] - say so. Offer the top candidates as *possibilities*,
 *   clearly labelled as not an identification.
 *
 * In the last two cases there is no `best` candidate in the model at all, so it
 * is not possible for this screen to accidentally render a species headline.
 */
@Composable
fun ResultScreen(
    identification: Identification,
    detail: SpeciesRepository.SpeciesDetail?,
    onOpenSpecies: (Long) -> Unit,
    onCorrect: () -> Unit,
    onSave: () -> Unit,
    modifier: Modifier = Modifier,
    /**
     * Warnings for every candidate on screen. Defaults to empty, in which case
     * the screen falls back to the winning candidate's own warnings.
     */
    hazards: List<SpeciesRepository.CandidateHazard> = emptyList(),
) {
    LazyColumn(
        modifier = modifier.fillMaxWidth(),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        item { CertaintyHeader(identification, detail) }

        // Safety first, literally: if a warning exists it appears above
        // everything except the identification itself.
        val ownWarnings = hazards.filter { it.isForBestCandidate }.map { it.warning }
            .ifEmpty { detail?.safetyWarnings.orEmpty() }
        if (ownWarnings.isNotEmpty()) {
            item { SafetyCard(ownWarnings) }
        }

        // Warnings belonging to candidates that did *not* win. Separated and
        // attributed rather than merged into the card above, because "this
        // fish is venomous" and "one of the other possibilities is venomous"
        // are different claims and the app must not conflate them. This is not
        // hypothetical: the model calls a venomous Trachinus draco a harmless
        // Mullus barbatus in 30.8% of test images.
        val otherHazards = hazards.filterNot { it.isForBestCandidate }
        if (otherHazards.isNotEmpty()) {
            item { AlternativeHazardCard(otherHazards, onOpenSpecies) }
        }

        if (identification.certainty == Certainty.AMBIGUOUS &&
            identification.alternatives.isNotEmpty()
        ) {
            item {
                ComparisonPrompt(
                    best = identification.best!!,
                    runnerUp = identification.alternatives.first(),
                    detail = detail,
                )
            }
        }

        if (identification.alternatives.isNotEmpty()) {
            item {
                Text(
                    if (identification.isUncertain) "Possibilities" else "Other candidates",
                    style = MaterialTheme.typography.titleMedium,
                    modifier = Modifier.padding(top = 8.dp),
                )
            }
            items(identification.alternatives) { c ->
                CandidateRow(c, onClick = { onOpenSpecies(c.taxon.id) })
            }
        }

        detail?.let { d ->
            if (d.diagnosticFeatures.isNotEmpty()) {
                item { HowToVerify(d) }
            }
            if (d.similarSpecies.isNotEmpty()) {
                item { SimilarSpeciesCard(d, onOpenSpecies) }
            }
            item { EvidenceCard(d, identification) }
        }

        item {
            Row(
                Modifier.fillMaxWidth().padding(top = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                OutlinedButton(onClick = onCorrect, modifier = Modifier.weight(1f)) {
                    Icon(Icons.Default.Edit, null, Modifier.size(18.dp))
                    Spacer(Modifier.width(8.dp))
                    Text("Not right?")
                }
                androidx.compose.material3.Button(
                    onClick = onSave, modifier = Modifier.weight(1f)
                ) { Text("Save catch") }
            }
        }

        item { ProvenanceFooter(identification) }
    }
}

@Composable
private fun CertaintyHeader(
    id: Identification,
    detail: SpeciesRepository.SpeciesDetail?,
) {
    val (container, accent, label) = when (id.certainty) {
        Certainty.CONFIDENT -> Triple(
            CertaintyColors.confidentContainer, CertaintyColors.confident, "Identified"
        )
        Certainty.AMBIGUOUS -> Triple(
            CertaintyColors.ambiguousContainer, CertaintyColors.ambiguous,
            "Likely, but check"
        )
        Certainty.COARSE_ONLY -> Triple(
            CertaintyColors.uncertainContainer, CertaintyColors.uncertain,
            "Species uncertain"
        )
        Certainty.UNKNOWN -> Triple(
            CertaintyColors.unknownContainer, CertaintyColors.unknown,
            "Identification uncertain"
        )
    }

    Surface(
        color = container,
        shape = RoundedCornerShape(16.dp),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(Modifier.padding(18.dp)) {
            Text(
                label.uppercase(),
                style = MaterialTheme.typography.labelLarge,
                color = accent,
            )
            Spacer(Modifier.height(8.dp))

            when (id.certainty) {
                Certainty.CONFIDENT, Certainty.AMBIGUOUS -> {
                    val best = id.best!!
                    Text(
                        detail?.displayName ?: best.taxon.commonName
                            ?: best.taxon.scientificName,
                        style = MaterialTheme.typography.headlineLarge,
                        color = Color(0xFF171C1F),
                    )
                    Text(
                        best.taxon.scientificName,
                        style = MaterialTheme.typography.bodyLarge,
                        fontStyle = FontStyle.Italic,
                        color = Color(0xFF41484D),
                    )
                    Spacer(Modifier.height(12.dp))
                    ConfidenceBar(best.probability, accent)
                }

                Certainty.COARSE_ONLY -> {
                    val coarse = id.coarseFallback
                    Text(
                        coarse?.taxon?.scientificName ?: "Unknown",
                        style = MaterialTheme.typography.headlineLarge,
                        fontStyle = FontStyle.Italic,
                        color = Color(0xFF171C1F),
                    )
                    Text(
                        "genus - the exact species could not be determined "
                            + "from this photograph",
                        style = MaterialTheme.typography.bodyMedium,
                        color = Color(0xFF41484D),
                    )
                    Spacer(Modifier.height(12.dp))
                    coarse?.let { ConfidenceBar(it.probability, accent) }
                }

                Certainty.UNKNOWN -> {
                    Text(
                        "Not enough confidence to name this fish",
                        style = MaterialTheme.typography.headlineMedium,
                        color = Color(0xFF171C1F),
                    )
                    Spacer(Modifier.height(6.dp))
                    Text(
                        "Try a clear side-on photograph of the whole fish, filling "
                            + "the frame, with the fins spread if you can.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = Color(0xFF41484D),
                    )
                }
            }

            if (id.geoApplied) {
                Spacer(Modifier.height(10.dp))
                Text(
                    "Ranking used your location. Visual evidence still came first.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = Color(0xFF41484D),
                )
            }
            detail?.takeIf { it.isLowConfidenceClass }?.let {
                Spacer(Modifier.height(10.dp))
                Text(
                    "This species had limited training data "
                        + "(${it.trainImages} photos from ${it.trainObservations} "
                        + "sightings), so treat the result with extra caution.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = CertaintyColors.uncertain,
                )
            }
        }
    }
}

@Composable
private fun ConfidenceBar(probability: Float, accent: Color) {
    Column {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                "${Math.round(probability * 100)}%",
                style = MaterialTheme.typography.titleLarge,
                color = accent,
            )
            Spacer(Modifier.width(10.dp))
            Text(
                "confidence",
                style = MaterialTheme.typography.bodyMedium,
                color = Color(0xFF41484D),
            )
        }
        Spacer(Modifier.height(6.dp))
        LinearProgressIndicator(
            progress = { probability.coerceIn(0f, 1f) },
            modifier = Modifier.fillMaxWidth().height(8.dp),
            color = accent,
            trackColor = accent.copy(alpha = 0.18f),
        )
    }
}

@Composable
private fun SafetyCard(warnings: List<SpeciesRepository.SafetyWarning>) {
    val worst = warnings.first()
    val danger = worst.isDangerous
    Surface(
        color = if (danger) CertaintyColors.dangerContainer
        else CertaintyColors.ambiguousContainer,
        shape = RoundedCornerShape(14.dp),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    Icons.Default.Warning, null,
                    tint = if (danger) CertaintyColors.danger else CertaintyColors.ambiguous,
                )
                Spacer(Modifier.width(8.dp))
                Text(
                    if (danger) "Handle with care" else "Note",
                    style = MaterialTheme.typography.titleMedium,
                    color = if (danger) CertaintyColors.danger else CertaintyColors.ambiguous,
                )
            }
            for (w in warnings) {
                Spacer(Modifier.height(8.dp))
                Text(w.summary, style = MaterialTheme.typography.bodyLarge)
                w.detail?.let {
                    Text(it, style = MaterialTheme.typography.bodyMedium)
                }
                if (w.appliesTo != "species") {
                    Text(
                        "Applies to the whole ${w.appliesTo}.",
                        style = MaterialTheme.typography.bodyMedium,
                        fontStyle = FontStyle.Italic,
                    )
                }
            }
        }
    }
}

/**
 * Warnings that belong to candidates other than the winner.
 *
 * Deliberately visually quieter than [SafetyCard] but still present: the point
 * is that the user is told, not that they are alarmed about a fish they
 * probably do not have. Each entry names the species it belongs to, so the
 * warning is a statement about a possibility rather than about their catch.
 */
@Composable
private fun AlternativeHazardCard(
    hazards: List<SpeciesRepository.CandidateHazard>,
    onOpenSpecies: (Long) -> Unit,
) {
    val anyDanger = hazards.any { it.warning.isDangerous }
    Surface(
        color = MaterialTheme.colorScheme.surfaceVariant,
        shape = RoundedCornerShape(14.dp),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    Icons.Default.Warning, null,
                    tint = if (anyDanger) CertaintyColors.danger
                    else CertaintyColors.ambiguous,
                )
                Spacer(Modifier.width(8.dp))
                Text(
                    "If it is something else",
                    style = MaterialTheme.typography.titleMedium,
                )
            }
            Spacer(Modifier.height(4.dp))
            Text(
                "These warnings belong to species this photograph could also be.",
                style = MaterialTheme.typography.bodyMedium,
                fontStyle = FontStyle.Italic,
            )
            for (h in hazards) {
                Spacer(Modifier.height(10.dp))
                Text(
                    h.commonName?.let { "$it (${h.scientificName})" } ?: h.scientificName,
                    style = MaterialTheme.typography.titleSmall,
                    color = if (h.warning.isDangerous) CertaintyColors.danger
                    else MaterialTheme.colorScheme.onSurface,
                    modifier = Modifier.clickable { onOpenSpecies(h.fwTaxonId) },
                )
                // Say where the claim comes from. "The model also considered
                // this" and "the model is known to mix these up" are different
                // strengths of evidence and the user is entitled to both.
                val provenance = when (h.reason) {
                    SpeciesRepository.HazardReason.ALTERNATIVE_CANDIDATE ->
                        "One of the other candidates for this photograph."
                    SpeciesRepository.HazardReason.KNOWN_LOOKALIKE ->
                        h.confusionRate?.let {
                            "This app mistakes it for the species above in " +
                                "${Math.round(it * 100)}% of test photographs."
                        } ?: "Known to be confused with the species above."
                }
                Text(
                    provenance,
                    style = MaterialTheme.typography.bodySmall,
                    fontStyle = FontStyle.Italic,
                )
                Text(h.warning.summary, style = MaterialTheme.typography.bodyMedium)
                if (h.warning.appliesTo != "species") {
                    Text(
                        "Applies to the whole ${h.warning.appliesTo}.",
                        style = MaterialTheme.typography.bodySmall,
                        fontStyle = FontStyle.Italic,
                    )
                }
            }
        }
    }
}

/**
 * The comparison that a human can actually settle.
 *
 * When two species are close, telling the user "91% vs 7%" is much less useful
 * than telling them the one feature that separates them, so this leads with the
 * difference and puts the numbers second.
 */
@Composable
private fun ComparisonPrompt(
    best: Candidate,
    runnerUp: Candidate,
    detail: SpeciesRepository.SpeciesDetail?,
) {
    val difference = detail?.similarSpecies
        ?.firstOrNull { it.taxon.id == runnerUp.taxon.id }?.difference
    Card(
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        ),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(Modifier.padding(16.dp)) {
            Text("Close call", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(6.dp))
            Text(
                "${best.taxon.scientificName} (${best.percent}%) vs "
                    + "${runnerUp.taxon.scientificName} (${runnerUp.percent}%)",
                style = MaterialTheme.typography.bodyLarge,
            )
            if (difference != null) {
                Spacer(Modifier.height(8.dp))
                Text("How to tell them apart", style = MaterialTheme.typography.labelLarge)
                Text(difference, style = MaterialTheme.typography.bodyMedium)
            }
        }
    }
}

@Composable
private fun CandidateRow(candidate: Candidate, onClick: () -> Unit) {
    Row(
        Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick)
            .padding(vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(
            Modifier
                .width(46.dp)
                .height(28.dp)
                .background(
                    MaterialTheme.colorScheme.surfaceVariant, RoundedCornerShape(6.dp)
                ),
            contentAlignment = Alignment.Center,
        ) {
            Text("${candidate.percent}%", style = MaterialTheme.typography.labelLarge)
        }
        Spacer(Modifier.width(12.dp))
        Column(Modifier.weight(1f)) {
            Text(
                candidate.taxon.commonName ?: candidate.taxon.scientificName,
                style = MaterialTheme.typography.bodyLarge,
            )
            Text(
                candidate.taxon.scientificName,
                style = MaterialTheme.typography.bodyMedium,
                fontStyle = FontStyle.Italic,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        // Make the geographic contribution visible rather than a hidden nudge.
        if (candidate.geoFactor != 1f) {
            Text(
                if (candidate.geoFactor > 1f) "local" else "unusual here",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
    Divider(color = MaterialTheme.colorScheme.surfaceVariant)
}

@Composable
private fun HowToVerify(detail: SpeciesRepository.SpeciesDetail) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp)) {
            Text("How to verify", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(8.dp))
            for (f in detail.diagnosticFeatures) {
                Row(Modifier.padding(vertical = 4.dp)) {
                    Text("•  ", style = MaterialTheme.typography.bodyLarge)
                    Column {
                        Text(f.feature, style = MaterialTheme.typography.bodyLarge)
                        f.bodyPart?.let {
                            Text(
                                it,
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun SimilarSpeciesCard(
    detail: SpeciesRepository.SpeciesDetail,
    onOpenSpecies: (Long) -> Unit,
) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp)) {
            Text("Similar species", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(8.dp))
            for (s in detail.similarSpecies) {
                Column(
                    Modifier
                        .fillMaxWidth()
                        .clickable { onOpenSpecies(s.taxon.id) }
                        .padding(vertical = 6.dp)
                ) {
                    Text(
                        s.taxon.commonName ?: s.taxon.scientificName,
                        style = MaterialTheme.typography.bodyLarge,
                    )
                    Text(s.difference, style = MaterialTheme.typography.bodyMedium)
                    s.confusionRate?.let {
                        Text(
                            "Confused with this species in ${Math.round(it * 100)}% "
                                + "of our test photographs.",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun EvidenceCard(
    detail: SpeciesRepository.SpeciesDetail,
    id: Identification,
) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp)) {
            Text("Habitat and range", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.height(8.dp))
            if (detail.habitats.isNotEmpty()) {
                Text(
                    detail.habitats.joinToString(", ") { it.first },
                    style = MaterialTheme.typography.bodyLarge,
                )
            }
            if (detail.regions.isNotEmpty()) {
                Spacer(Modifier.height(6.dp))
                for (r in detail.regions.take(4)) {
                    Text(
                        "${r.regionName} - ${r.observations} recorded sightings",
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
            }
            val iucn = detail.traits.firstOrNull { it.key == "iucn_status" }?.text
            if (iucn != null) {
                Spacer(Modifier.height(10.dp))
                Text("Conservation status", style = MaterialTheme.typography.labelLarge)
                Text(
                    iucn.replaceFirstChar { it.uppercase() },
                    style = MaterialTheme.typography.bodyLarge,
                )
            }
            if (detail.habitats.isEmpty() && detail.regions.isEmpty() && iucn == null) {
                // Absent rather than invented; say so plainly.
                Text(
                    "No sourced habitat or range information is available for this "
                        + "species in the installed pack.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun ProvenanceFooter(id: Identification) {
    Column(Modifier.fillMaxWidth().padding(top = 16.dp)) {
        Divider(color = MaterialTheme.colorScheme.surfaceVariant)
        Spacer(Modifier.height(8.dp))
        Text(
            "Identified entirely on this device. No photograph left your phone.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            "Pack ${id.packId} v${id.packVersion} • ${id.inferenceMillis} ms"
                + if (id.photoCount > 1) " • ${id.photoCount} photos" else "",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}
