package com.fisherwiki.app.ui

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

/**
 * Visual language: deep water blues with a warm accent.
 *
 * The palette is chosen for one practical reason above aesthetics - this app is
 * used outdoors, often in bright sun, sometimes at dusk with wet hands. So it
 * favours high contrast and large touch targets over subtlety, and the
 * uncertainty colours are deliberately unmistakable rather than tasteful.
 */

private val DeepWater = Color(0xFF0E3A53)
private val Shallow = Color(0xFF1E6F8E)
private val Foam = Color(0xFFE6F1F5)
private val Sand = Color(0xFFD9A441)
private val Kelp = Color(0xFF2E6B4F)

/** Certainty colours, used by result surfaces. Not part of the M3 scheme. */
object CertaintyColors {
    val confident = Color(0xFF1F7A4D)
    val ambiguous = Color(0xFFB8730B)
    val uncertain = Color(0xFF8A4B12)
    val unknown = Color(0xFF5B5B5B)
    val danger = Color(0xFFA8231C)

    val confidentContainer = Color(0xFFDCF0E5)
    val ambiguousContainer = Color(0xFFFBEEDA)
    val uncertainContainer = Color(0xFFF6E7DA)
    val unknownContainer = Color(0xFFECECEC)
    val dangerContainer = Color(0xFFFBE3E1)
}

private val LightColors = lightColorScheme(
    primary = DeepWater,
    onPrimary = Color.White,
    primaryContainer = Foam,
    onPrimaryContainer = DeepWater,
    secondary = Shallow,
    onSecondary = Color.White,
    tertiary = Sand,
    onTertiary = Color(0xFF2A1C00),
    background = Color(0xFFFBFCFD),
    onBackground = Color(0xFF171C1F),
    surface = Color.White,
    onSurface = Color(0xFF171C1F),
    surfaceVariant = Color(0xFFE9EEF1),
    onSurfaceVariant = Color(0xFF41484D),
    error = CertaintyColors.danger,
)

private val DarkColors = darkColorScheme(
    primary = Color(0xFF8FCCE6),
    onPrimary = Color(0xFF00344A),
    primaryContainer = Color(0xFF004C69),
    onPrimaryContainer = Color(0xFFC4E7F8),
    secondary = Color(0xFF8FCCE6),
    tertiary = Color(0xFFE9C06B),
    background = Color(0xFF0F1416),
    onBackground = Color(0xFFDFE3E6),
    surface = Color(0xFF161C1F),
    onSurface = Color(0xFFDFE3E6),
    surfaceVariant = Color(0xFF3F484C),
    onSurfaceVariant = Color(0xFFBFC8CC),
    error = Color(0xFFFFB4AB),
)

private val AppTypography = Typography(
    headlineLarge = TextStyle(fontSize = 30.sp, fontWeight = FontWeight.SemiBold),
    headlineMedium = TextStyle(fontSize = 24.sp, fontWeight = FontWeight.SemiBold),
    titleLarge = TextStyle(fontSize = 21.sp, fontWeight = FontWeight.SemiBold),
    titleMedium = TextStyle(fontSize = 17.sp, fontWeight = FontWeight.Medium),
    bodyLarge = TextStyle(fontSize = 16.sp, lineHeight = 24.sp),
    bodyMedium = TextStyle(fontSize = 14.sp, lineHeight = 20.sp),
    labelLarge = TextStyle(fontSize = 14.sp, fontWeight = FontWeight.Medium),
)

@Composable
fun FisherWikiTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    /**
     * Material You is off by default. A wallpaper-derived palette can make the
     * certainty colours ambiguous, and "how sure is this" must not depend on
     * the user's wallpaper.
     */
    dynamicColor: Boolean = false,
    content: @Composable () -> Unit,
) {
    val colors = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val ctx = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(ctx) else dynamicLightColorScheme(ctx)
        }
        darkTheme -> DarkColors
        else -> LightColors
    }
    MaterialTheme(colorScheme = colors, typography = AppTypography, content = content)
}
