package com.fisherwiki.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.List
import androidx.compose.material.icons.filled.PhotoCamera
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavGraph.Companion.findStartDestination
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.fisherwiki.app.ui.AboutScreen
import com.fisherwiki.app.ui.CatchLogScreen
import com.fisherwiki.app.ui.FisherWikiTheme
import com.fisherwiki.app.ui.HomeScreen
import com.fisherwiki.app.ui.IdentifyScreen
import com.fisherwiki.app.ui.PacksScreen
import com.fisherwiki.app.ui.SettingsScreen
import com.fisherwiki.app.ui.SpeciesDetailScreen
import com.fisherwiki.app.ui.IdentifyViewModel

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            FisherWikiTheme {
                AppShell()
            }
        }
    }
}

private enum class Tab(val route: String, val label: String, val icon: ImageVector) {
    Home("home", "Home", Icons.Default.Home),
    Identify("identify", "Identify", Icons.Default.PhotoCamera),
    Catches("catches", "Catches", Icons.Default.List),
    Packs("packs", "Packs", Icons.Default.Download),
    Settings("settings", "Settings", Icons.Default.Settings),
}

@Composable
private fun AppShell() {
    val nav = rememberNavController()
    val backStack by nav.currentBackStackEntryAsState()
    val current = backStack?.destination

    Scaffold(
        bottomBar = {
            NavigationBar {
                for (tab in Tab.entries) {
                    NavigationBarItem(
                        selected = current?.hierarchy?.any { it.route == tab.route } == true,
                        onClick = {
                            nav.navigate(tab.route) {
                                popUpTo(nav.graph.findStartDestination().id) {
                                    saveState = true
                                }
                                launchSingleTop = true
                                restoreState = true
                            }
                        },
                        icon = { Icon(tab.icon, contentDescription = tab.label) },
                        label = { Text(tab.label) },
                    )
                }
            }
        }
    ) { inner ->
        NavHost(
            navController = nav,
            startDestination = Tab.Home.route,
            modifier = Modifier.padding(inner),
        ) {
            composable(Tab.Home.route) {
                HomeScreen(
                    onIdentify = { nav.navigate(Tab.Identify.route) },
                    onOpenPacks = { nav.navigate(Tab.Packs.route) },
                    onOpenSpecies = { id -> nav.navigate("species/$id") },
                )
            }
            composable(Tab.Identify.route) {
                val vm: IdentifyViewModel = viewModel()
                IdentifyScreen(
                    vm = vm,
                    onOpenSpecies = { id -> nav.navigate("species/$id") },
                    onOpenPacks = { nav.navigate(Tab.Packs.route) },
                )
            }
            composable(Tab.Catches.route) {
                CatchLogScreen(onOpenSpecies = { id -> nav.navigate("species/$id") })
            }
            composable(Tab.Packs.route) { PacksScreen() }
            composable(Tab.Settings.route) {
                SettingsScreen(onOpenAbout = { nav.navigate("about") })
            }
            composable("about") { AboutScreen() }
            composable("species/{taxonId}") { entry ->
                val id = entry.arguments?.getString("taxonId")?.toLongOrNull()
                SpeciesDetailScreen(
                    taxonId = id,
                    onOpenSpecies = { other -> nav.navigate("species/$other") },
                )
            }
        }
    }
}
