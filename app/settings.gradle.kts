pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "fisherwiki"

// The identification engine: pure Kotlin/JVM, no Android types. Kept separate
// so it can be unit-tested on a desktop JVM against the same ONNX Runtime and
// the same model file the phone uses.
include(":core")

// The Android application. Thin UI + platform adapters over :core.
include(":android")

// Desktop manual-testing CLI. Same :core engine, same real .fwpack path, same
// preprocessing/calibration/taxonomy/open-set logic as the phone - a platform
// adapter over :core exactly like :android is, for when there is no device.
include(":cli")
