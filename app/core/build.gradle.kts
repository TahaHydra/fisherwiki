import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.serialization)
}

// Target Java 17 bytecode without demanding a JDK 17 *toolchain*. Using
// jvmToolchain(17) would force every contributor to have that exact JDK
// installed or let Gradle download one; targeting 17 from whatever modern JDK
// is present is equivalent for our purposes and builds on a stock Android
// Studio install (which bundles JBR 21).
kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
        freeCompilerArgs.addAll("-Xjvm-default=all")
    }
}

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

tasks.withType<JavaCompile>().configureEach {
    options.release.set(17)
}


// The pack species schema lives in tools/fwdata/species_schema.sql and is the
// single source of truth. Copying it into test resources means the Kotlin
// fixtures are built from exactly the DDL that ships, so the two cannot drift.
tasks.named<ProcessResources>("processTestResources") {
    from(rootProject.layout.projectDirectory.dir("../tools/fwdata")) {
        include("species_schema.sql")
        into("schema")
    }
}

dependencies {
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.kotlinx.coroutines.core)

    // compileOnly: :core codes against the ai.onnxruntime API but does not
    // bundle a native runtime. The Android module supplies onnxruntime-android
    // and the tests below supply the desktop build, so exactly one native
    // implementation is present in any given configuration.
    compileOnly(libs.onnxruntime.jvm)

    testImplementation(libs.onnxruntime.jvm)
    testImplementation(libs.sqlite.jdbc)
    testImplementation(libs.junit.jupiter)
    testImplementation(libs.truth)
    testRuntimeOnly(libs.junit.platform.launcher)
}

// ONNX Runtime's Windows native library fails to initialise under the
// JetBrains Runtime that Android Studio bundles:
//
//   UnsatisfiedLinkError: onnxruntime.dll: A dynamic link library (DLL)
//   initialization routine failed
//
// The identical jar loads fine on a stock Temurin JDK, and the Android app is
// unaffected (it uses onnxruntime-android on the device runtime), so this is
// purely a test-JVM problem. Prefer a non-JBR toolchain for tests when one is
// discoverable, and fall back to the default rather than failing the build for
// someone whose only JDK is the bundled one.
val nonJbrLauncher = listOf(25, 24, 23, 22, 21, 17).firstNotNullOfOrNull { major ->
    runCatching {
        javaToolchains.launcherFor {
            languageVersion.set(JavaLanguageVersion.of(major))
            vendor.set(JvmVendorSpec.ADOPTIUM)
        }.get()
    }.getOrNull()
}

tasks.test {
    nonJbrLauncher?.let { javaLauncher.set(it) }
    useJUnitPlatform()
    testLogging {
        events("passed", "skipped", "failed")
        showStandardStreams = false
    }
}
