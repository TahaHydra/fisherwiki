import org.gradle.jvm.toolchain.JavaLanguageVersion
import org.gradle.jvm.toolchain.JvmVendorSpec
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.jvm)
    application
}

// Same reasoning as :core/build.gradle.kts: target 17 bytecode without
// demanding a JDK 17 toolchain for the *compiler*. The runtime toolchain
// requirement (Temurin 25, below) is a separate, stricter concern that only
// applies to actually launching the JVM process, not to compiling against it.
kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

dependencies {
    // The real production engine. Nothing in this module reimplements
    // preprocessing, calibration, ranking, taxonomy resolution or open-set
    // rejection - all of that is :core, unmodified, the same jar the Android
    // app and :core's own tests use.
    implementation(project(":core"))

    // :core only *compiles* against the ONNX Runtime API (`compileOnly` in
    // :core/build.gradle.kts); every consumer supplies its own native runtime.
    // Android supplies onnxruntime-android on device; this is the identical
    // desktop artefact :core's own tests use, so `ai.onnxruntime.*` behaves
    // exactly like it does under test, not like a separate implementation.
    implementation(libs.onnxruntime.jvm)

    // Runtime backing for JdbcSqliteDriver (see core/db/JdbcSqliteDriver.kt):
    // registers itself with java.sql.DriverManager, no compile-time reference.
    implementation(libs.sqlite.jdbc)

    testImplementation(libs.junit.jupiter)
    testImplementation(libs.truth)
    testRuntimeOnly(libs.junit.platform.launcher)
}

application {
    mainClass.set("com.fisherwiki.cli.MainKt")
    // ONNX Runtime's native loading trips the JDK's "restricted method called"
    // warning on Java 22+ (JEP 472). It is harmless and not something this
    // project's code can fix - the call is inside onnxruntime's own native
    // binding - so it is silenced here rather than left to alarm every run.
    applicationDefaultJvmArgs = listOf("--enable-native-access=ALL-UNNAMED")
}

tasks.test {
    useJUnitPlatform()
}

// -----------------------------------------------------------------------
// ONNX Runtime's Windows native library fails to initialise under the
// JetBrains Runtime that Android Studio bundles - see :core/build.gradle.kts
// for the exact UnsatisfiedLinkError. :core's tests work around this by
// preferring any discoverable non-JBR toolchain and silently falling back
// otherwise, because a failing unit test is a clear enough signal on its own.
//
// This CLI is different: it is meant to be run directly by a person, on
// demand, often after weeks away from the project. A silent fallback here
// would surface as a cryptic UnsatisfiedLinkError three frames deep inside
// ONNX Runtime, with no clue that the JDK is the actual problem. So `run`
// fails at Gradle configuration time instead, with an actionable message,
// and requires specifically Temurin 25 - the exact build validated (see
// docs/engineering-log.md) to load onnxruntime.dll correctly on this stack.
// -----------------------------------------------------------------------
val temurin25 = runCatching {
    javaToolchains.launcherFor {
        languageVersion.set(JavaLanguageVersion.of(25))
        vendor.set(JvmVendorSpec.ADOPTIUM)
    }.get()
}.getOrNull()

tasks.named<JavaExec>("run") {
    javaLauncher.set(
        temurin25 ?: throw GradleException(
            "Eclipse Temurin 25 JDK not found.\n\n" +
                "This CLI loads ONNX Runtime's native library directly, which fails " +
                "under the JetBrains Runtime that Android Studio bundles " +
                "(UnsatisfiedLinkError: onnxruntime.dll). Install Temurin 25 from\n" +
                "  https://adoptium.net/temurin/releases/?version=25\n" +
                "Gradle's toolchain auto-detection finds installed JDKs on its own " +
                "(no JAVA_HOME change needed) once it is on disk."
        )
    )
}
