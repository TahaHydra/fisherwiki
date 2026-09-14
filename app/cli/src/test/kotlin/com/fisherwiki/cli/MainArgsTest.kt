package com.fisherwiki.cli

import com.google.common.truth.Truth.assertThat
import java.io.File
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir

/**
 * `parseArgs` returns [ArgsResult] instead of calling `exitProcess` directly
 * (see the kdoc on [ArgsResult]) specifically so its flag/value consumption,
 * numeric validation and the `--lat`/`--lon` pairing rule can be exercised
 * here without terminating the test JVM.
 */
class MainArgsTest {

    private fun ok(vararg args: String): CliArgs {
        val r = parseArgs(arrayOf(*args, "--pack", "somepack.fwpack"))
        assertThat(r).isInstanceOf(ArgsResult.Ok::class.java)
        return (r as ArgsResult.Ok).args
    }

    private fun errorMessage(vararg args: String): String {
        val r = parseArgs(arrayOf(*args))
        assertThat(r).isInstanceOf(ArgsResult.Error::class.java)
        return (r as ArgsResult.Error).message
    }

    @Test
    fun `no arguments shows help rather than erroring`() {
        assertThat(parseArgs(arrayOf())).isEqualTo(ArgsResult.ShowHelp)
    }

    @Test
    fun `--help and -h show help regardless of position`() {
        assertThat(parseArgs(arrayOf("--help"))).isEqualTo(ArgsResult.ShowHelp)
        assertThat(parseArgs(arrayOf("photo.jpg", "-h"))).isEqualTo(ArgsResult.ShowHelp)
    }

    @Test
    fun `a bare image path with defaults`() {
        val args = ok("photo.jpg")
        assertThat(args.target.path).isEqualTo("photo.jpg")
        assertThat(args.threads).isEqualTo(4)
        assertThat(args.topK).isEqualTo(5)
        assertThat(args.recursive).isFalse()
        assertThat(args.language).isEqualTo("en")
        assertThat(args.lat).isNull()
        assertThat(args.lon).isNull()
    }

    @Test
    fun `every flag overrides its default`() {
        val args = ok(
            "photo.jpg",
            "--lat", "52.1", "--lon", "5.2",
            "--threads", "8", "--topk", "10",
            "--lang", "fr", "--recursive",
        )
        assertThat(args.lat).isEqualTo(52.1)
        assertThat(args.lon).isEqualTo(5.2)
        assertThat(args.threads).isEqualTo(8)
        assertThat(args.topK).isEqualTo(10)
        assertThat(args.language).isEqualTo("fr")
        assertThat(args.recursive).isTrue()
    }

    @Test
    fun `flag order does not matter`() {
        val a = ok("--recursive", "--threads", "2", "photo.jpg")
        assertThat(a.target.path).isEqualTo("photo.jpg")
        assertThat(a.threads).isEqualTo(2)
        assertThat(a.recursive).isTrue()
    }

    @Test
    fun `an explicit --pack overrides the default path`() {
        val r = parseArgs(arrayOf("photo.jpg", "--pack", "custom.fwpack"))
        assertThat(r).isInstanceOf(ArgsResult.Ok::class.java)
        assertThat((r as ArgsResult.Ok).args.packPath.path).isEqualTo("custom.fwpack")
    }

    @Test
    fun `omitting --pack uses the default path when present, else errors`() {
        val result = parseArgs(arrayOf("photo.jpg"))
        if (File(DEFAULT_PACK_PATH).isFile) {
            assertThat(result).isInstanceOf(ArgsResult.Ok::class.java)
            assertThat((result as ArgsResult.Ok).args.packPath.path).isEqualTo(DEFAULT_PACK_PATH)
        } else {
            assertThat(result).isInstanceOf(ArgsResult.Error::class.java)
            assertThat((result as ArgsResult.Error).message).contains("--pack is required")
        }
    }

    @Test
    fun `no positional argument is an error`() {
        assertThat(errorMessage("--pack", "p.fwpack")).contains("got 0")
    }

    @Test
    fun `two positional arguments is an error, not a silent pick of one`() {
        assertThat(errorMessage("a.jpg", "b.jpg", "--pack", "p.fwpack")).contains("got 2")
    }

    @Test
    fun `--lat without --lon is an error`() {
        assertThat(errorMessage("photo.jpg", "--pack", "p.fwpack", "--lat", "52.1"))
            .contains("--lat and --lon must be given together")
    }

    @Test
    fun `--lon without --lat is an error`() {
        assertThat(errorMessage("photo.jpg", "--pack", "p.fwpack", "--lon", "5.2"))
            .contains("--lat and --lon must be given together")
    }

    @Test
    fun `a non-numeric --lat is an error, not a silent zero`() {
        assertThat(errorMessage("photo.jpg", "--pack", "p.fwpack", "--lat", "north", "--lon", "5.2"))
            .contains("--lat needs a number")
    }

    @Test
    fun `a non-integer --threads is an error`() {
        assertThat(errorMessage("photo.jpg", "--pack", "p.fwpack", "--threads", "many"))
            .contains("--threads needs an integer")
    }

    @Test
    fun `a flag missing its value is an error, not an index-out-of-bounds crash`() {
        assertThat(errorMessage("photo.jpg", "--pack")).contains("--pack needs a value")
    }

    @Test
    fun `a value-shaped token after a value-taking flag is consumed as the value`() {
        // "--recursive" here is the *value* of --lang, not a separate flag -
        // regression guard for the explicit index-arithmetic parser.
        val args = ok("photo.jpg", "--lang", "--recursive")
        assertThat(args.language).isEqualTo("--recursive")
        assertThat(args.recursive).isFalse()
    }

    // ------------------------------------------------------ discoverImages

    @Test
    fun `a single image file is found directly`(@TempDir tmp: File) {
        val f = File(tmp, "fish.JPG").apply { writeText("x") }
        assertThat(discoverImages(f, recursive = false)).containsExactly(f)
    }

    @Test
    fun `a non-image file given directly is not an image`(@TempDir tmp: File) {
        val f = File(tmp, "notes.txt").apply { writeText("x") }
        assertThat(discoverImages(f, recursive = false)).isEmpty()
    }

    @Test
    fun `a folder finds jpg jpeg and png, case-insensitively, sorted`(@TempDir tmp: File) {
        for (name in listOf("c.png", "a.JPG", "b.jpeg", "ignore.txt", "ignore.gif")) {
            File(tmp, name).writeText("x")
        }
        val found = discoverImages(tmp, recursive = false).map { it.name }
        assertThat(found).containsExactly("a.JPG", "b.jpeg", "c.png").inOrder()
    }

    @Test
    fun `a folder without --recursive does not look in subfolders`(@TempDir tmp: File) {
        File(tmp, "top.jpg").writeText("x")
        val sub = File(tmp, "sub").apply { mkdir() }
        File(sub, "nested.jpg").writeText("x")

        assertThat(discoverImages(tmp, recursive = false).map { it.name }).containsExactly("top.jpg")
        assertThat(discoverImages(tmp, recursive = true).map { it.name })
            .containsExactly("top.jpg", "nested.jpg")
    }
}
