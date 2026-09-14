# Security

## Reporting a vulnerability

Open a GitHub security advisory, or email the maintainers. Please do not open a
public issue for anything exploitable.

Findings that contradict any claim in [`PRIVACY.md`](PRIVACY.md) — for example
evidence that a photograph, a location or any identifier leaves the device
outside an explicit user-initiated export — are treated as security issues and
prioritised accordingly.

---

## 1. Threat model

FisherWiki has an unusually small attack surface because it does almost nothing
online. What remains:

| Asset | Threat | Mitigation |
|---|---|---|
| **Offline packs** | A malicious `.fwpack` from a CDN, a sideload, or a file the user was sent | §2 — the main one |
| **Catch log** | Another app, or a backup service, reading fishing locations | app-private storage; all backup domains excluded |
| **Photographs** | Exfiltration | no network code in the identification path; no analytics or ad SDK in the dependency graph |
| **Model output** | An identification the user trusts more than it deserves | calibration + three-signal rejection + coarse fallback (this is a *safety* property as much as a security one) |
| **Build pipeline** | A poisoned training image | licence-gated provenance store; every image traceable to a source record |

Explicitly **out of scope**: a rooted device, a malicious OS, and physical
access to an unlocked phone. None of those can be defended against by an app.

---

## 2. Packs are untrusted input

This is where essentially all of the risk lives, because a pack is a binary blob
the app is asked to trust with naming things to a user.

### 2.1 Nothing in a pack is code

A pack contains a model, a database, a binary histogram, a JSON manifest and a
CSV. None of them is ever executed, loaded as a library, added to a class path,
or used to resolve a path to load code from. There is no plugin mechanism, no
scripting, and no field whose value becomes an executable path.

### 2.2 Archive extraction

Implemented in [`SafeZip`](../app/core/src/main/kotlin/com/fisherwiki/core/pack/SafeZip.kt).
Each guard maps to a real, published attack:

| Attack | Guard |
|---|---|
| **Zip Slip** — `../../databases/app.db` | entry names rejected if absolute, drive-qualified (`C:`), containing a backslash, a `..` or `.` segment, an empty segment, or NUL; **and** the resolved canonical path must lie inside the destination |
| **Zip bomb** | per-entry cap 512 MB, total cap 2 GB, both enforced **while streaming**. A crafted archive can declare a small uncompressed size and then emit gigabytes, so the header value is only used as an early reject |
| **Declared-ratio abuse** | entries declaring >200:1 compression are rejected before any I/O |
| **Entry-count exhaustion** | 256 entries maximum |
| **Symlink entries** | `java.util.zip` exposes no unix mode and this extractor calls no link-creating API, so a symlink entry becomes an ordinary file containing the target string — inert. We do not attempt to honour them, because honouring them is the only way they become dangerous |
| **Unlisted payload** | extraction takes the manifest's entry names as an allow-list, so a pack cannot smuggle a file past verification by omitting it from the manifest |
| **Oversized sideload** | 1.5 GB cap while copying a `content://` URI into the cache |

### 2.3 Manifest validation

Implemented in [`PackVerifier`](../app/core/src/main/kotlin/com/fisherwiki/core/pack/PackVerifier.kt).
Order matters: each check is cheap relative to the next.

1. `manifest.json` read alone, bounded to 4 MB.
2. `format_version` and `min_engine_version` checked **before** anything else is
   interpreted.
3. Structural checks: `pack_id` against `^[a-z0-9_]{1,64}$`, plausible
   `input_size` and `num_classes`, 3-channel `input_std` with no zero (a zero
   would divide by zero during preprocessing), known `quantization`,
   calibration thresholds in range, `sha256` fields matching `^[0-9a-f]{64}$`,
   declared sizes within limits, no duplicate paths,
   `corpus.class_count <= model.num_classes`.
4. Extraction.
5. **Exact byte length and SHA-256** verified for every payload file.

JSON parsing is strict — unknown keys are an error, not ignored. A pack carrying
fields we do not understand may rely on semantics we will not apply.

### 2.4 Atomic installation

Extraction goes to a staging directory; only a fully verified result is moved
into place. An interrupted or failed install therefore cannot leave a
half-extracted pack that the engine would later load and trust. After the move,
sizes are re-checked from the final location rather than trusting that a
cross-filesystem copy was faithful.

### 2.5 Runtime consistency

`IdentificationEngine.open()` refuses to start if the manifest's `num_classes`
disagrees with the database's `model_classes` count. That mismatch would not
crash — it would quietly name the wrong species, which is worse.

Class indices are required to be **dense from 0**. A gap would shift the meaning
of every index after it.

---

## 3. Database handling

* The pack database is opened **read-only** (`SQLiteDatabase.OPEN_READONLY`). It
  is an immutable hash-verified artefact; writing to it would break
  re-verification and could corrupt data whose whole value is being exactly what
  shipped.
* The catch log is a **separate** writable database, so a pack install or
  upgrade can never touch user data.
* `onDowngrade` **throws**. `SQLiteOpenHelper`'s default is to delete the
  database and start over, which for a catch log means destroying records the
  user cannot recreate.
* All queries are parameterised. No SQL is built by string concatenation with
  user input; the only interpolated values are integer placeholders counts
  generated from list lengths.

---

## 4. Supply chain

* Dependencies are pinned in a version catalogue
  ([`libs.versions.toml`](../app/gradle/libs.versions.toml)) rather than
  floating.
* The dependency graph contains **no** analytics, crash-reporting or advertising
  library. This is verifiable with `./gradlew :android:dependencies` and is what
  makes the privacy claims checkable rather than promissory.
* Python dependencies are pinned in [`requirements.txt`](../requirements.txt).
* The GBIF backbone is pinned to a specific dated release and its SHA-256 is
  recorded, not fetched from a moving `current/` alias.
* Training-data provenance: every image in the corpus has a row recording its
  source, record id, URL, creator, licence and download timestamp. A poisoned or
  mislabelled image can be traced to its origin and removed by hash.

---

## 5. Network

**The application has none.** It declares no `INTERNET` permission and links no
HTTP client, so the platform will not let any code in the process open a socket.
Packs arrive as files the user chooses.

This is a deliberate reduction of the attack surface as well as a privacy
property: with no network path, a malicious pack cannot exfiltrate anything even
if it manages to influence execution, and there is no transport to attack.

It also removes a class of threat rather than mitigating it. The intended
product downloads packs, and when that is built this section gets substantially
longer — TLS, endpoint trust, resume-state handling, partial-download validation
— and `INTERNET` returns to the manifest. Any change to this section should be
reviewed against that manifest diff.

The dataset tooling (`tools/`) makes many more network requests, but it runs on
a build machine, never on a user's device. It uses official bulk endpoints only,
rate-limits per host, honours `Retry-After`, and identifies itself with a
contactable User-Agent.

---

## 6. Known gaps

Stated rather than omitted:

* **Packs are not signed.** Integrity is verified (SHA-256 per file against the
  manifest) but authenticity is not: a manifest and its payload can both be
  replaced by an attacker who controls the distribution channel. Detached
  signature verification with a pinned public key is the intended v2 work. Until
  then, a sideloaded pack is exactly as trustworthy as the person who gave it to
  you, and the app says so.
* **No in-app pack download**, so no transport security to speak of — see §5.
  Whatever channel a user obtains a pack through is outside this threat model,
  which is precisely why content verification does not rely on transport.
* **The model itself is an attack surface** in the sense that a malicious pack
  could ship a model that is simply wrong about dangerous species. Hash
  verification proves a pack is *intact*, not that it is *correct*. Signing
  addresses this; so does only installing packs from a source you trust.
* **No instrumented device testing yet.** The security properties above are
  covered by JVM unit tests (adversarial archives, traversal, bombs, allow-list,
  hash mismatch), but have not been exercised on a physical Android device. See
  [`README.md`](../README.md) for the current verification status.
