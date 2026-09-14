# Privacy

FisherWiki is offline-first because that is the only way to make a privacy
promise that does not depend on trusting us.

This document states what the app does, and — more usefully — **how you can
check it yourself**, since a privacy policy nobody can verify is just a
sentence.

---

## 1. The promises

* **No account.** There is no sign-up, no login, no user id, no device id.
* **No cloud inference.** Recognition runs entirely on your phone. It works in
  airplane mode, and it works in a valley with no signal.
* **No telemetry.** No analytics, no crash reporting, no "anonymous usage
  statistics", no remote configuration.
* **No advertising SDK.**
* **Photographs never leave your phone.** Not for identification, not for
  "improving the model", not ever, unless you explicitly export or share one.
* **Location stays local.** It is off by default, used only to rank candidates
  on-device, and never transmitted.
* **Your corrections stay local.** When you tell the app it got a fish wrong,
  that correction is stored on your phone and is never uploaded automatically.

---

## 2. What the app can access, and why

| Permission | Requested when | What it is used for | Works without it? |
|---|---|---|---|
| `CAMERA` | you first take a photograph | capturing a photo to identify | yes, use the gallery instead |
| `READ_MEDIA_IMAGES` | you first import a photo | reading the photo you picked | yes, use the camera instead |
| `ACCESS_COARSE_LOCATION` / `ACCESS_FINE_LOCATION` | you enable location ranking | ranking candidates on-device | **yes** — identification is fully functional without it |
`INTERNET` is **not requested**. Neither is `ACCESS_NETWORK_STATE`,
`ACCESS_BACKGROUND_LOCATION`, contacts, phone state, accounts or
nearby-devices.

### Why there is no `INTERNET` permission

Because the app cannot use the network, so asking for permission to would be
asking for capability it does not need.

This is stronger than a promise. Android enforces `INTERNET` at the socket
layer: without it, no code in the process can open a connection — not the app,
not a library, not something added by a future dependency without that removal
being visible in the manifest diff. You can check it yourself in
[`AndroidManifest.xml`](../app/android/src/main/AndroidManifest.xml) in about
ten seconds, which is the point.

Two corroborating checks if you want them. The dependency list in
[`app/android/build.gradle.kts`](../app/android/build.gradle.kts) contains no
HTTP client, no analytics library, no crash reporter and no ad SDK — confirm
with `./gradlew :android:dependencies`. And the whole app works in airplane
mode, because there is no path through it that does not.

### Check the APK, not this document

Reading the source manifest is not sufficient, and we know that because it was
not sufficient here.

Android's manifest merger unions in the manifest of every dependency.
`onnxruntime-android` — the inference library, the one component that most
obviously has no business on the network — declares `INTERNET` and
`ACCESS_NETWORK_STATE` in its own manifest. So the built APK **carried both**,
in an app requesting neither, until an explicit `tools:node="remove"` was added.

For a while, then, this page could have told you the app had no network access
while the artefact on your phone had exactly that. The lesson is that the
verifiable object is the APK:

```bash
python scripts/check_apk_permissions.py --apk <your.apk>
```

That reads the permission set out of the binary manifest inside the APK and
fails on anything not explicitly allowed — including permissions a future
dependency introduces without anyone noticing. It is pinned by
`tests/test_apk_permissions.py`, which cross-checks the parser against Gradle's
own merged manifest so that a passing run means something.

**The trade-off, stated plainly:** packs cannot be downloaded inside the app.
You install them from a file (see §3). Downloading packs over the network was
the intended design and is not built; when it is, this permission has to come
back, and that will be a visible change to this file and to the manifest.

---

## 3. What is stored on your phone

| Data | Location | Notes |
|---|---|---|
| Installed packs | app-private `files/packs/` | read-only once verified |
| Catch log | app-private `catch_log.db` | separate database from any pack |
| Catch photographs | app-private `files/catches/` | copied from the gallery so a later gallery deletion does not orphan the record |
| Settings | app-private `SharedPreferences` | |

All of it is in the app's private storage. Other apps cannot read it, and
uninstalling removes all of it.

### Backups are disabled

[`data_extraction_rules.xml`](../app/android/src/main/res/xml/data_extraction_rules.xml)
excludes **every** domain from both cloud backup and device-to-device transfer.

That is a deliberate trade. It means a new phone starts empty. It also means
your catch log — which can contain the coordinates of your fishing spots —
is never copied to a backup service you did not think about. For an app whose
whole point is that your data stays where you put it, one simple rule is worth
more than a convenience.

---

## 4. Location, specifically

Location is the most sensitive thing this app touches, so it is handled with
more care than the rest:

* **Two separate settings, both off by default.** "Use location to rank
  candidates" and "Save location with catches" are independent. You can use
  location to improve identification without recording where you were.
* **Asked in context.** The app does not demand location on first launch, when
  you have no idea why it would help. It asks the first time you photograph a
  fish, and explains what it does with it.
* **Used on-device only.** Ranking consults a lookup table shipped inside the
  pack. No request is made, so there is nothing to intercept.
* **Never the deciding factor.** The geographic prior is a bounded multiplier,
  clamped so it can never eliminate a species, and it is attenuated further when
  the model is visually confident. If you have genuinely caught something that
  "shouldn't be there", the app will still tell you so.
* **Exports omit it by default.** Exporting the catch log asks before including
  coordinates, and the default is to leave them out — an export is far more
  likely to be shared than the database on your phone.

---

## 5. Your corrections

When the app is wrong and you correct it, that correction is the single most
valuable signal for improving the model. It is stored locally, in your catch
log, and it stays there.

There is no automatic upload. If a voluntary contribution flow is added later it
will be:

* **opt-in per export**, never a background sync;
* **reviewable** — you see exactly what would be sent before it is sent;
* **location-free by default**;
* **photograph-optional** — a correction is useful even as
  `(predicted, corrected)` with no image attached.

Until such a flow exists and you use it, `exportJson()` writing a file you chose
to write is the only way a correction leaves the device.

---

## 6. How to verify all of this

1. **Airplane mode.** Turn it on. Identify a fish. It works. That is the whole
   claim, demonstrated.
2. **Inspect the manifest.**
   `aapt dump permissions android-arm64-v8a-release.apk` lists exactly the
   permissions in the table above.
3. **Inspect the dependencies.** `./gradlew :android:dependencies` — there is no
   analytics or ads library to find.
4. **Watch the network.** Run the app behind a proxy or with a network monitor.
   Identification generates no traffic.
5. **Read the code.** The whole identification path is
   [`IdentificationEngine`](../app/core/src/main/kotlin/com/fisherwiki/core/engine/IdentificationEngine.kt)
   and the four classes it calls. There is no HTTP client anywhere in `:core`.
6. **Build it yourself.** See [`README.md`](../README.md). The build is
   reproducible from this repository.

---

## 7. What we cannot promise

Being straight about the limits:

* **Your operating system is not ours.** Android may back up, index or log
  things at a level the app does not control. Disabling app backup is what we
  can do; we cannot audit the whole OS for you.
* **Photograph EXIF.** Photographs you took with your own camera app may
  already contain GPS coordinates in their EXIF. The app does not add location
  to a photograph, but if you share a photo you took, whatever your camera put
  in it goes too.
* **Getting a pack onto the phone happens outside this app.** The app has no
  network access, so you download a pack with a browser or copy it over USB.
  Whatever you use for that has its own privacy properties — a browser download
  shows a server your IP address exactly as any download does. Nothing about
  that is worse than an in-app download would be; it is just somewhere this
  document cannot make promises for you.
* **This is software.** It can have bugs. If you find one that contradicts
  anything above, that is a security issue — see
  [`SECURITY.md`](SECURITY.md).
